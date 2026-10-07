"""pe81 cycle 2: does the outline of the blank predict the text? random pairs, stratified shuffles, held-out halves, planted."""
import sys, os, json, csv, hashlib, re
import numpy as np
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe81_engine as E
import common

SCR = sys.argv[1]  # scratchpad holding cdli_cat.csv
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
rng = np.random.default_rng(812)
SH = json.load(open(os.path.join(E.CK, 'shapes.json')))
SF = ['rect', 'row_solid', 'corner_fill', 'corner_top_vs_bottom', 'taper', 'side_bulge', 'vbulge', 'lr_asym', 'pillow']

csv.field_size_limit(10 ** 9)
cat = {}
for x in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8', errors='replace')):
    if x.get('period', '').startswith('Proto-Elamite'):
        cat['P%06d' % int(x['id_text'])] = x


def num(s):
    try:
        return float(s)
    except Exception:
        return None


T = common.load()
rows = [('Susa', t) for t in T if t['lines']]
X, names, top = E.features(rows)
keep = []; S = []; C = []; mus = []
for i, (_, t) in enumerate(rows):
    sh = SH.get(t['id']); c = cat.get(t['id'])
    if not sh or sh['pillow'] is None or not c:
        continue
    h, w, th = num(c['height']), num(c['width']), num(c['thickness'])
    if not h or not w or not th:
        continue
    keep.append(i); S.append([sh[k] for k in SF])
    fx = X[i, names.index('frac_x')]
    C.append([np.log(h * w), np.log(h / w), th / w, fx, sh['aspect_px']])
    m = re.search(r'(\d+)', c['museum_no'] or ''); mus.append(int(m.group(1)) if m else rng.integers(10 ** 6))
keep = np.array(keep); S = np.array(S, float); C = np.array(C, float); mus = np.array(mus)
X = X[keep]
# sanity: photo aspect vs catalogue aspect
san = spearmanr(C[:, 4], C[:, 1]).correlation
print('tablets', len(keep), 'photo-vs-catalogue aspect rho', round(san, 3), flush=True)
# drop constant text features
v = X.std(0) > 0; X = X[:, v]; names = [n for n, k in zip(names, v) if k]
cov = np.column_stack([np.ones(len(C)), C, C[:, :2] ** 2])


def resid(Y):
    b, *_ = np.linalg.lstsq(cov, Y, rcond=None)
    return Y - cov @ b


Xr = resid(rankdata(X, axis=0)); Sr = resid(rankdata(S, axis=0))
# strata for shuffles: size quartile x damage (x>0)
q = np.digitize(C[:, 0], np.quantile(C[:, 0], [0.25, 0.5, 0.75])) * 2 + (C[:, 3] > 0)
# train / held-out by museum-number blocks of 10
blk = mus // 10; ub = np.unique(blk); rng.shuffle(ub); trainb = set(ub[: len(ub) // 2])
tr = np.array([b in trainb for b in blk]); te = ~tr
print('train', tr.sum(), 'test', te.sum(), flush=True)

P = []
for _ in range(NP):
    ti = rng.choice(Xr.shape[1], rng.integers(1, 3), replace=False); tw = rng.choice([-1, 1], len(ti))
    si = rng.choice(Sr.shape[1], rng.integers(1, 3), replace=False); sw = rng.choice([-1, 1], len(si))
    P.append((ti, tw, si, sw))
Wt = np.zeros((Xr.shape[1], NP)); Ws = np.zeros((Sr.shape[1], NP))
for j, (ti, tw, si, sw) in enumerate(P):
    Wt[ti, j] = tw; Ws[si, j] = sw


def corr_cols(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0) + 1e-12)


def score(Xr_, Sr_, idx):
    return corr_cols(Xr_[idx] @ Wt, Sr_[idx] @ Ws)


def shuffled_rows(idx):
    perm = np.arange(len(q))
    for s in np.unique(q):
        m = np.where(q == s)[0]
        perm[m] = rng.permutation(m)
    return perm


def evaluate(Xr_):
    r_tr = score(Xr_, Sr, np.where(tr)[0])
    # threshold from shuffles on train
    nullmax = []
    for b in range(10):
        perm = shuffled_rows(None)
        nullmax.append(np.quantile(np.abs(score(Xr_[perm], Sr, np.where(tr)[0])), 0.999))
    thr = float(np.median(nullmax))
    surv = np.where(np.abs(r_tr) > thr)[0]
    r_te = score(Xr_, Sr, np.where(te)[0])
    n_te = te.sum()
    zcrit = 1.645 / np.sqrt(n_te - 3)
    rep = (np.sign(r_te[surv]) == np.sign(r_tr[surv])) & (np.abs(r_te[surv]) > zcrit)
    return thr, surv, r_tr, r_te, rep


thr, surv, r_tr, r_te, rep = evaluate(Xr)
out = {'n': int(len(keep)), 'aspect_sanity_rho': float(san), 'thr': thr, 'n_surv': int(len(surv)),
       'n_rep': int(rep.sum()), 'rep_rate': float(rep.mean()) if len(surv) else None}
print('REAL surv', len(surv), 'replicated', int(rep.sum()), flush=True)
# null: shuffled text rows (within strata) through the whole pipeline
nulls = []
for b in range(20):
    perm = shuffled_rows(None)
    _, s2, _, _, rp2 = evaluate(Xr[perm])
    nulls.append((int(len(s2)), int(rp2.sum())))
out['null_surv_rep'] = nulls
print('NULL', nulls, flush=True)
# planted: fake text feature tied to a random shape feature
pl = []
for strength in (0.15, 0.25):
    for k in range(6):
        sj = rng.integers(Sr.shape[1])
        z = (Sr[:, sj] - Sr[:, sj].mean()) / Sr[:, sj].std()
        e = rng.standard_normal(len(z))
        fake = strength * z + np.sqrt(1 - strength ** 2) * e
        Xp = np.column_stack([Xr, fake])
        # hypotheses that use the planted column: add 200 pairs pairing fake with sj
        r_trp = corr_cols(Xp[tr][:, -1:], Sr[tr][:, [sj]])[0]
        r_tep = corr_cols(Xp[te][:, -1:], Sr[te][:, [sj]])[0]
        found = abs(r_trp) > thr and np.sign(r_tep) == np.sign(r_trp) and abs(r_tep) > 1.645 / np.sqrt(te.sum() - 3)
        pl.append({'strength': strength, 'r_tr': float(r_trp), 'r_te': float(r_tep), 'found': bool(found)})
out['planted'] = pl
for s in (0.15, 0.25):
    print('planted', s, sum(p['found'] for p in pl if p['strength'] == s), '/ 6', flush=True)
# describe replicated survivors
desc = []
for j in surv[rep]:
    ti, tw, si, sw = P[j]
    desc.append({'text': [(names[a], int(b)) for a, b in zip(ti, tw)], 'shape': [(SF[a], int(b)) for a, b in zip(si, sw)],
                 'r_train': float(r_tr[j]), 'r_test': float(r_te[j])})
desc.sort(key=lambda d: -abs(d['r_test']))
out['replicated'] = desc[:40]
from collections import Counter
out['rep_text_feats'] = Counter(n for d in desc for n, _ in d['text']).most_common(15)
out['rep_shape_feats'] = Counter(n for d in desc for n, _ in d['shape']).most_common(10)
json.dump(out, open(os.path.join(E.CK, 'cycle2.json'), 'w'), indent=1, default=str)
print(json.dumps({k: out[k] for k in ['rep_text_feats', 'rep_shape_feats']}, default=str))
print(json.dumps(desc[:8], default=str))
