"""pe81 cycle 3: (a) phase or form? header<->corner fill within museum-number batches (Sb blocks) and within publication volume;
(b) which other text features ride on square blanks beyond the header (random search, held out);
(c) freeze header predictions for the 87 Tehran photographs."""
import sys, os, json, csv, re, hashlib
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe81_engine as E, common
SCR = sys.argv[1]; rng = np.random.default_rng(8130)
SH = json.load(open(os.path.join(E.CK, 'shapes.json')))
csv.field_size_limit(10 ** 9)
cat = {'P%06d' % int(x['id_text']): x for x in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8', errors='replace')) if x.get('period', '').startswith('Proto-Elamite')}
T = common.load()
rows = [('S', t) for t in T if t['lines']]
X, names, _ = E.features(rows)
R = []; Xi = []
for i, (_, t) in enumerate(rows):
    sh = SH.get(t['id']); c = cat.get(t['id'])
    if not sh or not c: continue
    m = re.match(r'Sb (\d+)', c['museum_no'] or '')
    R.append(dict(id=t['id'], cf=sh['corner_fill'], hd=float(common.header(t) is not None), sb=int(m.group(1)) if m else None,
                  vol=t.get('volume'), la=np.log(max(1, sum(len(l['signs']) for l in t['lines']))), nl=np.log(len(t['lines'])),
                  asp=np.log(sh['aspect_px']), dx=float(X[i, names.index('frac_x')] > 0)))
    Xi.append(i)
X = X[Xi]
cf = rankdata([r['cf'] for r in R]); hd = np.array([r['hd'] for r in R])
Zc = np.column_stack([np.ones(len(R))] + [np.array([r[k] for r in R]) for k in ('la', 'nl', 'asp', 'dx')])
res = lambda v, Z=Zc: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
def within(groups, B=5000):
    g = np.array(groups); ok = np.array([x is not None for x in groups])
    keys = [k for k in set(g[ok]) if (g == k).sum() >= 3 and 0 < hd[g == k].sum() < (g == k).sum()]
    m = np.isin(g, keys)
    a = res(cf)[m]; b = res(hd)[m]; gg = g[m]
    # remove group means (fixed effects)
    for k in keys:
        s = gg == k; a[s] -= a[s].mean(); b[s] -= b[s].mean()
    r = float(np.corrcoef(a, b)[0, 1]); cnt = 0
    for _ in range(B):
        bp = b.copy()
        for k in keys:
            s = np.where(gg == k)[0]; bp[s] = rng.permutation(bp[s])
        cnt += np.corrcoef(a, bp)[0, 1] >= r
    return dict(n=int(m.sum()), groups=len(keys), r=round(r, 3), p_one_sided=(cnt + 1) / (B + 1))
out = {}
out['within_Sb_block20'] = within([r['sb'] // 20 if r['sb'] else None for r in R])
out['within_Sb_block5'] = within([r['sb'] // 5 if r['sb'] else None for r in R])
out['within_volume'] = within([r['vol'] for r in R])
print(json.dumps(out, indent=1), flush=True)
# (b) random search beyond header: text features vs corner fill, header + size in the covariates, held-out by Sb blocks
Z2 = np.column_stack([Zc, hd]); Xr = np.apply_along_axis(lambda v: res(rankdata(v), Z2), 0, X); cr = res(cf, Z2)
blk = np.array([(r['sb'] or rng.integers(10 ** 6)) // 10 for r in R]); ub = np.unique(blk); rng.shuffle(ub)
tr = np.isin(blk, ub[: len(ub) // 2]); te = ~tr
def cc(A, b):
    A = A - A.mean(0); b = b - b.mean(); return (A * b[:, None]).sum(0) / np.sqrt((A * A).sum(0) * (b * b).sum() + 1e-12)
NH = 5000
H = [(rng.choice(X.shape[1], rng.integers(1, 3), replace=False)) for _ in range(NH)]
W = np.zeros((X.shape[1], NH)); sg = rng.choice([-1, 1], (NH, 2))
for j, h in enumerate(H): W[h, j] = sg[j, :len(h)]
def run(Xr_):
    rt = cc(Xr_[tr] @ W, cr[tr]); re_ = cc(Xr_[te] @ W, cr[te]); return rt, re_
rt, rtest = run(Xr)
nullq = []
for b in range(10):
    p = rng.permutation(len(R)); a, _ = run(Xr[p]); nullq.append(np.quantile(np.abs(a), 0.999))
thr = float(np.median(nullq)); surv = np.where(np.abs(rt) > thr)[0]
zc = 1.645 / np.sqrt(te.sum() - 3)
rep = surv[(np.sign(rtest[surv]) == np.sign(rt[surv])) & (np.abs(rtest[surv]) > zc)]
nulls = []
for b in range(10):
    p = rng.permutation(len(R)); a, c2 = run(Xr[p]); s2 = np.where(np.abs(a) > thr)[0]
    nulls.append((int(len(s2)), int(((np.sign(c2[s2]) == np.sign(a[s2])) & (np.abs(c2[s2]) > zc)).sum())))
from collections import Counter
cnt = Counter()
for j in rep:
    for k in H[j]: cnt[names[k]] += 1
single = {names[k]: (round(float(cc(Xr[tr][:, [k]], cr[tr])[0]), 3), round(float(cc(Xr[te][:, [k]], cr[te])[0]), 3)) for k, _ in [(names.index(n), 0) for n, _ in cnt.most_common(12)]}
out['beyond_header'] = dict(thr=thr, surv=int(len(surv)), replicated=int(len(rep)), null_surv_rep=nulls, top=cnt.most_common(12), single_feature_train_test=single)
print(json.dumps(out['beyond_header'], indent=1), flush=True)
# (c) freeze Tehran predictions: logistic of header on corner fill (+ aspect), fitted on all transliterated Susa tablets
from math import exp
x1 = np.array([r['cf'] for r in R]); x2 = np.array([r['asp'] for r in R])
A = np.column_stack([np.ones(len(R)), x1, x2]); w = np.zeros(3)
for _ in range(500):
    pr = 1 / (1 + np.exp(-A @ w)); w += 0.5 * A.T @ (hd - pr) / len(R)
teh = {k: v for k, v in SH.items() if k.startswith('P5202') and v}
pred = {k: round(float(1 / (1 + exp(-(w[0] + w[1] * v['corner_fill'] + w[2] * np.log(v['aspect_px']))))), 3) for k, v in sorted(teh.items())}
cfs = sorted(teh, key=lambda k: teh[k]['corner_fill'])
n3 = len(cfs) // 3
frozen = {'what': 'pe81 frozen predictions for the untransliterated Tehran Susa tablets (CDLI P520211-P520300), from photo outline only',
          'model': {'logit_header': [float(v) for v in w], 'features': ['1', 'corner_fill', 'log aspect_px']},
          'per_tablet_P_header': pred,
          'pooled_prediction': {'bottom_third_by_corner_fill': cfs[:n3], 'top_third_by_corner_fill': cfs[-n3:],
              'claim': 'once transliterated, the top third (squarest corners) has a header (first line signs, no numerals) at a rate at least 0.20 higher than the bottom third',
              'kill': 'difference < 0.05 or reversed (fragments with a lost first line excluded from both thirds)'},
          'caveat': 'Tehran photos are full-size CDLI photos resized to 450 px height; Louvre features come from ~300-550 px thumbnails'}
b = json.dumps(frozen, sort_keys=True, indent=1).encode(); open(os.path.join(common.DATA, 'pe81_frozen_tehran.json'), 'wb').write(b)
out['tehran_frozen_sha256'] = hashlib.sha256(b).hexdigest(); out['tehran_n'] = len(pred)
out['louvre_header_rate_by_cf_third'] = [float(hd[np.argsort(x1)][i * len(x1) // 3:(i + 1) * len(x1) // 3].mean()) for i in range(3)]
print('tehran', len(pred), out['tehran_frozen_sha256'], out['louvre_header_rate_by_cf_third'])
json.dump(out, open(os.path.join(E.CK, 'cycle3.json'), 'w'), indent=1, default=str)
