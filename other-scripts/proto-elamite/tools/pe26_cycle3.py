"""pe26 cycle 3: CLAY TWINS. If a tablet's clay look records the lump / pit / day it came from, tablets photographed
in the same batch that look alike should also have been written alike (same office, same scribe, same day).
Statistic: within each photo batch, Spearman between clay similarity (-distance on batch-centred, standardised colour+texture)
and text similarity (Jaccard of base-sign sets; separately numeral-system sets), after regressing both on nuisance pair
covariates (|log area difference|, |log line-count difference|, |museum-number gap| (log), same-publication flag).
Null: clay vectors permuted among tablets inside each batch (Mantel, 1,000 draws). Planted control: 40 random within-batch
pairs with high text similarity get a shared clay offset; must be detected."""
import sys, os, json, numpy as np, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe26_common import *
import pe17_common as P17
from scipy.stats import spearmanr, rankdata
rng = np.random.default_rng(2603)
rows = [r for r in load() if r['group'] == 'PE' and r['site'] == 'Susa' and r.get('is_grey', 0) < 0.5]
cat = json.load(open(os.path.join(D, 'pe17_ckpt', 'pe_cat.json')))
tabs = {t['id']: t for t in P17.load()}
rows = [r for r in rows if r['id'] in tabs]
bc = collections.Counter(r['batch'] for r in rows)
rows = [r for r in rows if bc[r['batch']] >= 8]
print('tablets', len(rows), 'batches', len(set(r['batch'] for r in rows)))
FE = os.environ.get('FEATS', 'clay'); COLS = {'clay': CLAY, 'colour': COLOUR, 'texture': TEXTURE}[FE]
X = within_batch_centre(X_of(rows, COLS), [r['batch'] for r in rows])
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
tok = [P17.tablet_tokens(tabs[r['id']])[0] for r in rows]
def num(s):
    try: return float(s)
    except Exception: return np.nan
area = np.array([np.log(max(num(cat[r['id']]['height']) * num(cat[r['id']]['width']), 1)) if not np.isnan(num(cat[r['id']]['height']) * num(cat[r['id']]['width'])) else np.nan for r in rows])
area[np.isnan(area)] = np.nanmedian(area)
nlines = np.log1p([len(tabs[r['id']]['lines']) for r in rows])
def mno(r):
    d = re.findall(r'\d+', r['museum_no'] or ''); return float(d[0]) if d else np.nan
mn = np.array([mno(r) for r in rows])
pub = [re.sub(r',?\s*\d+.*$', '', tabs[r['id']]['des']) for r in rows]
b = np.array([r['batch'] for r in rows])
pairs = [(i, j) for k in set(b) for ii in [np.where(b == k)[0]] for x, i in enumerate(ii) for j in ii[x + 1:]]
pairs = np.array(pairs)
MINGAP = float(os.environ.get('MINGAP', 0))
if MINGAP:
    gap = np.abs(mn[pairs[:, 0]] - mn[pairs[:, 1]]); pairs = pairs[np.nan_to_num(gap, nan=1e9) >= MINGAP]
I, J = pairs[:, 0], pairs[:, 1]
BLOCK = int(os.environ.get('BLOCK', 0))
blk = np.zeros(len(rows), int)
if BLOCK:  # permutation blocks: runs of BLOCK consecutive museum numbers inside a batch
    for k in set(b):
        ii = np.where(b == k)[0]; ii = ii[np.argsort(np.nan_to_num(mn[ii], nan=1e9))]
        for q, i in enumerate(ii): blk[i] = q // BLOCK
print('pairs', len(pairs))
def jac(a, c): return len(a & c) / max(len(a | c), 1)
textS = np.array([jac(tok[i]['SIGN'], tok[j]['SIGN']) for i, j in pairs])
numS = np.array([jac(tok[i]['NUM'], tok[j]['NUM']) for i, j in pairs])
nuis = np.column_stack([np.abs(area[I] - area[J]), np.abs(nlines[I] - nlines[J]),
                        np.log1p(np.nan_to_num(np.abs(mn[I] - mn[J]), nan=1e4)), [pub[i] == pub[j] for i, j in pairs],
                        (area[I] + area[J]) / 2, (nlines[I] + nlines[J]) / 2])
nuis = np.column_stack([np.ones(len(pairs)), nuis.astype(float)])
Qn, _ = np.linalg.qr(nuis)
def resid(v):
    return v - Qn @ (Qn.T @ v)
def clayS(Xm): return -np.sqrt(((Xm[I] - Xm[J]) ** 2).sum(1))
rt, rn = resid(rankdata(textS)), resid(rankdata(numS))
if os.environ.get('NULLTEXT'):
    # size-preserving random texts: each tablet gets a random sign set of its own size, drawn by corpus frequency
    fq = collections.Counter(x for t_ in tok for x in t_['SIGN']); vocab = list(fq); pr_ = np.array([fq[v] for v in vocab], float); pr_ /= pr_.sum()
    rs = []
    for rep in range(int(os.environ.get('NULLTEXT'))):
        fake = [set(rng.choice(len(vocab), size=min(len(t_['SIGN']), len(vocab)), replace=False, p=pr_)) for t_ in tok]
        tS = np.array([jac(fake[i], fake[j]) for i, j in pairs]); rt_ = resid(rankdata(tS))
        rs.append(np.corrcoef(resid(rankdata(clayS(X))), rt_)[0, 1])
    print('size-preserving random texts: r mean', round(float(np.mean(rs)), 4), 'max', round(float(np.max(rs)), 4), flush=True)
    sys.exit()
def stat(Xm):
    c = resid(rankdata(clayS(Xm))); return np.corrcoef(c, rt)[0, 1], np.corrcoef(c, rn)[0, 1]
obs = stat(X)
def perm_X(Xm):
    Y = Xm.copy()
    for k in set(zip(b, blk)):
        ii = np.where((b == k[0]) & (blk == k[1]))[0]; Y[ii] = Xm[rng.permutation(ii)]
    return Y
NP = int(os.environ.get('NPERM', 1000))
import time; t0 = time.time(); _ = stat(X); print("one stat s", round(time.time() - t0, 2), flush=True)
nul = np.array([stat(perm_X(X)) for _ in range(NP)])
p = [(np.sum(nul[:, k] >= obs[k]) + 1) / (NP + 1) for k in range(2)]
print('clay~sign-set r', round(obs[0], 4), 'null q95', round(np.percentile(nul[:, 0], 95), 4), 'p', round(p[0], 4))
print('clay~numeral-set r', round(obs[1], 4), 'null q95', round(np.percentile(nul[:, 1], 95), 4), 'p', round(p[1], 4))
# raw, without nuisance regression, for transparency
c0 = rankdata(clayS(X)); print('raw clay~sign r', round(np.corrcoef(c0, rankdata(textS))[0, 1], 4), 'raw clay~mno-gap r', round(np.corrcoef(c0, rankdata(nuis[:, 3]))[0, 1], 4),
      'raw clay~same-pub r', round(np.corrcoef(c0, nuis[:, 4])[0, 1], 4))
# planted: tablets in the top text-similar pairs share a clay offset
plant = []
top = np.argsort(-textS)[:400]
for rep in range(int(os.environ.get('NREP', 10))):
    Xp = X.copy(); used = set(); k = 0
    for q in rng.permutation(top):
        i, j = pairs[q]
        if i in used or j in used: continue
        off = rng.normal(0, 0.5, X.shape[1]); Xp[i] += off; Xp[j] += off; used |= {i, j}; k += 1
        if k >= 40: break
    s = stat(Xp)[0]; nl2 = np.array([stat(perm_X(Xp))[0] for _ in range(int(os.environ.get('NPL', 40)))])
    plant.append((s, (np.sum(nl2 >= s) + 1) / (len(nl2) + 1)))
plant = np.array(plant)
print('planted 40 twin pairs (offset sd 0.5): mean r', round(plant[:, 0].mean(), 4), 'share p<0.05', (plant[:, 1] < 0.05).mean())
json.dump(dict(n_tab=len(rows), n_pairs=len(pairs), obs=obs, p=p, null_q95=list(np.percentile(nul, 95, axis=0)),
               planted_mean_r=float(plant[:, 0].mean()), planted_power=float((plant[:, 1] < 0.05).mean())),
          open(os.path.join(CK, f'cycle3_gap{int(MINGAP)}_blk{BLOCK}_{FE}.json'), 'w'), default=float, indent=1)
