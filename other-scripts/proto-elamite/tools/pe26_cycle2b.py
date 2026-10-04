"""pe26 cycle 2b: is the cycle-2 hit real? (1) ST-11 percentile under 2,000 pseudo-plateau directions built from 3 random
Susa tablets of the SAME photo batch as the real Yahya tablets (batch-shuffled labels). (2) ST-11 generic outlierness.
(3) which features drive it. (4) clay-vs-pe17 rank correlation partialled on size/length/pixels, and its null under the
same pseudo directions. Uses the frozen cycle-2 pipeline (month campaigns)."""
import sys, json, os, numpy as np, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe26_common import *
from scipy.stats import spearmanr, rankdata
rng = np.random.default_rng(26262)
rows = [r for r in load() if r['group'] == 'PE' and r.get('is_grey', 0) < 0.5]
def bkey(r):
    parts = r['batch'].split('|'); return parts[2][:22] + '|' + parts[3][:7]
bat = [bkey(r) for r in rows]
X = within_batch_centre(X_of(rows, CLAY), bat); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
ids = [r['id'] for r in rows]; ix = {p: i for i, p in enumerate(ids)}
HOLD = {'P009536', 'P009157'}
Y = [i for i, r in enumerate(rows) if r['museum'] == 'NMI' and r['site'] == 'Yahya' and r['id'] not in HOLD]
S = [i for i, r in enumerate(rows) if r['museum'] == 'NMI' and r['site'] == 'Susa' and r['id'] not in HOLD]
NMI_S = S + [ix['P009157']]
rng2 = np.random.default_rng(2626); K = X.shape[1]
subs = [rng2.choice(K, size=rng2.integers(3, 9), replace=False) for _ in range(2000)]
CNT = np.zeros(K)
for s_ in subs: CNT[s_] += 1
CNT /= 2000
def w_of(Yi, Si):
    d = X[Yi].mean(0) - X[Si].mean(0); sd = X[Si].std(0) + 1e-6; return d / sd ** 2 * CNT
st = ix['P009157']
def pct_st(w):
    s = X[NMI_S] @ w; return np.mean(s <= X[st] @ w)
real = pct_st(w_of(Y, S)); print('ST-11 percentile real', round(real, 4))
ybatch = bat[Y[0]]
pool = [i for i in S if bat[i] == ybatch]
print('pseudo pool (Susa, same campaign as Yahya):', len(pool), ybatch)
ps = []
for k in range(2000):
    fake = list(rng.choice(pool, 3, replace=False)); ps.append(pct_st(w_of(fake, [i for i in S if i not in fake])))
ps = np.array(ps)
print('pseudo-plateau directions: ST-11 percentile mean', round(ps.mean(), 3), 'share >= real', round(np.mean(ps >= real), 4))
# random directions in feature space (pure outlier control)
rd = []
for k in range(2000):
    w = rng.normal(size=K) * CNT; rd.append(pct_st(w))
rd = np.array(rd); print('random directions: ST-11 share >= real', round(np.mean(rd >= real), 4), 'share >= 0.99', round(np.mean(rd >= .99), 3))
# Mahalanobis-type outlierness (diagonal)
Z = X[NMI_S]; d2 = (Z ** 2).sum(1); print('ST-11 generic outlier percentile (sum z^2)', round(np.mean(d2 <= (X[st] ** 2).sum()), 3))
w = w_of(Y, S); contrib = X[st] * w; o = np.argsort(-contrib)[:8]
print('top contributions:', [(CLAY[j], round(float(X[st, j]), 2), round(float(w[j]), 3)) for j in o])
print('Yahya-minus-Susa centred means (top |d|):', sorted([(CLAY[j], round(float(X[Y, j].mean() - X[S, j].mean()), 2)) for j in range(K)], key=lambda t: -abs(t[1]))[:8])
# pe17 partial correlation
cat = json.load(open(os.path.join(D, 'pe17_ckpt', 'pe_cat.json')))
t = json.load(open(os.path.join(D, 'pe17_frozen_ranking.json'))); tx = {s['id']: s for s in t['susa']}
def num(v):
    try: return float(v)
    except Exception: return np.nan
L = [i for i in range(len(rows)) if rows[i]['site'] == 'Susa' and ids[i] in tx and rows[i]['museum'] in ('NMI', 'Louvre')]
h = np.array([num(cat[ids[i]]['height']) for i in L]); wd = np.array([num(cat[ids[i]]['width']) for i in L]); th = np.array([num(cat[ids[i]]['thickness']) for i in L])
for v in (h, wd, th): v[np.isnan(v)] = np.nanmedian(v)
nl = np.array([tx[ids[i]]['lines'] for i in L], float)
nuis = np.column_stack([np.ones(len(L)), np.log(h * wd), np.log(th), np.log1p(nl), np.log([rows[i]['n_core'] for i in L]),
                        [rows[i]['museum'] == 'NMI' for i in L]]).astype(float)
if os.environ.get('PUB'):
    pubs = [re.sub(r'[ ,]*[0-9]+[a-z]?$', '', tx[ids[i]]['des'].split(',')[0]).strip() for i in L]
    pc = collections.Counter(pubs); keep = [p for p, n in pc.items() if n >= 5]
    print('publication series as nuisance:', len(keep), pc.most_common(8))
    nuis = np.column_stack([nuis] + [np.array([p == q for p in pubs], float) for q in keep[1:]])
def res(v):
    v = rankdata(v); b, *_ = np.linalg.lstsq(nuis, v, rcond=None); return v - nuis @ b
tsc = res(np.array([tx[ids[i]]['plateau_score'] for i in L]))
def rho_w(w): c = res(X[L] @ w); return np.corrcoef(c, tsc)[0, 1]
real_r = rho_w(w_of(Y, S)); print('partial rho clay~pe17 (size, thickness, lines, pixels, museum removed):', round(real_r, 4))
pr = np.array([rho_w(w_of(list(rng.choice(pool, 3, replace=False)), S)) for _ in range(1000)])
print('pseudo-plateau directions partial rho: mean', round(pr.mean(), 4), 'q95', round(np.percentile(pr, 95), 4), 'share >= real', round(np.mean(pr >= real_r), 4))
raw_r = np.corrcoef(rankdata(X[L] @ w_of(Y, S)), rankdata([tx[ids[i]]['plateau_score'] for i in L]))[0, 1]
prr = np.array([np.corrcoef(rankdata(X[L] @ w_of(list(rng.choice(pool, 3, replace=False)), S)), rankdata([tx[ids[i]]['plateau_score'] for i in L]))[0, 1] for _ in range(1000)])
print('raw rho', round(raw_r, 4), 'pseudo raw rho mean', round(prr.mean(), 4), 'q95', round(np.percentile(prr, 95), 4), 'share >= real', round(np.mean(prr >= raw_r), 4))
json.dump(dict(st11_real=real, st11_pseudo_share=float(np.mean(ps >= real)), st11_random_dir_share=float(np.mean(rd >= real)),
               st11_outlier_pct=float(np.mean(d2 <= (X[st] ** 2).sum())), partial_rho=real_r, partial_null_share=float(np.mean(pr >= real_r)),
               raw_rho=raw_r, raw_null_share=float(np.mean(prr >= raw_r))), open(os.path.join(CK, 'cycle2b' + ('_pub' if os.environ.get('PUB') else '') + '.json'), 'w'), default=float, indent=1)
