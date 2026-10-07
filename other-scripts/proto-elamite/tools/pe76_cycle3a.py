"""pe76 cycle 3a: span from between-tablet habit sharing only (difference statistics vs each world's own
BLOCK null).  Calibration, plants (drift, office), real with 20 BLOCK draws, nulls, held-out sign halves,
seed stability."""
import os, sys, json, collections
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
import pe76_common as P, common
import pe76_bank2 as B2
from pe76_bank3 import dstats, block, load_bank3

T = common.load(); n = len(T)


def rf(TH, ST, col=0, seed=0):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=seed)
    return m.fit(ST, TH[:, col])


def vshuf(tokpairs, rng):
    by = collections.defaultdict(list)
    for i, s in tokpairs:
        by[common.base(s)].append((i, s))
    out = []
    for b, lst in by.items():
        lab = [s for _, s in lst]; rng.shuffle(lab); out += [(i, s) for (i, _), s in zip(lst, lab)]
    return out


def est(m, toks_fn, k=20, seed=0):
    """mean and spread of predicted logS over k independent BLOCK draws (and k draws of the input if random)."""
    v = []
    for j in range(k):
        r = np.random.default_rng(seed * 1000 + j)
        v.append(float(m.predict(dstats(toks_fn(r), n, r)[None])[0]))
    return dict(mean=float(np.mean(v)), sd=float(np.std(v)), S=float(np.exp(np.mean(v))))


res = {}
TH, ST = load_bank3('all', 1)
rng = np.random.default_rng(3); perm = rng.permutation(len(TH)); nh = len(TH) // 10
h, tr = perm[:nh], perm[nh:]
m = rf(TH[tr], ST[tr])
pr = m.predict(ST[h]); y = TH[h, 0]
res['n_sims'] = int(len(TH))
res['calib'] = dict(spearman=float(spearmanr(pr, y).correlation), r2=float(1 - ((pr - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                    auc_short_long=float(__import__('sklearn.metrics', fromlist=['x']).roc_auc_score(y[(y < 0) | (y > np.log(5))] > 0, pr[(y < 0) | (y > np.log(5))])))
for lo, hi in [(0, .33), (.33, .66), (.66, 1)]:
    sel = (TH[h, 2] >= lo) & (TH[h, 2] < hi)
    res['calib']['idio_%.2f' % lo] = float(spearmanr(pr[sel], y[sel]).correlation)
m = rf(TH, ST)
imp = sorted(zip(B2.STAT_NAMES2, m.feature_importances_), key=lambda x: -x[1])[:6]
res['importance'] = [(a, round(float(b), 3)) for a, b in imp]
toks = B2.token_table(T)
real = B2.real_tokens(T)
res['real'] = est(m, lambda r: real, 20, 1)
res['BLOCK_null'] = [est(m, lambda r: block(real, r), 4, 10 + k) for k in range(5)]
res['VSHUF_null'] = [est(m, lambda r: vshuf(real, r), 4, 20 + k) for k in range(5)]
# drift plants (same family) and misspecified plants
L = []
for S in (0.2, 1, 3, 10, 25):
    for k in range(3):
        r0 = np.random.default_rng(int(S * 10) * 10 + k)
        th = dict(logS=np.log(S), logM=np.log(8), idio=0.7, loyal=0.8, logW=np.log(0.7), mix=0.3)
        tk = B2.sim(th, toks, n, r0)
        L.append(dict(S=S, **est(m, lambda r: tk, 4, 30 + k)))
res['DRIFT_PLANT'] = L
L = []
for S in (0.2, 1, 3, 10):
    for k in range(2):
        r0 = np.random.default_rng(500 + int(S * 10) + k)
        th = dict(logS=np.log(S), logM=np.log(8), idio=0.6, loyal=0.8, logW=np.log(1.5), mix=0.0)
        tk = B2.sim(th, toks, n, r0)
        L.append(dict(S=S, **est(m, lambda r: tk, 4, 40 + k)))
res['DRIFT_PLANT_wide_nomix'] = L
def office_plant(K, rng, strength=0.8):
    from sklearn.cluster import KMeans
    import scipy.sparse as sp
    rows, cols, bidx = [], [], {}
    for t_i, t in enumerate(T):
        for l in t['lines']:
            for s in l['signs']:
                if common.is_sign(s):
                    rows.append(t_i); cols.append(bidx.setdefault(common.base(s), len(bidx)))
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(bidx))).toarray()
    X = X / np.maximum(X.sum(1, keepdims=True), 1)
    lab = KMeans(K, n_init=3, random_state=int(rng.integers(1e6))).fit_predict(X)
    out = []
    for b, lst in toks.items():
        cnt = collections.Counter(s for _, s in lst); V = sorted(cnt)
        pi = np.array([cnt[v] for v in V], float); pi /= pi.sum()
        fav = {k: rng.choice(len(V), p=pi) for k in range(K)}
        for i, _ in lst:
            k = fav[lab[i]] if rng.random() < strength else rng.choice(len(V), p=pi)
            out.append((i, V[k]))
    return out


L = []
for K in (4, 12, 30):
    for k in range(2):
        tk = office_plant(K, np.random.default_rng(60 + K + k))
        L.append(dict(K=K, **est(m, lambda r: tk, 4, 70 + k)))
res['OFFICE_PLANT'] = L
for half in ('A', 'B'):
    try:
        TA, SA = load_bank3(half, 1)
    except Exception as e:
        res['half' + half] = str(e); continue
    mA = rf(TA, SA)
    p2 = np.random.default_rng(4).permutation(len(TA)); nh2 = len(TA) // 10
    mA_cal = rf(TA[p2[nh2:]], SA[p2[nh2:]])
    rt = B2.real_tokens(T, half)
    res['half' + half] = dict(n_sims=int(len(TA)), calib=float(spearmanr(mA_cal.predict(SA[p2[:nh2]]), TA[p2[:nh2], 0]).correlation),
                              real=est(mA, lambda r: rt, 20, 2), BLOCK=[est(mA, lambda r: block(rt, r), 4, 50 + k) for k in range(3)])
try:
    T2, S2 = load_bank3('all', 2)
    m2 = rf(T2, S2, seed=7)
    res['seed2'] = dict(n_sims=int(len(T2)), real=est(m2, lambda r: real, 20, 1))
except Exception as e:
    res['seed2'] = str(e)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(P.CKPT, 'cycle3a.json'), 'w'), indent=1)
