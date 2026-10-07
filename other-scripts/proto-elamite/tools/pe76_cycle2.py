"""pe76 cycle 2: through-corpus ABC.  parts: main (calibration, real, nulls, confound plants), halves, stability."""
import os, sys, json, collections
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
import pe76_common as P, common
import pe76_bank2 as B

T = common.load(); n = len(T)


class RF:
    def __init__(self, TH, ST, col, seed=0):
        self.m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=seed)
        self.m.fit(ST, TH[:, col])

    def __call__(self, s):
        return float(self.m.predict(np.atleast_2d(s))[0])


def rej(TH, ST, s, col=0, q=0.01):
    med = np.median(ST, 0); mad = np.median(np.abs(ST - med), 0) + 1e-9
    Z = (ST - med) / mad; z = (s - med) / mad
    d = np.sqrt(((Z - z) ** 2).sum(1)); k = max(50, int(q * len(d))); idx = np.argsort(d)[:k]
    y = TH[idx, col]; Xz = Z[idx] - z; w = 1 - (d[idx] / d[idx].max()) ** 2
    beta = np.linalg.lstsq(np.c_[np.ones(k), Xz] * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
    ya = y - Xz @ beta[1:]
    return dict(med=float(np.average(ya, weights=w)), q10=float(np.quantile(ya, .1)), q90=float(np.quantile(ya, .9)), dist=float(d[idx].mean()))


def vshuf(toks, rng, block=False):
    """block=False: variant letters permuted over all tokens of a base.  block=True: each tablet's whole
    list of variants for a base is moved as a block to another tablet with the same number of tokens of
    that base (keeps within-tablet consistency, destroys between-tablet habit sharing)."""
    out = []
    for b, lst in toks.items():
        if not block:
            lab = [s for _, s in lst]; rng.shuffle(lab)
            out += [(i, s) for (i, _), s in zip(lst, lab)]
        else:
            per = collections.defaultdict(list)
            for i, s in lst:
                per[i].append(s)
            bysize = collections.defaultdict(list)
            for i, v in per.items():
                bysize[len(v)].append(i)
            for sz, tl in bysize.items():
                blocks = [per[i] for i in tl]; order = rng.permutation(len(tl))
                for i, j in zip(tl, order):
                    out += [(i, s) for s in blocks[j]]
    return out


def office_plant(toks, K, rng, strength=0.8):
    """No time at all: K offices = blocks of tablets clustered on base-sign content; each office has its
    own favourite variant per base (used with prob strength)."""
    from sklearn.cluster import KMeans
    import scipy.sparse as sp
    bs = sorted(toks); bi = {b: j for j, b in enumerate(bs)}
    rows, cols = [], []
    bidx = {}
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


part = sys.argv[1]
res = {}
if part == 'main':
    TH, ST = B.load_bank2('all', 1)
    rng = np.random.default_rng(1); perm = rng.permutation(len(TH)); nh = len(TH) // 10
    h, tr = perm[:nh], perm[nh:]
    rfs = {k: RF(TH[tr], ST[tr], j) for j, k in enumerate(B.KEYS) if k in ('logS', 'idio', 'mix', 'logM', 'logW')}
    cal = {}
    for k, m in rfs.items():
        j = B.KEYS.index(k); pr = m.m.predict(ST[h]); y = TH[h, j]
        cal[k] = dict(spearman=float(spearmanr(pr, y).correlation), r2=float(1 - ((pr - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    pr = rfs['logS'].m.predict(ST[h]); y = TH[h, 0]
    # does S stay identifiable when habits are weak?  split held-out by idio
    for lo, hi in [(0, .33), (.33, .66), (.66, 1)]:
        sel = (TH[h, 2] >= lo) & (TH[h, 2] < hi)
        cal['logS_idio_%.2f' % lo] = float(spearmanr(pr[sel], y[sel]).correlation)
    sel = TH[h, 5] < 0.5
    cal['logS_mix<0.5'] = float(spearmanr(pr[sel], y[sel]).correlation)
    res['calib'] = cal
    toks = B.token_table(T)
    s_real = B.tok_stats(B.real_tokens(T), n, np.random.default_rng(0))
    res['real'] = {k: m(s_real) for k, m in rfs.items()}
    res['real']['S_rf'] = float(np.exp(res['real']['logS']))
    res['real']['rej_logS'] = rej(TH, ST, s_real, 0)
    res['real']['rej_idio'] = rej(TH, ST, s_real, 2)
    res['real']['rej_mix'] = rej(TH, ST, s_real, 5)
    dh = [rej(TH[tr], ST[tr], ST[h][i])['dist'] for i in range(100)]
    res['ppc'] = dict(real=res['real']['rej_logS']['dist'], sim_q50=float(np.median(dh)), sim_q90=float(np.quantile(dh, .9)))
    res['real_stats'] = dict(zip(B.STAT_NAMES2, map(float, s_real)))
    for name, block in [('VSHUF', False), ('BLOCK', True)]:
        L = []
        for k in range(5):
            s = B.tok_stats(vshuf(toks, np.random.default_rng(4000 + k), block), n, np.random.default_rng(k))
            L.append({kk: m(s) for kk, m in rfs.items()} | {'dist': rej(TH, ST, s)['dist'], 'coh': float(s[-1]), 'agree': float(s[-5])})
        res[name] = L
    # confound plants: office-only world (no time) and drift-only world
    L = []
    for K in (4, 12, 30):
        for k in range(2):
            s = B.tok_stats(office_plant(toks, K, np.random.default_rng(50 + K + k)), n, np.random.default_rng(k))
            L.append(dict(K=K, logS=rfs['logS'](s), idio=rfs['idio'](s), mix=rfs['mix'](s), dist=rej(TH, ST, s)['dist']))
    res['OFFICE_PLANT'] = L
    L = []
    for S in (0.2, 1, 5, 20):
        for k in range(2):
            r = np.random.default_rng(int(S * 10) + k)
            th = dict(logS=np.log(S), logM=np.log(8), idio=0.5, loyal=0.7, logW=np.log(0.7), mix=0.3)
            s = B.tok_stats(B.sim(th, toks, n, r), n, r)
            L.append(dict(S=S, logS=rfs['logS'](s)))
    res['DRIFT_PLANT'] = L
    imp = sorted(zip(B.STAT_NAMES2, rfs['logS'].m.feature_importances_), key=lambda x: -x[1])[:6]
    res['importance'] = [(a, round(float(b), 3)) for a, b in imp]
elif part in ('A', 'B'):
    TH, ST = B.load_bank2(part, 1)
    rng = np.random.default_rng(2); perm = rng.permutation(len(TH)); nh = len(TH) // 10
    m = RF(TH[perm[nh:]], ST[perm[nh:]], 0)
    pr = m.m.predict(ST[perm[:nh]]); y = TH[perm[:nh], 0]
    s = B.tok_stats(B.real_tokens(T, part), n, np.random.default_rng(0))
    toks = B.token_table(T, B.halves(T)[0 if part == 'A' else 1])
    res = dict(half=part, n_sims=len(TH), calib_spearman=float(spearmanr(pr, y).correlation), real_logS=m(s), real_S=float(np.exp(m(s))),
               rej=rej(TH, ST, s, 0), idio=RF(TH, ST, 2)(s),
               BLOCK=[m(B.tok_stats(vshuf(toks, np.random.default_rng(6000 + k), True), n, np.random.default_rng(k))) for k in range(5)])
elif part == 'stab':
    T1, S1 = B.load_bank2('all', 1); T2, S2 = B.load_bank2('all', 2)
    s = B.tok_stats(B.real_tokens(T), n, np.random.default_rng(0))
    res = dict(seed1=RF(T1, S1, 0, 11)(s), seed2=RF(T2, S2, 0, 22)(s), seed1_n=len(T1), seed2_n=len(T2),
               seed2_on_seed1_sims=float(spearmanr(RF(T2, S2, 0, 22).m.predict(S1[:1000]), T1[:1000, 0]).correlation))
print(json.dumps(res, indent=1, default=float))
json.dump(res, open(os.path.join(P.CKPT, 'cycle2_%s.json' % part), 'w'), indent=1, default=float)
