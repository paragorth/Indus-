"""pe75 cycle 1b: quantum test with a SHAPE-PRESERVING null.
Cycle 1 showed the cosine quantogram (and the random-guess pipeline) is driven by the marginal shape
(thickness is narrow, so q ~ median thickness always scores).  Null here: values redrawn from a smooth
log-normal mixture (K=1 and K=2 components, EM on log x) fitted to the same dimension; it keeps location,
spread and skew but has no ripple.  Statistic: max over q in [8,30] of the quantogram excess
z(q) - mean_null z(q), standardised by the null sd at that q (FWER by the max over q).
Planted: 17 mm quantum (sd 3% shrink + 2 mm erosion) on 60% / 30% of tablets.
"""
import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_common as C
from pe75_cycle1 import plant

QS = np.arange(8.0, 30.0001, 0.1)
NN = 200


def qg(x):
    return np.sqrt(2.0 / len(x)) * np.cos(2 * np.pi * x[:, None] / QS[None, :]).sum(0)


def fit_mix(lx, K, rng, it=200):
    mu = np.percentile(lx, np.linspace(20, 80, K)); sd = np.full(K, lx.std()); pi = np.full(K, 1.0 / K)
    for _ in range(it):
        p = pi * np.exp(-0.5 * ((lx[:, None] - mu) / sd) ** 2) / sd
        r = p / p.sum(1, keepdims=True)
        nk = r.sum(0); pi = nk / len(lx)
        mu = (r * lx[:, None]).sum(0) / nk
        sd = np.sqrt((r * (lx[:, None] - mu) ** 2).sum(0) / nk) + 1e-3
    return pi, mu, sd


def draw(par, n, rng):
    pi, mu, sd = par
    k = rng.choice(len(pi), n, p=pi)
    return np.round(np.exp(rng.normal(mu[k], sd[k])))   # catalogue values are whole mm


def test(x, K, seed):
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    par = fit_mix(np.log(x), K, rng)
    z = qg(x)
    N = np.array([qg(draw(par, len(x), rng)) for _ in range(NN)])
    m, s = N.mean(0), N.std(0) + 1e-9
    ex = (z - m) / s
    nmax = ((N - m) / s).max(1)
    i = ex.argmax()
    return {'n': len(x), 'best_q': round(float(QS[i]), 1), 'excess_z': round(float(ex[i]), 2),
            'p_fwer': float((nmax >= ex[i]).mean()), 'excess_at_17': round(float(ex[(QS >= 15.5) & (QS <= 17.5)].max()), 2)}


def job(a):
    name, d, K, x, seed = a
    return name, d, K, test(x, K, seed)


def main():
    R = [r for r in C.pe_table() if r['complete_cat'] and r['h'] and r['w'] and r['t']]
    D = {k: np.array([r[k] for r in R], float) for k in 'hwt'}
    arch = {'PE': D, 'PE_strict': {k: np.array([r[k] for r in R if r['pres'] == 'complete'], float) for k in 'hwt'}}
    cat = C.catalogue()
    for name, pre in (('PC', 'Uruk'), ('UR3', 'Ur III')):
        rows = [v for v in cat.values() if v['period'].startswith(pre) and v['h'] and v['w'] and v['t']
                and not C.BAD_PRES.search(v['pres'] or '')]
        rr = np.random.default_rng(5).permutation(len(rows))[:1500]
        arch[name] = {k: np.array([rows[i][k] for i in rr], float) for k in 'hwt'}
    rng = np.random.default_rng(11)
    arch['PLANT_60'] = plant(D, rng, 2.0, 0.6)
    arch['PLANT_30'] = plant(D, rng, 2.0, 0.3)
    jobs = []
    for name, A in arch.items():
        for d in 'hwt':
            for K in (1, 2):
                jobs.append((name, d, K, A[d], hash((name, d, K)) % 10000))
    with Pool(2) as P:
        res = P.map(job, jobs)
    out = {}
    for name, d, K, r in res:
        out['%s:%s:K%d' % (name, d, K)] = r
        print(name, d, K, r)
    json.dump(out, open(os.path.join(C.CK, 'cycle1b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
