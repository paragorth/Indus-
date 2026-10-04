"""pe11 cycle 1b: POWER CURVE.  How strong must a season be before the blind ring search
sees it?  (i) simulated PE herd office at amplitudes a = 2, 4, 8, 16 (f = 1);
(ii) Drehem month-bundles: frames = mean feature vector of all tablets of one
(year, month) with >= 5 tablets (real data, dates used only to build the bundles and
to score recovery, never in the search); (iii) Drehem 4-feature set (fawn, gazelle,
ewe, logTot = the most seasonal features) on single tablets, n = 2000."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, pickle, time
import numpy as np
from multiprocessing import Pool
from pe11_common import *  # noqa
from pe11_cycle1 import sim_tabs, task, month_r2

def main():
    P, U = pickle.load(open(os.path.join(CKPT, 'frames.pkl'), 'rb'))
    pool = pe_pool(P)
    jobs, meta = [], {}
    def add(name, X, m, nrep=10):
        meta[name] = {'n': len(X), 'R2_month': month_r2(X, m)[0]}
        jobs.append(('real', name + '_real', X, m, 7, {'set': name}))
        for k in ('copula', 'shuffle', 'rw'):
            for q in range(nrep):
                jobs.append((k, '%s_%s%d' % (name, k, q), X, None, 3000 + q, {'set': name}))
    for a in (2, 4, 8, 16):
        Ps, ms = sim_tabs(P, pool, a, 1.0, seed=500 + a)
        X, _, _ = pe_features(Ps, pool)
        add('sim_a%g' % a, X, ms)
    Xu, nu, idu = ur3_features(U)
    mu = np.array([U[i]['month'] for i in idu])
    yr = [U[i]['year'] for i in idu]
    g = defaultdict(list)
    for r, (y, m) in enumerate(zip(yr, mu)):
        g[(y, m)].append(r)
    keys = [k for k in g if len(g[k]) >= 5]
    Xb = np.stack([Xu[g[k]].mean(0) for k in keys]); mb = np.array([k[1] for k in keys])
    add('ur3_bundles', Xb, mb)
    rng = np.random.default_rng(4)
    sel = rng.choice(len(idu), 2000, replace=False)
    cols = [nu.index(c) for c in ('sh_fawn', 'sh_gazelle', 'sh_u8', 'logTot')]
    add('ur3_seas4', Xu[sel][:, cols], mu[sel])
    with Pool(2) as pl:
        res = list(pl.imap_unordered(task, jobs))
    summ = {}
    for s in meta:
        R = [r for r in res if r['set'] == s]
        real = [r for r in R if r['kind'] == 'real'][0]
        row = dict(meta[s]); row.update(D=real['D'], R2c=real['R2c'], acc=real['acc'], acc_null=real['acc_null'],
                                        acc_null95=real['acc_null95'], cc=real['cc'])
        for k in ('copula', 'shuffle', 'rw'):
            d = np.array([r['D'] for r in R if r['kind'] == k]); c = np.array([r['R2c'] for r in R if r['kind'] == k])
            row[k] = {'D_mean': float(d.mean()), 'D_max': float(d.max()), 'p_D': float((1 + (d >= real['D']).sum()) / (1 + len(d))),
                      'R2c_mean': float(c.mean()), 'R2c_max': float(c.max()), 'p_R2c': float((1 + (c >= real['R2c']).sum()) / (1 + len(c)))}
        summ[s] = row
        print(s, json.dumps(row, default=lambda x: round(x, 4)), flush=True)
    save('pe11_cycle1b.json', summ)

if __name__ == '__main__':
    main()
