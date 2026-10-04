"""pe11 cycle 1c: detection threshold of the label-free ring statistics.
Pure ring: 4 features, phases lagged by 0, 3, 6, 9 months (cosines), Gaussian noise;
season share of variance R2s = 0.05 ... 0.8; n = 638 (PE size).  Statistics D and R2c
vs 10 copula surrogates; order recovery vs permuted months."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np
from multiprocessing import Pool
from pe11_common import *  # noqa
from pe11_cycle1 import task

def main():
    rng = np.random.default_rng(9)
    jobs, meta = [], {}
    for r2 in (0.05, 0.1, 0.2, 0.4, 0.8):
        n = 638
        m = rng.integers(0, S, n)
        sig = np.stack([np.cos(2 * np.pi * (m - l) / S) for l in (0, 3, 6, 9)], 1)
        sig /= sig.std(0)
        X = np.sqrt(r2) * sig + np.sqrt(1 - r2) * rng.normal(size=sig.shape)
        name = 'ring_r%g' % r2
        meta[name] = r2
        jobs.append(('real', name + '_real', X, m, 7, {'set': name}))
        for q in range(10):
            jobs.append(('copula', '%s_copula%d' % (name, q), X, None, 3000 + q, {'set': name}))
    with Pool(2) as pl:
        res = list(pl.imap_unordered(task, jobs))
    out = {}
    for s in meta:
        R = [r for r in res if r['set'] == s]
        real = [r for r in R if r['kind'] == 'real'][0]
        d = np.array([r['D'] for r in R if r['kind'] == 'copula']); c = np.array([r['R2c'] for r in R if r['kind'] == 'copula'])
        out[s] = {'R2s': meta[s], 'D': real['D'], 'D_null_mean': float(d.mean()), 'D_null_max': float(d.max()),
                  'p_D': float((1 + (d >= real['D']).sum()) / 11), 'R2c': real['R2c'], 'R2c_null_max': float(c.max()),
                  'p_R2c': float((1 + (c >= real['R2c']).sum()) / 11), 'acc': real['acc'], 'acc_null95': real['acc_null95'], 'cc': real['cc']}
        print(s, json.dumps(out[s], default=lambda x: round(x, 4)), flush=True)
    save('pe11_cycle1c.json', out)

if __name__ == '__main__':
    main()
