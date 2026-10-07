"""pe75 cycle 1d: calibrate 'a ripple at a fixed q replicates in two independent halves' on archives
with no claimed ruler (proto-cuneiform, Ur III heights, same catalogue) and on PE at random q."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_common as C
from pe75_cycle1b import qg, QS, fit_mix, draw


def exc_all(x, rng, NN=150):
    par = fit_mix(np.log(x), 2, rng)
    z = qg(x); N = np.array([qg(draw(par, len(x), rng)) for _ in range(NN)])
    return (z - N.mean(0)) / (N.std(0) + 1e-9)


def job(a):
    name, x, seed = a
    rng = np.random.default_rng(seed)
    m = rng.random(len(x)) < 0.5
    e1, e2 = exc_all(x[m], rng), exc_all(x[~m], rng)
    # fraction of q (in 0.6 mm windows) where both halves exceed 2.7
    both = np.minimum(e1, e2)
    w = [both[(QS >= q - .3) & (QS <= q + .3)].max() for q in np.arange(8.3, 29.8, 0.6)]
    return name, float(np.mean(np.array(w) >= 2.7)), float(np.max(w))


cat = C.catalogue()
R = [r for r in C.pe_table() if r['complete_cat'] and r['h'] and r['w'] and r['t']]
sets = {'PE_h': np.array([r['h'] for r in R], float)}
for name, pre in (('PC', 'Uruk'), ('UR3', 'Ur III')):
    rows = [v for v in cat.values() if v['period'].startswith(pre) and v['h'] and v['w']
            and not C.BAD_PRES.search(v['pres'] or '')]
    for d in 'hw':
        for k in range(3):
            rr = np.random.default_rng(50 + k).permutation(len(rows))[:781]
            sets['%s_%s_%d' % (name, d, k)] = np.array([rows[i][d] for i in rr], float)
jobs = [(n, x, 100 + i) for i, (n, x) in enumerate(sets.items())]
with Pool(2) as P:
    res = P.map(job, jobs)
for r in res:
    print(r)
json.dump(res, open(os.path.join(C.CK, 'cycle1d.json'), 'w'))
