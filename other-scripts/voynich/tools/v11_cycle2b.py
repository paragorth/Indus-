"""v11 cycle 2b: the one-way-route loophole. A map walk could be directional (a route, a volvelle turned one way),
which the symmetric dist model cannot express. Model drift-d: logit = -|x_a + v - x_b|^2 + beta_b + g[a=b] (one global
displacement v per step). Compare dist2, drift2, drift3, bilin2 held-out on Voynich ZL/IT, Latin, planted grid, and a planted
ROUTE (grid walk that always moves 1-2 rows forward: must be caught by drift2 >> dist2). Checkpoints c2b_<corpus>_<fold>.json."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_lib as L
import numpy as np
from multiprocessing import Pool
CORPORA = ['Planted-route-30', 'Voynich-ZL', 'Planted-grid-30', 'Latin-Isidore', 'Voynich-IT', 'Italian-Manzoni']
LN2 = np.log(2)

def job(a):
    name, fold = a; ck = f'c2b_{name}_{fold}.json'
    r = L.jload(ck)
    if r: return r
    lines, truth = L.corpus(name); voc = L.vocab(lines, 300); idx = {w: i for i, w in enumerate(voc)}
    tr, te = L.split(lines, fold); Ctr = L.bigram_counts(tr, idx); Cte = L.bigram_counts(te, idx)
    base = L.test_ll(L.fit(Ctr, 'uni', 0, iters=300), Cte)
    out = {'corpus': name, 'fold': fold, 'models': {}}
    for kind, d, rs in (('dist', 2, 2), ('drift', 2, 2), ('drift', 3, 1), ('bilin', 2, 1)):
        f = L.best_fit(Ctr, kind, d, rs, iters=500, lr=0.08)
        rec = {'gain_bits': (L.test_ll(f, Cte) - base) / Cte.sum() / LN2, 'rho': f['rho']}
        if kind == 'drift':
            X = f['P'][0]; v = f['P'][1]
            rec['drift_over_spread'] = float(np.linalg.norm(v) / np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))
        out['models'][f'{kind}{d}'] = rec
    L.jdump(out, ck); return out

if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(job, [(c, f) for c in CORPORA for f in (0, 1)]):
            print(r['corpus'], r['fold'], {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r['models'].items()}, flush=True)
