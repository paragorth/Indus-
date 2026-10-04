"""v11 cycle 5: the anti-map (volvelle-across-the-wheel) variant. Cycle 4 found Voynich succession nearly symmetric (unlike
language), yet a symmetric attractive map (dist2) loses to a 2-D bilinear model. A symmetric model that a distance kernel cannot
express is a REPULSIVE one: each step jumps to the opposite side of the space (logit = -x_a.x_b), or a mixed signature
(attract on one axis, repel on the other: logit = x1a x1b - x2a x2b). Models (2-D, same parameter count): sym(+,+) (= attractive map),
sym(-,-) (antipodal), sym(+,-) (mixed), bilin2 (asymmetric). Positive control: Planted-anti-30 (each step lands within radius 2 of
the point reflection of the current cell through the centre). Held-out gains, 2 folds. Checkpoints c5_<corpus>_<fold>.json."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_lib as L
import numpy as np
from multiprocessing import Pool
CORPORA = ['Planted-anti-30', 'Voynich-ZL', 'Planted-grid-30', 'Voynich-IT', 'Latin-Isidore', 'Italian-Manzoni', 'Shuffle-line']
LN2 = np.log(2)

def job(a):
    name, fold = a; ck = f'c5_{name}_{fold}.json'
    r = L.jload(ck)
    if r: return r
    lines, _ = L.corpus(name); voc = L.vocab(lines, 300); idx = {w: i for i, w in enumerate(voc)}
    tr, te = L.split(lines, fold); Ctr = L.bigram_counts(tr, idx); Cte = L.bigram_counts(te, idx)
    base = L.test_ll(L.fit(Ctr, 'uni', 0, iters=300), Cte)
    out = {'corpus': name, 'fold': fold}
    for kind, d in (('sympp', 2), ('symmm', 2), ('sympm', 2), ('symmm', 1), ('bilin', 2)):
        f = L.best_fit(Ctr, kind, d, 2, iters=500, lr=0.05)
        out[f'{kind}{d}'] = (L.test_ll(f, Cte) - base) / Cte.sum() / LN2
    L.jdump(out, ck); return out

if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(job, [(c, f) for c in CORPORA for f in (0, 1)]):
            print(r['corpus'], r['fold'], {k: round(v, 3) for k, v in r.items() if k not in ('corpus', 'fold')}, flush=True)
