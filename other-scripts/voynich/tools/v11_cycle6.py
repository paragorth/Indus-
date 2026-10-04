"""v11 cycle 6: what is the 'space' then? If succession is governed by the junction between the last glyph of word a and the
first glyph of word b (v6: 0.17-0.19 bits of junction information), the best 2-D model would be bilinear (end-class x start-class)
and no map would be needed. Models: junc (logit = J[last glyph of a, first glyph of b], ~20x20 table), juncdist2 (junction +
attractive 2-D map), dist2, bilin2. Question 1: does junc reach bilin2? Question 2: does a map add anything on top of the junction?
Controls: planted grid (map must add a lot over junc), Latin and Italian letters (junction = last/first letter), line-shuffle.
Held-out, 2 folds. Checkpoints c6_<corpus>_<fold>.json."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_lib as L
import numpy as np
from multiprocessing import Pool
CORPORA = ['Voynich-ZL', 'Planted-grid-30', 'Voynich-IT', 'Latin-Isidore', 'Italian-Manzoni', 'Shuffle-line']
LN2 = np.log(2)

def job(a):
    name, fold = a; ck = f'c6_{name}_{fold}.json'
    r = L.jload(ck)
    if r: return r
    lines, _ = L.corpus(name); voc = L.vocab(lines, 300); idx = {w: i for i, w in enumerate(voc)}
    tr, te = L.split(lines, fold); Ctr = L.bigram_counts(tr, idx); Cte = L.bigram_counts(te, idx)
    feats = L.junction_feats(voc)
    base = L.test_ll(L.fit(Ctr, 'uni', 0, iters=300), Cte)
    out = {'corpus': name, 'fold': fold, 'n_end': feats[0].shape[1], 'n_start': feats[1].shape[1]}
    for kind, d, rs in (('junc', 0, 1), ('juncdist', 2, 2), ('dist', 2, 2), ('bilin', 2, 1)):
        f = L.best_fit(Ctr, kind, d, rs, iters=500, lr=0.08, feats=feats)
        out[f'{kind}{d}'] = (L.test_ll(f, Cte) - base) / Cte.sum() / LN2
    L.jdump(out, ck); return out

if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(job, [(c, f) for c in CORPORA for f in (0, 1)]):
            print(r['corpus'], r['fold'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ('corpus', 'fold')}, flush=True)
