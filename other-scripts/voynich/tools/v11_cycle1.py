"""v11 cycle 1: fit hidden-map (distance) models vs non-geometric low-rank models on within-line bigrams.
Per corpus x fold (page-level 2-fold split): uni, dist d=1,2,3,5 (spectral + random restarts), bilin d=2,16.
Checkpoint per job in data/results/v11/c1_<corpus>_<fold>.json. 2 workers."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_lib as L
import numpy as np
from multiprocessing import Pool

CORPORA = ['Planted-grid-30', 'Planted-grid-60', 'Planted-graph-30', 'Voynich-ZL', 'Voynich-IT', 'Latin-Isidore',
           'Italian-Manzoni', 'Spanish-Cervantes', 'Shuffle-line', 'SelfCitation', 'Voynich-A', 'Voynich-B']
MODELS = [('uni', 0, 1), ('dist', 1, 3), ('dist', 2, 3), ('dist', 3, 2), ('dist', 5, 2), ('bilin', 2, 1), ('bilin', 16, 1)]
LN2 = np.log(2)


def job(args):
    name, fold = args
    ck = f'c1_{name}_{fold}.json'
    r = L.jload(ck)
    if r: return r
    t = time.time()
    lines, truth = L.corpus(name)
    voc = L.vocab(lines, 400); idx = {w: i for i, w in enumerate(voc)}
    tr, te = L.split(lines, fold)
    Ctr = L.bigram_counts(tr, idx); Cte = L.bigram_counts(te, idx)
    out = {'corpus': name, 'fold': fold, 'n_train': Ctr.sum(), 'n_test': Cte.sum(), 'models': {}}
    base = None; emb = {}
    for kind, d, rs in MODELS:
        f = L.best_fit(Ctr, kind, d, rs, iters=800)
        tl = L.test_ll(f, Cte)
        if kind == 'uni': base = tl
        rec = {'test_bits': tl / Cte.sum() / LN2, 'gain_bits': (tl - base) / Cte.sum() / LN2,
               'train_bits': f['train_ll'] / Ctr.sum() / LN2, 'rho': f['rho']}
        if kind == 'dist' and truth and d == 2:
            ws = [w for w in voc if w in truth]
            Xw = f['P'][0][[idx[w] for w in ws]]; Yw = np.array([truth[w] for w in ws], float)
            rec['procrustes_R2'] = L.procrustes_r2(Xw, Yw); rec['knn_hit'], rec['knn_chance'] = L.knn_recovery(Xw, Yw)
        if kind == 'dist' and d in (2, 3):
            emb[d] = f['P'][0].tolist()
        out['models'][f'{kind}{d}'] = rec
    out['vocab'] = voc; out['emb'] = emb; out['secs'] = time.time() - t
    L.jdump(out, ck)
    return out


if __name__ == '__main__':
    jobs = [(c, f) for c in CORPORA for f in (0, 1)]
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            m = r['models']
            print(r['corpus'], r['fold'], ' '.join(f"{k}:{v['gain_bits']:.3f}" for k, v in m.items()),
                  'rho2=%.2f' % m['dist2']['rho'], 'R2=%s' % m['dist2'].get('procrustes_R2'), flush=True)
