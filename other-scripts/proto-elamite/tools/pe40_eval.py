"""pe40 evaluation engine: one dataset -> RING/LINE/BLOCK/KNN held-out MRR (+ second-order subset) and truth recovery."""
import os as _o
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): _o.environ[_v]='1'
import numpy as np
from pe40_common import *

BW = (2, 5, 10, 25)
KS = (4, 8, 16, 32)


def evaluate(X, truth=None, period=None, nsplit=3, seed=0, iters=8000, restarts=3, rand_order=False):
    rng = np.random.default_rng(seed)
    g = giant(X); X = X[g][:, X[g].sum(0) >= 2]
    tr = None if truth is None else np.asarray(truth)[g]
    out = dict(n=int(X.shape[0]), p=int(X.shape[1]), occ=int(X.sum()))
    W = weights(X)
    oc, sc = seriate(W, True, restarts, iters, rng=rng)
    ol, sl = seriate(W, False, restarts, iters, rng=rng)
    out['band_circ'] = sc; out['band_line'] = sl
    if tr is not None:
        out['cc_circ'] = circ_corr(oc, tr, period)
        out['cc_line'] = circ_corr(ol, tr, period)
        out['cc_spec'] = circ_corr(spectral(W, True), tr, period)
    res = collections.defaultdict(list)
    for s in range(nsplit):
        Xtr, hid = split_hidden(X, rng)
        Wt = weights(Xtr)
        o1, _ = seriate(Wt, True, restarts, iters, rng=rng)
        o2, _ = seriate(Wt, False, restarts, iters, rng=rng)
        if rand_order:
            o1 = rng.permutation(len(Xtr)); o2 = rng.permutation(len(Xtr))
        K = knn_score(Xtr)
        second = [(i, j) for i, j in hid if K[i, j] == 0]
        res['KNN'].append(ranks(K, Xtr, hid))
        res['FREQ'].append(ranks(np.tile(Xtr.sum(0), (len(Xtr), 1)), Xtr, hid))
        for name, o, circ in (('RING', o1, True), ('LINE', o2, False)):
            best = max((ranks(kernel_score(o, Xtr, circ, bw), Xtr, hid), bw) for bw in BW)
            res[name].append(best[0])
            bw = best[1]
            res[name + '_2nd'].append(ranks(kernel_score(o, Xtr, circ, bw), Xtr, second) if second else np.nan)
        res['BLOCK'].append(max(ranks(block_score(Xtr, k, rng), Xtr, hid) for k in KS))
        bk = max(KS, key=lambda k: ranks(block_score(Xtr, k, np.random.default_rng(1)), Xtr, hid))
        res['BLOCK_2nd'].append(ranks(block_score(Xtr, bk, rng), Xtr, second) if second else np.nan)
        res['FREQ_2nd'].append(ranks(np.tile(Xtr.sum(0), (len(Xtr), 1)), Xtr, second) if second else np.nan)
        res['n_2nd'].append(len(second)); res['n_hid'].append(len(hid))
    for k, v in res.items():
        out[k] = float(np.nanmean(v))
    out['R-L'] = out['RING'] - out['LINE']; out['R-B'] = out['RING'] - out['BLOCK']; out['R-K'] = out['RING'] - out['KNN']
    out['R-B_2nd'] = out['RING_2nd'] - out['BLOCK_2nd']; out['R-L_2nd'] = out['RING_2nd'] - out['LINE_2nd']
    return out
