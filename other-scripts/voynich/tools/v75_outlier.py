"""v75: is the Voynich inside the reference cloud at all? Nearest-reference-chunk distance (z-scored features) for
Voynich chunks and Voynich-generator chunks, against the same distance for held-out reference chunks (nearest chunk
from ANY OTHER system). Usage: python3 v75_outlier.py REP feature-set(all|type|type_nolayout)"""
import os, sys
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_cls as C, v75_lib as X

SYM = ['wl_mean', 'wl_sd', 'h1', 'h2', 'rigid', 'h_first', 'h_last', 'junc_mi', 'lenfreq', 'short', 'fl_mi', 'h_len', 'typelen_gap']


def run(rep, fs):
    feats, rows, F = C.load(rep)
    if fs == 'all': cols = [i for i, f in enumerate(feats) if f not in X.LAYOUT]
    elif fs == 'type': cols = [i for i, f in enumerate(feats) if f not in SYM]
    else: cols = [i for i, f in enumerate(feats) if f not in SYM and f not in X.LAYOUT]
    ref = np.array([r['kind'] != '?' for r in rows])
    A = F[:, cols]; mu = A[ref].mean(0); sd = A[ref].std(0) + 1e-9; Z = (A - mu) / sd
    base = np.array([r['base'] for r in rows])
    RZ = Z[ref]; rb = base[ref]
    out = defaultdict(list)
    for i, r in enumerate(rows):
        d = np.sqrt(((RZ - Z[i]) ** 2).sum(1))
        if r['kind'] != '?':
            if r['role'] != 'real': continue
            d = d[rb != r['base']]; out['REF:' + r['kind']].append(d.min())
        else:
            out['%s:%s' % (r['base'], r['role'])].append(d.min())
    allref = np.concatenate([v for k, v in out.items() if k.startswith('REF')])
    q = np.quantile(allref, [0.5, 0.95, 0.99])
    res = {k: (float(np.median(v)), float(np.mean(np.array(v) > q[1]))) for k, v in sorted(out.items())}
    return q, res


if __name__ == '__main__':
    rep = sys.argv[1]; fs = sys.argv[2]
    q, res = run(rep, fs)
    print('reference held-out nearest distance: median %.2f q95 %.2f q99 %.2f' % tuple(q))
    for k, (m, f) in res.items(): print('%-28s median %.2f  share beyond ref q95 %.2f' % (k, m, f))
