"""v75: kind specificity. Distance (z-scored type-level features, layout excluded or included) from Voynich chunks
and from Voynich-generator chunks to each kind's centroid of real reference chunks. A kind is 'specific' to the
Voynich only if the Voynich is closer to it than every one of its generators (and the global shuffle) is."""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_cls as C, v75_lib as X
from v75_outlier import SYM
for rep in ('A', 'B'):
    for lay in (False, True):
        f, rows, F = C.load(rep)
        cols = [i for i, x in enumerate(f) if x not in SYM and (lay or x not in X.LAYOUT)]
        ref = np.array([r['kind'] not in ('?', 'GEN') for r in rows])
        Z = (F[:, cols] - F[ref][:, cols].mean(0)) / (F[ref][:, cols].std(0) + 1e-9)
        kinds = sorted({r['kind'] for r in rows if r['kind'] not in ('?', 'GEN')})
        cen = {k: Z[[i for i, r in enumerate(rows) if r['kind'] == k]].mean(0) for k in kinds}
        groups = {}
        for i, r in enumerate(rows):
            if r['kind'] != '?': continue
            key = r['base'][4:8] + ':' + ('VOY' if r['role'].startswith('voy') else r['role'][5:])
            groups.setdefault(key, []).append(i)
        print('== rep %s layout %s' % (rep, lay))
        print('%-14s' % '' + ' '.join('%6s' % k[:6] for k in kinds))
        for g, idx in sorted(groups.items()):
            d = [np.sqrt(((Z[idx] - cen[k]) ** 2).sum(1)).mean() for k in kinds]
            print('%-14s' % g + ' '.join('%6.2f' % x for x in d), ' nearest', kinds[int(np.argmin(d))])
