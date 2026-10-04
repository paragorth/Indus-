#!/usr/bin/env python3
"""X-3 matrix analysis: usage x3_matrix.py TAG [SLICE]. For the 10 x 10 rank-mode matrices of
G_real, G_body (structure), G_relab and ID: teacher strength (row mean), learner receptivity
(column mean) and the pair-specific residual (double-centred), with its z against the residual
spread over all 90 pairs. Pair-specific identity = residual of ID."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3_sum as S
N = ['LA', 'PE', 'VOY', 'LB', 'PC', 'UR3', 'AKK', 'ELX', 'LAT', 'GRC']
tag = sys.argv[1]; sl = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
T = S.table(S.load([tag]))
for met in ('G_real', 'G_body', 'G_relab', 'ID'):
    M = np.full((10, 10), np.nan)
    for i, x in enumerate(N):
        for j, y in enumerate(N):
            t = T.get((x, y, sl, 'rank'))
            if t and t[met]:
                M[i, j] = t[met][0]
    r = np.nanmean(M, 1); c = np.nanmean(M, 0); g = np.nanmean(M)
    R = M - r[:, None] - c[None, :] + g
    sd = np.nanstd(R)
    print(f'\n== {met} (rank mode, slice {sl}); grand mean {g:+.3f}')
    print('teacher (row mean): ' + ', '.join(f'{n} {v:+.3f}' for n, v in sorted(zip(N, r), key=lambda a: -a[1])))
    print('learner (col mean): ' + ', '.join(f'{n} {v:+.3f}' for n, v in sorted(zip(N, c), key=lambda a: -a[1])))
    pairs = [(R[i, j] / sd, N[i], N[j], M[i, j]) for i in range(10) for j in range(10) if not np.isnan(R[i, j])]
    pairs.sort(reverse=True)
    print('top pair residuals (z): ' + ', '.join(f'{a}>{b} {z:+.1f} ({m:+.3f})' for z, a, b, m in pairs[:6]))
    U = [p for p in pairs if p[1] in ('LA', 'PE', 'VOY') and p[2] in ('LA', 'PE', 'VOY')]
    print('undeciphered pairs (z): ' + ', '.join(f'{a}>{b} {z:+.1f} ({m:+.3f})' for z, a, b, m in U))
