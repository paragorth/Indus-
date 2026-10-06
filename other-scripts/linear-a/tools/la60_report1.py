#!/usr/bin/env python3
"""LA-60 cycle 1 summary: bits on held-out documents relative to G0 (the grammar with an empty reading)."""
import json, os, statistics as st
from la60_common import CK

for which, names in [('LA', ['INDUCED', 'PRIOR']), ('LB', ['INDUCED', 'LBPRIOR'])]:
    R = json.load(open(os.path.join(CK, 'c1_%s.json' % which)))
    nt = sum(r['ntest'] for r in R) / len(R)
    print(which, 'splits', len(R), 'mean test docs', round(nt, 1), 'mean test bits G0', round(st.mean(r['G0'] for r in R)))
    for k in ['B0', 'B1']:
        print('  %s - G0: %+.0f bits per test half' % (k, st.mean(r[k] - r['G0'] for r in R)))
    for n in names:
        d0 = [r[n] - r['G0'] for r in R]
        dsh = [r[n] - st.mean(r[n + '-shuf']) for r in R]
        P = [sum(1 for x in r[n + '-shuf'] if x <= r[n]) / len(r[n + '-shuf']) for r in R]
        Pk = [sum(1 for x in r[n + '-shufkeepTOT'] if x <= r[n]) / len(r[n + '-shufkeepTOT']) for r in R]
        ar = [r[n + '-noarith'] - r[n] for r in R]; od = [r[n + '-noorder'] - r[n] for r in R]
        mx = [r[n + '-mixB1'] - r['G0'] for r in R]
        hits = [tuple(r[n + '_arith']) for r in R]
        print('  %s: vs G0 %+.0f (better in %d/%d); vs shuffled roles %+.0f (better than every shuffle in %d/%d splits; median P %.2f); '
              'vs shuffles keeping TOT median P %.2f; arithmetic saves %.0f; order saves %.0f; mix with B1 vs G0 %+.0f; '
              'train arithmetic hits %s; reading size %s' % (
                  n, st.mean(d0), sum(x < 0 for x in d0), len(d0), st.mean(dsh), sum(p == 0 for p in P), len(P), st.median(P),
                  st.median(Pk), st.mean(ar), st.mean(od), st.mean(mx), hits, [r[n + '_size'] for r in R]))
