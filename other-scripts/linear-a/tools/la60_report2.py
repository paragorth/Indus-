#!/usr/bin/env python3
"""LA-60 cycle 2 summary: decoded held-out documents under frozen readings vs controls."""
import json, os, statistics as st
from la60_common import CK

KEYS = ['full', 'strict', 'close', 'agree', 'viol', 'cover']
for which, names in [('LA', ['INDUCED', 'PRIOR']), ('LB', ['INDUCED', 'LBPRIOR'])]:
    R = json.load(open(os.path.join(CK, 'c2_%s.json' % which)))
    rows = [r for r in R if r['split'] >= 0]
    print(which, 'splits', len(rows), 'mean test docs %.1f' % st.mean(r['ntest'] for r in rows))
    print('  EMPTY', {k: round(st.mean(r['EMPTY'].get(k, 0) for r in rows), 1) for k in KEYS})
    for n in names:
        line = []
        for k in KEYS:
            real = st.mean(r[n]['real'].get(k, 0) for r in rows)
            sh = st.mean(r[n]['shuf_' + k][0] for r in rows)
            rd = st.mean(r[n]['rand_' + k][0] for r in rows)
            psh = st.median(r[n]['shuf_' + k][1] for r in rows)
            line.append('%s %.1f (shuf %.1f P~%.2f, rand %.1f)' % (k, real, sh, psh, rd))
        nt = st.mean(r['ntest'] for r in rows)
        gs = st.mean(r[n]['real']['strict'] - r[n]['shuf_strict'][0] for r in rows)
        print('  %s: ' % n + '; '.join(line) + ' | strict gain over shuffles %.1f docs = %.1f %% of test docs' % (gs, 100 * gs / nt))
    if which == 'LA':
        p = [r for r in R if r['split'] == -1][0]
        print('  PRIMARY frozen split:', {n: p[n]['real'] for n in names})
