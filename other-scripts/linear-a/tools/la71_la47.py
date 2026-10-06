#!/usr/bin/env python3
"""la71 cycle 2: la47 layout facts on one corpus version (run via la71_run.py VERSION la71_la47.py).
Rebuilds the la47 layout (lineara.xyz GORILA line layout + corpus tokens + SigLA boxes) from the version's
corpus, then (1) the line-break census of la47_c1b (share of breaks that split a word or number, fM;
share at entry boundaries vs the re-flow null, fEc) and (2) the layout-cell z-scores of la47_c3b for the
la45 classes (2,000 within-page permutations).  Grammar searches are not re-run."""
import os, sys, json, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la47_layout
la47_layout.build()
import la47_common as C
from la47_c1b import classify
from la47_c3b import cells_test

ver = os.environ.get('LA71_VERSION')
rng = random.Random(7)
LA = C.la_pages(supports=('Tablet',))
out = {'ver': ver, 'pages': len(LA), 'LA': classify(LA)}
nulls = [classify(C.null_reflow(LA, rng)) for _ in range(200)]
for k in ('fEc', 'fM'):
    v = np.array([n[k] for n in nulls]); out['null_' + k] = [float(v.mean()), float(v.std())]
out['fEc_ratio'] = out['LA']['fEc'] / out['null_fEc'][0]
r2 = np.random.default_rng(9)
LA3 = C.la_pages(); rows = C.table(LA3)
cells = cells_test(rows, lambda r: r['cls'], C.SFEAT, r2)
out['cells'] = cells
print(json.dumps({k: v for k, v in out.items() if k != 'cells'}))
for k, v in sorted(cells.items(), key=lambda kv: -abs(kv[1]['z'])):
    print(' ', k, v)
json.dump(out, open(os.path.join(HERE, '..', 'data', 'la71_ckpt', 'c2_la47_%s.json' % ver), 'w'), indent=1)
