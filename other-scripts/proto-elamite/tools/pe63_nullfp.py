#!/usr/bin/env python3
"""pe63 cycle 1 false-positive check: each null run scored against thresholds from the other null runs."""
import sys, json, glob, os
from collections import defaultdict
import pe63_common as C
from pe63_report1 import components

corp = sys.argv[1]
fns = sorted(glob.glob(os.path.join(C.CK, 'c1_%s_null_*.json' % corp)))
runs = [json.load(open(f)) for f in fns]
for i, R in enumerate(runs):
    thr = defaultdict(float)
    for j, Q in enumerate(runs):
        if j == i:
            continue
        for g in Q['groups']:
            k = C.size_class(len(g['G']))
            thr[k] = max(thr[k], g['h'])
    sig, comps = components(R['groups'], thr)
    print(corp, 'null', R['seed'], 'sig groups', len(sig), 'dossiers', len(comps), 'tablets', sum(len(d['ids']) for d in comps))
