#!/usr/bin/env python3
"""print a compact table of an la47 engine log (name json per line)"""
import sys, json
K = ('sat_S', 'sat_O', 'sat_OS', 'top10_S', 'top10_O', 'best_S', 'best_O', 'incr_S_over_O', 'n')
for l in open(sys.argv[1]):
    if not l[:1].isalpha() or ' {' not in l: continue
    n, j = l.split(' ', 1)
    try: d = json.loads(j)
    except Exception: continue
    for t, o in d.items():
        if isinstance(o, dict): print(n, t, ' '.join('%s=%s' % (k, o[k]) for k in K if k in o))
