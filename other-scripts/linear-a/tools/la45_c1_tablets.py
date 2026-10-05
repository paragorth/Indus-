#!/usr/bin/env python3
"""LA-45 cycle 1 follow-up: does word order add information on Linear A TABLETS? Rescore the frozen meanings of
the c1 LA and LA_S2 populations on the final-test tablets only (S2 meanings scored on S2 tablets)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C
sys.argv = [sys.argv[0], '120', '6', 'c1']
import la45_c1 as J

res = {}
for name in ['LA', 'LA_S2']:
    docs, _ = J.corpus(name)
    B = C.build(docs)
    gs = []
    for k in range(6):
        r = json.load(open(os.path.join(C.CK, 'c1_%s_%d.json' % (name, k))))
        g = C.Game(docs, B, C.seed('la45-c1-%s-%d' % (name, k)), rounds=1)
        a = np.zeros(B.T, np.int64)
        for w, m in r['tclus'].items():
            a[B.ti[w]] = m
        tab = np.array([d for d in g.final if docs[d]['support'] == 'Tablet'])
        oth = np.array([d for d in g.final if docs[d]['support'] != 'Tablet'])
        st, ct, nt = g.score(a, None, tab); so, co, no = g.score(a, None, oth)
        gs.append((st + ct, nt, so + co, no))
        print(name, k, 'tablets gain %.3f n=%d | other gain %.3f n=%d' % gs[-1])
    gs = np.array(gs)
    res[name] = gs[:, [0, 2]].mean(0)
    print(name, 'mean gain tablets %.3f other %.3f' % tuple(res[name]))
