#!/usr/bin/env python3
"""LA-45 cycle 3 check: is LA's near-zero gain a search failure? Score a hand-made two-meaning partition
(logogram 'L:' vs syllabic word; no game) and a three-meaning partition (+ the total words of each script, KU-RO /
to-so / szu-nigin2) on the same final-test documents as each c3 population, real and S2."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C
sys.argv = [sys.argv[0], '120', '6', 'c3']
import la45_c3 as J
TOT = {'KU-RO', 'PO-TO-KU-RO', 'to-so', 'to-sa', 'szu-nigin2', 'szunigin', 'szu-nigin'}
COMM_UR = set(C.UR_TRUTH_W['COMMODITY'].split())
for name in ['LA_TAB', 'LA_TAB_S2', 'LB_TAB', 'LB_TAB_S2', 'UR3_TAB', 'UR3_TAB_S2']:
    docs, _ = J.corpus(name)
    B = C.build(docs)
    base = name[:-3] if name.endswith('_S2') else name
    res = []
    for k in range(6):
        g = C.Game(docs, B, C.seed('la45-c3-%s-%d' % (base, k)), rounds=1)
        a2 = np.array([1 if (w.startswith('L:') or w in COMM_UR) else 0 for w in B.types], np.int64)
        a3 = a2.copy()
        for i, w in enumerate(B.types):
            if w in TOT: a3[i] = 2
        s2, c2, n2 = g.score(a2, None, g.final); s3, c3, n3 = g.score(a3, None, g.final)
        res.append((s2 + c2, s3 + c3, n2, c2, c3))
    r = np.array(res)
    print('%-11s 2-meaning gain %.3f (contra %.3f) | 3-meaning %.3f (contra %.3f) | n %.0f' % (
        name, r[:, 0].mean(), r[:, 3].mean(), r[:, 1].mean(), r[:, 4].mean(), r[:, 2].mean()))
