#!/usr/bin/env python3
"""LA-67 kill sweep, test K-54: the la54 grade-C guess '~0.16 bits/doc about the unwritten commodity
beyond site (P 1/9 vs within-site label shuffles)'. Kill line: P > 0.2 with more shuffles.
Runs, with FRESH seeds (prefix la67-), the real pipeline again (2 seeds) and more within-site label
permutations (WS) into data/la67_ckpt/la54/.  usage: la67_ws.py JOB [JOB ...]  (LA_a, LA_b, WS_k)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la54_common as C
C.CK = os.path.join(HERE, '..', 'data', 'la67_ckpt', 'la54'); os.makedirs(C.CK, exist_ok=True)
_s = C.seed
C.seed = lambda name: _s('la67-' + name)
import la54_c1 as C1, la54_c3 as C3
for j in sys.argv[1:]:
    if j.startswith('LA_'):
        fn = os.path.join(C.CK, 'c1_LA.json')
        if not os.path.exists(fn.replace('LA', j)):
            _s2 = C.seed
            C.seed = lambda name, j=j: _s('la67-' + j + name)
            C1.run('LA')
            os.replace(fn, fn.replace('LA', j))
            C.seed = _s2
    elif not os.path.exists(os.path.join(C.CK, 'c3_%s.json' % j)):
        C3.run(j)
