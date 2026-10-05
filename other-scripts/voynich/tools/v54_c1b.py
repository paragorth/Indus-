"""v54 cycle 1b: distance spectrum of near-repeat pairs (total edits 0..3) real vs nulls (3 seeds each)."""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V
from v54_c1 import null_posshuf, null_pageshuf
out = {}
for nm, P in [('ZL', V.voynich('ZL3b')), ('IT', V.voynich('IT2a')), ('BRU_planted', V.brumati()[0]), ('GER_real', V.german())]:
    for tn, f in [('real', None), ('posshuf', null_posshuf), ('pageshuf', null_pageshuf), ('markov', V.null_markov)]:
        hs = []
        for s in ([0] if f is None else [1, 2, 3]):
            Q = P if f is None else f(P, s)
            toks, pairs, _ = V.families(Q)
            h = collections.Counter(sum(x[2:]) for x in pairs); hs.append([h[d] for d in range(4)])
        m = [sum(x[d] for x in hs) / len(hs) for d in range(4)]
        out[nm + ':' + tn] = m; print(nm, tn, [round(x) for x in m], flush=True)
json.dump(out, open(os.path.join(V.CK, 'c1b.json'), 'w'))
