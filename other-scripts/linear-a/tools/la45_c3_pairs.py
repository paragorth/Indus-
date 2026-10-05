#!/usr/bin/env python3
"""LA-45 cycle 3: paired real-vs-S2 final-test gains (same held-out documents) and the pre-set decision rule."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C
tag = sys.argv[1] if len(sys.argv) > 1 else 'c3'
out = {}
for base in ['PLANT_TAB', 'UR3_TAB', 'LB_TAB', 'LA_TAB']:
    g, s = [], []
    for k in range(6):
        a = os.path.join(C.CK, '%s_%s_%d.json' % (tag, base, k)); b = os.path.join(C.CK, '%s_%s_S2_%d.json' % (tag, base, k))
        if os.path.exists(a) and os.path.exists(b):
            g.append(json.load(open(a))['final_gain']); s.append(json.load(open(b))['final_gain'])
    g, s = np.array(g), np.array(s)
    d = g - s
    wins = int((d > 0).sum())
    verdict = 'ORDER CARRIES MEANING' if wins >= 5 and d.mean() >= 0.3 else 'no order effect'
    out[base] = {'real': g.round(3).tolist(), 's2': s.round(3).tolist(), 'drop_mean': float(d.mean()), 'wins': wins,
                 'verdict': verdict}
    print('%-10s real %s | S2 %s | drop %.3f (sd %.3f) wins %d/%d -> %s' % (base, g.round(2), s.round(2), d.mean(),
                                                                     d.std(), wins, len(d), verdict))
json.dump(out, open(os.path.join(C.CK, tag + '_pairs.json'), 'w'), indent=1)
