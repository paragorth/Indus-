#!/usr/bin/env python3
"""LA-7 power control: plant a picture->commodity effect into the real LA pairs and ask how often the
cycle-1 permutation test (H statistic, tokens) detects it at p < 0.05.
Planting: each token whose first sign has a predictive class (plant/vessel/animal/body) has, with
probability q, its commodity redrawn from that class's predicted families (LA marginal weights)."""
import os, random, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la7_common import *
import la7_test as T

T.N = 300
rng = random.Random(5)
lab, _ = picture_codes(False)
P = [(s, COM_FAMILY[c]) for s, k, c, d, w in la_pairs() if s in lab]
marg = Counter(f for _, f in P)
for q in (0.1, 0.2, 0.3, 0.5):
    hits = 0; R = 60
    for _ in range(R):
        Q = []
        for s, f in P:
            c = lab[s]
            if c in PRED and rng.random() < q:
                fams = [x for x in PRED[c] if marg[x]] or list(PRED[c])
                w = [marg[x] or 1 for x in fams]
                f = rng.choices(fams, w)[0]
            Q.append((s, f))
        r = T.perm_test(Q, lab)
        hits += r['pH'] < 0.05
    print('q=%.1f detected %d/%d' % (q, hits, R))
