#!/usr/bin/env python3
"""LA-24 cycle 3b: the HT 9a / HT 9b proportional core. Within-side permutation P for one pair,
and which values of J allow the 3-word 4/5 group (pool of 22 candidate values)."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la24_c3 as c3
from la24_common import CK, POOL
la = c3.la_sides()
a, b = 'HT9a', 'HT9b'
def k_of(A, B):
    A = {(w, k): v for w, k, v in A}; B = {(w, k): v for w, k, v in B}
    r = np.log(np.array([B[x] / A[x] for x in A if x in B]))
    return max([int((np.abs(r - x) < c3.TOL).sum()) for x in r if abs(x) >= c3.TOL] or [0])
obs = k_of(la[a], la[b])
rng = random.Random(3); N = 20000; ge = 0
for _ in range(N):
    A = [e for e in la[a]]; va = [e[2] for e in A]; rng.shuffle(va)
    B = [e for e in la[b]]; vb = [e[2] for e in B]; rng.shuffle(vb)
    if k_of([(w, k, v) for (w, k, _), v in zip(A, va)], [(w, k, v) for (w, k, _), v in zip(B, vb)]) >= obs: ge += 1
# J scan: HT 9a *324-DI-RA = 2 J, TA-I = 2 J (corpus letters), HT 9b 2 and 2
Jok = [str(J) for J in POOL if abs((2 + float(J)) * 0.8 - 2) < 1e-9]
out = {'pair': [a, b], 'k_obs': obs, 'P_within_side_perm': (ge + 1) / (N + 1),
       'pairs_tested_in_corpus': 15, 'J_values_allowing_group': Jok,
       'sides': {a: la[a], b: la[b]}}
print(json.dumps(out, default=str))
json.dump(out, open(os.path.join(CK, 'c3b.json'), 'w'), indent=1, default=str)
