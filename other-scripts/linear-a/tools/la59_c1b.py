#!/usr/bin/env python3
"""LA-59 cycle 1b: corrected rewired-graph null for the unconstrained fit (the cycle-1 stub rewiring spread
each pair's weight over many pairs and gave gain 0 by construction). Double-edge swaps, pair weights kept,
8 rewirings, best of 4 restarts each (as the cycle-1 key nulls)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import *
tag = sys.argv[1]
rng = np.random.default_rng(5911)
o = json.load(open(os.path.join(CK, f'c1_{tag}.json')))
A = np.array(o['A'])
K = kernel(consonant_key(), vowel_key()[0], 0.0)
res = []
for k in range(8):
    B = rewire(A, rng)
    res.append(max(anneal(B, K, rng, 30)[1] for _ in range(4)))
    print(tag, k, round(res[-1], 4), flush=True)
o['null_rewire'] = res
json.dump(o, open(os.path.join(CK, f'c1_{tag}.json'), 'w'))
