#!/usr/bin/env python3
"""LA-59 cycle 4: how much alternation data would the human key need? Planted realistic syllabaries
(Miller-Nicely + formant key, sound share 1.0 and 0.5) on Linear A's 76-sign degree profile at 1x, 3x, 10x and
30x LA's alternation weight. Search temperature scaled with the weight (cycle 3 froze at 10x); 3 restarts,
50 sweeps. Recovery = consonant / vowel label accuracy and co-assignment; identifiability = does the true
assignment score at least as well as the fit."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import *
rng = np.random.default_rng(594)
o = json.load(open(os.path.join(CK, 'c1_LA.json'))); A = np.array(o['A'])
K = kernel(consonant_key(), vowel_key()[0], 0.0)
d = A.sum(1); ne = int(round(A.sum() / 2)); res = []; t0 = time.time()
for share in (1.0, 0.5):
    for mult in (1, 3, 10, 30):
        for rep in range(2):
            st = planted_grid(len(d), rng)
            B = sample_planted(d, st, K, ne * mult, rng, share)
            sf, g, l = max((anneal(B, K, rng, 50, T0=3.0 * mult, T1=0.03 * mult) for _ in range(3)), key=lambda x: x[1])
            r = dict(share=share, mult=mult, g=g, g_true=gain(B, st, K), acc_c=float(np.mean(sf // 5 == st // 5)),
                     acc_v=float(np.mean(sf % 5 == st % 5)), co_c=co_assign_agreement(sf, list(st // 5), lambda x: x // 5),
                     co_v=co_assign_agreement(sf, list(st % 5), lambda x: x % 5))
            res.append(r); print({k: round(v, 3) for k, v in r.items()}, round(time.time() - t0), 's', flush=True)
json.dump(res, open(os.path.join(CK, 'c4_ceiling.json'), 'w'))
print('done')
