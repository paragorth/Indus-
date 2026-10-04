"""v6 cycle 5b: is the left-edge link only 'do not start a line with the glyph that started the line above'?
Control generator: Voynich lines permuted within paragraph, then re-ordered so that a line repeating the first glyph of
the line above is swapped away with probability p (repeat avoidance only, no preferred successions).
p is tuned so the same-glyph rate matches the real 8.5%. Compare total MI and off-diagonal MI excess with the real text."""
import sys, os, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from v6_gridlib import units_voy as U
from v6_cycle4 import voy_meta, perm_within, pairs, offdiag_mi, same_rate, zstat
from v6_gridlib import mi

def avoid(paras, p, rng):
    out = []
    for para in perm_within(paras, rng):
        q = list(para)
        for _ in range(3):
            for i in range(1, len(q) - 1):
                if U(q[i][0])[0] == U(q[i + 1][0])[0] and rng.random() < p:
                    j = rng.randrange(1, len(q)); q[i + 1], q[j] = q[j], q[i + 1]
        out.append(q)
    return out

paras = [p['lines'] for p in voy_meta('ZL3b')]
res = {}
real = pairs(paras, U)
nulls = [pairs(perm_within(paras, random.Random(s)), U) for s in range(200)]
res['real'] = {'MI': zstat(mi(real), [mi(x) for x in nulls]), 'offdiag': zstat(offdiag_mi(real), [offdiag_mi(x) for x in nulls]),
               'same': same_rate(real)}
for p in (0.6, 0.8, 0.95):
    sims = [pairs(avoid(paras, p, random.Random(1000 + s)), U) for s in range(30)]
    m = lambda f: sum(f(x) for x in sims) / len(sims)
    nm = lambda f: sum(f(x) for x in nulls) / len(nulls)
    res['avoid p=%.2f' % p] = {'same': m(same_rate), 'MI excess': m(mi) - nm(mi), 'offdiag excess': m(offdiag_mi) - nm(offdiag_mi)}
for k, v in res.items(): print(k, json.dumps(v, default=lambda x: round(x, 4)))
json.dump(res, open(os.path.join(vlib.RES, 'v6_cycle5b.json'), 'w'), indent=0)
