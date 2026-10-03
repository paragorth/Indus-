#!/usr/bin/env python3
"""PE-1.5: power of the 'stable ration level' test (PE-1.2, control C).
Positive control: for a fraction f of recurring designations (cross-tablet,
same system), overwrite all their occurrence values with the first
occurrence's value (a planted fixed ration), then rerun qty_equal and
qty_dlog against control C (partner replaced by a random same-tablet entry).
Reports the smallest f detected at p < 0.05 in >= 80% of 100 plantings."""
import json, random, math
from pe1_lib import *
T = load()
DE0 = designation_entries(T, minlen=2)
bt = collections.defaultdict(list)
for i, e in enumerate(DE0): bt[e['tablet']].append(i)
g = collections.defaultdict(list)
for i, e in enumerate(DE0): g[e['des']].append(i)
P = [(ix[a], ix[b]) for d, ix in g.items() if len({DE0[i]['tablet'] for i in ix}) >= 2
     for a in range(len(ix)) for b in range(a + 1, len(ix)) if DE0[ix[a]]['tablet'] != DE0[ix[b]]['tablet']]
recs = [d for d, ix in g.items() if len({DE0[i]['tablet'] for i in ix}) >= 2]
def stat(vals, PP):
    q = [(i, j) for i, j in PP if DE0[i]['sys'] and DE0[i]['sys'] == DE0[j]['sys'] and vals[i] and vals[j]]
    return sum(vals[i] == vals[j] for i, j in q) / max(1, len(q))
def test(vals, rng, R=200):
    o = stat(vals, P)
    nl = [stat(vals, [(i, rng.choice(bt[DE0[j]['tablet']])) for i, j in P]) for _ in range(R)]
    return (1 + sum(x >= o for x in nl)) / (R + 1)
rng = random.Random(2)
base = [e['val'] for e in DE0]
res = {'observed_p': test(base, rng, 1000)}
print('observed p', res['observed_p'])
for f in [0.1, 0.2, 0.3, 0.5]:
    hits = 0
    for k in range(100):
        v = base[:]
        for d in rng.sample(recs, int(f * len(recs))):
            ix = g[d]; v0 = v[ix[0]]
            for i in ix:
                if DE0[i]['sys'] == DE0[ix[0]]['sys']: v[i] = v0
        hits += test(v, rng) < 0.05
    res[f'power_f{f}'] = hits / 100; print('f', f, 'power', hits / 100)
json.dump(res, open(os.path.join(DATA, 'pe1_c5_power.json'), 'w'), indent=1)
