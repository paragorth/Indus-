#!/usr/bin/env python3
"""la74: re-score cycle-1 survivors against a stronger no-vessel null: the no-vessel model mixed with a
smoothed lookup table of whole fraction strings seen in training (mixing weight fitted on train by
leave-one-out)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from la74_common import *
S = load_strings(); V = vocab([r['s'] for r in S]); st = [norm(r['s'], V) for r in S]; grp = [r['doc'] for r in S]
def lookup_ll(train, test, nv):
    c = Counter(train); n = len(train)
    best = None
    for a in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):  # weight on NV
        ll = sum(math.log((1 - a) * (c[s] - 1) / (n - 1) + a * math.exp(nv.logp(s))) for s in train)  # leave-one-out
        if best is None or ll > best[0]: best = (ll, a)
    a = best[1]
    return sum(math.log((1 - a) * c[s] / n + a * math.exp(nv.logp(s))) for s in test), a
for nm, seed in [('real', 1), ('real2', 2), ('real3', 3)]:
    rng = np.random.default_rng(seed); docs = sorted(set(grp)); rng.shuffle(docs); tr = set(docs[:len(docs) // 2])
    train = [s for s, g in zip(st, grp) if g in tr]; test = [s for s, g in zip(st, grp) if g not in tr]
    nv = NoVessel(train, V); lte = [nv.logp(s) for s in test]
    lk, a = lookup_ll(train, test, nv)
    gl = (lk - sum(lte)) / len(test)
    top = json.load(open(os.path.join(CK, f'c1_{nm}.json')))['top']
    g = [r['gte'] for r in top]
    print(nm, 'lookup gain over NV %.3f (a=%.1f)' % (gl, a), 'vessel top mean %.3f' % np.mean(g), 'vessel>lookup %d/%d' % (sum(x > gl for x in g), len(g)))
