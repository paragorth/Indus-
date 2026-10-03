#!/usr/bin/env python3
"""LA-1 cycle 3: writing order of compound fractions vs the sub-unit sizes implied by la1_assign.py.

Pairs: every ordered pair (a written before b, a != b) inside the 61 compound fractions
(attack_fractions.compounds). Implied size rank: T (1/10) > V > Z ; S (1/3) > V > Z.
Count pair instances that are larger-first, smaller-first, or ties (same role -> same value,
which the strict writing order forbids, since a sign is never followed by one of equal value except itself).
Null A: permute the role labels among letters (role counts kept), 20,000 runs.
Null B: each letter draws a random value from the la1_totals pool (unit fractions 1/2..1/16 and pair sums).
"""
import sys, os, json, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack_fractions as AF
import la1_common as L
import la1_totals as LT

random.seed(31)
A = json.load(open(os.path.join(L.D, 'la1_assign.json')))['cap']['assign']
RANK = {'T': 3, 'S': 4, 'V': 2, 'Z': 1}
comps = AF.compounds()
pairs = AF.pairs_of(comps)
known = {(a, b): n for (a, b), n in pairs.items() if a in A and b in A}
print(f'{len(comps)} compounds, {sum(pairs.values())} pair instances, {sum(known.values())} with both letters assigned')


def tally(rank):
    lf = sf = tie = 0
    for (a, b), n in known.items():
        if rank[a] > rank[b]: lf += n
        elif rank[a] < rank[b]: sf += n
        else: tie += n
    return lf, sf, tie


obs = tally({f: RANK[r] for f, r in A.items()})
print('implied ranks: larger-first %d, smaller-first %d, tie (forbidden by strict order) %d' % obs)
for (a, b), n in sorted(known.items(), key=lambda x: -x[1]):
    print(f'   {a}({A[a]}) before {b}({A[b]}) x{n}')
letters = list(A); roles = [A[f] for f in letters]
nA = []
for _ in range(20000):
    random.shuffle(roles); nA.append(tally({f: RANK[r] for f, r in zip(letters, roles)}))
def rep(name, nl, ob):
    print(f'{name}: ties mean {sum(x[2] for x in nl)/len(nl):.1f}, P(ties <= {ob[2]}) = {sum(x[2] <= ob[2] for x in nl)/len(nl):.3f}; '
          f'directional score (LF - SF) mean {sum(x[0]-x[1] for x in nl)/len(nl):.1f}, '
          f'P(|LF-SF| >= {abs(ob[0]-ob[1])}) = {sum(abs(x[0]-x[1]) >= abs(ob[0]-ob[1]) for x in nl)/len(nl):.3f}')
rep('null A (role permutation)', nA, obs)
nB = []
for _ in range(20000):
    v = {f: random.choice(LT.POOL) for f in letters}; nB.append(tally(v))
rep('null B (random pool values)', nB, obs)
# how often do random pool values give a fully consistent order (0 ties, 0 reversals in one direction)?
full = sum(1 for x in nB if x[2] == 0 and (x[0] == 0 or x[1] == 0)) / len(nB)
print(f'null B fully ordered share {full:.4f}')
# which sign pairs would have to be equal under the assignment?
print('letters forced equal by the assignment but written in a fixed order:',
      sorted({tuple(sorted((a, b))) for (a, b) in known if A[a] == A[b]}))
