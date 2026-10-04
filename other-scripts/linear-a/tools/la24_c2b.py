#!/usr/bin/env python3
"""LA-24 cycle 2b: is the CONV advantage in fraction absorption (F3) about the letters, or only
about simple values? Controls: (a) CONV's values permuted among the letters (same value
multiset, so the same 'simplicity'), 3,000 permutations; (b) letters shuffled among the
fraction-bearing amounts of the corpus, CONV values kept (keeps every value, breaks which list
and which entry carries which letter), 3,000 shuffles; (c) random V restricted to the 'simple'
pool {1/2, 1/4, 3/4, 1/8, 1/3, 2/3, 1/5, 1/6, 1/16} (simplicity-matched random sets)."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la24_common import *
from la24_c2 import f3

N = int(os.environ.get('N', 3000))
la = la_lists()
fl = [l for l in la if any(a[1] for a in l['amts'])]
score = lambda lists, V: sum(f3([val(t, V) for t in l['amts']]) for l in lists)
obs = score(fl, CONV)
rng = random.Random(9)
present = sorted({x for l in fl for a in l['amts'] for x in a[1]})
vals = [CONV[k] for k in present]
perm = []
for _ in range(N):
    v = vals[:]; rng.shuffle(v)
    V = dict(CONV); V.update(dict(zip(present, v)))
    perm.append(score(fl, V))
# (b) letter-combination shuffle across fraction-bearing amounts
slots = [(i, j) for i, l in enumerate(fl) for j, a in enumerate(l['amts']) if a[1]]
combos = [fl[i]['amts'][j][1] for i, j in slots]
lsh = []
for _ in range(N):
    c = combos[:]; rng.shuffle(c)
    L2 = [{'amts': list(l['amts'])} for l in fl]
    for (i, j), cc in zip(slots, c): L2[i]['amts'][j] = (L2[i]['amts'][j][0], cc)
    lsh.append(score(L2, CONV))
SIMPLE = [Fr(1, 2), Fr(1, 4), Fr(3, 4), Fr(1, 8), Fr(1, 3), Fr(2, 3), Fr(1, 5), Fr(1, 6), Fr(1, 16)]
simp = [score(fl, {k: rng.choice(SIMPLE) for k in CONV}) for _ in range(N)]
out = {'present_letters': present, 'obs_CONV': obs}
for nm, a in (('value_permutation', perm), ('letter_shuffle', lsh), ('simple_random_V', simp)):
    a = np.array(a)
    out[nm] = {'mean': float(a.mean()), 'sd': float(a.std()), 'max': int(a.max()),
               'P': float(((a >= obs).sum() + 1) / (N + 1))}
print(json.dumps(out))
json.dump(out, open(os.path.join(CK, 'c2b.json'), 'w'), indent=1)
