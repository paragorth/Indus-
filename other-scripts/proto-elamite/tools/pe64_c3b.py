#!/usr/bin/env python3
"""pe64 cycle 3b: where does the held-out ladder prediction come from?
usage: pe64_c3b.py CORPUS VARIANT DIRECTION
Re-scores the cycle 3 predictions (markers learnt on half A) on half B, split by count (1 vs >= 2), against
  swap  : capacity values from other tablets with the same notation shape (as in cycles 1-3)
  free  : capacity values from other tablets, any notation shape (breaks the shape -> value link too)
For count 1 the notation shape nearly fixes the value (2(N39B) 1(N24) = 60), so only count >= 2 lines test a RATE.
Output: data/pe64_ckpt/c3b_<CORPUS>_<VARIANT>_<DIR>.json
"""
import sys, os, json, random
import numpy as np
import pe64_lib as L

C, V, D = sys.argv[1], sys.argv[2], sys.argv[3]
rng = random.Random(L.seed('pe64c1' + C + V + D))
T = L.load(C)
if V == 'small':
    T = [T[i] for i in sorted(rng.sample(range(len(T)), 120))]
elif V.startswith('plant'):
    T, truth = L.plant_ladder(L.null_swap(T, random.Random(7)), rng, rungs=(20, 40, 80), per=int(V[5:] or 10))
elif V.startswith('nullreal'):
    T = L.null_swap(T, random.Random(int(V[8:] or 11)))
A, B = L.split(T, 'pe64' + C)
if D == 'BA':
    A, B = B, A
TB = [T[i] for i in B]
c3 = json.load(open(os.path.join(L.CK, 'c3_%s_%s_%s.json' % (C, V, D))))
markers = c3['markers']


def null_free(TT, r):
    TT = json.loads(json.dumps(TT))
    pool = [l['cap'] for t in TT for l in t['lines'] if l['cap'] is not None and not l['amb']]
    for t in TT:
        for l in t['lines']:
            if l['cap'] is not None and not l['amb']:
                l['cap'] = pool[r.randrange(len(pool))]
    return TT


def score(TT):
    out = {'c1': [0, 0], 'c2': [0, 0]}
    for t in TT:
        Ls = t['lines']
        for i, l in enumerate(Ls):
            if not l['cnt']:
                continue
            ms = [s for s in l['s'] if s in markers]
            if not ms:
                continue
            key = 'c1' if l['cnt'] == 1 else 'c2'
            out[key][1] += 1
            want = l['cnt'] * markers[ms[0]]
            if any(j != i and Ls[j]['cap'] is not None and abs(Ls[j]['cap'] - want) < 1e-6
                   for j in range(max(0, i - 5), min(len(Ls), i + 6))):
                out[key][0] += 1
    return out


real = score(TB)
res = {'real': real}
for name, fn in (('swap', lambda z: L.null_swap(TB, random.Random(8000 + z))),
                 ('free', lambda z: null_free(TB, random.Random(9000 + z)))):
    ns = [score(fn(z)) for z in range(200)]
    for key in ('c1', 'c2'):
        arr = np.array([n[key][0] for n in ns])
        res['%s_%s' % (name, key)] = {'mean': float(arr.mean()),
                                      'p': float((1 + (arr >= real[key][0]).sum()) / 201)}
res['markers'] = markers
json.dump(res, open(os.path.join(L.CK, 'c3b_%s_%s_%s.json' % (C, V, D)), 'w'))
print(C, V, D, json.dumps(res))
