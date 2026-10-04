"""pe34 cycle 2: does the PE transfer survive (A) tablet-disjoint estimation,
(B) which feature families carry it, (C) which modifiers carry it (jackknife),
(D) size-matched planted modifiers (realistic compound sizes)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe34_common import tokens, prep, perm_test, transfer, plant, sub, CK

EXC = ('X', 'xX')
T = tokens()
res = {'A': {}, 'B': {}, 'C': [], 'D': []}
PP = {}
for c in ['PE', 'ARCH', 'LINB']:
    P = prep(T[c], c)
    PP[c] = P
    real, nm, ns, p = perm_test(P, 2000, seed=3, exclude=EXC, stat='dot', disjoint=True)
    res['A'][c] = {'n': real['n'], 'dot': real.get('dot'), 'cos': real.get('cos'),
                   'null': [nm, ns], 'p': p}
    print('A', c, res['A'][c], flush=True)

P = PP['PE']
fam = {}
for i, k in enumerate(P['names']):
    f = k.split(':')[0] if ':' in k else {'first': 'pos', 'last': 'pos', 'alone': 'pos', 'logn': 'pos',
                                          'logq': 'qty', 'num': 'qty', 'line0': 'place', 'rev': 'place'}.get(k, k)
    fam.setdefault(f, []).append(i)
for f, cols in fam.items():
    for dj in (False, True):
        real, nm, ns, p = perm_test(sub(P, cols), 1000, seed=4, exclude=EXC, stat='dot', disjoint=dj)
        res['B'][f + ('_disj' if dj else '')] = {'n': real['n'], 'dot': real.get('dot'), 'null': nm, 'p': p}
        print('B', f, dj, res['B'][f + ('_disj' if dj else '')], flush=True)

mods = [m for _, _, m in P['comp']]
full = transfer(P, mods, EXC)
per = {}
for m in sorted(set(mods) - set(EXC)):
    tg = {c for c, _, mm in P['comp'] if mm == m}
    r = transfer(P, mods, EXC, targets=tg)
    if not r['n']:
        continue
    drop = [('X' if x == m else x) for x in mods]
    rd = transfer(P, drop, EXC)
    res['C'].append({'mod': m, 'n': r['n'], 'dot_own': r['dot'], 'cos_own': r['cos'],
                     'dot_without': rd['dot'], 'bases': sorted({b for _, b, mm in P['comp'] if mm == m})})
res['C'].sort(key=lambda x: -x['dot_own'])
for x in res['C']:
    print('C', x, flush=True)

sizes = [P['n'][c] for c, _, _ in P['comp']]
for strength in (0.0, 0.2, 0.4, 0.6):
    for seed in range(5):
        toks, planted, eff = plant(T['PE'], 'PE', 50 + seed, strength, n_mod=10, n_base=4, sizes=sizes)
        Q = prep(toks, 'PE')
        tg = set(planted)
        Q['comp'] = [x for x in Q['comp'] if x[0] in tg]
        real, nm, ns, p = perm_test(Q, 500, seed=seed, exclude=EXC, stat='dot')
        res['D'].append({'strength': strength, 'seed': seed, 'n': real['n'], 'dot': real.get('dot'),
                         'null': nm, 'p': p})
        print('D', res['D'][-1], flush=True)
json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), indent=1)
