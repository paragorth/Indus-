#!/usr/bin/env python3
"""la74 cycle 2c: the big-amount sign narrowing, with the commodity held fixed. Commodity = base of the
last logogram before the quantity on the same document (carried over); permutation of integers within
site x commodity. Also: which signs carry the narrowing (share on N>=5 vs N==0)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from la74_common import *
import la74_entropy as E
d = json.load(open(os.path.join(DATA, 'corpus_ra.json')))
rows = []
for doc in d:
    com = 'none'
    for t in doc['tokens']:
        if t['t'] == 'logo': com = str(t['v']).split('+')[0]
        if t['t'] == 'num' and t['frac'] and t['st'] in ('read', 'damaged'):
            rows.append((tuple(t['frac']), t['v'], doc['site'], com))
V = vocab([r[0] for r in rows]); st = [norm(r[0], V) for r in rows]; N = np.array([r[1] for r in rows])
E.site = np.array([r[2] + '|' + r[3] for r in rows])
print('commodities of N>=5 fraction quantities:', Counter(r[3] for r in rows if r[1] >= 5).most_common(8))
print('commodities of N==0:', Counter(r[3] for r in rows if r[1] == 0).most_common(8))
print('REAL within site x commodity', json.dumps(E.perm_test(st, N, 3000, 7)))
# strata with both big and zero
for com in ['GRA', 'VIN', 'OLE', 'CYP', 'none', 'OLIV', '*308']:
    idx = [i for i, r in enumerate(rows) if r[3] == com]
    nb = sum(N[i] >= 5 for i in idx); nz = sum(N[i] == 0 for i in idx)
    if nb >= 5 and nz >= 5:
        E.site = np.array([rows[i][2] for i in idx])
        print(com, json.dumps(E.perm_test([st[i] for i in idx], N[idx], 2000, 3)))
cb = Counter(x for s, v in zip(st, N) if v >= 5 for x in s); cz = Counter(x for s, v in zip(st, N) if v == 0 for x in s)
tb, tz = sum(cb.values()), sum(cz.values())
print('share big vs zero:', [(k, round(cb[k] / tb, 2), round(cz[k] / tz, 2)) for k in V[:12]])
