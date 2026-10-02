#!/usr/bin/env python3
"""2b'': Do fraction letters depend on the commodity they measure?
Commodity of a quantity = the last commodity logogram seen earlier on the same
tablet (logograms carry over down a list). Control: permute letters across quantities.
"""
import json, os, random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, '..', 'data', 'corpus.json')))
random.seed(4)
MAIN = {'GRA', 'VIN', 'OLE', 'OLIV', 'CYP', 'VIR', 'AROM', 'HIDE', 'NI'}

rows = []
for i in C:
    cur = None
    for t in i['tokens']:
        if t['t'] == 'logo':
            b = t['v'].split('+')[0].lstrip('*')
            if b in MAIN: cur = b
        elif t['t'] == 'word' and t['s'] == ['NI']:
            cur = 'NI'                                   # sign NI used alone in commodity lists
        elif t['t'] == 'num' and t['frac'] and cur:
            for f in t['frac']: rows.append((cur, f))
tab = defaultdict(Counter)
for c, f in rows: tab[c][f] += 1
letters = sorted({f for _, f in rows}, key=lambda f: -sum(1 for _, g in rows if g == f))
print('fraction letters by commodity (rows=commodity):')
print('       ' + ' '.join(f'{l:>3s}' for l in letters))
for c in sorted(tab, key=lambda c: -sum(tab[c].values())):
    print(f'{c:6s} ' + ' '.join(f'{tab[c].get(l,0):3d}' for l in letters))

def chi(rows):
    tc = Counter(c for c, _ in rows); tf = Counter(f for _, f in rows); n = len(rows)
    ob = Counter(rows); x = 0
    for c in tc:
        for f in tf:
            e = tc[c] * tf[f] / n
            x += (ob.get((c, f), 0) - e) ** 2 / e
    return x
x0 = chi(rows)
cs = [c for c, _ in rows]; fs = [f for _, f in rows]
null = []
for _ in range(5000):
    random.shuffle(fs); null.append(chi(list(zip(cs, fs))))
print(f'n={len(rows)} chi2 obs {x0:.1f}, permutation null mean {sum(null)/len(null):.1f}, P(>=obs) {sum(v >= x0 for v in null)/len(null):.4f}')
