#!/usr/bin/env python3
"""LA-1 cycle 4: dry/liquid exclusivity. In Linear B, T is dry-only and S liquid-only.
Does any Linear A fraction letter behave like that? Liquid = OLE, VIN; dry = GRA, NI, OLIV, CYP, AROM.
Statistic: number of letters (n >= 8) whose family share is >= 0.9 one way. Control: shuffle letters
across quantities (10,000 runs). Planted control: relabel 20% of the J tokens on dry commodities as a
new dry-only letter and check the statistic detects it.
"""
import sys, os, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la1_common as L
random.seed(41)
LIQ = {'OLE', 'VIN'}; DRY = {'GRA', 'NI', 'OLIV', 'CYP', 'AROM'}
lb = [(c, u) for _, c, u in L.lb_rows() if c in LIQ | DRY and u in 'TSVZ']
t = L.table(lb, 1, 0)
print('Linear B liquid share by unit:', {u: f"{sum(t[u][c] for c in LIQ)}/{sum(t[u].values())}" for u in 'TSVZ'})
la = [(c, f) for c, f in L.la_rows() if c in LIQ | DRY]


def excl(rows, nmin=8):
    tt = L.table(rows, 1, 0); out = []
    for f in tt:
        n = sum(tt[f].values())
        if n < nmin: continue
        s = sum(tt[f][c] for c in LIQ) / n
        if s >= 0.9 or s <= 0.1: out.append((f, round(s, 2), n))
    return out


tt = L.table(la, 1, 0)
print('Linear A liquid share by letter:', {f: f"{sum(tt[f][c] for c in LIQ)}/{sum(tt[f].values())}" for f in sorted(tt, key=lambda f: -sum(tt[f].values()))})
ob = excl(la); print('exclusive letters (n>=8):', ob)
cs = [c for c, _ in la]; fs = [f for _, f in la]; nl = []
for _ in range(10000):
    random.shuffle(fs); nl.append(len(excl(list(zip(cs, fs)))))
print(f'null mean {sum(nl)/len(nl):.2f}, P(>= {len(ob)}) = {sum(v >= len(ob) for v in nl)/len(nl):.4f}')
hit = 0
for _ in range(200):
    rows = [(c, 'NEW' if f == 'J' and c in DRY and random.random() < 0.2 else f) for c, f in la]
    hit += any(f == 'NEW' for f, _, _ in excl(rows))
print(f'planted dry-only letter (~11 tokens) detected in {hit}/200')
