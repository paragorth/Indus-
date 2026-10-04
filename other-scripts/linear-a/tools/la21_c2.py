#!/usr/bin/env python3
"""LA-21 cycle 2 summary: real vs null spreads (100 restarts each), LL-only ablation, LA pair lists for the outside check."""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la21_report import load, analyse
from la21_common import *

KEYS = ['row70', 'col70', 'rowdec', 'coldec', 'crow_auc', 'col_auc']

def fam(tag, prefix):
    out = []
    for p in sorted(glob.glob(os.path.join(CK, f'{tag}_{prefix}*.npz'))):
        j = os.path.basename(p)[len(tag) + 1:-4]
        if j[len(prefix):].isdigit() or j == prefix: out.append((j, analyse(load(tag, j))[0]))
    return out

def show(name, rows):
    for k in KEYS:
        v = [o[k] for _, o in rows]
        print(f'  {name:8s} {k:9s} ' + ' '.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in v) +
              f'   mean {np.mean(v):.3f} sd {np.std(v):.3f}')

if __name__ == '__main__':
    print('== real vs nulls, tag c2 (100 restarts each)')
    groups_ = [('LA', 'LA'), ('LAshW', 'LAshW'), ('LAshG', 'LAshG'), ('LBs', 'LBs'), ('LBshW', 'LBshW'), ('LBshG', 'LBshG'),
               ('JPN', 'JPN'), ('GRC', 'GRC'), ('HAW', 'HAW')]
    F = {}
    for n, p in groups_:
        F[n] = fam('c2', p); print(n, [j for j, _ in F[n]]); show(n, F[n])
    la = F['LA'][0][1]
    for k in KEYS:
        for nn in ('LAshW', 'LAshG'):
            v = np.array([o[k] for _, o in F[nn]], float)
            print(f'LA {k} = {la[k]:.3f}; {nn} null {v.mean():.3f} +- {v.std():.3f}; z = {(la[k] - v.mean()) / (v.std() + 1e-9):.1f}; exceed {int((v >= la[k]).sum())}/{len(v)}')
    for k in ('crow_auc', 'col_auc'):
        lb = np.array([o[k] for _, o in F['LBs']]); lw = np.array([o[k] for _, o in F['LBshW']]); lg = np.array([o[k] for _, o in F['LBshG']])
        print(f'LB {k}: real {lb.mean():.3f} +- {lb.std():.3f}; W {lw.mean():.3f} +- {lw.std():.3f}; G {lg.mean():.3f} +- {lg.std():.3f}')
    print('== LL-only ablation (w_ocp = w_vin = 0), tag c2ll vs c2')
    for j in ('LA', 'LBs1', 'LAshW2', 'LBshW2'):
        try:
            a = analyse(load('c2ll', j))[0]
            print(f'  {j:7s} ' + ' '.join(f'{k} {a[k]:.3f}' if isinstance(a[k], float) else f'{k} {a[k]}' for k in KEYS))
        except FileNotFoundError: pass
    print('== LA (c1, 400 runs + c2, 100 runs): sign pairs with co-assignment >= 0.5, with LB-derived labels (outside check)')
    rows = np.concatenate([load('c1', 'LA')['rows'], load('c2', 'LA')['rows']]); cols = np.concatenate([load('c1', 'LA')['cols'], load('c2', 'LA')['cols']])
    signs = [str(s) for s in load('c1', 'LA')['signs']]
    PR, PC, pv = coassign(rows, cols)
    for nm, P, ix in (('ROW', PR, 0), ('COL', PC, 1)):
        prs = sorted(((P[i, j], signs[i], signs[j]) for i in range(len(signs)) for j in range(i + 1, len(signs)) if P[i, j] >= .5), reverse=True)
        ok = 0; tot = 0
        for p, a, b in prs:
            ta, tb = truth_lb(a), truth_lb(b)
            agree = None if (ta is None or tb is None) else (ta[ix] == tb[ix])
            if agree is not None: tot += 1; ok += agree
            print(f'  {nm} {p:.2f} {a}-{b} {"agree" if agree else ("-" if agree is None else "differ")}')
        print(f'  {nm}: {ok}/{tot} pairs agree with LB-derived values')
