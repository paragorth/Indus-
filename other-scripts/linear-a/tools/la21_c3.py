#!/usr/bin/env python3
"""LA-21 cycle 3: held-out replication across disjoint halves and grid-size sensitivity."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la21_report import load, analyse
from la21_common import *

def rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r

def pair(ja, jb, tag='c3'):
    oa, sa, PRa, PCa, _ = analyse(load(tag, ja)); ob, sb, PRb, PCb, _ = analyse(load(tag, jb))
    sh = [s for s in sa if s in sb]; ia = [sa.index(s) for s in sh]; ib = [sb.index(s) for s in sh]
    iu = np.triu_indices(len(sh), 1)
    out = {}
    for nm, A, B in (('row', PRa, PRb), ('col', PCa, PCb)):
        a = A[np.ix_(ia, ia)][iu]; b = B[np.ix_(ib, ib)][iu]
        rho = np.corrcoef(rank(a), rank(b))[0, 1]
        sel = a >= .5
        out[nm] = (rho, int(sel.sum()), float(b[sel].mean()) if sel.any() else float('nan'), float(b.mean()))
    return len(sh), out, oa, ob

if __name__ == '__main__':
    print('== held-out halves: Spearman rho of co-assignment over shared signs; pairs >= 0.5 in half A -> mean in half B (vs all pairs)')
    for ja, jb in (('LAHT', 'LAnonHT'), ('LBKN1', 'LBPY1'), ('LBKN2', 'LBPY2'), ('LAHTsh1', 'LAnonHTsh1'), ('LBKNsh1', 'LBPYsh1')):
        try: n, o, oa, ob = pair(ja, jb)
        except FileNotFoundError: print(ja, jb, 'missing'); continue
        print(f'{ja:8s} vs {jb:10s} shared {n}: ' + '; '.join(
            f"{k} rho {v[0]:.3f}, {v[1]} pairs -> {v[2]:.3f} (all {v[3]:.3f})" for k, v in o.items()) +
            f" | truth cRowAUC {oa['crow_auc']:.3f}/{ob['crow_auc']:.3f} colAUC {oa['col_auc']:.3f}/{ob['col_auc']:.3f}"
            f" row70 {oa['row70']}/{ob['row70']}")
    print('== grid size (100 restarts)')
    for g in ('c3g13x4', 'c3g18x5', 'c3g15x3', 'c3g17x6'):
        for j in ('LA', 'LBs1', 'LAshW2'):
            try: o = analyse(load(g, j))[0]
            except FileNotFoundError: print(g, j, 'missing'); continue
            print(f"{g:8s} {j:7s} row70 {o['row70']:3d} col70 {o['col70']:3d} cRowAUC {o['crow_auc']:.3f} colAUC {o['col_auc']:.3f}")
