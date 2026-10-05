"""v63 cycle 2b: is the Voynich's oriented near-pair excess adjacency-specific (slip-like) or an order
preference that also holds one word apart and in position-matched shuffles (syntax/layout-like)?
Per operator: orientation z at lag 1, lag 2, lag 3, and in 20 within-folio column shuffles (mean, sd); examples."""
import os, sys, json, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v63_lib as L
from v63_c2 import counts, agg, zset, inv_full, column_shuffle_folio, script

def table(name, zl, nsh=20):
    F = set(s['folio'] for s in zl)
    c = {lag: agg(counts(zl, F, lag), F) for lag in (1, 2, 3)}
    sh = [agg(counts(column_shuffle_folio(zl, k), F, 1), F) for k in range(nsh)]
    ops = [o for o in c[1] if '+' not in o and c[1][o] + c[1][inv_full(o)] >= 25 and c[1][o] > c[1][inv_full(o)]]
    rows = []
    for o in ops:
        z1 = zset(c[1], [o]); z2 = zset(c[2], [o]); z3 = zset(c[3], [o])
        zs = [zset(s, [o]) for s in sh]
        rows.append((z1, o, c[1][o], c[1][inv_full(o)], z2, z3, float(np.mean(zs)), float(np.std(zs))))
    rows.sort(reverse=True)
    print(name)
    for r in rows[:20]:
        print('  %-10s n %4d vs %4d  z1 %+5.2f  z2 %+5.2f  z3 %+5.2f  colshuf %+5.2f +- %.2f  excess %+5.2f' % (r[1], r[2], r[3], r[0], r[4], r[5], r[6], r[7], (r[0]-r[6])/max(r[7],0.3)))
    # aggregate: sum over all ops with |z1|>=2 of (z1 - z_shuffle) and of (z1 - z2)
    tot1 = sum(c[1][o] - c[1][inv_full(o)] for _, o, *x in rows); 
    return rows

zl = L.voynich_paras('ZL3b'); it = L.voynich_paras('IT2a')
R = {'ZL': table('V-ZL3b', zl), 'IT': table('V-IT2a', it)}
# examples for top ZL ops
F = set(s['folio'] for s in zl)
ex = defaultdict(Counter)
for s in zl:
    ws = s['words']
    for a, b in zip(ws, ws[1:]):
        sc = script(b, a)
        if sc and len(sc) == 1: ex[sc[0]][(''.join(a), ''.join(b))] += 1
for _, o, *x in R['ZL'][:10]:
    print(o, ex[o].most_common(8), ' | inverse:', ex[inv_full(o)].most_common(4))
json.dump({k: v[:60] for k, v in R.items()}, open(os.path.join(L.CK, 'c2b_ops.json'), 'w'))
