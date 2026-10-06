"""v80 cycle 2b: what is the interior-order excess? Held-out MI between token (pre2) and interior relative position
(3 bins: early/middle/late interior), real vs interior shuffles; and a discreteness test: does the position law
look like columns (sharp per-slot sets, token concentrated in one slot) or a gradient (smooth drift)?"""
import sys, json, os
import numpy as np
import v80_lib as L
from collections import Counter, defaultdict

C = L.corpora()


def mi_heldout(T, rep='pre2', nb=3):
    t = T['r_' + rep]; m = (T['i'] > 0) & (T['i'] < T['n'] - 1) & (T['n'] >= 5)
    rel = (T['i'] - 1) / np.maximum(T['n'] - 3, 1)
    b = np.minimum((rel * nb).astype(int), nb - 1)
    d = (T['half'] == 0) & m; h = (T['half'] == 1) & m
    V = t.max() + 1
    Cn = np.zeros((nb, V)) + 0.5; np.add.at(Cn, (b[d], t[d]), 1)
    U = np.bincount(t[d], minlength=V) + 0.5
    Pc = Cn / Cn.sum(1, keepdims=True); Pu = U / U.sum()
    g = np.log2(Pc[b[h], t[h]] / Pu[t[h]]).mean()
    return float(g), Cn


def run(nm, sec=None, seeds=5):
    P = C[nm][1]
    if sec: P = [p for p in P if p['sec'] in sec]
    g, Cn = mi_heldout(L.flatten(P))
    ns = [mi_heldout(L.flatten(L.shuffle_interior(P, 9000 + s)))[0] for s in range(seeds)]
    return g, np.mean(ns), np.std(ns), Cn


out = {}
for nm, sec in [('VOY_ZL', None), ('VOY_IT', None), ('VOY_ZL', 'S'), ('VOY_ZL', 'H'), ('VOY_ZL', 'B'), ('VOY_ZL', 'P'),
                ('VOY_ZL', 'CTA'), ('VOY_IT', 'S'), ('P_ISID', None), ('P_GERM', None), ('P_BRUM', None), ('G_MK2', None),
                ('G_JUNC', None), ('G_STACK', None), ('T_ALF', None), ('T_ALFd', None), ('T_KALd', None), ('T_HORd', None),
                ('T_CONd', None), ('T_DOSd', None)]:
    g, mu, sd, Cn = run(nm, sec)
    key = nm + ('@' + sec if sec else '')
    out[key] = dict(g=g, null=mu, sd=sd, z=(g - mu) / (sd + 1e-9))
    print('%-12s heldout pos-MI %.4f null %.4f +- %.4f excess %+.4f z %.1f' % (key, g, mu, sd, g - mu, (g - mu) / (sd + 1e-9)), flush=True)
# token-level read-off for the stars section: which pre2 classes lean early / late in the interior
P = [p for p in C['VOY_ZL'][1] if p['sec'] == 'S']
T = L.flatten(P)
X = L.V72.extract(P, L.E1C)
toks = [w[:2] for p in X for l in p['lines'] for w in l['w']]
m = (T['i'] > 0) & (T['i'] < T['n'] - 1) & (T['n'] >= 5)
rel = (T['i'] - 1) / np.maximum(T['n'] - 3, 1)
by = defaultdict(list)
for k, r, mm in zip(toks, rel, m):
    if mm: by[k].append(r)
lean = sorted(((np.mean(v) - 0.5, len(v), k) for k, v in by.items() if len(v) >= 40))
out['S_lean'] = [(k, n, round(float(x), 3)) for x, n, k in lean]
print('early:', lean[:6]); print('late:', lean[-6:])
json.dump(out, open(os.path.join(L.CK, 'c2b.json'), 'w'), default=str)
