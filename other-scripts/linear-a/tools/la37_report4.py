#!/usr/bin/env python3
"""LA-37 cycle 4 post-analysis: frequency-matched merge gains, LB control, LA outside check, agreement with T."""
import os, sys, json, collections
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr
import la37_common as K, la32_common as C

R = json.load(open(os.path.join(K.CK, 'c4.json')))
c2 = json.load(open(os.path.join(K.CK, 'c2.json')))
LA = K.la_units(); _, cla = K.alphabet(LA, 8)
LB = K.lb_units(); _, clb = K.alphabet(LB, 10)


def perm_auc(al, v, nperm=10000, seed=0):
    iu = np.triu_indices(len(al), 1)
    cv = [C.lb_cv(s.lower()) for s in al]
    ok = [i for i, x in enumerate(cv) if x is not None]
    cons = np.array([x[0] if x else None for x in cv], dtype=object); vow = np.array([x[1] if x else None for x in cv], dtype=object)
    m = np.array([cv[a] is not None and cv[b] is not None for a, b in zip(*iu)])
    def a_(c, w):
        sc = (c[iu[0]] == c[iu[1]]) & m; sv = (w[iu[0]] == w[iu[1]]) & m & ~sc; o = m & ~sc & ~sv
        return K.auc(v[sc], v[o]), K.auc(v[sv], v[o])
    obs = a_(cons, vow); rng = np.random.default_rng(seed); nc = []; nv = []
    for _ in range(nperm):
        c = cons.copy(); c[ok] = rng.permutation(cons[ok]); w = vow.copy(); w[ok] = rng.permutation(vow[ok])
        nc.append(a_(c, vow)[0]); nv.append(a_(cons, w)[1])
    nc, nv = np.array(nc), np.array(nv)
    return [round(obs[0], 3), round(float((1 + (nc >= obs[0]).sum()) / (nperm + 1)), 4), round(obs[1], 3), round(float((1 + (nv >= obs[1]).sum()) / (nperm + 1)), 4)]


for r in R:
    al = r['alph']; G = np.array(r['G']); iu = np.triu_indices(len(al), 1)
    if r['kind'] == 'LA':
        fm = K.freq_matched_pct(G, iu, al, cla)
        print('LA outside check (sameC AUC, P, sameV AUC, P) raw G', perm_auc(al, G), 'fm', perm_auc(al, fm))
        T = np.array(c2['T']); assert al == c2['alph']
        print('LA Spearman(G, T)', round(spearmanr(G, T).correlation, 3), 'fm', round(spearmanr(fm, T).correlation, 3))
        o = np.argsort(-fm)[:15]
        print('LA top fm', [(al[iu[0][j]], al[iu[1][j]], round(float(fm[j]), 3), round(float(G[j]), 2)) for j in o])
        for p in [('A', 'DA'), ('A', 'PA'), ('KU', 'KI'), ('TI', 'TE'), ('SI', 'TI'), ('RE', 'ME'), ('A', 'KU'), ('NA', 'NE')]:
            j = [k for k, (a, b) in enumerate(zip(*iu)) if {al[a], al[b]} == set(p)][0]
            print('  cand', p, 'G fm pct', round(float(fm[j]), 3), 'raw pct', round(float((G < G[j]).mean()), 3))
    if r['kind'] == 'LBfull':
        fm = K.freq_matched_pct(G, iu, al, clb)
        e = K.lb_eval(dict(alph=al, iu=iu, T=fm))
        print('LBfull fm', {g: e[g] for g in ('doublet', 'sameC', 'sameV')}, e['top20'], e['top20_pairs'][:10])
        st = K.stats(LB, al, K.doc_halves(LB, 0)); T = K.all3(st, iu)
        print('LB Spearman(G, T)', round(spearmanr(G, T).correlation, 3), 'fm', round(spearmanr(fm, T).correlation, 3))
