"""v41 cycle 4: does the drift clock run along the binding order?
Cycle 3 found the hand-1 A herbal clock rising with foliation. Decompose and attack it:
  total rho(clock, folio order); within-quire rho (both demeaned by quire; null = pages
  permuted inside quires); between-quire rho of quire means (null = whole quires permuted).
  Orientation control: 1000 random sign patterns over the same 11 traits. If any composite
  correlates with foliation as well, the effect is generic topic drift along the book, not
  an A->B arrow. Per-trait rho shows which habits carry it. ZL3b and IT2a.
"""
import sys, os, json, random
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from v41_lib import *

GROUPS = {
    'H1A_herbal': (lambda p: p['hand'] == '1' and p['lang'] == 'A' and p['illus'] == 'H', 'none'),
    'H1A_all': (lambda p: p['hand'] == '1' and p['lang'] == 'A', 'illus'),
    'H2B_all': (lambda p: p['hand'] == '2' and p['lang'] == 'B', 'illus'),
    'H3B_stars': (lambda p: p['hand'] == '3' and p['lang'] == 'B' and p['illus'] == 'S', 'none'),
    'B_all': (lambda p: p['lang'] == 'B' and (p['hand'] or '') in ('2', '3', '5'), 'hand_illus'),
}
res, rows = {}, []
rng = np.random.default_rng(41)


def decomp(clock, order, quire):
    clock = np.asarray(clock, float); order = np.asarray(order, float); quire = np.asarray(quire)
    tot = spearmanr(clock, order)[0]
    c2 = clock.copy(); o2 = order.copy()
    qs = sorted(set(quire.tolist()))
    for q in qs:
        m = quire == q
        c2[m] -= c2[m].mean(); o2[m] -= o2[m].mean()
    within = spearmanr(c2, o2)[0]
    qm = [clock[quire == q].mean() for q in qs]; qo = [order[quire == q].mean() for q in qs]
    between = spearmanr(qm, qo)[0] if len(qs) >= 4 else float('nan')
    return tot, within, between, qs


for name in ('ZL3b', 'IT2a'):
    P = vpages(name)
    T, _ = make_traits(P)
    o = orient_from([p for p in P if p['lang'] == 'A'], [p for p in P if p['lang'] == 'B'], T)
    for g, (pred, sm) in GROUPS.items():
        pages = [p for p in P if pred(p)]
        strata = ['x'] * len(pages) if sm == 'none' else [p['illus'] if sm == 'illus' else p['hand'] + p['illus'] for p in pages]
        (X1, X2), names = rate_matrix(pages, T)
        D = demean((X1 + X2) / 2, strata); Zs = D / (D.std(0) + 1e-12)
        clock = (Zs * o).mean(1)
        order = np.array([p['order'] for p in pages]); quire = np.array([p['quire'] or '?' for p in pages])
        tot, within, between, qs = decomp(clock, order, quire)
        # nulls
        wn, bn = [], []
        for _ in range(2000):
            pi = perm_within(len(pages), quire, rng)
            wn.append(decomp(clock[pi], order, quire)[1])
        # whole quires permuted: reassign quire blocks to the quire positions
        qpos = {q: order[quire == q].mean() for q in qs}
        for _ in range(2000):
            sh = rng.permutation(len(qs))
            newpos = {q: qpos[qs[sh[i]]] for i, q in enumerate(qs)}
            o3 = np.array([newpos[q] + (order[i] - qpos[q]) * 1e-3 for i, q in enumerate(quire)])
            bn.append(spearmanr(clock, o3)[0])
        wn, bn = np.array(wn), np.array(bn)
        # orientation control: random sign patterns
        rs = []
        for _ in range(1000):
            s = rng.choice([-1., 1.], len(o))
            rs.append(abs(spearmanr((Zs * s).mean(1), order)[0]))
        rs = np.array(rs)
        per = {nm: round(float(spearmanr(D[:, j] * o[j], order)[0]), 2) for j, nm in enumerate(names)}
        r = dict(n=len(pages), nq=len(qs), total=float(tot), within=float(within), between=float(between),
                 p_within=float((1 + (np.abs(wn) >= abs(within)).sum()) / 2001),
                 p_blocks=float((1 + (np.abs(bn) >= abs(tot)).sum()) / 2001),
                 randsign_q=float((rs >= abs(tot)).mean()), per_trait=per)
        res[f'{name}_{g}'] = r
        print(name, g, r, flush=True)
        rows.append(f"| V-41.4.{len(rows)+1} | {name} {g}: oriented 11-trait clock (strata {sm}) against folio order; within-quire (pages permuted in quires, 2000x) and between-quire (whole quires permuted, 2000x) decomposition; 1000 random trait-sign patterns as orientation control | n {r['n']} in {r['nq']} quires; total rho {tot:+.2f} (quire-block p {r['p_blocks']:.3f}); within-quire {within:+.2f} (p {r['p_within']:.3f}); between quires {between:+.2f}; random signs reaching |rho| {r['randsign_q']:.3f}; per trait {', '.join(f'{k} {v:+.2f}' for k, v in per.items())} | see verdict |")
json.dump(res, open(os.path.join(CK, 'c4.json'), 'w'), default=float)
write_rows(os.path.join(CK, 'c4_rows.txt'), rows)
