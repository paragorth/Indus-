"""pe27 cycle 3: rate LISTS inside one tablet (>= 2 disjoint count/capacity pairs at the same exact
ratio, as in 'N workers x rate' lists), neighbouring-line pairs, tablet sums, all system pairs.
Null: Y values re-dealt among all Y lines of the tablets carrying both systems (line positions
kept), 200 times.  Controls: Ur III, planted rate lists."""
import json, sys, os, time
import numpy as np
from fractions import Fraction as Fr
from collections import Counter
from scipy.stats import poisson
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe27_common import *  # noqa

NP = int(os.environ.get('NP', 200))
res = {}
t0 = time.time()


def kmax(qx, qy):
    """max over r of the number of disjoint (x, y) pairs with y = r x, and that r."""
    cx = Counter(q['v'] for q in qx)
    cy = Counter(q['v'] for q in qy)
    best, br = 0, None
    for r in {y / x for x in cx for y in cy}:
        m = sum(min(n, cy.get(x * r, 0)) for x, n in cx.items())
        if m > best or (m == best and br is not None and r < br):
            best, br = m, r
    return best, br


def shuffle_vals(ys, rng):
    vals = [q['v'] for qy in ys for q in qy]
    perm = rng.permutation(len(vals))
    out, k = [], 0
    for qy in ys:
        nq = []
        for q in qy:
            q = dict(q); q['v'] = vals[perm[k]]; k += 1
            nq.append(q)
        out.append(nq)
    return out


def stats(xs, ys):
    K = [kmax(a, b) for a, b in zip(xs, ys)]
    k2 = sum(1 for k, _ in K if k >= 2)
    k3 = sum(1 for k, _ in K if k >= 3)
    rs = Counter(str(r) for k, r in K if k >= 2)
    # neighbouring-line exact ratios
    adj = Counter()
    for a, b in zip(xs, ys):
        for r in pairs_ratios(a, b, adj=True):
            adj[r] += 1
    # tablet sums
    sums = Counter(sum(q['v'] for q in b) / sum(q['v'] for q in a) for a, b in zip(xs, ys))
    return {'k2': k2, 'k3': k3, 'rs': rs, 'adj': adj, 'sums': sums,
            'adj_max': max(adj.values()) if adj else 0, 'sum_max': max(sums.values()) if sums else 0}


def run(xs, ys, name, nperm=NP, seed=0):
    rng = np.random.default_rng(seed)
    R = stats(xs, ys)
    N = [stats(xs, shuffle_vals(ys, rng)) for _ in range(nperm)]
    out = {'n_tab': len(xs)}
    for key in ('k2', 'k3', 'adj_max', 'sum_max'):
        nv = np.array([n[key] for n in N])
        out[key] = {'real': R[key], 'null_mean': round(float(nv.mean()), 2),
                    'null_q95': float(np.quantile(nv, 0.95)), 'p': float((1 + (nv >= R[key]).sum()) / (1 + nperm))}
    # ratios most used in within-tablet rate lists vs null
    tot = Counter()
    for n in N:
        tot.update(n['rs'])
    out['rate_list_ratios'] = [(r, c, round(tot[r] / nperm, 2)) for r, c in R['rs'].most_common(10)]
    tot = Counter()
    for n in N:
        tot.update(n['adj'])
    sc = []
    for r, c in R['adj'].items():
        mu = max(tot[r] / nperm, 0.05)
        sc.append((float(-np.log10(max(poisson.sf(c - 1, mu), 1e-300))), str(r), c, round(mu, 2)))
    sc.sort(reverse=True)
    out['adj_top'] = sc[:8]
    tot = Counter()
    for n in N:
        tot.update(n['sums'])
    out['sum_top'] = [(str(r), c, round(tot[r] / nperm, 2)) for r, c in R['sums'].most_common(6)]
    print(name, json.dumps(out, default=str), round(time.time() - t0), flush=True)
    return out


if __name__ == '__main__':
    U = ur3_tablets()
    ux, uy, _ = split_sys(U, 'CNT', 'CAP')
    rng = np.random.default_rng(3)
    idx = rng.choice(len(ux), 257, replace=False)
    res['ur3_257'] = run([ux[i] for i in idx], [uy[i] for i in idx], 'UR3_257', seed=1)
    T = pe_tablets()
    # planted rate lists: on 8% of tablets with >= 2 count and >= 2 capacity lines, two capacity
    # lines = 30 x two count lines
    Tp = []
    rp = np.random.default_rng(4)
    for t in T:
        Q = [dict(q) for q in t['Q']]
        a = [q for q in Q if q['sys'] == 'CNT']
        b = [q for q in Q if q['sys'] == 'CAP']
        if len(a) >= 2 and len(b) >= 2 and rp.random() < 0.15:
            b[0]['v'] = a[0]['v'] * 30
            b[1]['v'] = a[1]['v'] * 30
        Tp.append({'id': t['id'], 'site': t['site'], 'Q': Q})
    a, b, _ = split_sys(Tp, 'CNT', 'CAP')
    res['plant_list30'] = run(a, b, 'PLANT_LIST30', seed=2)
    for X, Y in (('CNT', 'CAP'), ('CNT', 'CAP@'), ('CAP', 'CAP@'), ('CNT', 'CNT@'), ('CNT@', 'CAP'), ('CNT', 'B')):
        a, b, ids = split_sys(T, X, Y)
        if len(a) >= 5:
            res[f'pe_{X}_{Y}'] = run(a, b, f'PE_{X}_{Y}', seed=5)
    # sites
    for nm, f in (('susa', lambda s: s.startswith('Susa')), ('plateau', lambda s: not s.startswith('Susa'))):
        Ts = [t for t in T if f(t['site'])]
        a, b, _ = split_sys(Ts, 'CNT', 'CAP')
        if len(a) >= 5:
            res[f'site_{nm}'] = run(a, b, f'SITE_{nm}', seed=6)
    json.dump(res, open(os.path.join(CK, 'cycle3.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
