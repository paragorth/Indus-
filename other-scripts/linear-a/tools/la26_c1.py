"""LA-26 cycle 1: do ring-linked sites share more words / signs / ligature variants / entry
structures than distance and corpus size predict?

Pipeline per node set x measure:
  Z_ij = (observed similarity - mean under document-label permutation within support) / sd
  T    = mean distance-residual of Z on ring-linked pairs - on unlinked pairs
Nulls for T: (a) QAP node relabelling of the ring graph (exact for <= 7 nodes),
             (b) degree-preserving curveball rewiring of the seal x site bipartite graph (5,000),
             (c) distance-only model (T is computed on residuals of Z ~ log distance).
Positive control: m planted word types shared along each real ring edge (and, as a false-positive
control, along the same number of random non-ring pairs), at real corpus size.
"""
import sys, json, itertools, time
import numpy as np
from la26_common import *

rng = np.random.default_rng(2601)
NPERM = 1000
OUT = os.path.join(CK, 'c1.json')
res = {}


def qap_p(Z, R, D, nodes, nsamp=5000):
    t0 = ring_stat(Z, R, D)
    n = len(nodes)
    if n <= 7:
        perms = list(itertools.permutations(range(n)))
    else:
        perms = [rng.permutation(n) for _ in range(nsamp)]
    ts = []
    for p in perms:
        p = np.array(p)
        ts.append(ring_stat(Z, R[np.ix_(p, p)], D))
    ts = np.array(ts); ts = ts[np.isfinite(ts)]
    return t0, float((ts >= t0 - 1e-12).mean()), len(ts)


def rewire_p(Z, inc, D, nodes, nsamp=5000):
    t0 = ring_stat(Z, ring_matrix(inc, nodes), D)
    ts = []
    for _ in range(nsamp):
        inc2 = curveball_inc(inc, None, rng, iters=60)
        t = ring_stat(Z, ring_matrix(inc2, nodes), D)
        if np.isfinite(t):
            ts.append(t)
    ts = np.array(ts)
    return float((ts >= t0 - 1e-12).mean()), len(ts)


t_start = time.time()
for nset_name, nodes in [('A', NODES_A), ('B', NODES_B), ('C', NODES_C)]:
    for admin in (False, True):
        docs = load_docs(nodes, admin_only=admin)
        D = dist_matrix(nodes)
        for mname, (key, fn) in MEASURES.items():
            Z, obs, mu = excess_z(docs, nodes, key, fn, NPERM, rng)
            for ver in ('core', 'strict', 'broad'):
                inc = seal_incidence(ver)
                R = ring_matrix(inc, nodes)
                t0, pq, nq = qap_p(Z, R, D, nodes)
                pr, nr = rewire_p(Z, inc, D, nodes, nsamp=2000)
                zr = upper(Z); rr = upper(R) > 0; dd = upper(D)
                ok = np.isfinite(zr)
                rho_d = float(np.corrcoef(zr[ok], np.log(dd[ok]))[0, 1])
                k = f'{nset_name}|{"adm" if admin else "all"}|{mname}|{ver}'
                res[k] = dict(T=t0, p_qap=pq, n_qap=nq, p_rewire=pr, n_rewire=nr,
                              z_ring=float(np.nanmean(zr[rr & ok])) if (rr & ok).any() else None,
                              z_non=float(np.nanmean(zr[~rr & ok])) if (~rr & ok).any() else None,
                              r_dist=rho_d, n_ring_pairs=int(rr.sum()), n_pairs=len(rr),
                              Z=[[None if not np.isfinite(x) else round(float(x), 2) for x in row] for row in Z])
                print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k].items() if a != 'Z'}, flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

# ------------------------------------------------------------- positive / false-positive control
print('planted controls', time.time() - t_start, flush=True)
ctrl = {}


def plant(docs, pairs, m, rng):
    out = [dict(d, words=list(d['words'])) for d in docs]
    bysite = collections.defaultdict(list)
    for i, d in enumerate(out):
        bysite[d['site']].append(i)
    c = 0
    for (a, b) in pairs:
        for _ in range(m):
            w = f'PLANT-{c}'; c += 1
            for s in (a, b):
                i = bysite[s][rng.integers(len(bysite[s]))]
                ws = out[i]['words']
                if ws:
                    ws[rng.integers(len(ws))] = w
                else:
                    ws.append(w)
    return out


for nset_name, nodes in [('A', NODES_A), ('C', NODES_C)]:
    docs = load_docs(nodes)
    D = dist_matrix(nodes)
    R = ring_matrix(seal_incidence('core'), nodes)
    ring_pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if R[i, j] > 0]
    non_pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if R[i, j] == 0]
    for m in (1, 2, 4, 8, 16):
        for mode in ('ring', 'random'):
            hits = 0; ps = []
            for rep in range(60):
                if mode == 'ring':
                    pairs = ring_pairs
                else:
                    allp = ring_pairs + non_pairs
                    sel = rng.choice(len(allp), size=len(ring_pairs), replace=False)
                    pairs = [allp[k] for k in sel]
                d2 = plant(docs, pairs, m, rng)
                Z, _, _ = excess_z(d2, nodes, 'words', shared_types, 150, rng)
                _, pq, _ = qap_p(Z, R, D, nodes)
                ps.append(pq); hits += pq <= 0.05
            k = f'{nset_name}|m={m}|{mode}'
            ctrl[k] = dict(detect=hits / 60, median_p=float(np.median(ps)))
            print(k, ctrl[k], flush=True)
        json.dump(dict(main=res, control=ctrl), open(OUT, 'w'), indent=1)
print('done', time.time() - t_start)
