"""LA-26 cycle 3: let the texts guess the ring graph blind.

Every one of the 2^15 = 32,768 possible site graphs on the six sites of the seal system
(HT KN ZA KH GO TH) is scored against Linear A only (no seal data), on a random half of the
documents: score = mean distance-residual excess similarity on the graph's edges minus off-edges,
summed over the four measures (words, signs, ligature variants, entry structure).
The top 1 % of graphs on half A vote an edge consensus; it is re-tested on held-out half B
(does it predict B's excess similarity?) and compared with the real seal graph (Jaccard),
against the Jaccard of random graphs of the same density.
Controls: planted shared words along the real ring edges (m = 4, 8 per edge) must make the
consensus converge on the ring graph; a document-label-shuffled corpus must not.
20 random splits per arm.
"""
import json, itertools, time
import numpy as np
from la26_common import *

rng = np.random.default_rng(2603)
OUT = os.path.join(CK, 'c3.json')
nodes = NODES_A
n = len(nodes)
pairs = list(itertools.combinations(range(n), 2))
D = dist_matrix(nodes)
ld = np.log(np.array([D[i, j] for i, j in pairs]))
R = ring_matrix(seal_incidence('core'), nodes)
ring_vec = np.array([R[i, j] > 0 for i, j in pairs])
G = np.array([[(g >> k) & 1 for k in range(len(pairs))] for g in range(1, 2 ** len(pairs) - 1)], dtype=bool)
dens = G.sum(1)


def pair_resid(docs):
    tot = np.zeros(len(pairs))
    for key, fn in MEASURES.values():
        Z, _, _ = excess_z(docs, nodes, key, fn, 60, rng)
        z = np.array([Z[i, j] for i, j in pairs])
        z[~np.isfinite(z)] = 0
        tot += resid_on(z, [ld])
    return tot


def score_all(e):
    on = (G * e).sum(1) / dens
    off = ((~G) * e).sum(1) / (len(pairs) - dens)
    return on - off


def jacc(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def plant(docs, m):
    out = [dict(d, words=list(d['words'])) for d in docs]
    bysite = collections.defaultdict(list)
    for i, d in enumerate(out):
        bysite[d['site']].append(i)
    c = 0
    for (i, j), r in zip(pairs, ring_vec):
        if not r:
            continue
        for _ in range(m):
            w = f'PLANT-{c}'; c += 1
            for s in (nodes[i], nodes[j]):
                k = bysite[s][rng.integers(len(bysite[s]))]
                ws = out[k]['words']
                if ws:
                    ws[rng.integers(len(ws))] = w
                else:
                    ws.append(w)
    return out


base = load_docs(nodes)
res = {}
k_ring = int(ring_vec.sum())
for arm in ('real', 'plant4', 'plant8', 'shuffled'):
    stats = []
    for sp in range(20):
        docs = base
        if arm.startswith('plant'):
            docs = plant(base, int(arm[5:]))
        if arm == 'shuffled':
            lab = perm_within_support([d['site'] for d in base], [d['support'] for d in base], rng)
            docs = [dict(d, site=l) for d, l in zip(base, lab)]
        # split stratified by site
        half = np.zeros(len(docs), bool)
        for s in nodes:
            idx = [i for i, d in enumerate(docs) if d['site'] == s]
            sel = rng.choice(idx, size=len(idx) // 2 if len(idx) > 1 else 1, replace=False)
            half[sel] = True
        A = [d for d, h in zip(docs, half) if h]
        B = [d for d, h in zip(docs, half) if not h]
        eA, eB = pair_resid(A), pair_resid(B)
        sc = score_all(eA)
        top = np.argsort(-sc)[: len(sc) // 100]
        vote = G[top].mean(0)
        cons = vote >= np.sort(vote)[-k_ring]      # consensus graph with as many edges as the ring graph
        # held-out: consensus edges vs others on half B
        heldout = float(eB[cons].mean() - eB[~cons].mean())
        rnd = np.array([jacc(G[g], ring_vec) for g in rng.choice(np.where(dens == cons.sum())[0], 2000)])
        J = jacc(cons, ring_vec)
        ring_rank_A = float((sc >= score_all(eA)[np.where((G == ring_vec).all(1))[0][0]]).mean())
        ring_T_B = float(eB[ring_vec].mean() - eB[~ring_vec].mean())
        stats.append(dict(J=float(J), p_J=float((rnd >= J).mean()), heldout=heldout, ring_rank_A=ring_rank_A,
                          ring_T_B=ring_T_B, vote=[round(float(v), 3) for v in vote]))
    J = np.array([s['J'] for s in stats]); pJ = np.array([s['p_J'] for s in stats])
    H = np.array([s['heldout'] for s in stats]); RR = np.array([s['ring_rank_A'] for s in stats])
    RT = np.array([s['ring_T_B'] for s in stats])
    vote = np.mean([s['vote'] for s in stats], 0)
    res[arm] = dict(J_mean=float(J.mean()), frac_pJ_le_05=float((pJ <= 0.05).mean()), heldout_mean=float(H.mean()),
                    heldout_frac_pos=float((H > 0).mean()), ring_rank_A_median=float(np.median(RR)),
                    ring_T_B_mean=float(RT.mean()), ring_T_B_frac_pos=float((RT > 0).mean()),
                    edge_vote={f'{nodes[i]}-{nodes[j]}': round(float(v), 3) for (i, j), v in zip(pairs, vote)})
    print(arm, json.dumps(res[arm]), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
res['ring_edges'] = [f'{nodes[i]}-{nodes[j]}' for (i, j), r in zip(pairs, ring_vec) if r]
json.dump(res, open(OUT, 'w'), indent=1)
