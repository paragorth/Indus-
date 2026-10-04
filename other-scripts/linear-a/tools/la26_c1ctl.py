"""LA-26 cycle 1, controls (split from la26_c1.py for speed on a loaded machine).
Positive control: m planted word types shared along each real core ring edge, inserted into random
documents of the two sites (replacing an existing word), at real corpus size; detection = QAP P <= 0.05
on the distance-residual statistic T. False-positive control: the same number of planted types on
randomly chosen site pairs (ring or not).
"""
import json, itertools, time, collections
import numpy as np
from la26_common import *

rng = np.random.default_rng(2611)
OUT = os.path.join(CK, 'c1ctl.json')
t_start = time.time()
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


for nset_name, nodes in [('A', NODES_A)]:
    docs = load_docs(nodes)
    D = dist_matrix(nodes)
    R = ring_matrix(seal_incidence('core'), nodes)
    ring_pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if R[i, j] > 0]
    non_pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if R[i, j] == 0]
    for m in (2, 4, 8, 16):
        for mode in ('ring', 'random'):
            hits = 0; ps = []
            for rep in range(30):
                if mode == 'ring':
                    pairs = ring_pairs
                else:
                    allp = ring_pairs + non_pairs
                    sel = rng.choice(len(allp), size=len(ring_pairs), replace=False)
                    pairs = [allp[k] for k in sel]
                d2 = plant(docs, pairs, m, rng)
                Z, _, _ = excess_z(d2, nodes, 'words', shared_types, 100, rng)
                _, pq, _ = qap_p(Z, R, D, nodes)
                ps.append(pq); hits += pq <= 0.05
            k = f'{nset_name}|m={m}|{mode}'
            ctrl[k] = dict(detect=hits / 30, median_p=float(np.median(ps)))
            print(k, ctrl[k], flush=True)
        json.dump(dict(main=res, control=ctrl), open(OUT, 'w'), indent=1)
print('done', time.time() - t_start)
