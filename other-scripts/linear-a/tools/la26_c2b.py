"""LA-26 cycle 2b: is the word-level ring signal (c2) the ring graph, or just 'Khania is a
lexical island'?

(1) Competing graphs scored with the same F statistic and document-permutation null:
    RING (core seal graph), KHISO (every pair linked except pairs with Khania),
    HUB (every pair with HT linked), DIST (pairs under the median distance), and RING minus KH-free
    information. The ring graph has to beat the graphs that encode no seal information.
(2) Site jackknife: F and its document-permutation P with each site dropped.
(3) Site-graph null that keeps each site's degree: double-edge swaps on the site graph (NODES_B),
    so isolated sites stay isolated and only the placement of edges among linked sites varies.
"""
import json, itertools
import numpy as np
from la26_common import *
from la26_c2_lib import site_sets, F_stat


class Fast:
    """Vectorized F: co-occurrence of word types between site pairs from a doc x type presence matrix."""
    def __init__(self, docs, nodes, key='words'):
        self.M, _ = doc_matrix(docs, key)
        self.M = (self.M > 0).astype(np.float32)
        self.M = self.M[:, self.M.sum(0) >= 2]
        self.nodes = nodes
        self.lab = [d['site'] for d in docs]; self.sup = [d['support'] for d in docs]

    def F(self, Rset, labels=None):
        S = site_sum(labels or self.lab, self.nodes, self.M) > 0
        S = S[:, S.sum(0) >= 2].astype(np.float32)
        co = (S.astype(np.float64) @ S.T.astype(np.float64))
        n = len(self.nodes)
        num = den = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                den += co[i, j]
                num += co[i, j] * ((self.nodes[i], self.nodes[j]) in Rset)
        return float(num / den) if den else np.nan


rng = np.random.default_rng(2612)
OUT = os.path.join(CK, 'c2b.json')
res = {}


def graph_sets(nodes):
    R = ring_matrix(seal_incidence('core'), nodes)
    D = dist_matrix(nodes)
    med = np.median([D[i, j] for i, j in itertools.combinations(range(len(nodes)), 2)])
    G = {}
    G['RING'] = {(a, b) for i, a in enumerate(nodes) for j, b in enumerate(nodes) if R[i, j] > 0}
    G['KHISO'] = {(a, b) for a in nodes for b in nodes if a != b and 'KH' not in (a, b)}
    G['HUB'] = {(a, b) for a in nodes for b in nodes if a != b and 'HT' in (a, b)}
    G['DIST'] = {(a, b) for i, a in enumerate(nodes) for j, b in enumerate(nodes) if i != j and D[i, j] <= med}
    return G


def docperm(docs, Rset, nperm=2000, nodes=None):
    fx = Fast(docs, nodes)
    f0 = fx.F(Rset)
    assert abs(f0 - F_stat(site_sets(docs), Rset)[0]) < 1e-5
    fa = np.array([fx.F(Rset, perm_within_support(fx.lab, fx.sup, rng)) for _ in range(nperm)])
    fa = fa[np.isfinite(fa)]
    return f0, float(fa.mean()), float((fa >= f0).mean()), float((f0 - fa.mean()) / (fa.std() + 1e-12))


for nset_name, nodes in [('A', NODES_A), ('B', NODES_B)]:
    docs = load_docs(nodes)
    for gname, Rset in graph_sets(nodes).items():
        f0, mu, p, z = docperm(docs, Rset, nodes=nodes)
        res[f'{nset_name}|{gname}'] = dict(F=f0, null=mu, p=p, z=z, n_edges=len(Rset) // 2)
        print(nset_name, gname, res[f'{nset_name}|{gname}'], flush=True)
    # jackknife
    for drop in nodes:
        nd = [s for s in nodes if s != drop]
        dd = [d for d in docs if d['site'] != drop]
        Rset = graph_sets(nd)['RING']
        if not Rset:
            continue
        f0, mu, p, z = docperm(dd, Rset, 1000, nodes=nd)
        res[f'{nset_name}|drop_{drop}'] = dict(F=f0, null=mu, p=p, z=z)
        print(nset_name, 'drop', drop, res[f'{nset_name}|drop_{drop}'], flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)

# (3) degree-preserving site-graph swaps
nodes = NODES_B
docs = load_docs(nodes)
sets = site_sets(docs)
R0 = graph_sets(nodes)['RING']
E0 = sorted({tuple(sorted(e)) for e in R0})
f0 = F_stat(sets, R0)[0]
fs = []
for _ in range(5000):
    E = [list(e) for e in E0]
    for _ in range(30):
        i, j = rng.integers(len(E)), rng.integers(len(E))
        if i == j:
            continue
        a, b = E[i]; c, d = E[j]
        if rng.random() < 0.5:
            c, d = d, c
        new1, new2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        cur = {tuple(sorted(e)) for e in E}
        if a == d or c == b or new1 in cur or new2 in cur or new1 == new2:
            continue
        E[i], E[j] = list(new1), list(new2)
    Rs = {tuple(e) for e in E} | {(e[1], e[0]) for e in E}
    fs.append(F_stat(sets, Rs)[0])
fs = np.array(fs)
res['B|degree_swap'] = dict(F=f0, null=float(fs.mean()), p=float((fs >= f0 - 1e-12).mean()),
                            distinct=int(len(set(np.round(fs, 6)))))
print(res['B|degree_swap'])
json.dump(res, open(OUT, 'w'), indent=1)

# (4) the Linear B calibration is in la26_c2b_lb.py
