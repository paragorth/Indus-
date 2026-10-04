"""LA-26 cycle 2e: Linear B calibration of the word-level ring-path null (split from la26_c2b.py)."""

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
OUT = os.path.join(CK, 'c2b_lb.json')
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


# (4) LB calibration of the word-level null (moved from c2): LB sites as nodes, random 'ring'
# graphs at the LA graph's density; the document-permutation P must be ~uniform. The site-pair
# co-occurrence matrices of 500 permutations are computed once (sparse) and shared by all graphs.
import la15_common
import scipy.sparse as sp
lb = la15_common.load_lb()
lbsites = [s for s, c in collections.Counter(d['site'] for d in lb).most_common() if c >= 20][:8]
lbd = [d for d in lb if d['site'] in lbsites]
M, _ = doc_matrix(lbd, 'words')
M = sp.csr_matrix((M > 0).astype(np.float32))
lab = np.array([lbsites.index(d['site']) for d in lbd])


def co_of(lb_lab):
    O = sp.csr_matrix((np.ones(len(lb_lab), np.float32), (lb_lab, np.arange(len(lb_lab)))), shape=(len(lbsites), len(lb_lab)))
    S = ((O @ M) > 0).astype(np.float32)
    S = S[:, np.asarray(S.sum(0)).ravel() >= 2]
    return (S @ S.T).toarray()


iu = list(itertools.combinations(range(len(lbsites)), 2))
co0 = co_of(lab)
cos = [co_of(rng.permutation(lab)) for _ in range(500)]
RA = graph_sets(NODES_A)['RING']
nE = int(round(len(RA) / 2 / 15 * len(iu)))


def Fco(co, edges):
    den = sum(co[i, j] for i, j in iu)
    return sum(co[i, j] for i, j in edges) / den


pv = []
for g in range(1000):
    sel = rng.choice(len(iu), size=nE, replace=False)
    edges = [iu[k] for k in sel]
    f0 = Fco(co0, edges)
    fa = np.array([Fco(c, edges) for c in cos])
    pv.append(float((fa >= f0 - 1e-12).mean()))
pv = np.array(pv)
res['lb_calibration'] = dict(sites=lbsites, n_edges=nE, n_docs=len(lbd), n_graphs=1000, n_perm=500,
                             frac_p_le_05=float((pv <= 0.05).mean()), frac_p_ge_95=float((pv >= 0.95).mean()),
                             p_quantiles=[float(x) for x in np.quantile(pv, [0.1, 0.25, 0.5, 0.75, 0.9])])
print('lb_calibration', res['lb_calibration'], flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
