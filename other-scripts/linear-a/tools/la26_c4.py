"""LA-26 cycle 4: separate 'ring-linked' from 'not Khania'.

Uses the cycle-1 excess-similarity matrices Z (document-label permutation within support,
1,000 permutations; data/la26_ckpt/c1.json). Two partial tests:
 (a) Khania removed: T over the 10 pairs among HT KN ZA GO TH (ring 6, unlinked 4), exact QAP
     over 5! relabellings.
 (b) Khania kept but given its own covariate: Z ~ log distance + KH-indicator, residual T on
     ring vs unlinked, exact QAP over 6! relabellings of the ring graph (KH indicator fixed).
Also a size covariate (log of the smaller site's document count) in both.
"""
import json, itertools
import numpy as np
from la26_common import *

r = json.load(open(os.path.join(CK, 'c1.json')))
nodes = NODES_A
docs = load_docs(nodes)
size = collections.Counter(d['site'] for d in docs)
D = dist_matrix(nodes)
R = ring_matrix(seal_incidence('core'), nodes)
out = {}


def T(Z, Rm, keep, covs):
    iu = [(i, j) for i, j in itertools.combinations(range(len(nodes)), 2) if keep(i, j)]
    z = np.array([Z[i][j] for i, j in iu], float)
    rr = np.array([Rm[i, j] > 0 for i, j in iu])
    X = [np.array([c(i, j) for i, j in iu], float) for c in covs]
    if rr.all() or not rr.any():
        return np.nan
    e = resid_on(z, X)
    return float(e[rr].mean() - e[~rr].mean())


cov_d = lambda i, j: math.log(D[i, j])
cov_kh = lambda i, j: float('KH' in (nodes[i], nodes[j]))
cov_sz = lambda i, j: math.log(min(size[nodes[i]], size[nodes[j]]))
ikh = nodes.index('KH')
for docset in ('all', 'adm'):
    for m in ('words', 'signs', 'lvar', 'struct'):
        Z = r[f'A|{docset}|{m}|core']['Z']
        Z = [[0.0 if x is None else x for x in row] for row in Z]
        for name, keep, covs, permset in [
            ('noKH', lambda i, j: ikh not in (i, j), [cov_d, cov_sz], [p for p in itertools.permutations(range(6)) if p[ikh] == ikh]),
            ('KHcov', lambda i, j: True, [cov_d, cov_sz, cov_kh], list(itertools.permutations(range(6)))),
        ]:
            t0 = T(Z, R, keep, covs)
            ts = []
            for p in permset:
                p = np.array(p)
                ts.append(T(Z, R[np.ix_(p, p)], keep, covs))
            ts = np.array(ts); ts = ts[np.isfinite(ts)]
            out[f'{docset}|{m}|{name}'] = dict(T=t0, p=float((ts >= t0 - 1e-12).mean()), n=len(ts))
            print(docset, m, name, out[f'{docset}|{m}|{name}'], flush=True)
json.dump(out, open(os.path.join(CK, 'c4.json'), 'w'), indent=1)
