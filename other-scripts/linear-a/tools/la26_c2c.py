"""LA-26 cycle 2f: graph-relabelling reference for the word-level F statistic.
The Linear B calibration (la26_c2b_lb.py) shows the document-permutation null rejects for ~38% of
RANDOM graphs, because real corpora have strong site-pair affinities. The fair reference is
therefore: among ALL graphs with as many edges as the ring graph, how many give F >= the ring
graph's F? Exhaustive for set A (C(15,6) = 5,005) and set B (C(36,6) = 1,947,792, sampled 200,000).
Also QAP (node relabellings) of the ring graph itself.
"""
import json, itertools
import numpy as np
from la26_common import *

rng = np.random.default_rng(2626)
out = {}
for name, nodes in (('A', NODES_A), ('B', NODES_B)):
    for admin in (False, True):
        docs = load_docs(nodes, admin_only=admin)
        M, _ = doc_matrix(docs, 'words')
        S = (site_sum([d['site'] for d in docs], nodes, (M > 0).astype(np.float32)) > 0)
        S = S[:, S.sum(0) >= 2].astype(np.float64)
        co = S @ S.T
        iu = list(itertools.combinations(range(len(nodes)), 2))
        w = np.array([co[i, j] for i, j in iu]); den = w.sum()
        R = ring_matrix(seal_incidence('core'), nodes)
        ring = np.array([R[i, j] > 0 for i, j in iu])
        f0 = w[ring].sum() / den
        k = int(ring.sum())
        if len(iu) <= 15:
            fs = np.array([w[list(c)].sum() / den for c in itertools.combinations(range(len(iu)), k)])
        else:
            fs = np.array([w[rng.choice(len(iu), k, replace=False)].sum() / den for _ in range(200000)])
        perms = list(itertools.permutations(range(len(nodes)))) if len(nodes) <= 7 else [rng.permutation(len(nodes)) for _ in range(20000)]
        fq = []
        for p in perms:
            Rp = R[np.ix_(p, p)]
            fq.append(sum(w[t] for t, (i, j) in enumerate(iu) if Rp[i, j] > 0) / den)
        fq = np.array(fq)
        key = f'{name}|{"adm" if admin else "all"}'
        out[key] = dict(F=float(f0), n_edges=k, p_all_graphs=float((fs >= f0 - 1e-12).mean()), n_graphs=len(fs),
                        p_qap=float((fq >= f0 - 1e-12).mean()), pair_words={f'{nodes[i]}-{nodes[j]}': int(c) for (i, j), c in zip(iu, w) if c})
        print(key, out[key], flush=True)
json.dump(out, open(os.path.join(CK, 'c2c.json'), 'w'), indent=1)
