"""v41 cycle 4b: kill test for the hand-1 herbal quire-level rise: drop the late quires
(those after quire H, i.e. the herbal pages bound next to pharma) and re-test; also leave-one-quire-out."""
import sys, os, json
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from v41_lib import *
out = {}
for name in ('ZL3b', 'IT2a'):
    P = vpages(name)
    T, _ = make_traits(P)
    o = orient_from([p for p in P if p['lang'] == 'A'], [p for p in P if p['lang'] == 'B'], T)
    H = [p for p in P if p['hand'] == '1' and p['lang'] == 'A' and p['illus'] == 'H']
    (X1, X2), _ = rate_matrix(H, T)
    D = (X1 + X2) / 2; D = D - D.mean(0); clock = ((D / (D.std(0) + 1e-12)) * o).mean(1)
    order = np.array([p['order'] for p in H]); q = np.array([p['quire'] for p in H])
    qs = sorted(set(q), key=lambda x: order[q == x].mean())
    qm = {x: (float(clock[q == x].mean()), int((q == x).sum())) for x in qs}
    early = np.array([x <= 'H' for x in q])
    r_early = spearmanr(clock[early], order[early])[0]
    qe = [x for x in qs if x <= 'H']
    b_early = spearmanr([qm[x][0] for x in qe], range(len(qe)))[0]
    loo = {x: float(spearmanr(clock[q != x], order[q != x])[0]) for x in qs}
    out[name] = dict(quire_means=qm, early_only_rho=float(r_early), early_between=float(b_early), n_early=int(early.sum()), loo=loo)
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(CK, 'c4b.json'), 'w'))
