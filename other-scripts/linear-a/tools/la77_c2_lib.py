"""LA-77 helper: outflow z of a corpus through the cycle-2 pipeline (surviving rooting rules)."""
import numpy as np
from la77_geo import Engine, type_table, permute_sites


def _outflow(M):
    m = 1 - np.eye(M.shape[0]); o = M * m; a, b = o.sum(1), o.sum(0)
    return np.where(a + b > 0, (a - b) / np.maximum(a + b, 1e-12), 0.0), a + b


def outflow_z_for(docs, codes, rules, surv, rng, H=150, nperm=100):
    eng = Engine(type_table(docs)[0], H, rng, codes, ntop_pool=5000)
    for h in range(eng.H):
        eng.root_coef[h] = rules[surv[h % len(surv)]]
    O, mass = _outflow(eng.transmissions(eng.X(docs)))
    NO = np.array([_outflow(eng.transmissions(eng.X(permute_sites(docs, rng))))[0] for _ in range(nperm)])
    return (O - NO.mean(0)) / np.maximum(NO.std(0), 1e-9), mass
