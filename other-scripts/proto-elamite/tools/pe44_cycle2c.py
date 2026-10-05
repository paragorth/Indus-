"""pe44 cycle 2c: held-out replication of the cycle-2 sign / position flags with z-based p-values
(the permutation p of cycle 2 bottoms out at 1/(reps+1), below BH resolution).
Flags: BH q < 0.05 on two-sided normal p from z3 (N3 = size-stratified shuffle) on tablet half A;
replicated if p3(z) < 0.05 with the same sign on half B.  6 random halves, 60 null reps each."""
import json, os, random, sys
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe44_common import *  # noqa
from pe44_cycle2 import survivors, fit_set, scan


def zq(res):
    ks = sorted(res)
    p = [2 * norm.sf(abs(res[k]['z3'])) for k in ks]
    return dict(zip(ks, p)), dict(zip(ks, P23.bh(p)))


def one(rep):
    surv, _ = survivors()
    tabs = pe_tabs()
    X, meta = corpus_matrix(tabs)
    ms = fit_set(X, surv, 1)
    rng = np.random.default_rng(1000 + rep)
    ids = sorted(tabs); rr = random.Random(100 + rep)
    A = {t for t in ids if rr.random() < 0.5}; B = set(ids) - A
    rA, _, _ = scan(tabs, ms, rng, tabset=A, reps=60)
    rB, _, _ = scan(tabs, ms, rng, tabset=B, reps=60)
    pA, qA = zq(rA); pB, _ = zq(rB)
    fl = [k for k in rA if qA[k] < 0.05 and not k.startswith('sys:')]
    ok = [k for k in fl if k in rB and pB[k] < 0.05 and np.sign(rB[k]['z3']) == np.sign(rA[k]['z3'])]
    return {'rep': rep, 'flagsA': {k: round(rA[k]['z3'], 2) for k in fl},
            'replicated': {k: round(rB[k]['z3'], 2) for k in ok}}


if __name__ == '__main__':
    with Pool(2) as pool:
        out = pool.map(one, range(6))
    for r in out:
        print(r, flush=True)
    json.dump(out, open(os.path.join(CK, 'c2c.json'), 'w'), indent=1)
