"""pe11 cycle 3b: sign-switch calibration with a RECOVERED order.  Same planted ring and
'HSPR' header as cycle 3, but the order is searched on the 4 planted features only
(what a successful random-subset search would hand over).  Also reports the planted
header's own rank, and its rank in the cycle-3 runs."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, pickle
import numpy as np
from pe11_common import *  # noqa
import pe11_cycle3 as c3

def main():
    P, U = pickle.load(open(os.path.join(CKPT, 'frames.pkl'), 'rb'))
    pool = pe_pool(P)
    X, names, ids = pe_features(P, pool)
    prng = np.random.default_rng(77)
    PL = sorted(prng.choice(len(names), 4, replace=False).tolist())
    mtrue = prng.integers(0, S, len(ids))
    Xp = standardize(X).copy()
    for li, c in enumerate(PL):
        sig = np.cos(2 * np.pi * (mtrue - 3 * li) / S); sig /= sig.std()
        Xp[:, c] = np.sqrt(0.5) * Xp[:, c] + np.sqrt(0.5) * sig
    Pp = {k: dict(v) for k, v in P.items()}
    hr = np.random.default_rng(5)
    for r, tid in enumerate(ids):
        if mtrue[r] in (2, 3, 4) and hr.random() < 0.5:
            Pp[tid]['hdr'] = 'HSPR'
    z = c3.ring_fit(Xp[:, PL])
    acc = align_acc(z, mtrue)[0]
    res = c3.test('plant_ring4', z, Pp, ids, pool, Xp[:, PL], 21)
    # explicit HSPR statistic
    mask = np.array([Pp[t]['hdr'] == 'HSPR' for t in ids])
    rng = np.random.default_rng(1)
    obs = c3.arc_share(z, mask)
    null = [c3.arc_share(z[rng.permutation(len(z))], mask) for _ in range(1000)]
    out = {'acc': acc, 'test': res, 'HSPR': {'n': int(mask.sum()), 'arc': obs, 'null_mean': float(np.mean(null)),
                                             'z': float((obs - np.mean(null)) / np.std(null))}}
    print(json.dumps(out['HSPR']), acc)
    save('pe11_cycle3b.json', out)

if __name__ == '__main__':
    main()
