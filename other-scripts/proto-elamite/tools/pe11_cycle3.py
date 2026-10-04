"""pe11 cycle 3: DO SIGNS SWITCH AT THE INFERRED SEASON BOUNDARIES?
Month words, season words or seasonal operations should sit in a contiguous arc of the
inferred year.  For every sign that is NOT a feature (header signs; any non-final sign on
>= 8 frame tablets), statistic = largest share of its tablets inside any 3 adjacent
inferred months (arc share).  Nulls: (A) sign labels permuted across tablets;
(B) permuted only within 12 k-means clusters of the feature vectors (removes the
commodity profile, which also drives the phase).  FWER over all signs from the max of
the null.  Calibration:
  PLANT-RING: PE features with a strong planted ring (half the variance, 4 features)
    and a planted header 'HSPR' on half the tablets of true months 2-4;
  PLANT-SIM: the cycle-1 simulated office (a = 4) with its 'HSPR' header.
Then the real PE order (ALL features, cycle 2) is tested the same way.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, pickle
import numpy as np
from multiprocessing import Pool
from pe11_common import *  # noqa
from pe11_cycle1 import sim_tabs

NPERM = 1000


def kmeans(X, k, rng, it=50):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def arc_share(z, mask):
    if mask.sum() == 0:
        return 0.0
    h = np.bincount(z[mask], minlength=S)
    return float(max(h[[(i + d) % S for d in range(3)]].sum() for i in range(S)) / mask.sum())


def sign_sets(tabs, ids, pool):
    """Candidate signs: headers and non-final signs present on >= 8 frame tablets."""
    feat = set(pool)
    occ = defaultdict(set)
    for r, tid in enumerate(ids):
        t = tabs[tid]
        if t['hdr']:
            occ['H:' + t['hdr']].add(r)
        finals = set(e[0] for e in t['entries'])
        for s in t['signs']:
            if s not in feat and s not in finals:
                occ['S:' + s].add(r)
    return {k: np.array(sorted(v)) for k, v in occ.items() if len(v) >= 8}


def test(name, z, tabs, ids, pool, X, seed):
    rng = np.random.default_rng(seed)
    n = len(ids)
    occ = sign_sets(tabs, ids, pool)
    keys = sorted(occ)
    M = np.zeros((len(keys), n), bool)
    for i, k in enumerate(keys):
        M[i, occ[k]] = True
    obs = np.array([arc_share(z, M[i]) for i in range(len(keys))])
    lab = kmeans(standardize(X), 12, rng)
    nullA = np.zeros((NPERM, len(keys)))
    nullB = np.zeros((NPERM, len(keys)))
    strata = [np.where(lab == j)[0] for j in range(12)]
    for p in range(NPERM):
        pa = rng.permutation(n)
        pb = np.arange(n)
        for st in strata:
            pb[st] = rng.permutation(st)
        zA, zB = z[pa], z[pb]
        nullA[p] = [arc_share(zA, M[i]) for i in range(len(keys))]
        nullB[p] = [arc_share(zB, M[i]) for i in range(len(keys))]
    # standardise per sign, FWER on max z
    out = []
    for nm, N in (('A', nullA), ('B', nullB)):
        mu, sd = N.mean(0), N.std(0) + 1e-9
        zs = (obs - mu) / sd
        zmax_null = ((N - mu) / sd).max(1)
        hits = []
        for i in np.argsort(-zs)[:8]:
            hits.append({'sign': keys[i], 'n': int(M[i].sum()), 'arc': float(obs[i]), 'null_mean': float(mu[i]),
                         'z': float(zs[i]), 'p_fwer': float((1 + (zmax_null >= zs[i]).sum()) / (1 + NPERM)),
                         'arc_months': int(np.argmax([np.bincount(z[M[i]], minlength=S)[[(j + d) % S for d in range(3)]].sum() for j in range(S)]))})
        out.append({'null': nm, 'n_signs': len(keys), 'top': hits})
    print(name, json.dumps(out, default=lambda x: round(x, 3))[:1500], flush=True)
    return out


def ring_fit(X):
    rng = np.random.default_rng(31)
    r = loop_score(X, rng, restarts=8, sweeps=30)
    return r['z']


def main():
    P, U = pickle.load(open(os.path.join(CKPT, 'frames.pkl'), 'rb'))
    pool = pe_pool(P)
    X, names, ids = pe_features(P, pool)
    res = {}
    # PLANT-RING (same planted ring as cycle 2)
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
    Ps, ms = sim_tabs(P, pool, 4, 1.0, seed=504)
    Xs, _, _ = pe_features(Ps, pool, ids)
    zreal = np.array(json.load(open(os.path.join(CKPT, 'c2_ALL_real.json')))['z'])
    with Pool(2) as pl:
        zp, zs = pl.map(ring_fit, [Xp, Xs])
    res['plant_ring'] = {'acc': align_acc(zp, mtrue)[0], 'test': test('plant_ring', zp, Pp, ids, pool, Xp, 11)}
    res['plant_sim'] = {'acc': align_acc(zs, ms)[0], 'test': test('plant_sim', zs, Ps, ids, pool, Xs, 12)}
    res['real_ALL'] = {'test': test('real_ALL', zreal, P, ids, pool, X, 13)}
    save('pe11_cycle3.json', res)


if __name__ == '__main__':
    main()
