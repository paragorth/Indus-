"""pe40 cycle 3: MASSIVE RANDOM ROTAS.  Thousands of random cyclic slot assignments of the players
(period P = 4..24), each hill-climbed on the fit half of the tablets so that co-occurring players sit in the
same or the next slot.  Survivors (best of each batch) are scored on held-out tablets by the ROTA INDEX
RI = P(held-out pair at cyclic distance 1) / mean P(distance >= 2).  A rota (handover between consecutive
groups) gives RI >> 1; disjoint teams give RI ~ 1 (no reason for neighbours).  Also SAME = P(d=0)/uniform.
Nulls: curveball (names shuffled across tablets, sizes and frequencies kept); random (un-climbed) orders.
Controls: planted rota / planted blocks at PE size; Ur III Drehem AS5 (players = rare words, truth month)."""
import os as _o
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'): _o.environ[_v] = '1'
import sys, json, time
import numpy as np
from multiprocessing import Pool
from pe40_common import *

PERIODS = (4, 6, 8, 12, 16, 24)


def cooc(X):
    C = X.T @ X; np.fill_diagonal(C, 0)
    return C


def climb(C, P, theta, wd, sweeps=30):
    n = len(theta)
    D = np.abs(np.arange(P)[:, None] - np.arange(P)[None, :]); D = np.minimum(D, P - D)
    Wm = wd[np.minimum(D, len(wd) - 1)]  # slot x slot weight
    for _ in range(sweeps):
        changed = 0
        for p in np.random.permutation(n):
            # score of each slot for p
            onehot = np.zeros((n, P)); onehot[np.arange(n), theta] = 1
            sc = (C[p] @ onehot) @ Wm
            s = int(np.argmax(sc + 1e-9 * np.random.random(P)))
            if s != theta[p]:
                theta[p] = s; changed += 1
        if not changed:
            break
    return theta


def climb_fast(C, P, theta, wd, sweeps=30, rng=None):
    n = len(theta)
    D = np.abs(np.arange(P)[:, None] - np.arange(P)[None, :]); D = np.minimum(D, P - D)
    Wm = wd[np.minimum(D, len(wd) - 1)]
    H = np.zeros((n, P)); H[np.arange(n), theta] = 1
    A = C @ H  # player x slot: co-occurrence mass in each slot
    for _ in range(sweeps):
        changed = 0
        for p in rng.permutation(n):
            sc = A[p] @ Wm
            s = int(np.argmax(sc + 1e-9 * rng.random(P)))
            if s != theta[p]:
                A[:, theta[p]] -= C[:, p]; A[:, s] += C[:, p]; theta[p] = s; changed += 1
        if not changed:
            break
    return theta


def objective(C, P, theta, wd):
    D = np.abs(theta[:, None] - theta[None, :]); D = np.minimum(D, P - D)
    return float((C * wd[np.minimum(D, len(wd) - 1)]).sum())


def dist_hist(Ct, P, theta, mask):
    D = np.abs(theta[:, None] - theta[None, :]); D = np.minimum(D, P - D)
    h = np.zeros(P // 2 + 1)
    M = Ct * mask
    for d in range(P // 2 + 1):
        h[d] = M[D == d].sum()
    # chance: number of slot-pairs at each distance
    cnt = np.array([P if d in (0, P / 2) else 2 * P for d in range(P // 2 + 1)], float) / P ** 2
    return h / max(h.sum(), 1e-9), cnt


def rota_index(X, P, rng, starts=200, keep=5, fit_frac=0.5, nsplit=3):
    wd = np.array([1.0, 0.5, 0.0])
    out = []
    for s in range(nsplit):
        idx = rng.permutation(len(X)); nf = int(fit_frac * len(X))
        Xf, Xt = X[idx[:nf]], X[idx[nf:]]
        seen = Xf.sum(0) > 0
        Cf, Ct = cooc(Xf), cooc(Xt)
        mask = np.outer(seen, seen)
        res = []
        for k in range(starts):
            th = rng.integers(0, P, X.shape[1])
            th = climb_fast(Cf, P, th, wd, rng=rng)
            res.append((objective(Cf, P, th, wd), th.copy()))
        res.sort(key=lambda r: -r[0])
        RI, SAME, RIr = [], [], []
        for _, th in res[:keep]:
            h, c = dist_hist(Ct, P, th, mask)
            rel = h / c
            RI.append(rel[1] / max(rel[2:].mean(), 1e-9)); SAME.append(rel[0])
        thr = rng.integers(0, P, X.shape[1])
        h, c = dist_hist(Ct, P, thr, mask); rel = h / c
        RIr.append(rel[1] / max(rel[2:].mean(), 1e-9))
        out.append(dict(RI=float(np.mean(RI)), SAME=float(np.mean(SAME)), RI_rand=float(np.mean(RIr)),
                        best_theta=res[0][1].tolist()))
    return dict(RI=float(np.mean([o['RI'] for o in out])), SAME=float(np.mean([o['SAME'] for o in out])),
                RI_rand=float(np.mean([o['RI_rand'] for o in out])), theta=out[0]['best_theta'])


def job(a):
    name, X, P, seed, truth = a
    rng = np.random.default_rng(seed)
    r = rota_index(X, P, rng)
    r['name'] = name; r['P'] = P
    if truth is not None:
        # circular recovery: players' slot vs players' mean month
        th = np.array(r['theta']); pm = []
        for j in range(X.shape[1]):
            t = np.nonzero(X[:, j])[0]
            z = np.exp(2j * np.pi * np.asarray(truth)[t] / 12).mean()
            pm.append(np.angle(z) / (2 * np.pi) * 12 % 12)
        a1 = 2 * np.pi * th / P; b1 = 2 * np.pi * np.array(pm) / 12
        r['cc_month'] = max(abs(np.exp(1j * (a1 - b1)).mean()), abs(np.exp(1j * (-a1 - b1)).mean()))
    del r['theta']
    return r


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    R = filter_players(pe_rounds('sign')); X, Pl = incidence(R)
    sizes = X.sum(1).astype(int); npl = X.shape[1]
    jobs = []
    if which in ('all', 'ctrl'):
        for per in (8, 16):
            Xp, _ = plant_rota(len(X), sizes, npl, per, overlap=1, noise=0.3, rng=np.random.default_rng(per))
            for P in (8, 16):
                jobs.append((f'PLANT_ROTA_p{per}', Xp, P, 1, None))
        Xb, _ = plant_block(len(X), sizes, npl, 16, noise=0.3, rng=np.random.default_rng(5))
        for P in (8, 16):
            jobs.append(('PLANT_BLOCK_k16', Xb, P, 1, None))
        from pe40_cycle1 import ur3_month
        Xu, mu = ur3_month()
        for P in (6, 12):
            jobs.append(('UR3_AS5', Xu, P, 1, mu))
    if which in ('all', 'pe'):
        for P in PERIODS:
            jobs.append(('PE', X, P, 11, None))
    if which in ('all', 'null'):
        rng = np.random.default_rng(3)
        for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 10):
            Xn = curveball(X, rng)
            for P in (6, 12):
                jobs.append((f'NULL{i}', Xn, P, 100 + i, None))
    t = time.time()
    with Pool(2) as pool:
        out = pool.map(job, jobs)
    json.dump(out, open(os.path.join(CK, f'cycle3_{which}.json'), 'w'), indent=1)
    for r in out:
        print(json.dumps(r))
    print('sec', round(time.time() - t))
