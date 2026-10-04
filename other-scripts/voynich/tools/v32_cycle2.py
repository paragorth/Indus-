"""v32 cycle 2: hidden cycle with slips and the missing bifolio.

(a) CHMM: circular HMM with K = 2..40 states; state advances by 1 (prob 1-2e), stays (e) or skips one (e), e = 0.05;
    at the lost central bifolio (between f108v and f111r) the state is reset to uniform (unknown number of lost
    entries). Emissions: product of independent categoricals (opening-glyph class, length tercile, gallows tercile,
    q tercile, star points when present). Baum-Welch, 3 restarts x 30 iterations. Score = per-unit log-likelihood
    gain over the K=1 model. Each K calibrated by paragraph shuffles (the gap position is kept fixed), family-wise max.
(b) FOLD-gap: epoch folding where the part after the gap may carry any phase offset (max over offsets), for every
    feature and period, family-wise against shuffles.
Run on the Voynich (star units), IT2a, the Ado calendar (letter dropped: date only, and body only), a planted
calendar, and the biological section."""
import os, sys, time, pickle, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit
from multiprocessing import Pool
import v32_lib as V

KS = np.arange(2, 41)
R = int(os.environ.get('V32_R2', 60))
EPS = 0.05


@njit(cache=True)
def _fb(LE, K, eps, gap):
    """LE: N x K emission likelihoods (scaled). Returns loglik, gamma."""
    N = LE.shape[0]
    a = np.zeros((N, K)); c = np.zeros(N)
    for k in range(K): a[0, k] = LE[0, k] / K
    c[0] = a[0].sum(); a[0] /= c[0]
    for i in range(1, N):
        for k in range(K):
            if i == gap:
                p = 1.0 / K
            else:
                p = (1 - 2 * eps) * a[i - 1, (k - 1) % K] + eps * a[i - 1, k] + eps * a[i - 1, (k - 2) % K]
            a[i, k] = p * LE[i, k]
        c[i] = a[i].sum(); a[i] /= c[i]
    b = np.ones((N, K))
    for i in range(N - 2, -1, -1):
        for k in range(K):
            if i + 1 == gap:
                s = 0.0
                for j in range(K): s += LE[i + 1, j] * b[i + 1, j] / K
            else:
                s = ((1 - 2 * eps) * LE[i + 1, (k + 1) % K] * b[i + 1, (k + 1) % K] + eps * LE[i + 1, k] * b[i + 1, k]
                     + eps * LE[i + 1, (k + 2) % K] * b[i + 1, (k + 2) % K])
            b[i, k] = s / c[i + 1]
    g = a * b
    for i in range(N):
        g[i] /= g[i].sum()
    return np.log(c).sum(), g


def chmm(X, ncat, K, gap, seed, iters=30, restarts=3):
    """X: N x F int codes; ncat: list of category counts. Returns best per-unit loglik."""
    N, F = X.shape
    rng = np.random.default_rng(seed)
    best = -1e18
    for r in range(restarts):
        # random initial responsibilities: a random phase walk
        g = rng.dirichlet(np.ones(K) * 0.3, size=N)
        for it in range(iters):
            E = []
            for f in range(F):
                T = np.full((K, ncat[f]), 0.5)
                for k in range(K): T[k] += np.bincount(X[:, f], weights=g[:, k], minlength=ncat[f])
                E.append(T / T.sum(1, keepdims=True))
            LE = np.ones((N, K))
            for f in range(F): LE *= E[f][:, X[:, f]].T
            ll, g = _fb(LE, K, EPS, gap)
        best = max(best, ll / N)
    return best


def iid_ll(X, ncat):
    N, F = X.shape; ll = 0
    for f in range(F):
        p = (np.bincount(X[:, f], minlength=ncat[f]) + 0.5); p /= p.sum()
        ll += np.log(p[X[:, f]]).sum()
    return ll / N


def codes(units, star):
    num, cat, vec = V.features(units, star=star)
    cols, nc = [], []
    c, n = V.encode(cat['w1_first']); cols.append(c); nc.append(n)
    c, n = V.encode(cat['w1_word']); cols.append(c); nc.append(n)
    for k in ('log_nwords', 'gallows_pw', 'q_frac'):
        x = V.detrend(num[k]); q = np.quantile(x, [1 / 3, 2 / 3])
        cols.append(np.digitize(x, q)); nc.append(3)
    if star and 'star_pts' in cat:
        c, n = V.encode(cat['star_pts']); cols.append(c); nc.append(n)
    return np.stack(cols, 1), nc, (num, cat, vec)


def gap_index(units):
    for i in range(1, len(units)):
        a, b = units[i - 1]['f'], units[i]['f']
        if a.startswith('f108') and b.startswith('f111'): return i
    return -1


def chmm_scan(X, nc, gap, seed):
    base = iid_ll(X, nc)
    return np.array([chmm(X, nc, int(K), gap, seed + int(K)) - base for K in KS])


def fold_gap_scan(num, cat, perm, gap, periods):
    out = {}
    N = len(next(iter(num.values())))
    for k, x in num.items():
        y = V.detrend(np.asarray(x)[perm])
        out[('FOLDGAP', k)] = np.array([_fg_num(y, P, gap) for P in periods])
    for k, lab in cat.items():
        c, nc = V.encode([lab[i] for i in perm])
        if nc < 2: continue
        out[('FOLDGAP', k)] = np.array([_fg_cat(c, nc, P, gap) for P in periods])
    return out


def _phases(N, P, gap, off):
    i = np.arange(N, dtype=float)
    if gap > 0: i[gap:] += off
    K = int(round(P))
    return np.minimum((K * ((i / P) % 1.0)).astype(int), K - 1), K


def _fg_num(y, P, gap):
    best = -1e9
    for off in range(int(round(P))) if gap > 0 else [0]:
        b, K = _phases(len(y), P, gap, off)
        n = np.bincount(b, minlength=K); s = np.bincount(b, weights=y, minlength=K)
        ok = n > 0; m = s / np.maximum(n, 1)
        ssb = (n[ok] * (m[ok] - y.mean()) ** 2).sum(); ssw = ((y - m[b]) ** 2).sum(); k = ok.sum()
        best = max(best, (ssb / (k - 1)) / (ssw / (len(y) - k) + 1e-12))
    return best


def _fg_cat(c, nc, P, gap):
    best = -1e9
    for off in range(int(round(P))) if gap > 0 else [0]:
        b, K = _phases(len(c), P, gap, off)
        T = np.zeros((K, nc)); np.add.at(T, (b, c), 1)
        E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum(); nz = T > 0
        best = max(best, 2 * (T[nz] * np.log(T[nz] / E[nz])).sum())
    return best


PER2 = np.array([7, 12, 14, 24, 27.32, 28, 29.53, 30, 30.44, 36] + list(range(2, 41, 1)))
PER2 = np.array(sorted(set(PER2.tolist())))


def run(args):
    name, units, star, seed = args
    ck = os.path.join(V.CK, f'c2_{name}.pkl')
    if os.path.exists(ck): return pickle.load(open(ck, 'rb'))
    t = time.time()
    gap = gap_index(units)
    X, nc, (num, cat, vec) = codes(units, star)
    N = len(X)
    obs_h = chmm_scan(X, nc, gap, seed)
    obs_f = fold_gap_scan(num, cat, np.arange(N), gap, PER2)
    rng = np.random.default_rng(seed)
    nh, nf = [], []
    for r in range(R):
        p = rng.permutation(N)
        nh.append(chmm_scan(X[p], nc, gap, seed + 7919 * (r + 1)))
        if r < R:
            nf.append(fold_gap_scan(num, cat, p, gap, PER2))
    nh = np.array(nh)
    zh = (obs_h - nh.mean(0)) / (nh.std(0) + 1e-12)
    nullmax_h = np.array([((nh[r] - np.delete(nh, r, 0).mean(0)) / (np.delete(nh, r, 0).std(0) + 1e-12)).max() for r in range(R)])
    p_h = (1 + (nullmax_h >= zh.max()).sum()) / (R + 1)
    rf = V.family_test(obs_f, nf)
    out = {'name': name, 'N': N, 'gap': gap, 'chmm_z': zh, 'chmm_gain': obs_h, 'chmm_best_K': int(KS[zh.argmax()]),
           'chmm_zmax': float(zh.max()), 'chmm_p': float(p_h),
           'fg_zmax': rf['zmax'], 'fg_p': rf['p_fw'], 'fg_top': V.fmt_top(rf, PER2), 'secs': time.time() - t}
    pickle.dump(out, open(ck, 'wb'))
    print(name, N, 'gap', gap, f"CHMM best K {out['chmm_best_K']} z {out['chmm_zmax']:.2f} p {p_h:.3f};",
          f"FOLDGAP zmax {rf['zmax']:.2f} p {rf['p_fw']:.3f}", out['fg_top'][:160], f"{out['secs']:.0f}s", flush=True)
    return out


if __name__ == '__main__':
    q = V.load_q20()
    jobs = [
        ('VOY-star', q, True, 11),
        ('VOY-IT2a', V.load_q20('IT2a', mode='para'), False, 12),
        ('ADO-date-noletter', V.ado_units(letter=False), False, 13),
        ('ADO-full', V.ado_units(), False, 14),
        ('PLANT-open-29.5-s0.3-gap', V.plant_calendar(q, 29.53, 0.3, 5, 'opening'), False, 15),
        ('SEC-bio-B', V.load_section('B', 'B'), False, 16),
    ]
    with Pool(2) as p:
        outs = p.map(run, jobs, chunksize=1)
    print('done')
