"""la73 ABC engine: reference tables (2 workers), rejection ABC with local-linear adjustment, RF check."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, json, sys, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la73_common as C
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'la73_ckpt')
DER = ['R_eff', 'persons_ever', 'surv_frac', 'tablets_ever', 'distinct_persons_seen']

def _work(args):
    seed, n, pool = args
    rng = np.random.default_rng(seed)
    TH = C.draw_prior(rng, n); S = []; Dv = []
    for th in TH:
        L = pool[rng.integers(len(pool))] if isinstance(pool, list) else pool
        d, der = C.simulate(th, L, rng)
        S.append(C.stats(d, rng)); Dv.append([der[k] for k in DER])
    return TH, np.array(S), np.array(Dv)

def table(name, n, lengths, seed=0, chunks=40):
    fn = os.path.join(CK, 'ref_%s.npz' % name)
    if os.path.exists(fn):
        z = np.load(fn); return z['th'], z['s'], z['d']
    with mp.Pool(2) as P:
        R = P.map(_work, [(seed * 1000 + i, n // chunks, lengths) for i in range(chunks)])
    th = np.vstack([r[0] for r in R]); s = np.vstack([r[1] for r in R]); d = np.vstack([r[2] for r in R])
    np.savez(fn, th=th, s=s, d=d)
    return th, s, d

def cols(names):
    return [C.SNAMES.index(k) for k in names]

def abc(s_obs, th, s, d=None, use=None, frac=0.005, adjust=True):
    """Rejection ABC on the `use` statistics (default TRAIN), MAD-scaled; local-linear adjustment.
    Returns dict param -> (median, q10, q90) for PNAMES (+ derived)."""
    ci = cols(use or C.TRAIN)
    X = s[:, ci]; x0 = s_obs[ci]
    mad = np.median(np.abs(X - np.median(X, 0)), 0) + 1e-9
    dist = np.sqrt((((X - x0) / mad) ** 2).sum(1))
    k = max(50, int(frac * len(X))); idx = np.argsort(dist)[:k]
    h = dist[idx[-1]] + 1e-12; w = 1 - (dist[idx] / h) ** 2
    Y = np.hstack([th, np.log10(np.maximum(d[:, 1:], 1e-9)), d[:, :1]]) if d is not None else th
    names = C.PNAMES + (['log_' + n for n in DER[1:]] + ['R_eff'] if d is not None else [])
    Yk = Y[idx]
    if adjust:
        Z = np.hstack([np.ones((k, 1)), (X[idx] - x0) / mad])
        WZ = Z * w[:, None]
        beta = np.linalg.lstsq(WZ.T @ Z + 1e-6 * np.eye(Z.shape[1]), WZ.T @ Yk, rcond=None)[0]
        Yk = Yk - ((X[idx] - x0) / mad) @ beta[1:]
    out = {}
    for j, n in enumerate(names):
        o = np.argsort(Yk[:, j]); cw = np.cumsum(w[o]) / w.sum()
        q = lambda a: float(Yk[o, j][np.searchsorted(cw, a)])
        out[n] = (q(0.5), q(0.1), q(0.9))
    out['_idx'] = idx; out['_w'] = w
    return out
