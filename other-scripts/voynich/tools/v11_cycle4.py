"""v11 cycle 4: one-way routes, redone. Cycle 2b's drift model did not beat the symmetric map on the planted ROUTE,
so its Voynich null was uninformative. Here:
 (a) model-free direction asymmetry on held-out pages: A = sum over unordered in-vocab pairs of the log-likelihood gain of
     observed direction split C_ab : C_ba over 50:50, in bits per bigram; null = each bigram's direction flipped by coin (200x).
     Excess A (obs - null) measures how one-way the succession is. A symmetric map walk gives ~0; a route gives a lot.
 (b) can a ROUTE map capture the asymmetry? drift2 refitted from the dist2 optimum with a random drift start (3 starts:
     v = 0, +-1 along each axis), keep best train fit. Score = drift2 gain minus dist2 gain (held-out), compared with bilin2 minus dist2
     (what a non-geometric asymmetric model of the same size gains). Must be large on the planted route.
Checkpoints c4_<corpus>_<fold>.json."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_lib as L
import numpy as np
from multiprocessing import Pool
CORPORA = ['Planted-route-30', 'Planted-grid-30', 'Voynich-ZL', 'Voynich-IT', 'Latin-Isidore', 'Italian-Manzoni', 'Shuffle-line']
LN2 = np.log(2)


def asym(C):
    S = C + C.T; iu = np.triu_indices(len(C), 1); a = C[iu]; b = C.T[iu]; s = S[iu]; m = s > 0
    a, b, s = a[m], b[m], s[m]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(a > 0, a * np.log2(2 * a / s), 0) + np.where(b > 0, b * np.log2(2 * b / s), 0)
    return float(t.sum() / C.sum()), a + b


def job(arg):
    name, fold = arg; ck = f'c4_{name}_{fold}.json'
    r = L.jload(ck)
    if r: return r
    lines, _ = L.corpus(name); voc = L.vocab(lines, 300); idx = {w: i for i, w in enumerate(voc)}
    tr, te = L.split(lines, fold); Ctr = L.bigram_counts(tr, idx); Cte = L.bigram_counts(te, idx)
    A, s = asym(Cte); rng = np.random.default_rng(fold)
    iu = np.triu_indices(len(Cte), 1); S = (Cte + Cte.T)[iu]; m = S > 0; S = S[m].astype(int)
    nulls = []
    for _ in range(200):
        a = rng.binomial(S, 0.5); b = S - a
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(a > 0, a * np.log2(2 * a / S), 0) + np.where(b > 0, b * np.log2(2 * b / S), 0)
        nulls.append(t.sum() / Cte.sum())
    nulls = np.array(nulls)
    base = L.test_ll(L.fit(Ctr, 'uni', 0, iters=300), Cte)
    g = lambda f: (L.test_ll(f, Cte) - base) / Cte.sum() / LN2
    d2 = L.best_fit(Ctr, 'dist', 2, 2, iters=500, lr=0.08)
    best = None
    for v0 in ([0, 0], [1, 0], [0, 1]):
        f = fit_drift_from(Ctr, d2['P'][0], np.array(v0, float))
        if best is None or f['train_ll'] > best['train_ll']: best = f
    b2 = L.best_fit(Ctr, 'bilin', 2, 1, iters=500, lr=0.08)
    X = best['P'][0]; v = best['P'][1]
    out = {'corpus': name, 'fold': fold, 'asym_obs': A, 'asym_null': float(nulls.mean()), 'asym_null_sd': float(nulls.std()),
           'asym_excess': A - float(nulls.mean()), 'dist2': g(d2), 'drift2': g(best), 'bilin2': g(b2),
           'drift_over_spread': float(np.linalg.norm(v) / np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))}
    L.jdump(out, ck); return out


def fit_drift_from(C, X0, v0, iters=800):
    """drift2 fit with the drift vector initialised to v0 (scaled to the map's spread)."""
    spread = np.sqrt(((X0 - X0.mean(0)) ** 2).sum(1).mean())
    return L.fit(C, 'drift', 2, iters=iters, lr=0.05, init=X0, v_init=v0 * 0.5 * spread)


if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(job, [(c, f) for c in CORPORA for f in (0, 1)]):
            print(r['corpus'], r['fold'], 'asym %.4f null %.4f excess %.4f' % (r['asym_obs'], r['asym_null'], r['asym_excess']),
                  'dist2 %.3f drift2 %.3f bilin2 %.3f |v|/spread %.2f' % (r['dist2'], r['drift2'], r['bilin2'], r['drift_over_spread']), flush=True)
