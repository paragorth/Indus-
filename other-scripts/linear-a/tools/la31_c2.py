"""LA-31 cycle 2: do words flow DOWNWIND? A directed gravity (pseudo-likelihood) model.

For each type t (word / rarer sign / logogram) and site j: logit P(t at j) = a + b log(size_j)
+ c log(1 + n_oth) + e [n_oth > 0] + d log(mean_{i != j, t at i} exp(-T[i -> j] / L)) (0 if n_oth = 0).
G(T) = max over L of the log-likelihood gain of the d term. Asymmetric wind times give
Delta = G(T) - G(T transposed): positive = types present where the wind would have carried them.
Controls: rewired network (site labels of T permuted, 300), planted downwind and symmetric
spread on the real geography (same pipeline), which season / travel model fits best.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la31_common import SITES, CRETE, OUT, CKPT, load_docs, euclid  # noqa
from la31_stats import doc_matrices  # noqa
from la31_c1 import plant  # noqa

rng = np.random.default_rng(3102)
LGRID = (4, 8, 16, 32, 64, 128, 256)


def incidence(docs, codes, layer):
    ci = {c: i for i, c in enumerate(codes)}
    M, tags, F = doc_matrices(docs, codes)
    sel = np.nonzero(tags == layer)[0]
    lab = np.array([ci[d['site']] for d in docs])
    K = len(codes)
    from scipy.sparse import csr_matrix
    A = csr_matrix((np.ones(len(lab)), (lab, np.arange(len(lab)))), shape=(K, len(lab)))
    B = (A @ M[:, sel]).toarray() > 0
    Y = B.T.astype(float)                      # types x sites
    Y = Y[Y.sum(1) > 0]
    size = np.log1p(np.asarray((A @ M[:, sel]).sum(1)).ravel())
    return Y, size


def irls(X, y, it=25):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        eta = X @ w; p = 1 / (1 + np.exp(-eta)); W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        w = w + np.linalg.solve(H, X.T @ (y - p))
    eta = X @ w; p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gain(Y, size, T, base=None):
    """max_L LL gain of the directed influence term; T[i, j] hours from i to j."""
    nT, K = Y.shape
    oth = Y.sum(1, keepdims=True) - Y
    y = Y.ravel()
    X0 = np.c_[np.ones(nT * K), np.repeat(size[None], nT, 0).ravel(), np.log1p(oth).ravel(),
               (oth > 0).ravel().astype(float)]
    if base is None:
        base = irls(X0, y)
    Tm = np.where(np.isfinite(T), T, 1e6).copy(); np.fill_diagonal(Tm, 1e6)
    best = -1e18; bestL = None
    for L in LGRID:
        I = Y @ np.exp(-Tm / L)            # (types x sites): sum_i Y[t,i] exp(-T[i,j]/L)
        x = np.where(oth > 0, np.log(1e-9 + I / np.maximum(oth, 1)), 0.0)   # mean closeness of holders
        ll = irls(np.c_[X0, x.ravel()], y)
        if ll > best:
            best, bestL = ll, L
    return best - base, bestL, base


def mats(travel, codes):
    tc = travel['codes']; ix = [tc.index(c) for c in codes]
    sub = lambda k: np.array(travel[k], float)[np.ix_(ix, ix)]  # noqa
    P = {'km': euclid(codes) / 5.0, 'calm': sub('calm')}
    for s in ('annual', 'sailing', 'etesian', 'spring', 'autumn', 'winter'):
        P['wind_' + s] = sub('wind_' + s)
    if all(c in CRETE for c in codes):
        P['walk'] = sub('walk')
    return P


def analyse(docs, codes, travel, layer, nrew=300, tag=''):
    Y, size = incidence(docs, codes, layer)
    P = mats(travel, codes)
    out = []; base = None; G = {}
    for k, T in P.items():
        g, L, base = gain(Y, size, T, base)
        G[k] = (g, L)
    for k in ('wind_etesian', 'wind_sailing', 'wind_winter', 'wind_annual'):
        gT, LT, _ = gain(Y, size, P[k].T, base)
        G[k + '^T'] = (gT, LT)
    # rewired network null for the best-season wind gain and for Delta
    K = len(codes)
    rew = []
    for _ in range(nrew):
        q = rng.permutation(K)
        Tq = P['wind_etesian'][np.ix_(q, q)]
        g1, _, _ = gain(Y, size, Tq, base); g2, _, _ = gain(Y, size, Tq.T, base)
        rew.append((g1, g1 - g2))
    rew = np.array(rew)
    d_et = G['wind_etesian'][0] - G['wind_etesian^T'][0]
    pg = (np.sum(rew[:, 0] >= G['wind_etesian'][0]) + 1) / (nrew + 1)
    pd_ = (np.sum(rew[:, 1] >= d_et) + 1) / (nrew + 1)
    s = (f'{tag} layer {layer} ({Y.shape[0]} types, {(Y.sum(1) > 1).sum()} multi-site): gains '
         + ', '.join(f'{k} {v[0]:.2f}@{v[1]}' for k, v in sorted(G.items(), key=lambda kv: -kv[1][0]))
         + f' | etesian gain vs rewired P {pg:.3f}; Delta(fwd-transposed) {d_et:+.2f}, rewired P {pd_:.3f} '
         f'(rewired Delta sd {rew[:, 1].std():.2f})')
    return s, G, d_et


def main():
    travel = json.load(open(os.path.join(OUT, 'travel.json')))
    docs = load_docs()
    lines = []
    for nm, codes in (('all29', list(SITES)), ('crete23', CRETE)):
        dd = [d for d in docs if d['site'] in codes]
        for layer in ('W', 'S', 'L'):
            s, _, _ = analyse(dd, codes, travel, layer, nrew=int(os.environ.get('NREW', 300)), tag=nm)
            lines.append(s); print(s, flush=True)
    # planted controls (all 29 sites, W layer): downwind vs symmetric spread on real geography
    codes = list(SITES)
    P = mats(travel, codes)
    Tet = P['wind_etesian']; Tsym = (Tet + Tet.T) / 2
    for nmT, T in (('downwind', Tet), ('upwind', Tet.T), ('symmetric', Tsym)):
        for L_h in (int(os.environ.get('LPL', 64)), 4 * int(os.environ.get('LPL', 64))):
            ds, win = [], []
            for rep in range(int(os.environ.get('NREP', 20))):
                pd_ = plant(docs, codes, T, L_h, rng)
                Y, size = incidence(pd_, codes, 'W')
                g1, _, base = gain(Y, size, Tet); g2, _, _ = gain(Y, size, Tet.T, base)
                gk, _, _ = gain(Y, size, P['km'], base)
                ds.append(g1 - g2); win.append(max(g1, g2) > gk)
            ds = np.array(ds)
            lines.append(f'planted {nmT} L={L_h}h: Delta mean {ds.mean():+.2f} sd {ds.std():.2f}; '
                         f'Delta>0 in {np.mean(ds > 0):.2f}; wind beats km in {np.mean(win):.2f}')
            print(lines[-1], flush=True)
    open(os.path.join(CKPT, 'c2_out.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
