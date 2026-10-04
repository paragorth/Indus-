#!/usr/bin/env python3
"""la34 cycle 2: massive random guessing of 'what a hand is'.
Every hypothesis is a random weighting of 42 per-feature distance layers (22 sign-agnostic ductus layers,
8 sign-matched grid-PCA layers + 22 sign-matched scalar layers, 12 box-geometry layers) with random sparsity.
Squared distances are linear in the weights, so thousands of hypotheses cost almost nothing.
Protocol: within each site, hands are split in two (train / test).  N hypotheses are scored by same-site AUC on
train-hand units only; the top K are averaged and scored on test-hand units only.  The whole search is repeated on
label sets permuted within site (null of the selected-and-retested score).  Splits repeated S times."""
import numpy as np, json, os, sys, collections
from la34_common import load, CK
from la34_img import occ_table
from la34_score import stats

occ, meta, cid = load()
feat = json.load(open(os.path.join(CK, 'feat.json')))
R = json.load(open(os.path.join(CK, 'rhythm.json')))


def layers(min_occ=3):
    o_, names, Z, P, codes = occ_table(feat, scale_free=True, occ=occ)
    nocc = collections.Counter(o['unit'] for o in o_)
    U = [u for u in sorted(nocc) if meta[u]['scribe'] and nocc[u] >= min_occ]
    c = collections.Counter(meta[u]['scribe'] for u in U)
    U = [u for u in U if c[meta[u]['scribe']] >= 2]
    ix = {u: i for i, u in enumerate(U)}; n = len(U)
    F = np.hstack([Z, P])
    # ductus (sign-agnostic): unit means of within-code z
    S = collections.defaultdict(list)
    for i, o in enumerate(o_):
        if o['unit'] in ix: S[o['unit']].append(i)
    V = np.array([Z[S[u]].mean(0) for u in U])
    Ld = [(V[:, None, j] - V[None, :, j]) ** 2 for j in range(V.shape[1])]
    # sign-matched: per (unit, code) means
    M = collections.defaultdict(dict); byc = collections.defaultdict(list)
    for i, o in enumerate(o_):
        if o['unit'] in ix: byc[(o['unit'], codes[i])].append(i)
    for (u, cc), ii in byc.items(): M[u][cc] = F[ii].mean(0)
    nf = F.shape[1]
    A = np.zeros((nf, n, n)); cnt = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            sh = set(M[U[a]]) & set(M[U[b]])
            if sh:
                d = np.mean([(M[U[a]][cc] - M[U[b]][cc]) ** 2 for cc in sh], 0)
                A[:, a, b] = A[:, b, a] = d; cnt[a, b] = cnt[b, a] = len(sh)
    # box geometry
    RX = np.array([R['units'].get(u, [np.nan] * 12) for u in U], float)
    mu, sd = np.nanmean(RX, 0), np.nanstd(RX, 0); RX = (RX - mu) / sd; RX[np.isnan(RX)] = 0
    Lr = [(RX[:, None, j] - RX[None, :, j]) ** 2 for j in range(12)]
    L = np.concatenate([np.array(Ld), A, np.array(Lr)])
    L = L / (np.array([l[np.triu_indices(n, 1)].mean() for l in L])[:, None, None] + 1e-12)
    shared = cnt > 0
    return U, L, shared


def auc_fast(Dm, same, mask):
    d = Dm[mask]; y = same[mask]
    if y.sum() == 0 or (~y).sum() == 0: return np.nan
    from scipy.stats import rankdata
    r = rankdata(d); n1 = y.sum(); n0 = (~y).sum()
    return 1 - (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def search(L, hands, sites, shared, rng, N=3000, K=20):
    n = len(hands); iu = np.triu_indices(n, 1)
    nl = L.shape[0]
    hands = np.asarray(hands); sites = np.asarray(sites)
    # split hands within site
    train = np.zeros(n, bool)
    for s in np.unique(sites):
        hs = np.unique(hands[sites == s]); rng.shuffle(hs)
        tr = set(hs[: (len(hs) + 1) // 2])
        train |= (sites == s) & np.isin(hands, list(tr))
    same = (hands[:, None] == hands[None])[iu]
    ssite = (sites[:, None] == sites[None])[iu]
    m_tr = ssite & train[iu[0]] & train[iu[1]] & shared[iu]
    m_te = ssite & ~train[iu[0]] & ~train[iu[1]] & shared[iu]
    Lf = L[:, iu[0], iu[1]]                     # layers x pairs
    W = rng.exponential(1, (N, nl)) * (rng.random((N, nl)) < rng.uniform(0.1, 0.9, (N, 1)))
    W[0] = 1.0                                   # hypothesis 0 = equal weights (the default guess)
    Dall = W @ Lf
    sc = np.array([auc_fast(d, same, m_tr) for d in Dall])
    top = np.argsort(-np.nan_to_num(sc, nan=-1))[:K]
    wsel = W[top].mean(0)
    te = auc_fast(wsel @ Lf, same, m_te)
    te_def = auc_fast(Dall[0], same, m_te)
    return sc[top].mean(), te, te_def, wsel


def main(S=10, NULL=20, N=3000):
    U, L, shared = layers()
    h = np.array([meta[u]['scribe'] for u in U]); s = np.array([meta[u]['site'] for u in U])
    print('units', len(U), 'hands', len(set(h)), 'layers', L.shape[0], flush=True)
    rng = np.random.default_rng(34)
    real = []; null = []; W = []
    for k in range(S):
        tr, te, td, w = search(L, h, s, shared, rng, N)
        real.append((tr, te, td)); W.append(w)
        print('real split', k, round(tr, 3), 'test', round(te, 3), 'default', round(td, 3), flush=True)
        for j in range(NULL):
            hp = h.copy()
            for site in np.unique(s):
                ix = np.where(s == site)[0]; hp[ix] = rng.permutation(hp[ix])
            tr0, te0, td0, _ = search(L, hp, s, shared, rng, N)
            null.append((tr0, te0, td0))
    real = np.array(real); null = np.array(null)
    Wm = np.mean(W, 0)
    names = ([f'D:{n}' for n in range(22)] + [f'M:{n}' for n in range(30)] + [f'R:{n}' for n in R['names']])
    out = {'units': len(U), 'real_train': real[:, 0].tolist(), 'real_test': real[:, 1].tolist(), 'real_default': real[:, 2].tolist(),
           'null_train': null[:, 0].tolist(), 'null_test': null[:, 1].tolist(),
           'p_test': float((1 + (null[:, 1].reshape(S, NULL).mean(0) >= real[:, 1].mean()).sum()) / (1 + NULL)),
           'weights': dict(sorted(zip(names, Wm.round(3).tolist()), key=lambda x: -x[1]))}
    json.dump(out, open(os.path.join(CK, 'c2.json'), 'w'), indent=1)
    print('real test mean', real[:, 1].mean().round(3), 'default', real[:, 2].mean().round(3),
          'null test mean', null[:, 1].mean().round(3), 'null train', null[:, 0].mean().round(3), 'P', out['p_test'])
    print('top weights', list(out['weights'].items())[:12])


if __name__ == '__main__':
    main()
