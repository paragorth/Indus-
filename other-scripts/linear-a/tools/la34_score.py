#!/usr/bin/env python3
"""la34 scoring of a unit x feature matrix against held-out published hands.
All comparisons are made WITHIN a site (different-site pairs are never scored), so site or drawing-campaign
differences cannot pass as hands.  Statistics:
  AUC  : P(dist same-hand pair < dist different-hand pair), same-site pairs only
  NN   : fraction of labelled units whose nearest labelled same-site neighbour has the same hand
  ARI  : agglomerative (average linkage) clustering per site with k = true number of hands at that site,
         pooled ARI over sites
Null: hand labels permuted among labelled units of the same site (n_perm)."""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def standardize(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    Z[np.isnan(Z)] = 0
    return Z


def dist(Z, metric='euclid'):
    if metric == 'cos':
        n = np.linalg.norm(Z, axis=1, keepdims=True); n[n == 0] = 1; Y = Z / n
        return 1 - Y @ Y.T
    sq = (Z ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * Z @ Z.T, 0))


def stats(Dm, hands, sites):
    hands = np.asarray(hands); sites = np.asarray(sites)
    n = len(hands)
    iu = np.triu_indices(n, 1)
    same_site = sites[iu[0]] == sites[iu[1]]
    same_hand = (hands[iu[0]] == hands[iu[1]])[same_site]
    d = Dm[iu][same_site]
    if same_hand.sum() == 0 or (~same_hand).sum() == 0: auc = np.nan
    else:
        from scipy.stats import rankdata
        r = rankdata(d); n1 = same_hand.sum(); n0 = (~same_hand).sum()
        auc = 1 - (r[same_hand].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    D2 = Dm.copy(); np.fill_diagonal(D2, np.inf)
    D2[sites[:, None] != sites[None]] = np.inf
    nn = D2.argmin(1); ok = np.isfinite(D2.min(1))
    nnacc = (hands[nn] == hands)[ok].mean()
    ari_num = []; 
    pred = np.empty(n, object)
    for s in np.unique(sites):
        ix = np.where(sites == s)[0]
        k = len(set(hands[ix]))
        if len(ix) < 3 or k < 2: pred[ix] = [f'{s}:0'] * len(ix); continue
        sub = Dm[np.ix_(ix, ix)]; sub = (sub + sub.T) / 2; np.fill_diagonal(sub, 0)
        L = linkage(squareform(sub, checks=False), 'average')
        lab = fcluster(L, k, 'maxclust')
        pred[ix] = [f'{s}:{l}' for l in lab]
    ari = adjusted_rand_score(hands, pred)
    return {'auc': float(auc), 'nn': float(nnacc), 'ari': float(ari)}


def perm_test(Dm, hands, sites, n_perm=500, seed=0):
    rng = np.random.default_rng(seed)
    hands = np.asarray(hands); sites = np.asarray(sites)
    real = stats(Dm, hands, sites)
    null = {k: [] for k in real}
    for _ in range(n_perm):
        h = hands.copy()
        for s in np.unique(sites):
            ix = np.where(sites == s)[0]; h[ix] = rng.permutation(h[ix])
        st = stats(Dm, h, sites)
        for k in st: null[k].append(st[k])
    out = {}
    for k in real:
        a = np.array(null[k])
        out[k] = real[k]; out[k + '_null'] = float(np.nanmean(a))
        out[k + '_p'] = float((1 + np.sum(a >= real[k])) / (1 + len(a)))
    return out
