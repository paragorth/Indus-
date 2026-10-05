#!/usr/bin/env python3
"""LA-55 cycle 3b: INVERT THE PROBLEM - census from infrastructure.
If some words saturate (a fixed number per archive whatever its size), their per-document rate in a surviving
fragment tells how big the whole archive was: rate_w(N) = exp(alpha_w) N^(beta_w - 1). Given n surviving documents
of an archive, estimate N by maximum likelihood over a grid (scaling fitted on the other archives only).
Validation: leave each archive out, draw n documents, estimate its size; score Spearman(estimate, true N) over
archives. Nulls: the same pipeline on N1 / N3 re-dealt corpora (size information destroyed), 200 each.
Then: Linear A fragments -> estimated size of the archives they came from.
usage: la55_c3b.py CORPUS NNULL"""
import sys, collections, json
import numpy as np
from scipy.stats import spearmanr
from la55_common import *

corpus = sys.argv[1]; NN = int(sys.argv[2])
rng = np.random.default_rng(seed('la55-c3b-' + corpus))
docs, mk = build_corpus(corpus)
data = Data(docs, min_unit=3, min_k=3)
GRID = np.exp(np.linspace(np.log(3), np.log(5000), 120))
NS = 8           # surviving documents drawn per archive
MINN = 15        # archives with at least this many documents are tested


def fit_ab(K, N):
    b = fit_beta(K, N); x = np.log(N)
    info_mu = np.exp(np.log(K.sum(1) / np.exp(b[:, None] * x[None, :]).sum(1))[:, None] + b[:, None] * x[None, :])
    info = (info_mu * (x[None, :] - (info_mu * x).sum(1, keepdims=True) / info_mu.sum(1, keepdims=True)) ** 2).sum(1)
    w = info / (info + 4); b = 1 + w * (b - 1)
    a = np.log(K.sum(1) / np.exp(b[:, None] * x[None, :]).sum(1))
    return a, b


def estimate(k, n, a, b):
    """k: counts of each word in n docs; log-lik over GRID of N (N >= n)."""
    lg = np.log(GRID)
    lam = np.exp(a[:, None] + (b[:, None] - 1) * lg[None, :])          # per-doc rate
    p = np.clip(1 - np.exp(-lam), 1e-6, 1 - 1e-6)                       # presence per doc
    ll = (k[:, None] * np.log(p) + (n - k)[:, None] * np.log(1 - p)).sum(0)
    ll[GRID < n] = -np.inf
    return GRID[np.argmax(ll)]


def run(u, reps=3, X=None):
    X = data.X if X is None else X
    K, N = data.counts(u)
    est, tru = [], []
    for A in range(data.A):
        if N[A] < MINN: continue
        m = np.ones(data.A, bool); m[A] = False
        ok = K[:, m].sum(1) >= 3
        a, b = fit_ab(K[ok][:, m], N[m])
        rows = np.where(u == A)[0]
        for _ in range(reps):
            pick = rng.choice(rows, size=min(NS, len(rows)), replace=False)
            k = np.asarray(X[pick].sum(0)).ravel()[ok]
            est.append(estimate(k, len(pick), a, b)); tru.append(N[A])
    if len(est) < 6: return np.nan, est, tru
    return spearmanr(np.log(est), np.log(tru)).correlation, est, tru


rho, est, tru = run(data.u)
n1 = [run(data.shuffle(rng), reps=1)[0] for _ in range(NN)]
n3 = [run(data.shuffle(rng, True), reps=1)[0] for _ in range(NN)]
n1, n3 = np.array(n1), np.array(n3)
p1 = (np.sum(n1 >= rho) + 1) / (NN + 1); p3 = (np.sum(n3 >= rho) + 1) / (NN + 1)
print('== %s archives tested %d (N>=%d, %d docs drawn): Spearman(est, true) %.3f | N1 %.3f+-%.3f p %.3f | N3 %.3f+-%.3f p %.3f' % (
    corpus, len(set(tru)), MINN, NS, rho, np.nanmean(n1), np.nanstd(n1), p1, np.nanmean(n3), np.nanstd(n3), p3), flush=True)
med = collections.defaultdict(list)
for e, t in zip(est, tru): med[t].append(e)
print('   true N -> median estimate:', [(int(t), int(np.median(v))) for t, v in sorted(med.items())])
out = dict(rho=rho, N1=n1, N3=n3, p1=p1, p3=p3, est=est, tru=tru)

if corpus == 'LA':
    # Linear A small sites / deposits: estimate the size of the archive each surviving set came from
    K, N = data.counts()
    sites = collections.defaultdict(list)
    for i, d in enumerate(data.docs): sites[data.u[i]].append(i)
    rows = []
    for A, ix in sorted(sites.items(), key=lambda x: -len(x[1])):
        m = np.ones(data.A, bool); m[A] = False          # scaling fitted WITHOUT this archive
        ok = K[:, m].sum(1) >= 3
        a, b = fit_ab(K[ok][:, m], N[m])
        k = np.asarray(data.X[ix].sum(0)).ravel()[ok]
        e = estimate(k, len(ix), a, b)
        bs = [estimate(np.asarray(data.X[rng.choice(ix, len(ix))].sum(0)).ravel()[ok], len(ix), a, b) for _ in range(200)]
        rows.append((data.units[A], len(ix), int(e), int(np.percentile(bs, 10)), int(np.percentile(bs, 90))))
    print('   LA unit, surviving docs, estimated archive size, scaling fitted without it (80% bootstrap):')
    for r in rows: print('     ', r)
    out['la_units'] = rows
jdump(out, 'c3b_%s.json' % corpus)
