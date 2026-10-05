#!/usr/bin/env python3
"""LA-55 cycle 3c: held-out test of the size-based prediction for 'saturating' words only.
Leave one small archive out (N <= 100); on the other archives fit shrunk beta; words with beta_shrunk < 0.75 and
>= 5 documents are 'predicted saturating'. Score presence log-loss in the held-out archive: scaling vs
proportional, bits per (word, archive). Nulls: N3 (within-genre) and N4 (within genre x length) re-dealings.
Also: Heaps exponent of ALL non-logogram words, real vs N3 (archive lexical closure), LA vs LBS vs LB.
usage: la55_c3c.py CORPUS NNULL"""
import sys, collections
import numpy as np
from la55_common import *

corpus = sys.argv[1]; NN = int(sys.argv[2])
rng = np.random.default_rng(seed('la55-c3c-' + corpus))
docs, mk = build_corpus(corpus)
data = Data(docs, min_unit=3, min_k=3)


def shrunk(Ko, No):
    b = fit_beta(Ko, No); x = np.log(No)
    mu = np.exp(np.log(Ko.sum(1) / np.exp(b[:, None] * x[None, :]).sum(1))[:, None] + b[:, None] * x[None, :])
    info = (mu * (x[None, :] - (mu * x).sum(1, keepdims=True) / mu.sum(1, keepdims=True)) ** 2).sum(1)
    w = info / (info + 4); b = 1 + w * (b - 1)
    a = np.log(Ko.sum(1) / np.exp(b[:, None] * x[None, :]).sum(1))
    return a, b


def sat_gain(u):
    K, N = data.counts(u); A = data.A
    tot, n, nw = 0.0, 0, []
    for a_ in range(A):
        if N[a_] > 100: continue
        m = np.ones(A, bool); m[a_] = False
        Ko, No = K[:, m], N[m]
        ok = Ko.sum(1) >= 5
        if not ok.any(): continue
        al, b = shrunk(Ko[ok], No)
        sel = b < 0.75
        if not sel.any(): continue
        mu1 = np.exp(al[sel] + b[sel] * np.log(N[a_])); mu0 = Ko[ok][sel].sum(1) / No.sum() * N[a_]
        p1 = np.clip(1 - np.exp(-mu1), 1e-6, 1 - 1e-6); p0 = np.clip(1 - np.exp(-mu0), 1e-6, 1 - 1e-6)
        y = K[ok][sel, a_] > 0
        tot += (np.where(y, np.log2(p1), np.log2(1 - p1)) - np.where(y, np.log2(p0), np.log2(1 - p0))).sum()
        n += sel.sum(); nw.append(int(sel.sum()))
    return tot / max(n, 1), n


g, n = sat_gain(data.u)
n3 = np.array([sat_gain(data.shuffle(rng, True))[0] for _ in range(NN)])
n4 = np.array([sat_gain(data.shuffle(rng, 'len'))[0] for _ in range(NN)])
p3 = (np.sum(n3 >= g) + 1) / (NN + 1); p4 = (np.sum(n4 >= g) + 1) / (NN + 1)
print('== %s saturating-word held-out presence gain %.4f bits (n %d word-archives) | N3 %.4f+-%.4f p %.3f | N4 %.4f+-%.4f p %.3f' % (
    corpus, g, n, n3.mean(), n3.std(), p3, n4.mean(), n4.std(), p4), flush=True)

# Heaps exponent of all non-logogram word types (min_k 1)
dall = Data(docs, min_unit=3, min_k=1)
wm = np.array([not t.startswith('L:') for t in dall.types])


def h_all(u):
    K, N = dall.counts(u)
    V = (K[wm] > 0).sum(0)
    return np.polyfit(np.log(N), np.log(V + 1), 1)[0]


h = h_all(dall.u)
hn3 = np.array([h_all(dall.shuffle(rng, True)) for _ in range(NN)])
hn4 = np.array([h_all(dall.shuffle(rng, 'len')) for _ in range(NN)])
print('   Heaps(all words) %.3f | N3 %.3f+-%.3f (p_lo %.3f) | N4 %.3f+-%.3f (p_lo %.3f); ratio h/hN3 %.3f' % (
    h, hn3.mean(), hn3.std(), (np.sum(hn3 <= h) + 1) / (NN + 1), hn4.mean(), hn4.std(), (np.sum(hn4 <= h) + 1) / (NN + 1),
    h / hn3.mean()))
jdump(dict(gain=g, n=n, N3=n3, N4=n4, p3=p3, p4=p4, heaps=h, heaps_N3=hn3, heaps_N4=hn4), 'c3c_%s.json' % corpus)
