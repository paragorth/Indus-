#!/usr/bin/env python3
"""LA-55 cycle 2: (a) Linear A scaling classes vs la45 classes (label-permutation test);
(b) held-out presence prediction for small archives (N <= 100): does a word's scaling exponent, fitted on the other
archives, predict whether it is present in a held-out small archive better than proportional sampling? Real vs N1
and N3 shuffles; controls LB, LBS, UR; (c) planted check.
usage: la55_c2.py CORPUS NNULL"""
import sys, ast, re, collections, json
import numpy as np
from la55_common import *

corpus = sys.argv[1]; NN = int(sys.argv[2])
rng = np.random.default_rng(seed('la55-c2-' + corpus))


def build(corpus):
    if corpus == 'LA': return la_units(), 3
    if corpus == 'LAG': return la_units(signs=True), 3
    if corpus == 'LB': return lb_units(), 3
    if corpus == 'LBS':
        docs = lb_units(); cnt = collections.Counter(d['unit'] for d in docs)
        us = sorted(u for u in cnt if cnt[u] >= 3); r = random.Random(seed('la55-lbs')); r.shuffle(us)
        keep, n = set(), 0
        for u in us:
            if n >= 1574: break
            keep.add(u); n += cnt[u]
        return [d for d in docs if d['unit'] in keep], 3
    if corpus == 'UR': return ur_units(max_docs=12000), 5


def presence_gain(K, N, maxN=100, shrink=True):
    """Leave-one-unit-out for units with N <= maxN. Model S: Poisson log-linear (alpha, beta) from the other units,
    beta shrunk toward 1 by its bootstrap-free precision (prior sd 0.5) if shrink; model P: proportional.
    Presence probability p = 1 - exp(-mu). Return mean bits gained per (word, unit) by S over P."""
    A = K.shape[1]; x = np.log(N)
    tot = []; n = 0
    for a in range(A):
        if N[a] > maxN: continue
        m = np.ones(A, bool); m[a] = False
        Ko, No = K[:, m], N[m]
        ok = Ko.sum(1) >= 3
        if not ok.any(): continue
        b = fit_beta(Ko[ok], No)
        xo = np.log(No)
        if shrink:
            mu = np.exp(np.log(Ko[ok].sum(1) / np.exp(b[:, None] * xo[None, :]).sum(1))[:, None] + b[:, None] * xo[None, :])
            info = (mu * (xo[None, :] - (mu * xo).sum(1, keepdims=True) / mu.sum(1, keepdims=True)) ** 2).sum(1)
            w = info / (info + 1 / 0.25)
            b = 1 + w * (b - 1)
        al = np.log(Ko[ok].sum(1) / np.exp(b[:, None] * xo[None, :]).sum(1))
        mu1 = np.exp(al + b * x[a]); mu0 = Ko[ok].sum(1) / No.sum() * N[a]
        p1 = np.clip(1 - np.exp(-mu1), 1e-6, 1 - 1e-6); p0 = np.clip(1 - np.exp(-mu0), 1e-6, 1 - 1e-6)
        y = K[ok, a] > 0
        l1 = np.where(y, np.log2(p1), np.log2(1 - p1)); l0 = np.where(y, np.log2(p0), np.log2(1 - p0))
        tot.append((l1 - l0).sum()); n += ok.sum()
    return float(np.sum(tot) / max(n, 1)), int(n)


docs, mk = build(corpus)
data = Data(docs, min_unit=3, min_k=mk)
K, N = data.counts()
out = {'corpus': corpus}
for shrink in (True, False):
    g, n = presence_gain(K, N, shrink=shrink)
    nul1 = [presence_gain(data.counts(data.shuffle(rng))[0], N, shrink=shrink)[0] for _ in range(NN)]
    nul3 = [presence_gain(data.counts(data.shuffle(rng, True))[0], N, shrink=shrink)[0] for _ in range(NN)]
    key = 'shrunk' if shrink else 'raw'
    out[key] = dict(real=g, n=n, N1=nul1, N3=nul3,
                    p1=(np.sum(np.array(nul1) >= g) + 1) / (NN + 1), p3=(np.sum(np.array(nul3) >= g) + 1) / (NN + 1))
    print(corpus, key, 'real %.5f bits (n %d)  N1 %.5f+-%.5f p %.3f  N3 %.5f+-%.5f p %.3f' % (
        g, n, np.mean(nul1), np.std(nul1), out[key]['p1'], np.mean(nul3), np.std(nul3), out[key]['p3']), flush=True)

if corpus == 'LA':
    # la45 stable classes (parsed from its LA report) vs la55 classes
    la45 = {}
    for line in open(os.path.join(D, 'la45_ckpt', 'c1_report_LA.txt')):
        m = re.match(r'\s+(HEADER|COMMODITY|ENTRY|RARE) (\[.*\])\s*$', line)
        if m:
            for w, ag, c in [t[:3] for t in ast.literal_eval(m.group(2))]:
                if w.startswith('L:'): k = 'L:' + w[2:].split('+')[0]
                elif '-' in w: k = 'w:' + w
                else: k = 's:' + w
                la45.setdefault(k, m.group(1))
    c1 = json.load(open(os.path.join(CK, 'c1_LA.json')))
    types = c1['types']
    for key in ('N1', 'N3'):
        cls = c1[key]['cls']
        tab = collections.defaultdict(collections.Counter)
        lab = [la45.get(t) for t in types]
        for l, c in zip(lab, cls):
            if l: tab[l][c] += 1
        # permutation: chi-square-like statistic = sum over cells (obs-exp)^2/exp, permuting la45 labels among typed words
        idx = [i for i, l in enumerate(lab) if l]
        L = np.array([lab[i] for i in idx]); C = np.array([cls[i] for i in idx])
        def stat(L, C):
            s = 0.0
            for a in set(L):
                for b in set(C):
                    o = np.sum((L == a) & (C == b)); e = np.sum(L == a) * np.sum(C == b) / len(L)
                    s += (o - e) ** 2 / e
            return s
        s0 = stat(L, C)
        rr = np.random.default_rng(seed('la55-c2-perm'))
        ps = [stat(rr.permutation(L), C) for _ in range(5000)]
        p = (np.sum(np.array(ps) >= s0) + 1) / 5001
        out['la45_' + key] = dict(table={k: dict(v) for k, v in tab.items()}, stat=s0, p=p, n=len(idx))
        print('la45 x la55', key, {k: dict(v) for k, v in tab.items()}, 'chi %.2f p %.4f n %d' % (s0, p, len(idx)))
jdump(out, 'c2_%s.json' % corpus)
