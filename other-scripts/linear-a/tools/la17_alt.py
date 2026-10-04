"""LA-17 'other-generator' planted outbreaks (misspecification control).
A scribal-copying world unlike the ABC simulator: each site, when founded, inherits a
random share of a donor site's CURRENT lexicon (donor chosen by distance among founded sites),
then coins new words and replaces a fraction of its lexicon each step; documents are written
by drawing tokens from the site's final lexicon with a site-specific Zipf law and the real
number of word tokens per site. Source and order are known."""
import numpy as np
from la17_common import *


def alt_world(E, D, src, t, rng, deposit=None, inherit=0.6, coin=300, turn=0.15, zipf=0.6, L=120):
    K = len(E)
    nst = 12
    order = np.argsort(t)
    lex = [None] * K
    nxt = [0]

    def new(n):
        a = np.arange(nxt[0], nxt[0] + n); nxt[0] += n; return list(a)
    founded = []
    tau = deposit if deposit is not None else np.ones(K)
    final = [None] * K
    for k in range(nst):
        now = k / nst
        for j in order:
            if lex[j] is None and t[j] < now + 1 / nst - 1e-9:
                if founded:
                    w = np.exp(-D[founded, j] / L); q = founded[rng.choice(len(founded), p=w / w.sum())]
                    keep = [x for x in lex[q] if rng.random() < inherit]
                else:
                    keep = []
                lex[j] = keep + new(int(coin * (E[j] / E.mean()) ** 0.5) + 5)
                founded.append(j)
        for j in founded:
            if final[j] is not None:
                continue
            L0 = lex[j]
            # turnover and borrowing from founded neighbours
            L1 = [x for x in L0 if rng.random() > turn]
            nb = [q for q in founded if q != j]
            if nb:
                w = np.exp(-D[nb, j] / L); q = nb[rng.choice(len(nb), p=w / w.sum())]
                L1 += [x for x in lex[q] if rng.random() < 0.05]
            L1 += new(int(coin * turn * (E[j] / E.mean()) ** 0.5) + 1)
            lex[j] = list(dict.fromkeys(L1))
            if tau[j] <= now + 1 / nst + 1e-9 and k >= int(t[j] * nst):
                final[j] = list(lex[j])
    for j in range(K):
        if final[j] is None:
            final[j] = list(lex[j])
    allw = sorted(set(x for f in final for x in f))
    idx = {w: i for i, w in enumerate(allw)}
    inc = np.zeros((len(allw), K), bool)
    for j in range(K):
        n = int(E[j]); f = final[j]
        pw = 1.0 / np.arange(1, len(f) + 1) ** zipf; pw /= pw.sum()
        perm = rng.permutation(len(f))
        draws = rng.choice(len(f), size=n, p=pw)
        for x in set(draws):
            inc[idx[f[perm[x]]], j] = True
    return inc
