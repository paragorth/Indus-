"""v19 cycle 2 description: which glyphs drift down the page? For words in non-paragraph-first lines, mean relative
line position on the page (0 = top line, 1 = bottom) by first glyph, z against 2,000 within-page shuffles of words
(composition per page kept). Same for line-initial words only. ZL, IT, Latin XVI (negative)."""
import json, os, random, sys
import numpy as np
from v19_lib import *
from v19_cycle1 import corpora


def drift(lines, which, R=2000, seed=1):
    rng = np.random.default_rng(seed)
    order, by = pages_of(lines)
    G, P, PG = [], [], []
    for pi, p in enumerate(order):
        idx = by[p]; n = len(idx)
        if n < 6:
            continue
        for k, i in enumerate(idx):
            if lines[i]['para_start']:
                continue
            ws = lines[i]['words'] if which == 'all' else lines[i]['words'][:1]
            for w in ws:
                G.append(w[0]); P.append(k / (n - 1)); PG.append(pi)
    G = np.array(G); P = np.array(P); PG = np.array(PG)
    syms = [g for g, c in Counter(G).most_common() if c >= 100]
    real = {g: P[G == g].mean() for g in syms}
    # null: permute positions within page
    sims = {g: [] for g in syms}
    pages = np.unique(PG); groups = [np.where(PG == q)[0] for q in pages]
    for r in range(R):
        Pn = P.copy()
        for gi in groups:
            Pn[gi] = P[rng.permutation(gi)]
        for g in syms:
            sims[g].append(Pn[G == g].mean())
    out = []
    for g in syms:
        m, s = np.mean(sims[g]), np.std(sims[g])
        out.append((g, int((G == g).sum()), real[g], (real[g] - m) / s))
    return sorted(out, key=lambda x: x[3])


if __name__ == '__main__':
    C, _ = corpora()
    res = {}
    for c in ['ZL', 'IT', 'LatXVI']:
        for which in ['all', 'init']:
            d = drift(C[c], which, R=500)
            res['%s_%s' % (c, which)] = d
            print(c, which, ' '.join('%s:%+.1f' % (g, z) for g, n, m, z in d))
    json.dump(res, open(os.path.join(RES, 'c2_drift.json'), 'w'))
