"""v69 cycle 6: settle the same-word likeness lead with a null that has power in Latin too.
c4's null (identities permuted within page x 4-line band x glyph count x first x last glyph) had no
power in Latin (a planted 0.25 SD per-word signature gave z 0.1): Latin strata mostly hold one word.
Here: residual features (linear glyph content removed), identities permuted within (page, 4-line
band, glyph count) only. Because nonlinear glyph content (ligatures, glyph pairs) can make two
tokens of one word alike without any word-level unit, the decisive control is a planted NEGATIVE
built from a glyph-BIGRAM additive model of the raw features (fitted on the real data) + smooth
line drift + noise, residualised in the same way: the real gap must exceed the plant's.
Planted POSITIVE: real residuals + 0.25 SD per-type signature. Out: data/v69_ckpt/c6.json"""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v69_lib as X
import v69_c1 as C1
import v69_c4 as C4

RNG = np.random.default_rng(6960)
C4.RNG = RNG


def perm_g(D, gls, rng):
    keys = collections.defaultdict(list)
    for i in range(D.N):
        keys[(D.page[i], D.li[i] // 4, len(gls[i]))].append(i)
    w = list(D.word); gg = list(gls)
    for idx in keys.values():
        if len(idx) > 1:
            pr = rng.permutation(idx)
            for a, b in zip(idx, pr):
                w[a] = D.word[b]; gg[a] = gls[b]
    return w, gg


C4.perm_ident = perm_g


def bigram_negative(D):
    gls = [['^'] + D.gl(w) + ['$'] for w in D.word]
    cnt = collections.Counter((a, b) for g in gls for a, b in zip(g, g[1:]))
    uni = collections.Counter(x for g in gls for x in g)
    keys = [k for k, c in cnt.items() if c >= 15] + [(u,) for u, c in uni.items() if c >= 5]
    ki = {k: i for i, k in enumerate(keys)}
    C = np.zeros((D.N, len(keys)))
    for i, g in enumerate(gls):
        for a, b in zip(g, g[1:]):
            if (a, b) in ki:
                C[i, ki[(a, b)]] += 1
        for x in g:
            if (x,) in ki:
                C[i, ki[(x,)]] += 1
    A = np.hstack([np.ones((D.N, 1)), C])
    raw = {}
    for f in C4.F8:
        src = 'slant' if f == 'upright' else f
        y = np.clip(D.raw[f], -4, 4)
        b = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y)
        pred = A @ b; sd = np.std(y - pred)
        drift = np.zeros(D.N)
        for p in np.unique(D.page):
            m = D.page == p
            walk = np.cumsum(RNG.normal(0, 0.15, D.li[m].max() + 1))
            drift[m] = walk[D.li[m]]
        raw[f] = pred + drift + RNG.normal(0, sd, D.N)
    # residualise exactly like Data (linear unigram content + position)
    Nm = D.nuis(); Xm = np.hstack([np.ones((D.N, 1)), Nm])
    M = np.zeros((D.N, len(C4.F8)))
    for j, f in enumerate(C4.F8):
        y = raw[f]
        b = np.linalg.solve(Xm.T @ Xm + 3 * np.eye(Xm.shape[1]), Xm.T @ y)
        r = y - Xm @ b
        M[:, j] = np.clip((r - r.mean()) / r.std(), -4, 4)
    return M


if __name__ == '__main__':
    vf, _ = X.voynich_freq(); lf = X.latin_freq()
    res = {}
    for which, freq in [('L', lf), ('V', vf)]:
        D = C1.Data(which)
        M = C4.resid_matrix(D)
        res[which + '_resid'] = C4.test(D, M, freq, which + '_resid', nperm=25)
        res[which + '_plant_pos'] = C4.test(D, C4.planted_positive(D, M), freq, which + '_plant_pos', nperm=15)
        for r in range(2):
            res[f'{which}_bigram_neg{r}'] = C4.test(D, bigram_negative(D), freq, f'{which}_bigram_neg{r}', nperm=15)
    json.dump(res, open(os.path.join(X.CK, 'c6.json'), 'w'), indent=1)
