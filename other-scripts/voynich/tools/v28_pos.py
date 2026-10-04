"""v28 positional-variant (allograph) test.

Alternative explanation for v25: some Voynich glyphs are one letter written in different shapes by position
(like long s / round s).  Allographs look alike AND share non-positional contexts, so they make shape
predict behaviour without any featural design.  Tests, on every corpus with cached results:
  (a) Mantel on all pairs vs Mantel excluding KNOWN allograph pairs (medieval sets; Voynich has none known);
  (b) Mantel restricted to pairs that overlap in word position (both glyphs occur in the same positions;
      overlap = sum_k min(P_ik, P_jk) over initial/medial/final/single profile, top half) vs the
      complementary (low-overlap) half - positional variants live in the low-overlap half;
  (c) drop the k most allograph-like pairs (shape similarity rank x low positional overlap) and recompute.
Null for every r: random glyph-to-shape permutation restricted to the same pair mask."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import rankdata
import v28_lib as X, v28_corpora as C, v28_run as R, v25_lib as L


def mantel_mask(Ssh, B, keep, nperm=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(B); iu = np.triu_indices(n, 1)
    k = keep
    rb = rankdata(B[iu][k]); rb -= rb.mean(); nb = np.sqrt((rb ** 2).sum())

    def stat(perm):
        a = rankdata(Ssh[perm][:, perm][iu][k]); a -= a.mean()
        d = np.sqrt((a ** 2).sum()) * nb
        return (a * rb).sum() / d if d > 0 else 0.0
    r0 = stat(np.arange(n))
    null = np.array([stat(rng.permutation(n)) for _ in range(nperm)])
    return float(r0), float((1 + (null >= r0).sum()) / (nperm + 1)), int(k.sum())


def allo_pairs(alph):
    fam = [C.family(g) for g in alph]
    n = len(alph); iu = np.triu_indices(n, 1)
    return np.array([fam[i] == fam[j] and alph[i] != alph[j] for i, j in zip(*iu)])


def test(name, simkey=None, models=('ppmi', 'potts'), kdrop=(3, 6, 10)):
    c = C.build(name)
    A, ws = c['alph'], c['words']
    beh = X.load(f'beh_{name.replace("/", "-")}.pkl')
    S = R.sims_for(c)
    simkey = simkey or f'img:{c["fonts"][0]}'
    Ssh = S[simkey]
    n = len(A); iu = np.triu_indices(n, 1)
    P = X.position_profile(ws, A)
    ov = np.minimum(P[:, None, :], P[None, :, :]).sum(2)[iu]
    hi = ov >= np.median(ov)
    ap = allo_pairs(A)
    sr = rankdata(Ssh[iu]) / len(ov); orank = rankdata(-ov) / len(ov)
    score = sr * orank  # allograph-likeness: very similar shape AND complementary position
    out = []
    rng = np.random.default_rng(11)
    for m in models:
        B = beh[m]
        cells = []
        r, p, k = mantel_mask(Ssh, B, np.ones(len(ov), bool), rng=rng); cells.append(f'all r {r:+.2f} p {p:.4f}')
        if ap.any():
            r, p, k = mantel_mask(Ssh, B, ~ap, rng=rng); cells.append(f'minus {ap.sum()} known allograph pairs r {r:+.2f} p {p:.4f}')
            r, p, k = mantel_mask(Ssh, B, ~ap & hi, rng=rng); cells.append(f'minus allographs, overlap-high r {r:+.2f} p {p:.4f}')
        r, p, k = mantel_mask(Ssh, B, hi, rng=rng); cells.append(f'position-overlap-high half r {r:+.2f} p {p:.4f}')
        r, p, k = mantel_mask(Ssh, B, ~hi, rng=rng); cells.append(f'overlap-low half r {r:+.2f} p {p:.4f}')
        order = np.argsort(-score)
        for kd in kdrop:
            keep = np.ones(len(ov), bool); keep[order[:kd]] = False
            r, p, _ = mantel_mask(Ssh, B, keep, rng=rng); cells.append(f'drop top-{kd} allograph-like r {r:+.2f} p {p:.4f}')
        out.append((m, '; '.join(cells)))
    top = [f'{A[iu[0][q]]}/{A[iu[1][q]]}' for q in np.argsort(-score)[:6]]
    return out, top, simkey


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        o, top, sk = test(nm)
        for m, s in o:
            print(nm, sk, m, '|', s, flush=True)
        print(nm, 'most allograph-like pairs:', top, flush=True)


def size_matched(name, k=23, models=('ppmi', 'svd', 'potts'), nperm=3000):
    """Mantel restricted to the k most frequent units (behaviour from the full set), all fonts averaged."""
    c = C.build(name)
    beh = X.load(f'beh_{name.replace("/", "-")}.pkl')
    S = R.sims_for(c)
    top = np.argsort(-beh['_freq'])[:k]
    ix = np.ix_(top, top)
    rng = np.random.default_rng(3)
    out = {}
    for m in models:
        rs = []; ps = []
        for key, Ssh in S.items():
            if not key.startswith('img:'):
                continue
            r, p, _, _ = L.mantel(Ssh[ix], beh[m][ix], nperm=nperm, rng=rng)
            rs.append(r); ps.append(p)
        out[m] = (float(np.mean(rs)), float(max(ps)))
    return out, [c['alph'][i] for i in top]


def twins(name, ks=(1, 3, 6, 10), models=('ppmi', 'potts'), nperm=3000):
    """Is the link carried by a few 'twin' pairs (look alike AND behave alike, as free allographs would) or spread
    over the inventory?  Drop the k pairs with the highest rank(shape) x rank(behaviour) and recompute r."""
    c = C.build(name)
    beh = X.load(f'beh_{name.replace("/", "-")}.pkl')
    S = R.sims_for(c)
    Ssh = S[f'img:{c["fonts"][0]}']
    A = c['alph']; n = len(A); iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(23)
    out = []
    for m in models:
        B = beh[m]
        sc = rankdata(Ssh[iu]) * rankdata(B[iu])
        order = np.argsort(-sc)
        cells = []
        for k in (0,) + tuple(ks):
            keep = np.ones(len(sc), bool); keep[order[:k]] = False
            r, p, _ = mantel_mask(Ssh, B, keep, nperm=nperm, rng=rng)
            cells.append(f'drop {k}: r {r:+.2f} p {p:.4f}')
        out.append((m, '; '.join(cells), [f'{A[iu[0][q]]}/{A[iu[1][q]]}' for q in order[:6]]))
    return out
