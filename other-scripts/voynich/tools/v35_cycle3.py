"""v35 cycle 3: WHICH MECHANISM?  Are look-alike glyphs behaviourally INDISTINGUISHABLE (homophones / free variants,
the cipher-designer mechanism) or merely SIMILAR but distinct (featural design, Hangul-like)?

Indistinguishability index for a glyph pair (a, b): split the corpus into alternating 400-word blocks; PPMI context
rows from each half; I(a,b) = mean(cos(a1,b2), cos(b1,a2)) / sqrt(cos(a1,a2) cos(b1,b2)).  I ~ 1: the two glyphs are
as alike as a glyph is to itself in the other half (homophones).  I << 1: distinct behaviour.
Statistic: mean I over the 10 most shape-similar pairs (hand decomposition if present, else image combo), vs 5,000
random 10-pair sets (p), and vs the mean over all pairs.
Calibration: planted homophonic cipher with shape-family homophones (must give I ~ 1 on top pairs), random-homophone
cipher (top pairs ~ all pairs), Hangul (featural, distinct jamo).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v35_lib as V, v28_lib as X, v25_lib as L, v25_shapes as S


def half_ppmi(words, alph):
    a, b = L.halves(words)
    out = []
    for h in (a, b):
        C, _, _ = L.contexts(h, alph)
        P = L.ppmi(C)
        n = np.linalg.norm(P, axis=1, keepdims=True); n[n == 0] = 1
        out.append(P / n)
    return out


def index(words, alph):
    P1, P2 = half_ppmi(words, alph)
    Xc = P1 @ P2.T
    d = np.sqrt(np.maximum(np.outer(np.diag(Xc), np.diag(Xc)), 1e-12))
    return 0.5 * (Xc + Xc.T) / d


def shape_of(name, c):
    S_ = X.load(f'sims_{name}.pkl')
    if S_ is not None and 'hand' in S_:
        return S_['hand'], 'hand'
    if S_ is not None:
        k = [k for k in S_ if k.startswith('img:')][0]
        return S_[k], k
    F, _ = L.shape_matrix(c['hand'], c['alph'])
    return L.jaccard_sim(F), 'hand'


def test(name, k=10, nperm=5000, seed=7):
    rng = np.random.default_rng(seed)
    if name == 'hangul':
        ws = L.corpus('hangul'); A = L.alphabet(ws, S.HANGUL)
        c = dict(words=ws, alph=A, hand={g: S.HANGUL[g] for g in A})
    else:
        c = V.build(name)
    A = c['alph']; n = len(A)
    Sh, sname = shape_of(name, c)
    I = index(c['words'], A)
    iu = np.triu_indices(n, 1)
    sv = Sh[iu] + rng.random(len(iu[0])) * 1e-9
    top = np.argsort(-sv)[:k]
    Iv = I[iu]
    m_top = float(Iv[top].mean())
    null = np.array([Iv[rng.choice(len(Iv), k, replace=False)].mean() for _ in range(nperm)])
    p = float((1 + (null >= m_top).sum()) / (nperm + 1))
    pairs = [(A[iu[0][t]], A[iu[1][t]], round(float(Iv[t]), 2)) for t in top]
    return dict(name=name, n=n, shape=sname, top=m_top, all=float(Iv.mean()), p=p, pairs=pairs,
                max_all=float(np.sort(Iv)[-k:].mean()))


def classsplit(name, nperm=2000, seed=9):
    """KILL CONTROL for a class confound (marks vs letters, tall vs short, finals vs syllables): split the inventory
    into the 2 top-level shape clusters (average linkage on the shape matrix); report (i) partial Spearman r of shape vs
    behaviour given same-cluster, (ii) Mantel with labels permuted only WITHIN clusters (p), (iii) the split."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    rng = np.random.default_rng(seed)
    c = V.build(name); A = c['alph']
    B = X.load(f'beh_{name}.pkl'); S_ = X.load(f'sims_{name}.pkl')
    out = {}
    iu = np.triu_indices(len(A), 1)
    for k in [k for k in S_ if k.startswith('img:')][:1] + (['hand'] if 'hand' in S_ else []):
        Sh = S_[k]
        D = Sh.max() - Sh; np.fill_diagonal(D, 0); D = (D + D.T) / 2
        cl = fcluster(linkage(squareform(D, checks=False), 'average'), 2, 'maxclust')
        same = (cl[:, None] == cl[None, :]).astype(float)
        small = min(np.bincount(cl)[1:])
        for m in ('ppmi', 'svd', 'potts'):
            pr = L.partial_spearman(Sh[iu], B[m][iu], [same[iu]])
            r, p, _, _ = L.mantel(Sh, B[m], nperm=nperm, rng=rng, strata=list(cl))
            out[(k, m)] = (pr, r, p)
        mino = 1 + int(np.argmin(np.bincount(cl)[1:]))
        out[(k, 'split')] = [A[i] for i in range(len(A)) if cl[i] == mino]
    return out


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'class':
    res = {}
    for nm in sys.argv[2:]:
        res[nm] = classsplit(nm)
        for kk, v in res[nm].items():
            print(nm, kk, v if kk[1] == 'split' else tuple(round(x, 4) for x in v), flush=True)
    X.save('c3_class.pkl', res)
    sys.exit()

if __name__ == '__main__':
    names = sys.argv[1:] or ['voy', 'hangul', 'plant_family', 'plant_rand', 'copiale', 'borg', 'tengwar', 'shavian',
                             'deseret', 'cherokee', 'cree']
    out = []
    for nm in names:
        r = test(nm)
        out.append(r)
        print(nm, r['n'], r['shape'], f"top10 I {r['top']:.3f} (p {r['p']:.4f}) all {r['all']:.3f} best10 {r['max_all']:.3f}",
              r['pairs'][:5], flush=True)
    X.save('c3.pkl', out)
