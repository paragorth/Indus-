"""v35 cycle 3b: KILL CONTROLS for the indistinguishability result (v35_cycle3.py).
(1) frequency-matched null: each of the 10 most shape-similar pairs (a,b) is replaced by a random pair (c,d) with c, d
    within +-2 frequency ranks of a and b (rare glyphs have noisy contexts, which could inflate or deflate I);
(2) Voynich with the gallows family (k t p f and benched gallows) removed: top-10 pairs among the rest;
(3) Voynich IT transliteration replication (v28 corpus voy:IT2a:v25, hand strokes)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v35_lib as V, v28_lib as X, v25_lib as L, v25_shapes as S
import v35_cycle3 as C3


def run(name, keep=None, k=10, nperm=5000, seed=13, corpus=None):
    rng = np.random.default_rng(seed)
    if name == 'hangul':
        ws = L.corpus('hangul'); A = L.alphabet(ws, S.HANGUL); c = dict(words=ws, alph=A, hand={g: S.HANGUL[g] for g in A})
    elif corpus is not None:
        c = corpus
    else:
        c = V.build(name)
    A = list(c['alph'])
    if c.get('hand') and all(g in c['hand'] for g in A):
        F, _ = L.shape_matrix(c['hand'], A); Sh = L.jaccard_sim(F); sname = 'hand'
    else:
        Sh, sname = C3.shape_of(name, c)
    I = C3.index(c['words'], A)
    from collections import Counter
    fr = Counter(g for w in c['words'] for g in w)
    rank = {g: r for r, g in enumerate(sorted(A, key=lambda g: -fr[g]))}
    ix = [i for i, g in enumerate(A) if keep is None or g in keep]
    pairs = [(i, j) for a, i in enumerate(ix) for j in ix[a + 1:]]
    sv = np.array([Sh[i, j] for i, j in pairs]) + rng.random(len(pairs)) * 1e-9
    top = [pairs[t] for t in np.argsort(-sv)[:k]]
    m_top = float(np.mean([I[i, j] for i, j in top]))
    byrank = {rank[A[i]]: i for i in ix}
    def near(i):
        r = rank[A[i]]
        return [byrank[q] for q in range(r - 2, r + 3) if q in byrank]
    null = []
    for _ in range(nperm):
        v = []
        for i, j in top:
            ci = rng.choice(near(i)); cj = rng.choice(near(j))
            while cj == ci:
                ci = rng.choice(near(i)); cj = rng.choice(near(j))
            v.append(I[ci, cj])
        null.append(np.mean(v))
    null = np.array(null)
    p = float((1 + (null >= m_top).sum()) / (nperm + 1))
    return dict(name=name, shape=sname, n=len(ix), top=m_top, null=float(null.mean()), p=p,
                pairs=[(A[i], A[j], round(float(I[i, j]), 2)) for i, j in top])


if __name__ == '__main__':
    import v28_corpora as C
    GAL = {'k', 't', 'p', 'f', 'ckh', 'cth', 'cph', 'cfh'}
    out = []
    jobs = [('voy', None, None), ('voy', 'nogallows', None), ('hangul', None, None), ('plant_family', None, None),
            ('tengwar', None, None), ('shavian', None, None), ('cree', None, None), ('borg', None, None),
            ('copiale', None, None)]
    for nm, opt, _ in jobs:
        keep = None
        if opt == 'nogallows':
            keep = set(V.build('voy')['alph']) - GAL
        r = run(nm, keep=keep); r['opt'] = opt; out.append(r)
        print(nm, opt, r['n'], r['shape'], f"top10 I {r['top']:.3f} freq-matched null {r['null']:.3f} p {r['p']:.4f}", r['pairs'][:6], flush=True)
    X.CK = os.path.join(L.DATA, 'v28_ckpt')
    it = C.build('voy:IT2a:v25')
    X.CK = V.CK
    it = dict(words=it['words'], alph=it['alph'], hand={g: S.VOYNICH[g] for g in it['alph'] if g in S.VOYNICH})
    for opt in (None, 'nogallows'):
        r = run('voy_it', keep=None if opt is None else set(it['alph']) - GAL, corpus=it); r['opt'] = opt; out.append(r)
        print('voy_IT', opt, r['n'], r['shape'], f"top10 I {r['top']:.3f} freq-matched null {r['null']:.3f} p {r['p']:.4f}", r['pairs'][:6], flush=True)
    X.save('c3b.pkl', out)
