"""v25 cycle 3: controls that could KILL a 'sound feature' reading of the shape-behaviour link.

K1 misreading: Latin written in arbitrary EVA shapes + shape-driven confusion noise (rate 0.5-10%);
   how much shape-behaviour r does confusion alone make?  Voynich consensus text (lines where ZL and
   IT agree glyph for glyph) and the ZL-IT single-glyph disagreement rate.
K2 copy-and-edit generator (self-citation with stroke edits): words copied from recent words and
   edited by swapping a glyph for a SHAPE-SIMILAR one (and a shape-blind version).  Does it reproduce
   the Voynich Mantel r, the swap Mantel, and the parallelograms?
K3 position: partial Mantel controlling for the glyph's word-position profile (initial/medial/final/
   alone) -- is the shape link only a positional class link?
K4 compositional prediction: leave-one-glyph-out ridge from stroke features (hand and image topo)
   to the glyph's behaviour embedding; score = mean cosine(pred, actual) vs relabelled null.
"""
import os, sys, pickle, random, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v25_lib as L, v25_shapes as S, v25_image as I
from v25_cycle1 import get_beh
from multiprocessing import Pool

OUT = os.path.join(L.LOOPS, 'v25_cycle3.txt')
AV = list(S.VOYNICH)


def img_sim(script, A):
    return I.image_sims(script, A)['combo']


def mantel_img_hand(words, A, rng, script='voynich', nperm=2000):
    B = L.behaviour(words, A, models=('ppmi', 'svd'))
    Si = img_sim(script, A)
    F, _ = L.shape_matrix(S.SHAPES[script], A)
    Sh = L.jaccard_sim(F)
    ri = L.mantel(Si, B['ppmi'], nperm=nperm, rng=rng)
    rh = L.mantel(Sh, B['ppmi'], nperm=nperm, rng=rng)
    return ri[:3], rh[:3], B


# ---------------------------------------------------------------- K1
def k1_confusion(_):
    rng = np.random.default_rng(21)
    rows = []
    w = L.corpus('latin')
    Al = L.alphabet(w, S.LATIN)
    Svi = img_sim('voynich', AV)
    res = {}
    for rate in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        rs = []
        for rep in range(12):
            a = list(rng.permutation(len(AV))[:len(Al)])
            m = {Al[i]: AV[a[i]] for i in range(len(Al))}
            Ag = [AV[i] for i in a]
            Ss = Svi[np.ix_(a, a)]
            P = np.exp(4 * (Ss - Ss.mean())); np.fill_diagonal(P, 0); P /= P.sum(1, keepdims=True)
            gi = {g: i for i, g in enumerate(Ag)}
            ww = []
            for x in w:
                y = []
                for c in x:
                    g = m[c]
                    if rng.random() < rate:
                        g = Ag[rng.choice(len(Ag), p=P[gi[g]])]
                    y.append(g)
                ww.append(tuple(y))
            B = L.behaviour(ww, Ag, models=('ppmi',))
            rs.append(L.mantel(Ss, B['ppmi'], nperm=300, rng=rng)[0])
        res[rate] = (np.mean(rs), np.std(rs))
    rows.append(('V-25.3.1 K1 confusion', 'Latin in 12 arbitrary EVA-shape assignments + misreading noise (each glyph replaced at rate x by a glyph drawn by exp(4 x image similarity)); image Mantel r (ppmi)',
                 '; '.join(f'rate {k:.3f}: r {v[0]:+.3f} +- {v[1]:.3f}' for k, v in res.items()), ''))
    return rows


def k1_consensus(_):
    rng = np.random.default_rng(22)
    zl = json.load(open(os.path.join(L.DATA, 'derived', 'ZL3b_lines.json')))
    it = json.load(open(os.path.join(L.DATA, 'derived', 'IT2a_lines.json')))
    key = lambda Ln: (Ln['folio'], Ln['n'])
    itd = {key(x): x for x in it}
    nglyph = ndiff = 0; cons = []; nl = nagree = 0
    keep = set(AV)
    for Ln in zl:
        if Ln['ltype'] != 'P' or key(Ln) not in itd:
            continue
        a = Ln['words']; b = itd[key(Ln)]['words']
        nl += 1
        if len(a) != len(b):
            continue
        ok = True
        for x, y in zip(a, b):
            gx, gy = L.vglyphs(x), L.vglyphs(y)
            if len(gx) == len(gy):
                nglyph += len(gx); d = sum(p != q for p, q in zip(gx, gy)); ndiff += d
                if d: ok = False
            else:
                ok = False
        if ok:
            nagree += 1
            for x, u in zip(a, Ln['uncertain']):
                g = tuple(L.vglyphs(x))
                if not u and '?' not in x and all(c in keep for c in g):
                    cons.append(g)
    A = L.alphabet(cons, S.VOYNICH)
    ri, rh, _ = mantel_img_hand(cons, A, rng)
    return [('V-25.3.2 K1 consensus', f'ZL-IT: single-glyph disagreement rate within equal-length aligned words; Voynich CONSENSUS text = {nagree} of {nl} paragraph lines where ZL and IT agree glyph for glyph ({len(cons)} words, {len(A)} glyphs)',
             f'disagreement {ndiff}/{nglyph} = {ndiff / max(nglyph, 1):.4f} per glyph; consensus image r {ri[0]:+.3f} p {ri[1]:.4f}; hand r {rh[0]:+.3f} p {rh[1]:.4f}', '')]


# ---------------------------------------------------------------- K2 generator
def gen_copy_edit(seed_words, n, Sim, A, rng, shape=True, p_edit=0.7, window=30, p_fresh=0.03):
    gi = {g: i for i, g in enumerate(A)}
    P = np.exp(5 * (Sim - Sim.mean())) if shape else np.ones_like(Sim)
    np.fill_diagonal(P, 0); P /= P.sum(1, keepdims=True)
    freq = np.ones(len(A)) / len(A)
    out = [seed_words[rng.integers(len(seed_words))] for _ in range(window)]
    while len(out) < n + window:
        if rng.random() < p_fresh:
            out.append(seed_words[rng.integers(len(seed_words))]); continue
        w = list(out[-1 - rng.integers(window)])
        if rng.random() < p_edit:
            for _ in range(1 + (rng.random() < 0.3)):
                k = rng.integers(len(w))
                w[k] = A[rng.choice(len(A), p=P[gi[w[k]]])]
        out.append(tuple(w))
    return out[window:]


def k2_generator(arg):
    shape, seedsrc = arg
    rng = np.random.default_rng(23 + shape)
    vw = L.corpus('voynich')
    A = L.alphabet(vw, S.VOYNICH)
    Sim = img_sim('voynich', A)
    if seedsrc == 'voynich':
        seeds = vw
    else:  # shape-blind seeds: Voynich words with glyph labels randomly permuted (no shape-behaviour link to start from)
        pm = dict(zip(A, rng.permutation(A)))
        seeds = [tuple(pm[g] for g in x) for x in vw]
    g = gen_copy_edit(seeds, len(vw), Sim, A, rng, shape=bool(shape))
    ri, rh, B = mantel_img_hand(g, A, rng)
    # swap Mantel
    lines = L.pseudo_lines(g, 9)
    ex = L.swap_excess(lines, nshuf=30)
    idx = {x: i for i, x in enumerate(A)}; n = len(A); Z = np.zeros((n, n))
    for (a, b), (o, e, z) in ex.items():
        if a in idx and b in idx:
            Z[idx[a], idx[b]] = Z[idx[b], idx[a]] = z
    rs = L.mantel(Sim, Z, nperm=2000, rng=rng)
    an = []
    for fam, prs in S.V_ANALOGIES.items():
        t = L.analogy_test(B['_emb'], A, prs, nperm=2000, rng=rng)
        if t: an.append(f'{fam.split()[0]} {t[0]:+.2f} (p {t[1]:.3f})')
    lab = f'{"shape-driven" if shape else "shape-blind"} edits, seeds {"Voynich words" if seedsrc == "voynich" else "relabelled Voynich words"}'
    return [(f'V-25.3.3 K2 copy-edit ({lab})', 'self-citation generator: each word copied from one of the last 30, 70% edited (1-2 glyph substitutions; shape-driven = drawn by exp(5 x image similarity)), 3% fresh seed words; same size as ZL',
             f'image r {ri[0]:+.3f} p {ri[1]:.4f}; hand r {rh[0]:+.3f} p {rh[1]:.4f}; swap Mantel r {rs[0]:+.3f} p {rs[1]:.4f}; parallelograms: ' + ', '.join(an), '')]


# ---------------------------------------------------------------- K3 position profile
def posprof(words, A):
    idx = {g: i for i, g in enumerate(A)}
    P = np.zeros((len(A), 4))
    for w in words:
        for k, g in enumerate(w):
            if g not in idx: continue
            c = 3 if len(w) == 1 else (0 if k == 0 else (2 if k == len(w) - 1 else 1))
            P[idx[g], c] += 1
    P = P / P.sum(1, keepdims=True)
    return -np.abs(P[:, None] - P[None]).sum(2)


def k3_position(arg):
    name, script = arg
    rng = np.random.default_rng(24)
    A, B, w = get_beh(name, 'voynich' if name.startswith('voynich') else name)
    iu = np.triu_indices(len(A), 1)
    Pp = posprof(w, A)
    Si = img_sim(script, A)
    F, _ = L.shape_matrix(S.SHAPES[script], A); Sh = L.jaccard_sim(F)
    cells = []
    for lab, Ssh in (('image', Si), ('hand', Sh)):
        r0 = L.spearman_vec(Ssh[iu], B['ppmi'][iu])
        pr = L.partial_spearman(Ssh[iu], B['ppmi'][iu], [Pp[iu]])
        # permutation null for the partial r
        null = []
        for _ in range(2000):
            pm = rng.permutation(len(A)); Sp = Ssh[pm][:, pm]
            null.append(L.partial_spearman(Sp[iu], B['ppmi'][iu], [Pp[iu]]))
        null = np.array(null)
        rpos = L.spearman_vec(Ssh[iu], Pp[iu])
        cells.append(f'{lab}: r {r0:+.3f} -> partial {pr:+.3f} (p {(1 + (null >= pr).sum()) / 2001:.4f}); shape~position r {rpos:+.3f}')
    return [(f'V-25.3.4 K3 {name}', 'partial Mantel: shape vs behaviour controlling for word-position profile similarity (initial/medial/final/alone); null relabels glyphs', '; '.join(cells), '')]


# ---------------------------------------------------------------- K4 compositional prediction
def k4_compose(arg):
    name, script = arg
    rng = np.random.default_rng(25)
    A, B, w = get_beh(name, 'voynich' if name.startswith('voynich') else name)
    E = B['_emb']; E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    F, _ = L.shape_matrix(S.SHAPES[script], A)
    T = I.image_sims(script, A)['_topo_raw']; T = (T - T.mean(0)) / (T.std(0) + 1e-9)

    def loo(X, lam=1.0):
        cs = []
        for i in range(len(A)):
            tr = np.arange(len(A)) != i
            Xt = np.column_stack([X[tr], np.ones(tr.sum())])
            W = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ E[tr])
            p = np.append(X[i], 1) @ W
            cs.append(p @ E[i] / (np.linalg.norm(p) * np.linalg.norm(E[i]) + 1e-12))
        return float(np.mean(cs))
    cells = []
    for lab, X in (('hand', F), ('image-topo', T)):
        s0 = loo(X)
        null = np.array([loo(X[rng.permutation(len(A))]) for _ in range(500)])
        cells.append(f'{lab}: LOO cosine {s0:+.3f} vs null {null.mean():+.3f} +- {null.std():.3f}, p {(1 + (null >= s0).sum()) / 501:.4f}')
    return [(f'V-25.3.5 K4 {name}', 'compositional prediction: a held-out glyph\'s 8-d behaviour embedding predicted from its stroke features by ridge trained on the other glyphs; null relabels glyph-to-features (500)', '; '.join(cells), '')]


if __name__ == '__main__':
    C = [('voynich', 'voynich'), ('voynich_it', 'voynich'), ('hangul', 'hangul'), ('latin', 'latin'), ('greek', 'greek')]
    with Pool(2) as pool:
        res = (pool.map(k1_confusion, [0]) + pool.map(k1_consensus, [0]) +
               pool.map(k2_generator, [(1, 'voynich'), (0, 'voynich'), (1, 'relab'), (0, 'relab')]) +
               pool.map(k3_position, C) + pool.map(k4_compose, C))
    rows = [r for rr in res for r in rr]
    hdr = ('# v25 cycle 3 - kill controls for a sound-feature reading: misreading noise, consensus text, copy-and-edit generator, position profile, compositional prediction.\n'
           '| row | method and control | result | verdict |\n|---|---|---|---|')
    L.write_rows(OUT, rows, hdr)
    print(open(OUT).read())
