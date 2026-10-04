"""v25 cycle 2: replace the hand-authored stroke decomposition by IMAGE-BASED shape similarity from
fonts (v25_image), so the shape side carries no knowledge of glyph behaviour.  All corpora, Voynich
split by Currier language, Latin under three fonts, planted featural control re-scored on images,
and swap pairs against image similarity."""
import os, sys, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v25_lib as L, v25_shapes as S, v25_image as I
from v25_cycle1 import get_beh, PHON
from multiprocessing import Pool

OUT = os.path.join(L.LOOPS, 'v25_cycle2.txt')
DESC = ('zone', 'frame', 'topo', 'combo')


def beh_lang(lang):
    fn = os.path.join(L.CK, f'beh_voy_{lang}.pkl')
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    w = L.voynich('ZL3b', lang=lang)
    A = L.alphabet(w, S.VOYNICH)
    B = L.behaviour(w, A)
    pickle.dump((A, B, w), open(fn, 'wb'))
    return A, B, w


def job(arg):
    tag, script, src = arg
    rng = np.random.default_rng(5)
    if src == 'langA':
        A, B, w = beh_lang('A')
    elif src == 'langB':
        A, B, w = beh_lang('B')
    else:
        A, B, w = get_beh(src, {'voynich_it': 'voynich'}.get(src, src if src != 'voynich' else 'voynich'))
    Im = I.image_sims(script, A)
    rows = []
    iu = np.triu_indices(len(A), 1)
    strata = L.freq_strata(B['_freq'])
    for d in DESC:
        cells = []
        for m in ('ppmi', 'svd', 'potts'):
            r, p, z, _ = L.mantel(Im[d], B[m], nperm=5000, rng=rng)
            r2, p2, z2, _ = L.mantel(Im[d], B[m], nperm=2000, rng=rng, strata=strata)
            cells.append(f'{m} r {r:+.3f} p {p:.4f} (freq-strat p {p2:.4f})')
        rows.append((f'V-25.2.1 {tag}/{d}', f'IMAGE shape similarity ({d}, font {os.path.basename(I.FONTS[script])}) vs behaviour; null random glyph-to-image assignment (5,000) and frequency-stratified', '; '.join(cells), ''))
    # agreement of image shape with the hand decomposition (for Voynich / reference)
    if script in S.SHAPES:
        F, _ = L.shape_matrix(S.SHAPES[script], A)
        rows.append((f'V-25.2.2 {tag}', 'agreement of image combo similarity with the hand-authored stroke Jaccard', f'Spearman {L.spearman_vec(Im["combo"][iu], L.jaccard_sim(F)[iu]):+.3f}', ''))
    # topology features one at a time: which automatic stroke feature carries behaviour
    topo = Im['_topo_raw']; names = ['holes', 'ends', 'junctions', 'components', 'ascender', 'descender', 'width', 'ink']
    cells = []
    for k, nm in enumerate(names):
        v = topo[:, k]
        if v.std() == 0:
            continue
        Sk = -np.abs(v[:, None] - v[None, :])
        r, p, z, _ = L.mantel(Sk, B['ppmi'], nperm=2000, rng=rng)
        cells.append(f'{nm} r {r:+.2f} p {p:.3f}')
    rows.append((f'V-25.2.3 {tag}', 'single automatic stroke features (|difference| as dissimilarity) vs PPMI behaviour', '; '.join(cells), ''))
    return rows


def swaps_img(arg):
    tag, script, src = arg
    rng = np.random.default_rng(9)
    A, B, w = get_beh(src, 'voynich' if src.startswith('voynich') else src)
    Im = I.image_sims(script, A)
    idx = {g: i for i, g in enumerate(A)}
    if src.startswith('voynich'):
        lines = []
        for Ln in L.voynich_lines('ZL3b' if src == 'voynich' else 'IT2a'):
            seg = []
            for x in Ln:
                if x is None:
                    if len(seg) > 1: lines.append(seg)
                    seg = []
                else:
                    seg.append(x)
            if len(seg) > 1: lines.append(seg)
    else:
        lines = L.pseudo_lines(w, 9)
    ex = L.swap_excess(lines, nshuf=60)
    n = len(A); Z = np.zeros((n, n))
    for (a, b), (o, e, z) in ex.items():
        if a in idx and b in idx:
            Z[idx[a], idx[b]] = Z[idx[b], idx[a]] = z
    r, p, zz, _ = L.mantel(Im['combo'], Z, nperm=5000, rng=rng)
    out = [(f'V-25.2.4 {tag}', 'swap-pair excess z (lag-1 Hamming-1, within-line shuffle) vs IMAGE combo similarity (Mantel)', f'r {r:+.3f} p {p:.4f}', '')]
    if script == 'voynich':
        iu = np.triu_indices(n, 1); ranks = {}
        allv = Im['combo'][iu]
        cells = []
        for a, b in S.SWAPS:
            v = Im['combo'][idx[a], idx[b]]
            cells.append(f'{a}/{b} {100 * (allv < v).mean():.0f}th pct')
        out.append((f'V-25.2.5 {tag}', 'named swap pairs: percentile of their image similarity among all 253 pairs', ', '.join(cells), ''))
    return out


def planted_img(_):
    rng = np.random.default_rng(13)
    w = L.corpus('latin')
    Al = L.alphabet(w, S.LATIN)
    feats = sorted({f for g in Al for f in PHON[g].split()})
    Ph = np.array([[1.0 if f in PHON[g].split() else 0 for f in feats] for g in Al])
    Sph = L.jaccard_sim(Ph)
    Av = list(S.VOYNICH)
    Sv = I.image_sims('voynich', Av)['combo']
    iu = np.triu_indices(len(Al), 1)

    def fit(a):
        return L.spearman_vec(Sph[iu], Sv[np.ix_(a, a)][iu])
    best = None
    for rep in range(6):
        a = list(rng.permutation(len(Av))[:len(Al)]); cur = fit(a); T = 0.05
        for it in range(20000):
            b = a[:]; i = rng.integers(len(Al)); j = rng.integers(len(Av))
            if j in b:
                k = b.index(j); b[i], b[k] = b[k], b[i]
            else:
                b[i] = j
            f = fit(b)
            if f > cur or rng.random() < np.exp((f - cur) / T):
                a, cur = b, f
            T *= 0.9997
        if best is None or cur > best[1]:
            best = (a, cur)

    def run(a):
        m = {Al[i]: Av[a[i]] for i in range(len(Al))}
        ww = [tuple(m[c] for c in x) for x in w]
        Ag = [Av[a[i]] for i in range(len(Al))]
        B = L.behaviour(ww, Ag, models=('ppmi', 'svd'))
        return L.mantel(Sv[np.ix_(a, a)], B['ppmi'], nperm=2000, rng=rng)[:3]
    rf = run(best[0])
    arb = np.array([run(list(rng.permutation(len(Av))[:len(Al)]))[0] for _ in range(40)])
    return [('V-25.2.6 planted featural (image)', f'Latin in Voynich EVA-font shapes, assignment annealed so sound-feature similarity matches IMAGE similarity (fit {best[1]:+.2f})', f'r {rf[0]:+.3f} p {rf[1]:.4f}', 'must be recovered'),
            ('V-25.2.7 planted arbitrary (image)', '40 arbitrary Latin-to-EVA-shape assignments', f'r mean {arb.mean():+.3f} sd {arb.std():.3f} max {arb.max():+.3f}', 'must be null')]


if __name__ == '__main__':
    jobs = [('voynich ZL', 'voynich', 'voynich'), ('voynich IT', 'voynich', 'voynich_it'), ('voynich ZL Currier A', 'voynich', 'langA'),
            ('voynich ZL Currier B', 'voynich', 'langB'), ('hangul', 'hangul', 'hangul'), ('latin DejaVuSerif', 'latin', 'latin'),
            ('latin DejaVuSans', 'latin_sans', 'latin'), ('latin FreeSerif', 'latin_free', 'latin'), ('greek', 'greek', 'greek')]
    sj = [('voynich ZL', 'voynich', 'voynich'), ('voynich IT', 'voynich', 'voynich_it'), ('hangul', 'hangul', 'hangul'), ('latin', 'latin', 'latin'), ('greek', 'greek', 'greek')]
    with Pool(2) as pool:
        res = pool.map(job, jobs) + pool.map(swaps_img, sj) + pool.map(planted_img, [0])
    rows = [r for rr in res for r in rr]
    hdr = ('# v25 cycle 2 - image-based shape (no hand decomposition). Glyphs rendered from fonts: Landini EVA Hand A (Voynich), DejaVu Serif/Sans, FreeSerif (Latin), DejaVu Serif (Greek), WenQuanYi Zen Hei (Hangul jamo).\n'
           '# Descriptors: zone (16x16 bbox density), frame (baseline-fixed 12x16), topo (holes, skeleton ends/junctions, components, ascender, descender, width, ink), combo = mean z.\n'
           '| row | method and control | result | verdict |\n|---|---|---|---|')
    L.write_rows(OUT, rows, hdr)
    print(open(OUT).read())
