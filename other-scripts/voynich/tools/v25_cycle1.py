"""v25 cycle 1: principled stroke decompositions vs behaviour (Mantel), random/perturbed decomposition
families, primitive-subset search with held-out fold, swap pairs, analogies, planted featural control."""
import os, sys, pickle, itertools, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v25_lib as L, v25_shapes as S
from multiprocessing import Pool

OUT = os.path.join(L.LOOPS, 'v25_cycle1.txt')
CORP = [('voynich', 'voynich'), ('voynich_it', 'voynich'), ('hangul', 'hangul'), ('latin', 'latin'), ('greek', 'greek')]
MODELS = ('ppmi', 'svd', 'potts')


def get_beh(name, sh, part=None):
    fn = os.path.join(L.CK, f'beh_{name}_{part}.pkl')
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    w = L.corpus(name)
    A = L.alphabet(w, S.SHAPES[sh])
    if part is not None:
        w = L.halves(w)[part]
    B = L.behaviour(w, A)
    pickle.dump((A, B, w), open(fn, 'wb'))
    return A, B, w


def prim_tensors(shapes, A):
    F, prims = L.shape_matrix(shapes, A)
    mn = np.minimum(F[:, None, :], F[None, :, :]); mx = np.maximum(F[:, None, :], F[None, :, :])
    iu = np.triu_indices(len(A), 1)
    return F, prims, mn[iu].T, mx[iu].T  # P x pairs


def rank_rows(X):
    from scipy.stats import rankdata
    R = rankdata(X, axis=1); R -= R.mean(1, keepdims=True)
    return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)


def subset_search(MN, MX, bvec, nmax=16384, seed=0):
    P = MN.shape[0]
    if 2 ** P <= nmax:
        masks = np.array(list(itertools.product([0, 1], repeat=P)))[1:]
    else:
        rng = np.random.default_rng(seed)
        masks = rng.integers(0, 2, size=(nmax, P)); masks = masks[masks.sum(1) > 0]
    num = masks @ MN; den = masks @ MX
    sim = np.where(den > 0, num / np.maximum(den, 1e-9), 0)
    ok = sim.std(1) > 0
    from scipy.stats import rankdata
    rb = rankdata(bvec); rb -= rb.mean(); rb /= np.linalg.norm(rb)
    r = np.full(len(masks), -1.0)
    r[ok] = rank_rows(sim[ok]) @ rb
    return masks, r


def job_corpus(arg):
    name, sh = arg
    rows = []
    rng = np.random.default_rng(7)
    A, B, w = get_beh(name, sh)
    F, prims, MN, MX = prim_tensors(S.SHAPES[sh], A)
    Ssh = L.jaccard_sim(F)
    iu = np.triu_indices(len(A), 1)
    freq = B['_freq']
    strata = L.freq_strata(freq)
    lf = np.log(freq)
    fd = np.abs(lf[:, None] - lf[None, :])[iu]; fs = (lf[:, None] + lf[None, :])[iu]
    res = {}
    for m in MODELS:
        r, p, z, _ = L.mantel(Ssh, B[m], nperm=5000, rng=rng)
        r2, p2, z2, _ = L.mantel(Ssh, B[m], nperm=5000, rng=rng, strata=strata)
        pr = L.partial_spearman(Ssh[iu], B[m][iu], [fd, fs])
        res[m] = (r, p, z)
        rows.append((f'V-25.1.1 {name}/{m}', f'principled stroke decomposition ({len(A)} glyphs, {len(prims)} primitives); Mantel Spearman shape-Jaccard vs behaviour cosine; null = random glyph-to-shape assignment (5,000) and frequency-tertile-stratified assignment; partial r given |dlogf| and sum logf',
                     f'r {r:+.3f} p {p:.4f} z {z:+.1f}; stratified p {p2:.4f} z {z2:+.1f}; partial r {pr:+.3f}', ''))
    # random decompositions family
    nR = 3000
    dens = (F > 0).mean()
    rr = {m: [] for m in MODELS}
    for k in range(nR):
        P = rng.integers(6, 17)
        Fr = (rng.random((len(A), P)) < dens).astype(float)
        Sr = L.jaccard_sim(Fr)
        if np.std(Sr[iu]) == 0:
            continue
        for m in MODELS:
            rr[m].append(L.spearman_vec(Sr[iu], B[m][iu]))
    # perturbed principled
    pt = {m: [] for m in MODELS}
    for k in range(3000):
        Fp = F.copy()
        for _ in range(rng.integers(1, 6)):
            i = rng.integers(len(A)); j = rng.integers(len(prims))
            Fp[i, j] = 0 if Fp[i, j] > 0 else 1
        Sp = L.jaccard_sim(Fp)
        for m in MODELS:
            pt[m].append(L.spearman_vec(Sp[iu], B[m][iu]))
    for m in MODELS:
        a = np.array(rr[m]); b = np.array(pt[m])
        rows.append((f'V-25.1.2 {name}/{m}', '3,000 random binary decompositions (6-16 primitives, density matched) and 3,000 perturbed principled ones (1-5 glyph-primitive entries flipped)',
                     f'random: mean {a.mean():+.3f}, 95th pct {np.quantile(a, .95):+.3f}, max {a.max():+.3f}; principled {res[m][0]:+.3f} beats {100 * (a < res[m][0]).mean():.1f}% of random; perturbed: median {np.median(b):+.3f}, 5th pct {np.quantile(b, .05):+.3f}',
                     ''))
    # primitive importance: single-primitive and drop-one
    imp = []
    for j, pnm in enumerate(prims):
        sh_j = (MN[j] > 0).astype(float)
        r1 = L.spearman_vec(sh_j, B['ppmi'][iu]) if sh_j.std() > 0 else 0.0
        keep = [q for q in range(len(prims)) if q != j]
        Sd = np.where(MX[keep].sum(0) > 0, MN[keep].sum(0) / np.maximum(MX[keep].sum(0), 1e-9), 0)
        rd = L.spearman_vec(Sd, B['ppmi'][iu])
        imp.append((pnm, r1, res['ppmi'][0] - rd))
    imp.sort(key=lambda x: -x[2])
    rows.append((f'V-25.1.3 {name}', 'which primitive carries it (ppmi): shared-primitive indicator r, and drop-one loss of r',
                 '; '.join(f'{p} {r1:+.2f}/{d:+.3f}' for p, r1, d in imp), ''))
    # subset search on half A, test on half B
    A0, B0, _ = get_beh(name, sh, 0)
    A1, B1, _ = get_beh(name, sh, 1)
    if A0 == A and A1 == A:
        for m in ('ppmi', 'potts'):
            masks, r = subset_search(MN, MX, B0[m][iu])
            best = masks[np.argmax(r)]
            sel = [prims[q] for q in range(len(prims)) if best[q]]
            Sb = np.where(MX[best > 0].sum(0) > 0, MN[best > 0].sum(0) / np.maximum(MX[best > 0].sum(0), 1e-9), 0)
            Sbm = np.zeros((len(A), len(A))); Sbm[iu] = Sb; Sbm = Sbm + Sbm.T + np.eye(len(A))
            rh, ph, zh, _ = L.mantel(Sbm, B1[m], nperm=5000, rng=rng)
            rp, pp, zp, _ = L.mantel(Ssh, B1[m], nperm=5000, rng=rng)
            rows.append((f'V-25.1.4 {name}/{m}', f'primitive-subset search ({len(masks)} subsets of {len(prims)}) on half A, best subset scored on held-out half B (permutation null on B)',
                         f'train best r {r.max():+.3f} (subset: {",".join(sel)}); held-out r {rh:+.3f} p {ph:.4f}; full principled on B r {rp:+.3f} p {pp:.4f}', ''))
    # analogies
    fams = S.ANALOGIES.get(sh, {})
    for fam, prs in fams.items():
        for rep, E in (('svd', B['_emb']), ('potts', B['_W'] - B['_W'].mean(0))):
            t = L.analogy_test(E, A, prs, nperm=3000, rng=rng)
            if t:
                rows.append((f'V-25.1.5 {name}/{rep}', f'parallelogram: offsets for "{fam}" {prs} share direction? null = same family under random glyph relabelling',
                             f'mean offset cosine {t[0]:+.3f} vs null {t[2]:+.3f}, p {t[1]:.4f}', ''))
    return name, rows


def swaps(arg):
    name, sh = arg
    rows = []
    rng = np.random.default_rng(3)
    A, B, w = get_beh(name, sh)
    F, prims = L.shape_matrix(S.SHAPES[sh], A)
    idx = {g: i for i, g in enumerate(A)}
    if name.startswith('voynich'):
        lines = []
        for Ln in L.voynich_lines('ZL3b' if name == 'voynich' else 'IT2a'):
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
    n = len(A)
    Z = np.zeros((n, n)); O = np.zeros((n, n))
    for (a, b), (o, e, z) in ex.items():
        if a in idx and b in idx:
            Z[idx[a], idx[b]] = Z[idx[b], idx[a]] = z; O[idx[a], idx[b]] = O[idx[b], idx[a]] = o
    Ssh = L.jaccard_sim(F)
    D = L.l1(F)
    r, p, zz, _ = L.mantel(Ssh, Z, nperm=5000, rng=rng)
    iu = np.triu_indices(n, 1)
    sig = [(A[i], A[j], Z[i, j], D[i, j]) for i, j in zip(*iu) if Z[i, j] > 2.5]
    sig.sort(key=lambda x: -x[2])
    one = (D[iu] == 1).mean()
    sig1 = np.mean([d == 1 for *_, d in sig]) if sig else float('nan')
    dmean_sig = np.mean([d for *_, d in sig]) if sig else float('nan')
    # null for mean stroke distance of the significant pairs: random relabelling
    nulld = []
    for _ in range(5000):
        pm = rng.permutation(n); Dp = D[pm][:, pm]
        nulld.append(np.mean([Dp[idx[a], idx[b]] for a, b, *_ in sig]) if sig else 0)
    nulld = np.array(nulld)
    pd = (1 + (nulld <= dmean_sig).sum()) / 5001
    rows.append((f'V-25.1.6 {name}', 'swap pairs: lag-1 Hamming-1 word neighbours per glyph pair, z vs 60 within-line shuffles; Mantel of swap z vs stroke similarity; stroke distance (L1 primitive count) of pairs with z > 2.5 vs random relabelling',
                 f'Mantel r {r:+.3f} p {p:.4f}; {len(sig)} pairs z>2.5: ' + ', '.join(f'{a}/{b} z{z:.1f} d{int(d)}' for a, b, z, d in sig[:10]) +
                 f'; share at d=1 {sig1:.2f} (all pairs {one:.2f}); mean d {dmean_sig:.2f} vs null {nulld.mean():.2f}, p(lower) {pd:.4f}', ''))
    if sh == 'voynich':
        nm = [(a, b, int(D[idx[a], idx[b]])) for a, b in S.SWAPS]
        rows.append((f'V-25.1.7 {name}', 'named swap pairs (v5/v12) stroke distance under the principled decomposition',
                     ', '.join(f'{a}/{b} d{d}' for a, b, d in nm) + f'; mean {np.mean([d for *_, d in nm]):.2f} vs all-pair mean {D[iu].mean():.2f}, share d=1 {np.mean([d == 1 for *_, d in nm]):.2f} vs {one:.2f}', ''))
    return name + '_swap', rows


# ---------------------------------------------------------------- planted featural control
PHON = {  # Latin letter -> phonological features (sound only)
    'a': 'V low', 'e': 'V mid front', 'i': 'V high front', 'o': 'V mid back round', 'u': 'V high back round',
    'b': 'C stop lab vd', 'p': 'C stop lab', 'm': 'C nas lab vd son', 'f': 'C fric lab',
    'd': 'C stop cor vd', 't': 'C stop cor', 'n': 'C nas cor vd son', 's': 'C fric cor', 'l': 'C liq cor vd son', 'r': 'C liq cor vd son trill',
    'c': 'C stop dor', 'g': 'C stop dor vd', 'q': 'C stop dor round', 'h': 'C fric glot', 'x': 'C stop dor fric cor', 'y': 'V high front round', 'z': 'C fric cor vd', 'k': 'C stop dor',
}


def planted(_):
    rows = []
    rng = np.random.default_rng(11)
    w = L.corpus('latin')
    Al = L.alphabet(w, S.LATIN)
    feats = sorted({f for g in Al for f in PHON[g].split()})
    Ph = np.array([[1.0 if f in PHON[g].split() else 0 for f in feats] for g in Al])
    Sph = L.jaccard_sim(Ph)
    Bl = L.behaviour(w, Al, models=('ppmi', 'svd'))
    r_ph, p_ph, z_ph, _ = L.mantel(Sph, Bl['ppmi'], nperm=5000, rng=rng)
    rows.append(('V-25.1.8 latin', 'calibration: does Latin SOUND-feature similarity predict Latin letter behaviour? (phonological features, Mantel)', f'r {r_ph:+.3f} p {p_ph:.4f} z {z_ph:+.1f}', 'upper bound for what a featural script with Latin underneath can show'))
    Av = list(S.VOYNICH)
    Fv, _ = L.shape_matrix(S.VOYNICH, Av)
    Sv = L.jaccard_sim(Fv)
    iu = np.triu_indices(len(Al), 1)

    def fit(assign):  # assign: Latin index -> Voynich index
        sv = Sv[np.ix_(assign, assign)]
        return L.spearman_vec(Sph[iu], sv[iu])
    # annealing for a featural assignment
    best = None
    for rep in range(6):
        a = list(rng.permutation(len(Av))[:len(Al)])
        cur = fit(a); T = 0.05
        for it in range(20000):
            b = a[:]
            i = rng.integers(len(Al))
            j = rng.integers(len(Av))
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

    def run(assign):
        m = {Al[i]: Av[assign[i]] for i in range(len(Al))}
        ww = [tuple(m[c] for c in x) for x in w]
        Ag = [Av[assign[i]] for i in range(len(Al))]
        B = L.behaviour(ww, Ag, models=('ppmi', 'svd'))
        F2, _ = L.shape_matrix(S.VOYNICH, Ag)
        return L.mantel(L.jaccard_sim(F2), B['ppmi'], nperm=2000, rng=rng)[:3]
    rf = run(best[0])
    mp = {Al[i]: Av[best[0][i]] for i in range(len(Al))}
    rows.append(('V-25.1.9 planted featural', f'Latin written in Voynich glyph shapes with a FEATURAL assignment (annealed so that sound-feature similarity matches stroke similarity, fit rho {best[1]:+.2f}); principled Voynich decomposition; Mantel', f'r {rf[0]:+.3f} p {rf[1]:.4f} z {rf[2]:+.1f}; map ' + ' '.join(f'{k}={v}' for k, v in sorted(mp.items())), 'must be recovered'))
    arb = [run(list(rng.permutation(len(Av))[:len(Al)])) for _ in range(40)]
    a_r = np.array([x[0] for x in arb]); a_p = np.array([x[1] for x in arb])
    rows.append(('V-25.1.10 planted arbitrary', 'same Latin text, 40 ARBITRARY letter-to-Voynich-shape assignments', f'r mean {a_r.mean():+.3f} sd {a_r.std():.3f}, max {a_r.max():+.3f}; share p<0.05 {np.mean(a_p < 0.05):.2f}', 'must be null'))
    # intermediate strength: assignments with fit rho ~ 0.2-0.4
    return 'planted', rows


if __name__ == '__main__':
    t0 = time.time()
    # behaviour caches first (two workers)
    jobs = [(n, s, p) for n, s in CORP for p in (None, 0, 1)]
    with Pool(2) as pool:
        pool.starmap(get_beh, jobs)
    print('behaviour cached', time.time() - t0, flush=True)
    with Pool(2) as pool:
        res = pool.map(job_corpus, CORP) + pool.map(swaps, CORP) + pool.map(planted, [0])
    rows = [r for _, rr in res for r in rr]
    hdr = ('# v25 cycle 1 - THE GLYPHS ARE BUILT FROM FEATURES. Stroke decompositions fixed in tools/v25_shapes.py before any statistics.\n'
           '# Corpora matched at ~120,000 glyph tokens: Voynich ZL3b and IT2a paragraph text (23 units), Hangul (NSMC review corpus, jamo), Latin (Isidore), Greek (3 Gutenberg texts, accents stripped).\n'
           '# Behaviour: L1/R1/L2/R2 context counts -> PPMI rows, 8-d SVD embedding, pseudolikelihood Potts couplings. Halves = alternating 400-word blocks.\n'
           '| row | method and control | result | verdict |\n|---|---|---|---|')
    L.write_rows(OUT, [(a, b, c, d) for a, b, c, d in rows], hdr)
    print(open(OUT).read())
