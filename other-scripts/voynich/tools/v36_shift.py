"""v36 cycle 3b: SEARCH-FREE test.  Does the frequency shift X -> Y move along stroke/feature lines?
For each glyph g and word slot s (initial / medial / final): shift D[g,s] = log((pY+a)/(pX+a)), pX, pY the
glyph's share of slot-s tokens, a = 2/n (shrink).  Statistics:
  MANTEL  Spearman between feature similarity and shift similarity (-||D_g - D_h||) over glyph pairs
  FEATMAX the largest |mean shift| over glyphs carrying one feature, any slot (max over features x slots)
Null = 2,000 frequency-stratified label permutations (random glyph classes), z and p; each pair is run on
5 equal-size page subsamples (5,600 words a side) and the z averaged.  Calibrated with the cycle-2
plants (featural vs arbitrary vs none) and real pairs.
Usage: python3 v36_shift.py
"""
import os, sys, json, random, collections
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as L
import v36_plant as P
import v25_shapes as S

NW = 5600


def slots(words, tok, alph):
    ix = {g: i for i, g in enumerate(alph)}
    C = np.zeros((len(alph), 3))
    for w in words:
        gs = tok(w)
        for k, g in enumerate(gs):
            if g in ix:
                s = 0 if k == 0 else (2 if k == len(gs) - 1 else 1)
                C[ix[g], s] += 1
    return C


def sample(pages, rng, n=NW):
    P_ = list(pages); rng.shuffle(P_); out = []
    for p in P_:
        out.extend(p)
        if len(out) >= n:
            break
    return out[:n]


def pair_data(name):
    C = L.corpora30()
    if name.startswith('plant_'):
        X, Y, _ = P.make(name[6:]); return X, Y, {'P': 'voynich', 'H': 'hangul', 'G': 'latin'}[name[6]]
    spec = {'V_AB': ('V_A', 'V_B'), 'V_AherbBherb': ('V_Aherb', 'V_Bherb'), 'V_B2B3': ('V_B2', 'V_B3'),
            'N_AA': ('V_A', None), 'N_BB': ('V_B', None), 'L_BavAlem': ('G_Bav1', 'G_Alem'), 'L_LatIta': ('I_Lat', 'I_Ita'),
            'L_CzReform': ('C_Mod', 'C_Old'), 'L_BavBav': ('G_Bav1', 'G_Bav2'), 'K_Sub': ('G_Bav2', 'K_Sub'),
            'K_Swap3': ('G_Bav2', 'K_Swap3'), 'N_BavBav': ('G_Bav1', None)}[name]
    x, y = spec
    if y is None:
        pg = C[x]; return pg[0::2], pg[1::2], ('voynich' if x.startswith('V') else 'latin')
    return C[x], C[y], ('voynich' if x.startswith('V') else 'latin')


def mantel_stats(D, F, rng, nperm=2000):
    n = len(F.alph); iu = np.triu_indices(n, 1)
    dist = np.sqrt(((D[:, None, :] - D[None, :, :]) ** 2).sum(2))
    bs = rankdata(-dist[iu])
    def stat(p):
        Mp = F.M[np.ix_(p, p)]
        return np.corrcoef(rankdata(Mp[iu]), bs)[0, 1]
    H = np.array([[1.0 if f in s else 0.0 for s in F.sets] for f in F.feats]); cnt = H.sum(1, keepdims=True)
    ok = (cnt[:, 0] >= 2) & (cnt[:, 0] <= n - 2)
    def fmax(p):
        Dp = D[p]
        m = (H @ Dp) / np.maximum(cnt, 1)
        return np.abs(m[ok]).max(), m
    o_m = stat(np.arange(n)); o_f, o_tab = fmax(np.arange(n))
    nm, nf, ntab = [], [], []
    for _ in range(nperm):
        p = F.perm(rng); nm.append(stat(p)); a, t = fmax(p); nf.append(a); ntab.append(t)
    nm, nf, ntab = np.array(nm), np.array(nf), np.array(ntab)
    zf = (o_tab - ntab.mean(0)) / (ntab.std(0) + 1e-12)
    return dict(mantel=o_m, mantel_z=(o_m - nm.mean()) / nm.std(), mantel_p=((nm >= o_m).sum() + 1) / (nperm + 1),
                fmax=o_f, fmax_z=(o_f - nf.mean()) / nf.std(), fmax_p=((nf >= o_f).sum() + 1) / (nperm + 1), zf=zf)


def run(name, kinds, reps=5):
    X, Y, script = pair_data(name)
    tok = L.tok_for('voynich' if script == 'voynich' else 'latin')
    allw = [w for p in X for w in p] + [w for p in Y for w in p]
    fr = L.glyph_freq(collections.Counter(allw), tok)
    if script == 'voynich':
        alph = [g for g in S.VOYNICH if fr[g] >= 40]
    elif script == 'hangul':
        alph = [g for g in S.HANGUL if fr[g] >= 40]
    else:
        alph = [g for g, c in fr.most_common() if c >= 40]
    out = {}
    for kind in kinds:
        F = L.Feat(kind, alph, fr)
        F.feats = sorted(set().union(*F.sets))
        rng = np.random.default_rng(1); prng = random.Random(1)
        R = []
        for r in range(reps):
            xs = sample(X, prng); ys = sample(Y, prng)
            CX = slots(xs, tok, alph); CY = slots(ys, tok, alph)
            a = 2.0 / CX.sum(0)
            D = np.log((CY / CY.sum(0) + a) / (CX / CX.sum(0) + a))
            R.append(mantel_stats(D, F, rng, nperm=1000))
        zf = np.mean([r['zf'] for r in R], 0)
        top = sorted(((float(zf[i, s]), F.feats[i], 'IMF'[s]) for i in range(len(F.feats)) for s in range(3)), key=lambda t: -abs(t[0]))[:5]
        res = {k: float(np.mean([r[k] for r in R])) for k in ('mantel', 'mantel_z', 'mantel_p', 'fmax', 'fmax_z', 'fmax_p')}
        res['top'] = top
        out[kind] = res
        print(f"{name:14s} {kind:6s} Mantel {res['mantel']:+.3f} z {res['mantel_z']:+.2f} p {res['mantel_p']:.3f} | FEATMAX z {res['fmax_z']:+.2f} p {res['fmax_p']:.3f} | top {[(f, s, round(z, 1)) for z, f, s in top[:3]]}", flush=True)
    return out


NAMES = [('plant_P_FEAT', ['vhand', 'vimg']), ('plant_P_ARB1', ['vhand', 'vimg']), ('plant_P_ARB2', ['vhand', 'vimg']), ('plant_P_NULL', ['vhand', 'vimg']),
         ('plant_H_FEAT', ['hangul']), ('plant_H_ARB1', ['hangul']), ('plant_H_ARB2', ['hangul']), ('plant_H_NULL', ['hangul']),
         ('plant_G_FEAT', ['phon']), ('plant_G_ARB1', ['phon']), ('plant_G_ARB2', ['phon']), ('N_BavBav', ['phon']),
         ('L_BavAlem', ['phon']), ('L_BavBav', ['phon']), ('L_LatIta', ['phon']), ('L_CzReform', ['phon']), ('K_Sub', ['phon']), ('K_Swap3', ['phon']),
         ('V_AB', ['vhand', 'vimg']), ('V_AherbBherb', ['vhand', 'vimg']), ('V_B2B3', ['vhand', 'vimg']), ('N_AA', ['vhand', 'vimg']), ('N_BB', ['vhand', 'vimg'])]

if __name__ == '__main__':
    sel = sys.argv[1:]
    res = {}
    for n, k in NAMES:
        if sel and n not in sel:
            continue
        res[n] = run(n, k)
    tag = sel[0] if sel else 'all'
    json.dump(res, open(os.path.join(L.CK, f'shift_{tag}.json'), 'w'), default=float)
