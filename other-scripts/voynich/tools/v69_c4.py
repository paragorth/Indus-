"""v69 cycle 4: does the same-word likeness of c3 survive line distance, residualisation and plants?
Same pair design as c3 (equal glyph count 3-7, same page, >= 2 lines apart, Hamming bins), but
 - features: the 8 c1 residuals (glyph content, length, position, line index removed linearly)
 - null: token identities permuted within (page, 4-line band, glyph count, first glyph, last glyph)
 - gap01 reported per line-distance bin (2-3, 4-7, 8-15, 16+)
 - planted NEGATIVE: features rebuilt as glyph-additive content + smooth per-page line drift +
   Gaussian noise (no word identity) -> gap01 must equal its null
 - planted POSITIVE: real residuals + a random per-type signature of 0.25 SD -> must be found
Corpora: Voynich (all, Q20), Latin control. Out: data/v69_ckpt/c4.json"""
import sys, os, json, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v69_lib as X
import v69_c1 as C1

RNG = np.random.default_rng(6940)
F8 = ['wpg', 'hgt', 'upright', 'ncomp', 'swid', 'gapcv', 'irr', 'dark']
DB = [(2, 3), (4, 7), (8, 15), (16, 999)]


def pairs(D, M, words, gls, freq, maxd=2):
    """sums[(band, dbin, d)] = [sum, n] over same-page pairs"""
    S = collections.defaultdict(lambda: [0.0, 0])
    page_idx = collections.defaultdict(list)
    for i in range(D.N):
        if 3 <= len(gls[i]) <= 7:
            page_idx[(D.page[i], len(gls[i]))].append(i)
    for (p, g), idx in page_idx.items():
        idx = np.array(idx)
        if len(idx) < 2:
            continue
        G = np.array([gls[i] for i in idx]); Mm = M[idx]; L = D.li[idx]
        Dm = np.sqrt(((Mm[:, None, :] - Mm[None, :, :]) ** 2).sum(-1))
        H = (G[:, None, :] != G[None, :, :]).sum(-1)
        LD = np.abs(L[:, None] - L[None, :])
        a, b = np.triu_indices(len(idx), 1)
        keep = (LD[a, b] >= 2) & (H[a, b] <= maxd)
        a, b = a[keep], b[keep]
        for x, y in zip(a, b):
            f = freq.get(words[idx[x]], 0) + freq.get(words[idx[y]], 0)
            fb = 0 if f < 20 else (1 if f < 200 else 2)
            ld = LD[x, y]
            db = next(j for j, (lo, hi) in enumerate(DB) if lo <= ld <= hi)
            h = int(H[x, y]); dv = Dm[x, y]
            for key in [('all', 'all', h), (fb, 'all', h), ('all', db, h)]:
                S[key][0] += dv; S[key][1] += 1
    return S


def gaps(S):
    out = {}
    for band in ['all', 0, 1, 2]:
        for db in ['all'] + list(range(len(DB))):
            if band != 'all' and db != 'all':
                continue
            m = [S[(band, db, d)][0] / S[(band, db, d)][1] if S[(band, db, d)][1] else np.nan for d in range(3)]
            out[f'{band}|{db}'] = (m[1] - m[0], (m[1] - m[0]) - (m[2] - m[1]), S[(band, db, 0)][1])
    return out


def perm_ident(D, gls, rng):
    keys = collections.defaultdict(list)
    for i in range(D.N):
        g = gls[i]
        keys[(D.page[i], D.li[i] // 4, len(g), g[0], g[-1])].append(i)
    w = list(D.word); gg = list(gls)
    for idx in keys.values():
        if len(idx) > 1:
            pr = rng.permutation(idx)
            for a, b in zip(idx, pr):
                w[a] = D.word[b]; gg[a] = gls[b]
    return w, gg


def test(D, M, freq, tag, nperm=25):
    gls = [tuple(D.gl(w)) for w in D.word]
    obs = gaps(pairs(D, M, D.word, gls, freq))
    nul = collections.defaultdict(list)
    for t in range(nperm):
        w, gg = perm_ident(D, gls, RNG)
        for k, v in gaps(pairs(D, M, w, gg, freq)).items():
            nul[k].append(v[0])
    out = {}
    for k, (g01, J, n0) in obs.items():
        a = np.array(nul[k])
        out[k] = {'gap01': round(float(g01), 4), 'J': round(float(J), 4), 'n_same': n0,
                  'null': round(float(np.nanmean(a)), 4), 'z': round(float((g01 - np.nanmean(a)) / (np.nanstd(a) + 1e-9)), 2)}
    print(tag, {k: (v['gap01'], v['null'], v['z'], v['n_same']) for k, v in out.items()}, flush=True)
    return out


def resid_matrix(D):
    M = np.vstack([D.res[f] for f in F8]).T
    return np.clip(M, -4, 4)


def planted_negative(D):
    """glyph-additive content + smooth line drift per page + noise, no word identity"""
    gls = [D.gl(w) for w in D.word]
    alph = sorted(set(x for g in gls for x in g)); ai = {a: i for i, a in enumerate(alph)}
    C = np.zeros((D.N, len(alph)))
    for i, g in enumerate(gls):
        for x in g:
            C[i, ai[x]] += 1
    out = np.zeros((D.N, len(F8)))
    for j, f in enumerate(F8):
        y = np.clip(D.raw[f], -4, 4)
        b = np.linalg.lstsq(np.hstack([np.ones((D.N, 1)), C]), y, rcond=None)[0]
        pred = np.hstack([np.ones((D.N, 1)), C]) @ b
        sd = np.std(y - pred)
        drift = np.zeros(D.N)
        for p in np.unique(D.page):
            m = D.page == p
            nl = D.li[m].max() + 1
            walk = np.cumsum(RNG.normal(0, 0.15, nl))
            drift[m] = walk[D.li[m]]
        out[:, j] = pred + drift + RNG.normal(0, sd, D.N)
    return out


def planted_positive(D, M, amp=0.25):
    sig = {w: RNG.normal(0, amp, M.shape[1]) for w in set(D.word)}
    return M + np.array([sig[w] for w in D.word])


if __name__ == '__main__':
    vf, _ = X.voynich_freq(); lf = X.latin_freq()
    res = {}
    V = C1.Data('V'); L = C1.Data('L')
    MV = resid_matrix(V); ML = resid_matrix(L)
    res['V_resid'] = test(V, MV, vf, 'V_resid')
    res['L_resid'] = test(L, ML, lf, 'L_resid')
    res['V_plant_neg'] = test(V, planted_negative(V), vf, 'V_plant_neg', nperm=15)
    res['V_plant_pos'] = test(V, planted_positive(V, MV), vf, 'V_plant_pos', nperm=15)
    res['L_plant_pos'] = test(L, planted_positive(L, ML), lf, 'L_plant_pos', nperm=15)
    Q = C1.Data('V', lambda f: f not in ('f58r', 'f58v'))
    res['VQ20_resid'] = test(Q, resid_matrix(Q), vf, 'VQ20_resid')
    json.dump(res, open(os.path.join(X.CK, 'c4.json'), 'w'), indent=1)
