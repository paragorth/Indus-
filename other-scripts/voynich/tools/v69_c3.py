"""v69 cycle 3: IS THE WORD A MOTOR UNIT? (care turned inside out)
A writer who knows a word writes it as one practised movement: two tokens of the same word look
more alike than two tokens of different words that share all but one glyph, by more than the one
glyph explains. Frequent words should show the bigger jump (practice). A writer composing glyph by
glyph (computing each glyph) has no word-level jump.
Pairs of tokens on the same page, both of glyph count g (3-7), at least 2 lines apart, binned by the
number of glyph positions that differ (Hamming d = 0..g). Distance = Euclidean on the 8 page-z
shape features (wpg, hgt, slant, ncomp, swid, gapcv, irr, dark). Jump J = D(d=1) - D(d=0) minus the
per-glyph slope D(d=2) - D(d=1) (a smooth trend predicts J = 0).
Nulls: (a) token labels permuted within (page, g, first glyph, last glyph): J must vanish;
(b) bootstrap over pages for CIs. Controls: Latin (CREMMA) must show J > 0 and a larger J for
frequent types. Out: data/v69_ckpt/c3.json"""
import sys, os, json, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v69_lib as X

RNG = np.random.default_rng(693)
F8 = ['wpg', 'hgt', 'slant', 'ncomp', 'swid', 'gapcv', 'irr', 'dark']


def prep(which, subset=None):
    R = X.extract(which)
    if subset:
        R = [r for r in R if subset(r['folio'])]
    gl = X.vglyphs if which == 'V' else X.lglyphs
    pages = collections.defaultdict(list)
    for r in R:
        pages[r['folio']].append(r)
    P = []
    for fo, rs in pages.items():
        M = np.array([[r[f] for f in F8] for r in rs], float)
        M = (M - np.median(M, 0)) / (M.std(0) + 1e-9)
        M = np.clip(M, -4, 4)
        P.append({'folio': fo, 'M': M, 'word': [r['word'] for r in rs], 'gl': [tuple(gl(r['word'])) for r in rs],
                  'li': np.array([r['li'] for r in rs])})
    return P


def pair_stats(P, freq, words=None, gmin=3, gmax=7, maxd=3):
    """returns sums[(fbin, d)] = [sum distance, n]; fbin by corpus frequency of the d=0 type"""
    S = collections.defaultdict(lambda: [0.0, 0])
    for pi, p in enumerate(P):
        w = p['word'] if words is None else words[pi]
        gls = p['gl'] if words is None else [tuple(x) for x in words[pi + len(P)]]
        byg = collections.defaultdict(list)
        for i, g in enumerate(gls):
            if gmin <= len(g) <= gmax:
                byg[len(g)].append(i)
        for g, idx in byg.items():
            idx = np.array(idx)
            if len(idx) < 2:
                continue
            G = np.array([gls[i] for i in idx])  # (n, g) of str
            M = p['M'][idx]; L = p['li'][idx]
            D = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))
            H = (G[:, None, :] != G[None, :, :]).sum(-1)
            far = np.abs(L[:, None] - L[None, :]) >= 2
            iu = np.triu_indices(len(idx), 1)
            for a, b in zip(*iu):
                if not far[a, b] or H[a, b] > maxd:
                    continue
                f = freq.get(w[idx[a]], 0) + freq.get(w[idx[b]], 0)
                fb = 0 if f < 20 else (1 if f < 200 else 2)
                k = (fb, int(H[a, b]))
                S[k][0] += D[a, b]; S[k][1] += 1
                S[('all', int(H[a, b]))][0] += D[a, b]; S[('all', int(H[a, b]))][1] += 1
    return S


def summarise(S):
    out = {}
    for fb in ['all', 0, 1, 2]:
        m = {d: (S[(fb, d)][0] / S[(fb, d)][1] if S[(fb, d)][1] else float('nan'), S[(fb, d)][1]) for d in range(4)}
        J = (m[1][0] - m[0][0]) - (m[2][0] - m[1][0])
        out[str(fb)] = {'D': [round(m[d][0], 4) for d in range(4)], 'n': [m[d][1] for d in range(4)], 'J': round(J, 4),
                        'gap01': round(m[1][0] - m[0][0], 4), 'slope12': round(m[2][0] - m[1][0], 4)}
    return out


def permuted_words(P):
    """permute token identities within (g, first glyph, last glyph) on each page"""
    W, G = [], []
    for p in P:
        w = list(p['word']); g = list(p['gl'])
        groups = collections.defaultdict(list)
        for i, x in enumerate(g):
            groups[(len(x), x[0], x[-1])].append(i)
        nw, ng = list(w), list(g)
        for idx in groups.values():
            pr = RNG.permutation(idx)
            for a, b in zip(idx, pr):
                nw[a] = w[b]; ng[a] = g[b]
        W.append(nw); G.append(ng)
    return W + G


def run(which, freq, subset=None, nperm=30, nboot=200, tag=''):
    P = prep(which, subset)
    obs = summarise(pair_stats(P, freq))
    nulJ = collections.defaultdict(list)
    for t in range(nperm):
        s = summarise(pair_stats(P, freq, permuted_words(P)))
        for k in s:
            nulJ[k].append(s[k]['gap01'])
    # bootstrap pages for gap01 and J
    per = [pair_stats([p], freq) for p in P]
    boot = collections.defaultdict(list)
    for b in range(nboot):
        ix = RNG.integers(0, len(P), len(P))
        T = collections.defaultdict(lambda: [0.0, 0])
        for i in ix:
            for k, v in per[i].items():
                T[k][0] += v[0]; T[k][1] += v[1]
        s = summarise(T)
        for k in s:
            boot[k].append((s[k]['gap01'], s[k]['J']))
    for k in obs:
        a = np.array(nulJ[k]); bb = np.array(boot[k])
        obs[k]['gap01_null'] = [round(float(np.nanmean(a)), 4), round(float(np.nanstd(a)), 4)]
        obs[k]['gap01_z_vs_perm'] = round(float((obs[k]['gap01'] - np.nanmean(a)) / (np.nanstd(a) + 1e-9)), 2)
        obs[k]['gap01_ci'] = [round(float(np.nanpercentile(bb[:, 0], q)), 4) for q in (5, 95)]
        obs[k]['J_ci'] = [round(float(np.nanpercentile(bb[:, 1], q)), 4) for q in (5, 95)]
        print(tag, k, obs[k], flush=True)
    return obs


if __name__ == '__main__':
    vf, _ = X.voynich_freq(); lf = X.latin_freq()
    res = {'L': run('L', lf, tag='L'), 'V': run('V', vf, tag='V'),
           'V_Q20': run('V', vf, lambda f: f not in ('f58r', 'f58v'), tag='V_Q20')}
    json.dump(res, open(os.path.join(X.CK, 'c3.json'), 'w'), indent=1)
