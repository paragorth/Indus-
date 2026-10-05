"""v69 cycle 1: what drives per-word writing care, beyond length, glyph content and position?
Care features (v69_lib) are page-z-scored, then residualised (ridge) on glyph content (top-25 glyph
counts), glyph count, position in line (k, from-end, first, last), line index, paragraph start and
alignment cost. Pre-registered care index = wpg + hgt + upright + ncomp - gapcv - irr (residual z).
Tests, Voynich vs Latin control (CREMMA):
 T1 rarity: corr(care, -log corpus frequency)
 T2 page keyword: corr(care, log page over-representation), partial on rarity
 T3 first mention on page vs later repeats of the same word (paired within type)
 T4 corpus hapax vs rest
Nulls: care permuted within (glyph count, first glyph, last glyph) strata (T1, T2, T4); for T3 care
shuffled within (page, 5-line band, glyph count) strata. 2000 permutations.
Out: data/v69_ckpt/c1.json"""
import sys, os, json, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v69_lib as X

RNG = np.random.default_rng(69)
NP = int(os.environ.get('NPERM', 2000))


class Data:
    def __init__(self, which, subset=None):
        R = X.extract(which)
        if subset:
            R = [r for r in R if subset(r['folio'])]
        self.R = R; self.which = which
        self.gl = X.vglyphs if which == 'V' else X.lglyphs
        self.N = len(R)
        self.word = [r['word'] for r in R]
        self.page = np.array([hash(r['folio']) % 100003 for r in R])
        self.folio = [r['folio'] for r in R]
        self.li = np.array([r['li'] for r in R]); self.k = np.array([r['k'] for r in R])
        self.g = np.array([r['g'] for r in R])
        F = {f: np.array([r[f] for r in R], float) for f in X.FEATS}
        # orientation of slant: upright = less of the page's usual slant
        for p in np.unique(self.page):
            m = self.page == p
            for f in X.FEATS:
                v = F[f][m]
                F[f][m] = (v - np.median(v)) / (np.std(v) + 1e-9)
        sgn = np.sign(np.median([r['slant'] for r in R]))
        F['upright'] = -sgn * F['slant']
        self.raw = F
        C = self.nuis()
        Xm = np.hstack([np.ones((self.N, 1)), C])
        self.res = {}
        for f in list(X.FEATS) + ['upright']:
            y = np.clip(F[f], -4, 4)
            b = np.linalg.solve(Xm.T @ Xm + 3 * np.eye(Xm.shape[1]), Xm.T @ y)
            r = y - Xm @ b
            self.res[f] = (r - r.mean()) / r.std()
        self.care = (self.res['wpg'] + self.res['hgt'] + self.res['upright'] + self.res['ncomp']
                     - self.res['gapcv'] - self.res['irr'])
        self.care = (self.care - self.care.mean()) / self.care.std()
        self.res['care'] = self.care
        gls = [self.gl(w) for w in self.word]
        self.first = [g[0] for g in gls]; self.last = [g[-1] for g in gls]

    def nuis(self):
        gls = [self.gl(w) for w in self.word]
        cnt = collections.Counter(x for g in gls for x in g)
        top = [g for g, _ in cnt.most_common(25)]
        cols = []
        for i, g in enumerate(gls):
            c = collections.Counter(g); r = self.R[i]
            row = [c[t] for t in top] + [len(g), len(g) ** 2, r['k'], r['nw'] - 1 - r['k'], float(r['k'] == 0),
                   float(r['k'] == r['nw'] - 1), r['li'], float(bool(r['para_start'])), r['lcost'],
                   float(g[0] in top[:6]), float(g[-1] in top[:6])]
            cols.append(row)
        C = np.array(cols, float)
        return (C - C.mean(0)) / (C.std(0) + 1e-9)

    def strata(self, kind):
        if kind == 'content':
            keys = [(g, f, l) for g, f, l in zip(self.g, self.first, self.last)]
        else:
            keys = [(p, li // 5, min(g, 8)) for p, li, g in zip(self.page, self.li, self.g)]
        d = collections.defaultdict(list)
        for i, kk in enumerate(keys):
            d[kk].append(i)
        return [np.array(v) for v in d.values() if len(v) > 1]

    def perm(self, v, groups):
        out = v.copy()
        for gidx in groups:
            out[gidx] = v[RNG.permutation(gidx)]
        return out


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def partial(y, x, z):
    Z = np.vstack([np.ones_like(z), z]).T
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return corr(ry, rx)


_FM = {}


def first_mention(D, care):
    if id(D) not in _FM:
        byp = collections.defaultdict(list)
        for i in np.lexsort((D.k, D.li, D.page)):
            byp[(D.page[i], D.word[i])].append(i)
        _FM[id(D)] = [(v[0], np.array(v[1:])) for v in byp.values() if len(v) >= 2]
    G = _FM[id(D)]
    diffs = [care[a] - care[b].mean() for a, b in G]
    return float(np.mean(diffs)) if diffs else 0.0, len(diffs)


def run(D, freq, tag):
    tot = sum(freq.values())
    lf = np.array([math.log((freq.get(w, 0) + 1) / tot) for w in D.word])
    rar = -lf
    pc = collections.Counter(zip(D.page, D.word)); pn = collections.Counter(D.page)
    kw = np.array([math.log((pc[(p, w)] + 0.5) / (pn[p] * (freq.get(w, 0) + 1) / tot + 0.5)) for p, w in zip(D.page, D.word)])
    hap = np.array([freq.get(w, 0) <= 1 for w in D.word], float)
    Sc = D.strata('content'); Sp = D.strata('place')
    out = {'N': D.N}
    for f in ['care', 'wpg', 'hgt', 'upright', 'ncomp', 'gapcv', 'irr', 'swid', 'dark']:
        c = D.res[f]
        obs = {'rar': corr(c, rar), 'kw': partial(c, kw, rar), 'hap': float(c[hap == 1].mean() - c[hap == 0].mean())}
        fm, nfm = first_mention(D, c)
        obs['first'] = fm
        nul = collections.defaultdict(list)
        nperm = NP if f == 'care' else NP // 4
        for t in range(nperm):
            cp = D.perm(c, Sc)
            nul['rar'].append(corr(cp, rar)); nul['kw'].append(partial(cp, kw, rar))
            nul['hap'].append(float(cp[hap == 1].mean() - cp[hap == 0].mean()))
            nul['first'].append(first_mention(D, D.perm(c, Sp))[0])
        row = {}
        for kk, v in obs.items():
            a = np.array(nul[kk])
            row[kk] = {'obs': round(v, 4), 'z': round(float((v - a.mean()) / (a.std() + 1e-12)), 2),
                       'p_hi': round(float((a >= v).mean()), 4)}
        row['n_first_types'] = nfm
        out[f] = row
        print(tag, f, {k: (v['obs'], v['z']) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    return out


if __name__ == '__main__':
    vf, _ = X.voynich_freq(); lf = X.latin_freq()
    res = {}
    res['V_all'] = run(Data('V'), vf, 'V_all')
    res['V_Q20'] = run(Data('V', lambda f: f != 'f58r' and f != 'f58v'), vf, 'V_Q20')
    res['L_all'] = run(Data('L'), lf, 'L_all')
    json.dump(res, open(os.path.join(X.CK, 'c1.json'), 'w'), indent=1)
