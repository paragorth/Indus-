"""v18 cycle 3: (a) are the detected dips real pen events?  (b) pen-load coherence.

(a) INK BUDGET. A quill holds a roughly fixed load, so real dip-to-dip intervals should be
    more regular in ink spent than random marks. For every primary setting, coefficient
    of variation (CV) of the intervals measured in words, glyphs, ink pixels and pen travel
    (sum of word widths), against pseudo-dips at random positions (same number per page,
    same minimum spacing, 50 draws). Voynich and Latin.
(b) PEN-LOAD COHERENCE. If a generator / table / grille is consulted once per pen-load,
    words written on the same load should resemble each other more than words the same
    distance apart that straddle a dip. Same-line word pairs at distance 2-4: similarity
    (same first glyph, same last glyph, edit similarity, same length) within one load
    minus across a dip; mean over primary settings; null = 24 page-swap surrogates.
    Planted check: Voynich-like text whose generator draws a fresh 'theme' (a first-glyph
    preference) at every canonical dip.
Checkpoint: data/results/v18/c3.json
"""
import sys, os, json, time, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v18_dips import *
from v18_lib import glyphs
from v18_cycle1 import CANON
from v18_cycle2 import latin
D = os.path.join(HERE, '..', 'data', 'derived')
RES = os.path.join(HERE, '..', 'data', 'results', 'v18')


def voynich():
    pages = json.load(open(os.path.join(D, 'v18_words.json')))
    ZL = json.load(open(os.path.join(D, 'ZL3b_lines.json')))
    return Corpus(pages, glyphs, [r['words'] for r in ZL if r['ltype'] == 'P'])


def budget(C, prim, rng):
    g = np.array([len(C.glyphs(w)) for w in C.word], float)
    area = np.array([(w['m']['areag'] * max(1, len(C.glyphs(w['word'])))) if w['m'] else np.nan for w in C.W])
    area[np.isnan(area)] = np.nanmedian(area)
    width = np.array([w['x1'] - w['x0'] for w in C.W], float)
    units = {'words': np.ones(C.N), 'glyphs': g, 'ink': area, 'travel': width}
    cum = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in units.items()}

    def cvs(dmask):
        out = {k: [] for k in units}
        for p in range(C.npages):
            idx = np.where(dmask & (C.page == p))[0]
            if len(idx) < 3:
                continue
            for k in units:
                out[k] += list(np.diff(cum[k][idx]))
        return {k: float(np.std(v) / np.mean(v)) if len(v) > 5 else np.nan for k, v in out.items()}
    real = collections.defaultdict(list); null = collections.defaultdict(list)
    for st, idx in prim:
        d = np.zeros(C.N, bool); d[idx] = True
        r = cvs(d)
        for k in units:
            real[k].append(r[k])
        # random pseudo-dips, same count per page, same min spacing
        sp = st[5]
        rr = collections.defaultdict(list)
        for it in range(5):
            dm = np.zeros(C.N, bool)
            for p in range(C.npages):
                pidx = np.where(C.page == p)[0]; n = int(d[pidx].sum())
                if n == 0:
                    continue
                pos = []
                tries = 0
                while len(pos) < n and tries < 20 * n:
                    t = int(rng.integers(0, len(pidx))); tries += 1
                    if all(abs(t - u) >= sp for u in pos):
                        pos.append(t)
                dm[pidx[pos]] = True
            r2 = cvs(dm)
            for k in units:
                rr[k].append(r2[k])
        for k in units:
            null[k].append(float(np.nanmean(rr[k])))
    res = {}
    for k in units:
        a = np.array(real[k]); b = np.array(null[k])
        res[k] = {'cv_real': float(np.nanmean(a)), 'cv_random': float(np.nanmean(b)),
                  'frac_settings_real_below_random': float(np.nanmean(a < b))}
    return res


def pairs(C, dmax=4):
    I, J, Dd = [], [], []
    for t in range(C.N):
        for d in range(2, dmax + 1):
            u = t + d
            if u < C.N and C.page[u] == C.page[t] and C.li[u] == C.li[t]:
                I.append(t); J.append(u); Dd.append(d)
    return np.array(I), np.array(J), np.array(Dd)


def edsim(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def simmats(C, I, J):
    gl = [C.glyphs(w) for w in C.word]
    f = np.array([g[0] if g else '' for g in gl]); l = np.array([g[-1] if g else '' for g in gl])
    n = np.array([len(g) for g in gl])
    return {'first': (f[I] == f[J]).astype(float), 'last': (l[I] == l[J]).astype(float),
            'edit': np.array([edsim(C.word[i], C.word[j]) for i, j in zip(I, J)]),
            'len': (n[I] == n[J]).astype(float)}


def coherence(C, prim, maps, I, J, Dd, S):
    """within-load minus straddling similarity, distance-stratified, mean over settings."""
    def stat(dmask):
        cum = np.concatenate([[0], np.cumsum(dmask)])
        strad = (cum[J + 1] - cum[I + 1]) > 0     # a dip at some word in (I, J]
        out = {}
        for k, s in S.items():
            v = []
            for d in (2, 3, 4):
                m = Dd == d
                a = s[m & ~strad]; b = s[m & strad]
                if len(a) > 20 and len(b) > 20:
                    v.append((a.mean() - b.mean(), len(b)))
            out[k] = sum(x * w for x, w in v) / max(1, sum(w for _, w in v))
        return out
    nsur = len(maps)
    acc = [collections.defaultdict(list) for _ in range(nsur + 1)]
    for st, idx in prim:
        d = np.zeros(C.N, bool); d[idx] = True
        for j, dm in enumerate([d] + [f(d) for f in maps]):
            for k, v in stat(dm).items():
                acc[j][k].append(v)
    res = {}
    for k in S:
        real = float(np.mean(acc[0][k])); sur = [float(np.mean(acc[j][k])) for j in range(1, nsur + 1)]
        res[k] = {'real': real, 'sur_mean': float(np.mean(sur)), 'sur_sd': float(np.std(sur)),
                  'z': (real - np.mean(sur)) / (np.std(sur) + 1e-12),
                  'p_one_sided': (1 + sum(s >= real for s in sur)) / (1 + nsur)}
    return res


class ThemeGen(Generator):
    """At each reset a fresh 'theme' (a preferred first glyph, weight kappa) is drawn."""
    def text_theme(self, reset_mask, kappa=1.5):
        out = []; prev = None; theme = None
        for t in range(self.C.N):
            if t > 0 and self.C.page[t] != self.C.page[t - 1]:
                prev = None
            if reset_mask[t] or theme is None:
                theme = self.firsts[self.rng.choice(len(self.firsts), p=self.U / self.U.sum())]
            if prev is not None and self.rng.random() < self.rho:
                w = prev
            else:
                a = self.g(prev)[-1] if prev and self.g(prev) else None
                wt = self.U * np.exp(self.beta * np.array([self.C.jpmi.get((a, b), -1.0) for b in self.firsts]))
                wt = wt * np.where(np.array(self.firsts) == theme, np.exp(kappa), 1.0)
                wt /= wt.sum()
                b = self.firsts[self.rng.choice(len(self.firsts), p=wt)]
                w = self.vocab[self.byf[b][self.rng.choice(len(self.byf[b]), p=self.within[b])]]
            out.append(w); prev = w
        return out


def main():
    out = {}
    fn = os.path.join(RES, 'c3.json')
    rng = np.random.default_rng(11)
    for tag, maker in (('voynich', voynich), ('latin', lambda: latin()[0])):
        t0 = time.time()
        C = maker()
        ds = all_dips(C)
        prim = [x for x in ds if x[0][3] in CLEAN and x[0][1]]
        out[tag] = {'budget': budget(C, prim, rng)}
        print(tag, 'budget', out[tag]['budget'], round(time.time() - t0), flush=True)
        maps = [page_swap_map(C, k) for k in range(1, C.npages)]
        I, J, Dd = pairs(C)
        S = simmats(C, I, J)
        out[tag]['coherence'] = coherence(C, prim, maps, I, J, Dd, S)
        print(tag, 'coherence', out[tag]['coherence'], flush=True)
        json.dump(out, open(fn, 'w'), indent=1, default=str)
        if tag == 'voynich':
            can = [i for st, i in ds if st == CANON][0]
            for kappa in (1.5, 0.75, 0.0):
                mask = np.zeros(C.N, bool)
                for t in can:
                    u = t + (rng.choice([-1, 1]) if rng.random() < 0.2 else 0)
                    if 0 <= u < C.N:
                        mask[u] = True
                G = ThemeGen(C, C.lines_of_text(), seed=500)
                words = G.text_theme(mask, kappa)
                Cs = voynich(); Cs.word = words; Cs.fit_model(Cs.lines_of_text()); Cs.features()
                Ss = simmats(Cs, I, J)
                out[tag]['planted_theme_%s' % kappa] = coherence(Cs, prim, maps, I, J, Dd, Ss)
                print('planted theme', kappa, out[tag]['planted_theme_%s' % kappa], flush=True)
                json.dump(out, open(fn, 'w'), indent=1, default=str)
    print('done')


if __name__ == '__main__':
    main()
