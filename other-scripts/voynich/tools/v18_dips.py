"""v18 THE PEN REMEMBERS: pen-dip detection on per-word ink series and the 'reset' tests.

corpus(words_json) -> per-page word sequences (writing order: line by line, left to right)
residual darkness -> dip detection under thousands of settings -> text features at the
boundary entering each dip word, standardised within position-in-line strata ->
analytic z against a page x stratum resampling null; family-wise calibration by
page-swap surrogates (dips of page P laid on the text of page P+k by line/word coordinates).
"""
import json, math, itertools, collections
import numpy as np
from scipy import ndimage as ndi

MEAS = ['med', 'top', 'p90', '-gray', 'areag', 'rb', '-rb']
DET = ['jump1', 'jump2', 'saw', 'min5']
QS = [0.03, 0.05, 0.08, 0.12, 0.16, 0.20]
SPACE = [1, 3, 5]
FEATS = ['junc', 'bigr', 'len', 'dlen', 'freq', 'rep', 'init']


def edit1(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(b)):
        if b[:i] + b[i + 1:] == a:
            return True
    return False


class Corpus:
    def __init__(self, pages, glyphs, model_lines):
        """pages: list of {'folio', 'words': [...]} from the extractors.
        glyphs: word -> glyph list. model_lines: list of word lists (whole corpus) for
        the language model (glyph junction PMI, word bigram PMI, line-initial stats)."""
        self.glyphs = glyphs
        W = []
        for pi, p in enumerate(pages):
            ws = sorted(p['words'], key=lambda w: (w['li'], w['k']))
            for w in ws:
                w = dict(w); w['page'] = pi; W.append(w)
        self.W = W
        self.N = len(W)
        self.page = np.array([w['page'] for w in W])
        self.npages = len(pages)
        self.li = np.array([w['li'] for w in W]); self.k = np.array([w['k'] for w in W])
        self.nw = np.array([w['nw'] for w in W])
        self.word = [w['word'] for w in W]
        self.fit_model(model_lines)
        self.features()

    # ---------------- language model statistics
    def fit_model(self, lines):
        g = self.glyphs
        uni = collections.Counter(); big = collections.Counter(); first = collections.Counter()
        last = collections.Counter(); pair = collections.Counter(); init_first = collections.Counter()
        for ws in lines:
            for i, w in enumerate(ws):
                uni[w] += 1
                gl = g(w)
                if not gl:
                    continue
                first[gl[0]] += 1; last[gl[-1]] += 1
                if i == 0:
                    init_first[gl[0]] += 1
                if i > 0:
                    big[(ws[i - 1], w)] += 1
                    gp = g(ws[i - 1])
                    if gp:
                        pair[(gp[-1], gl[0])] += 1
        self.uni = uni; self.big = big; self.Nu = sum(uni.values())
        nf = sum(first.values()); nl = sum(last.values()); npair = sum(pair.values())
        self.jpmi = {}
        for (a, b), c in pair.items():
            self.jpmi[(a, b)] = math.log((c + 0.5) * npair / ((last[a] + 1) * (first[b] + 1)) )
        self.first = first; self.nf = nf
        ni = sum(init_first.values())
        self.initlr = {x: math.log((init_first[x] + 0.5) / (ni + 1)) - math.log((first[x] + 0.5) / (nf + 1)) for x in first}

    def features(self):
        g = self.glyphs; N = self.N
        F = {f: np.full(N, np.nan) for f in FEATS}
        for t in range(N):
            w = self.word[t]; gl = g(w)
            F['len'][t] = len(gl)
            F['freq'][t] = math.log(self.uni.get(w, 0) + 0.5)
            F['init'][t] = self.initlr.get(gl[0], 0.0) if gl else 0.0
            if t == 0 or self.page[t - 1] != self.page[t]:
                continue
            p = self.word[t - 1]; gp = g(p)
            F['dlen'][t] = len(gl) - len(gp)
            if gp and gl:
                F['junc'][t] = self.jpmi.get((gp[-1], gl[0]), math.log(0.5 / max(1, self.nf)))
            cw1 = self.uni.get(p, 0); pw2 = (self.uni.get(w, 0) + 0.5) / self.Nu
            F['bigr'][t] = math.log((self.big.get((p, w), 0) + 2.0 * pw2) / ((cw1 + 2.0) * pw2))
            rep = edit1(p, w) or (t >= 2 and self.page[t - 2] == self.page[t] and edit1(self.word[t - 2], w))
            F['rep'][t] = float(rep)
        # strata: line start, second word, line end, interior
        st = np.where(self.k == 0, 0, np.where(self.k == 1, 1, np.where(self.k == self.nw - 1, 2, 3)))
        self.stratum = st
        Z = {}
        for f, v in F.items():
            z = v.copy()
            for s in range(4):
                m = (st == s) & ~np.isnan(v)
                if m.sum() > 2:
                    z[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-9)
            Z[f] = z
        self.F = Z
        # resampling groups: page x stratum
        self.group = self.page * 4 + st
        G = self.npages * 4
        self.gstats = {}
        for f, z in Z.items():
            ok = ~np.isnan(z)
            mu = np.zeros(G); var = np.zeros(G); n = np.zeros(G)
            for gi in range(G):
                m = ok & (self.group == gi)
                if m.sum():
                    mu[gi] = z[m].mean(); var[gi] = z[m].var(); n[gi] = m.sum()
            self.gstats[f] = (mu, var, n, ok)

    def set_text(self, words):
        """Replace the word sequence (planted / synthetic text), same layout."""
        self.word = list(words)
        self.features()

    # ---------------- darkness and dips
    def residual(self, meas, content=True):
        sign = -1.0 if meas.startswith('-') else 1.0
        key = meas.lstrip('-')
        y = np.array([sign * w['m'][key] if w['m'] else np.nan for w in self.W])
        r = np.full(self.N, np.nan)
        xs = np.array([w['x0'] + w['x1'] for w in self.W]) / 4000.0
        ys = np.array([w['y'] for w in self.W]) / 3000.0
        for p in range(self.npages):
            m = (self.page == p) & ~np.isnan(y)
            if m.sum() < 10:
                continue
            X = np.vstack([np.ones(m.sum()), xs[m], ys[m], xs[m] ** 2, ys[m] ** 2, xs[m] * ys[m]]).T
            b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
            r[m] = y[m] - X @ b
        if content:
            C = self.content_matrix()
            m = ~np.isnan(r)
            X = np.hstack([np.ones((m.sum(), 1)), C[m]])
            b = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ r[m])
            r[m] = r[m] - X @ b
        for p in range(self.npages):
            m = (self.page == p) & ~np.isnan(r)
            if m.sum() > 2:
                r[m] = (r[m] - r[m].mean()) / (r[m].std() + 1e-9)
        return r

    def content_matrix(self):
        if getattr(self, '_C', None) is not None:
            return self._C
        cnt = collections.Counter()
        gls = [self.glyphs(w['word']) for w in self.W]
        for gl in gls:
            cnt.update(gl)
        top = [g for g, _ in cnt.most_common(20)]
        C = np.zeros((self.N, len(top) + 3))
        for i, gl in enumerate(gls):
            c = collections.Counter(gl)
            for j, g in enumerate(top):
                C[i, j] = c[g]
            C[i, -3] = len(gl)
            w = self.W[i]
            C[i, -2] = math.log(max(1, w['x1'] - w['x0']) / max(1.0, w['wexp']))
            C[i, -1] = float(w['k'] == 0)
        C = (C - C.mean(0)) / (C.std(0) + 1e-9)
        self._C = C
        return C

    def detect(self, r, det, q, space, smooth, lsmode):
        N = self.N
        s = r.copy()
        dips = np.zeros(N, bool)
        for p in range(self.npages):
            idx = np.where(self.page == p)[0]
            x = s[idx].copy()
            nanm = np.isnan(x)
            if nanm.all():
                continue
            x[nanm] = np.nanmean(x)
            if smooth:
                x = ndi.median_filter(x, 3, mode='nearest')
            n = len(x)
            d = np.full(n, -np.inf)
            if det == 'jump1':
                d[1:] = x[1:] - x[:-1]
            elif det == 'jump2':
                d[2:] = x[2:] - 0.5 * (x[1:-1] + x[:-2])
            elif det == 'saw':
                for t in range(3, n - 1):
                    d[t] = 0.5 * (x[t] + x[t + 1]) - x[t - 3:t].mean()
            elif det == 'min5':
                for t in range(1, n):
                    d[t] = x[t] - x[max(0, t - 5):t].min()
            d[nanm] = -np.inf
            if lsmode == 'noLS':
                d[self.k[idx] == 0] = -np.inf
            fin = np.isfinite(d)
            if fin.sum() < 5:
                continue
            thr = np.quantile(d[fin], 1 - q)
            cand = np.where(d >= thr)[0]
            cand = cand[np.argsort(-d[cand])]
            taken = np.zeros(n, bool); block = np.zeros(n, bool)
            for t in cand:
                if block[t]:
                    continue
                taken[t] = True
                block[max(0, t - space + 1):t + space] = True
            dips[idx[taken]] = True
        return dips

    def zscores(self, dips):
        out = {}
        for f in FEATS:
            mu, var, n, ok = self.gstats[f]
            d = dips & ok
            N = d.sum()
            if N < 10:
                out[f] = 0.0; continue
            z = self.F[f][d]
            cnt = np.bincount(self.group[d], minlength=len(mu))
            em = (cnt * mu).sum() / N
            fpc = np.where(n > 1, (n - np.minimum(cnt, n)) / np.maximum(n - 1, 1), 0)
            ev = (cnt * var * fpc).sum() / N ** 2
            out[f] = float((z.mean() - em) / math.sqrt(max(ev, 1e-12)))
        return out


def settings():
    for meas, cont, sm, det, q, sp, ls in itertools.product(MEAS, (False, True), (False, True), DET, QS, SPACE, ('all', 'noLS')):
        yield (meas, cont, sm, det, q, sp, ls)


def search(C, resid_cache=None, dipmap=None):
    """Run all settings. dipmap(dips)->dips lets surrogates relocate dips."""
    rows = []
    cache = resid_cache if resid_cache is not None else {}
    for st in settings():
        meas, cont, sm, det, q, sp, ls = st
        key = (meas, cont)
        if key not in cache:
            cache[key] = C.residual(meas, cont)
        dips = C.detect(cache[key], det, q, sp, sm, ls)
        if dipmap is not None:
            dips = dipmap(dips)
        z = C.zscores(dips)
        rows.append((st, int(dips.sum()), z))
    return rows


def page_swap_map(C, shift):
    """Lay dips of page p onto page (p+shift) mod P by (line index, word index)."""
    coord = {}
    for t in range(C.N):
        coord[(C.page[t], C.li[t], C.k[t])] = t

    def f(dips):
        out = np.zeros(C.N, bool)
        for t in np.where(dips)[0]:
            key = ((C.page[t] + shift) % C.npages, C.li[t], C.k[t])
            u = coord.get(key)
            if u is not None:
                out[u] = True
        return out
    return f


def maxabs(rows):
    return max(abs(z) for _, _, zz in rows for z in zz.values())


def spacing_stats(C, dips):
    gaps = []
    lines_per = []
    for p in range(C.npages):
        idx = np.where(dips & (C.page == p))[0]
        gaps += list(np.diff(idx))
    gaps = np.array(gaps)
    return {'n': int(dips.sum()), 'median_gap_words': float(np.median(gaps)) if len(gaps) else None,
            'mean_gap_words': float(gaps.mean()) if len(gaps) else None,
            'p25': float(np.percentile(gaps, 25)) if len(gaps) else None,
            'p75': float(np.percentile(gaps, 75)) if len(gaps) else None,
            'line_start_share': float((dips & (C.k == 0)).sum() / max(1, dips.sum())),
            'base_line_start_share': float((C.k == 0).mean())}
