"""v57 THE INK CLOCK: shared loader. Uses the per-word ink measures of v18
(data/derived/v18_words.json: 25 Voynich text pages; v18_latin_words.json: 28 Latin
manuscript pages, CREMMA-Medieval-Lat). Darkness is residualised per page against a
quadratic plane in (x, y) and, corpus-wide, against glyph content (ridge), then z-scored
per page. Words are kept in reading order with (page, line, k)."""
import json, os, math, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.join(HERE, '..', 'data', 'derived')
CK = os.path.join(HERE, '..', 'data', 'v57_ckpt')
os.makedirs(CK, exist_ok=True)
EVA_MULTI = ['cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']


def vglyphs(w):
    out, i = [], 0
    while i < len(w):
        for m in EVA_MULTI:
            if w.startswith(m, i):
                out.append(m); i += len(m); break
        else:
            out.append(w[i]); i += 1
    return out


def lglyphs(w):
    return [c for c in w if not c.isspace()]


class Corpus:
    def __init__(self, which, meas='top'):
        fn = 'v18_words.json' if which == 'V' else 'v18_latin_words.json'
        self.which = which
        self.gl = vglyphs if which == 'V' else lglyphs
        pages = json.load(open(os.path.join(DER, fn)))
        W = []
        for pi, p in enumerate(pages):
            for w in p['words']:
                if w['m'] is None:
                    continue
                w = dict(w); w['pi'] = pi; W.append(w)
        self.W = W
        self.pages = [p['folio'] for p in pages]
        self.N = len(W)
        self.page = np.array([w['pi'] for w in W])
        self.li = np.array([w['li'] for w in W])
        self.k = np.array([w['k'] for w in W])
        self.nw = np.array([w['nw'] for w in W])
        self.x = np.array([(w['x0'] + w['x1']) / 2 for w in W], float)
        self.y = np.array([w['y'] for w in W], float)
        self.word = [w['word'] for w in W]
        self.r = self.residual(meas)
        # lines: list of index arrays in reading order
        self.lines = []
        cur, key = [], None
        for i, w in enumerate(W):
            kk = (w['pi'], w['li'])
            if kk != key and cur:
                self.lines.append(np.array(cur)); cur = []
            key = kk; cur.append(i)
        if cur:
            self.lines.append(np.array(cur))
        self.line_page = np.array([self.page[l[0]] for l in self.lines])
        self.line_li = np.array([self.li[l[0]] for l in self.lines])

    def content(self):
        cnt = collections.Counter()
        gls = [self.gl(w) for w in self.word]
        for g in gls:
            cnt.update(g)
        top = [g for g, _ in cnt.most_common(20)]
        C = np.zeros((self.N, len(top) + 3))
        for i, g in enumerate(gls):
            c = collections.Counter(g)
            for j, t in enumerate(top):
                C[i, j] = c[t]
            C[i, -3] = len(g)
            w = self.W[i]
            C[i, -2] = math.log(max(1, w['x1'] - w['x0']) / max(1.0, w['wexp']))
            C[i, -1] = float(w['k'] == 0)
        return (C - C.mean(0)) / (C.std(0) + 1e-9)

    def residual(self, meas):
        y = np.array([w['m'][meas] for w in self.W], float)
        r = np.full(self.N, np.nan)
        xs, ys = self.x / 2000.0, self.y / 3000.0
        for p in np.unique(self.page):
            m = self.page == p
            X = np.vstack([np.ones(m.sum()), xs[m], ys[m], xs[m] ** 2, ys[m] ** 2, xs[m] * ys[m]]).T
            b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
            r[m] = y[m] - X @ b
        C = self.content()
        X = np.hstack([np.ones((self.N, 1)), C])
        b = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ r)
        r = r - X @ b
        for p in np.unique(self.page):
            m = self.page == p
            r[m] = (r[m] - r[m].mean()) / (r[m].std() + 1e-9)
        return r
