"""v30 dialect ladder: cheapest context-sensitive rewrite X -> Y, by stochastic search.

A rule is (lhs, rhs, lc, rc): lhs a 1-3 symbol string of the source, rhs a 0-3 symbol
string, lc/rc an optional left/right context symbol ('#' = word edge). Rules apply in one
simultaneous left-to-right pass (a deterministic transducer): at each position the first
matching rule (longest lhs first, then insertion order) fires and its output is not re-read.
Contexts are read on the input.

Distance D(X, Y) = JSD(char bigrams) + JSD(char trigrams) + JSD(word types), with '#'
padding at word edges. Everything is computed on word types with counts.
"""
import collections, math, random
import numpy as np

PAD = '#'


def grams(w):
    s = PAD + w + PAD
    return ['2' + s[i:i + 2] for i in range(len(s) - 1)] + ['3' + s[i:i + 3] for i in range(len(s) - 2)]


class Index:
    def __init__(self, cap=400000):
        self.ix = {}
        self.cap = cap

    def get(self, k):
        i = self.ix.get(k)
        if i is None:
            i = len(self.ix); self.ix[k] = i
        return i


def apply_word(w, rb):
    if not rb:
        return w
    s = PAD + w + PAD
    out = []; i = 0; n = len(w)
    while i < n:
        rs = rb.get(w[i])
        if rs:
            for lhs, rhs, lc, rc in rs:
                L = len(lhs)
                if w.startswith(lhs, i) and (lc is None or s[i] == lc) and (rc is None or s[i + L + 1] == rc):
                    out.append(rhs); i += L; break
            else:
                out.append(w[i]); i += 1
        else:
            out.append(w[i]); i += 1
    return ''.join(out)


def rules_by_first(rules):
    rb = collections.defaultdict(list)
    for r in rules:
        rb[r[0][0]].append(r)
    for k in rb:
        rb[k].sort(key=lambda r: -len(r[0]))  # stable: insertion order within same length
    return rb


def jsd_vec(p, q):
    """JSD in bits between two count vectors (same length)."""
    P = p / max(p.sum(), 1e-12); Q = q / max(q.sum(), 1e-12)
    M = 0.5 * (P + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(P > 0, P * np.log2(P / M), 0.0).sum()
        b = np.where(Q > 0, Q * np.log2(Q / M), 0.0).sum()
    return 0.5 * (a + b)


class Target:
    """Fixed target statistics (Y) and an index shared with the source."""
    def __init__(self, ywords, cap=60000):
        self.idx = Index(cap)
        self.cap = cap
        self.y2 = np.zeros(cap); self.y3 = np.zeros(cap); self.yw = np.zeros(cap)
        self.oov = self.idx.get('<OOV>')
        for w, c in collections.Counter(ywords).items():
            self.add(self.y2, self.y3, self.yw, w, c)
        self.frozen = True  # keys absent from Y share one bucket: exact for JSD (q=0 terms are linear in p)

    def gram_ids(self, w):
        g = grams(w)
        if getattr(self, 'frozen', False):
            ix = self.idx.ix; o = self.oov
            return [ix.get(x, o) for x in g], ix.get('W' + w, o)
        return [self.idx.get(x) for x in g], self.idx.get('W' + w)

    def add(self, a2, a3, aw, w, c):
        gi, wi = self.gram_ids(w)
        for k, i in zip(grams(w), gi):
            (a2 if k[0] == '2' else a3)[i] += c
        aw[wi] += c

    def dist(self, x2, x3, xw):
        n = len(self.idx.ix)
        d2 = jsd_vec(x2[:n], self.y2[:n]); d3 = jsd_vec(x3[:n], self.y3[:n]); dw = jsd_vec(xw[:n], self.yw[:n])
        return d2 + d3 + dw, (d2, d3, dw)


def distance(xwords, ywords):
    T = Target(ywords, cap=200000)
    x2 = np.zeros(T.cap); x3 = np.zeros(T.cap); xw = np.zeros(T.cap)
    for w, c in collections.Counter(xwords).items():
        T.add(x2, x3, xw, w, c)
    return T.dist(x2, x3, xw)


def eval_rules(rules, xwords, ywords):
    rb = rules_by_first(rules)
    cache = {}
    out = []
    for w in xwords:
        o = cache.get(w)
        if o is None:
            o = apply_word(w, rb); cache[w] = o
        out.append(o)
    return distance(out, ywords)


def substrings(w, kmax=3):
    s = set()
    for k in range(1, kmax + 1):
        for i in range(len(w) - k + 1):
            s.add(w[i:i + k])
    return s


class Search:
    """Stochastic greedy forward search for an ordered rule list, with a guided random proposal pool."""
    def __init__(self, xfit, yfit, seed=0, ctx_symbols=False, kmax_lhs=3, kmax_rhs=3):
        self.rng = random.Random(seed)
        self.T = Target(yfit)
        self.types = list(collections.Counter(xfit).items())
        self.inp = [w for w, _ in self.types]
        self.cnt = np.array([c for _, c in self.types], float)
        self.sub2t = collections.defaultdict(list)
        for t, w in enumerate(self.inp):
            for s in substrings(w, kmax_lhs):
                self.sub2t[s].append(t)
        self.rules = []
        self.rb = rules_by_first([])
        self.out = list(self.inp)
        cap = self.T.cap
        self.x2 = np.zeros(cap); self.x3 = np.zeros(cap); self.xw = np.zeros(cap)
        self.gid = []
        for t, w in enumerate(self.inp):
            gi, wi = self.T.gram_ids(w)
            self.gid.append((gi, wi))
            self._addids(self.x2, self.x3, self.xw, w, gi, wi, self.cnt[t])
        self.score, self.parts = self.T.dist(self.x2, self.x3, self.xw)
        # proposal pools
        xs = collections.Counter(); ys = collections.Counter()
        for w, c in self.types:
            for s in substrings(w, kmax_lhs):
                xs[s] += c
        for w, c in collections.Counter(yfit).items():
            for s in substrings(w, kmax_rhs):
                ys[s] += c
        self.lhs_pool = [s for s, c in xs.most_common(150) if c >= 5]
        self.lhs_set = set(s for s, c in xs.items() if c >= 3)
        self.ysub = ys; self.ny = sum(ys.values()) or 1
        self.rhs_pool = [''] + [s for s, c in ys.most_common(150) if c >= 5]
        syms = [s for s in self.lhs_pool if len(s) == 1]
        self.ctx_pool = [(None, None), (PAD, None), (None, PAD)]
        if ctx_symbols:
            self.ctx_pool += [(c, None) for c in syms[:12]] + [(None, c) for c in syms[:12]]
        self.ysyms = [s for s in self.rhs_pool if len(s) == 1]

    def _addids(self, a2, a3, aw, w, gi, wi, c):
        nb = len(w) + 1
        np.add.at(a2, gi[:nb], c); np.add.at(a3, gi[nb:], c); aw[wi] += c

    def _grow(self):
        n = len(self.x2)
        pad = lambda a: np.concatenate([a, np.zeros(max(n, len(self.T.idx.ix) - n + 1000))])
        self.x2, self.x3, self.xw = pad(self.x2), pad(self.x3), pad(self.xw)
        self.T.y2, self.T.y3, self.T.yw = pad(self.T.y2), pad(self.T.y3), pad(self.T.yw)
        self.T.cap = len(self.x2)

    def ids_of(self, w):
        r = self._idc.get(w)
        if r is None:
            gi, wi = self.T.gram_ids(w)
            nb = len(w) + 1
            r = (np.array(gi[:nb], int), np.array(gi[nb:], int), wi); self._idc[w] = r
        return r

    def try_rules(self, newrules, commit=False):
        if not hasattr(self, '_idc'):
            self._idc = {}
        rb = rules_by_first(newrules)
        lhss = {r[0] for r in set(newrules) ^ set(self.rules)}
        aff = set()
        for l in lhss:
            aff.update(self.sub2t.get(l, ()))
        i2, v2, i3, v3, iw, vw = [], [], [], [], [], []
        changes = []
        for t in aff:
            o = apply_word(self.inp[t], rb)
            if o == self.out[t]:
                continue
            c = self.cnt[t]
            a2, a3, aw = self.ids_of(self.out[t]); b2, b3, bw = self.ids_of(o)
            i2 += [a2, b2]; v2 += [np.full(len(a2), -c), np.full(len(b2), c)]
            i3 += [a3, b3]; v3 += [np.full(len(a3), -c), np.full(len(b3), c)]
            iw += [aw, bw]; vw += [-c, c]
            changes.append((t, o))
        if len(self.T.idx.ix) > len(self.x2) - 10:
            self._grow()
        if changes:
            I2 = np.concatenate(i2); V2 = np.concatenate(v2); I3 = np.concatenate(i3); V3 = np.concatenate(v3)
            IW = np.array(iw, int); VW = np.array(vw)
            np.add.at(self.x2, I2, V2); np.add.at(self.x3, I3, V3); np.add.at(self.xw, IW, VW)
        sc, parts = self.T.dist(self.x2, self.x3, self.xw)
        if commit:
            self.rules = list(newrules); self.rb = rb
            for t, o in changes:
                self.out[t] = o
            self.score, self.parts = sc, parts
            if getattr(self, 'guided', False):
                self.refresh_guide()
        elif changes:
            np.add.at(self.x2, I2, -V2); np.add.at(self.x3, I3, -V3); np.add.at(self.xw, IW, -VW)
        return sc, parts

    def refresh_guide(self, top=40):
        out = collections.Counter()
        for t, w in enumerate(self.out):
            for x in substrings(w, 3):
                out[x] += self.cnt[t]
        no = sum(out.values()) or 1
        d = {k: out.get(k, 0) / no - self.ysub.get(k, 0) / self.ny for k in set(out) | set(self.ysub)}
        over = sorted((k for k in d if d[k] > 0 and k in self.lhs_set), key=lambda k: -d[k])[:top]
        under = sorted((k for k in d if d[k] < 0), key=lambda k: d[k])[:top]
        self.g_over = (over, [d[k] for k in over]); self.g_under = (under, [-d[k] for k in under])

    def propose(self):
        if getattr(self, 'guided', False) and self.rng.random() < 0.6:
            if not hasattr(self, 'g_over'):
                self.refresh_guide()
            while True:
                lhs = self.rng.choices(*self.g_over)[0]
                rhs = '' if self.rng.random() < 0.1 else self.rng.choices(*self.g_under)[0]
                lc, rc = self.rng.choice(self.ctx_pool)
                if rhs != lhs and not any(r[0] == lhs and r[2] == lc and r[3] == rc for r in self.rules):
                    return (lhs, rhs, lc, rc)
        while True:
            lhs = self.rng.choice(self.lhs_pool)
            u = self.rng.random()
            if u < 0.5 and len(lhs) == 1:
                rhs = self.rng.choice(self.ysyms + [''])
            else:
                rhs = self.rng.choice(self.rhs_pool)
            lc, rc = self.rng.choice(self.ctx_pool)
            if rhs == lhs:
                continue
            if any(r[0] == lhs and r[2] == lc and r[3] == rc for r in self.rules):
                continue
            return (lhs, rhs, lc, rc)

    def step(self, n_cand=250, eps=1e-4, allow_drop=True):
        best = (self.score - eps, None)
        for _ in range(n_cand):
            r = self.propose()
            sc, _ = self.try_rules(self.rules + [r])
            if sc < best[0]:
                best = (sc, self.rules + [r])
        if allow_drop and self.rules:
            for i in range(len(self.rules)):
                nr = self.rules[:i] + self.rules[i + 1:]
                sc, _ = self.try_rules(nr)
                if sc < best[0] + 2e-4 and sc < self.score:  # drop if nearly as good (MDL: a rule costs)
                    best = (sc, nr)
                    break
        if best[1] is None:
            return False
        self.try_rules(best[1], commit=True)
        return True


def run_search(xfit, yfit, xheld, yheld, kmax=30, n_cand=250, seed=0, ctx_symbols=False, log=None, ctx_pool=None, guided=False):
    S = Search(xfit, yfit, seed=seed, ctx_symbols=ctx_symbols)
    S.guided = guided
    if ctx_pool is not None:
        S.ctx_pool = ctx_pool
    path = []
    h0 = distance(xheld, yheld)
    path.append(dict(k=0, fit=S.score, held=h0[0], held_parts=h0[1], rules=[]))
    fails = 0
    while len(S.rules) < kmax and fails < 2:
        ok = S.step(n_cand=n_cand)
        if not ok:
            fails += 1; continue
        fails = 0
        h = eval_rules(S.rules, xheld, yheld)
        path.append(dict(k=len(S.rules), fit=S.score, held=h[0], held_parts=h[1], rules=list(S.rules)))
        if log:
            log(f'  k={len(S.rules):2d} fit {S.score:.4f} held {h[0]:.4f} last {S.rules[-1] if S.rules else None}')
    return path


def split_pages(pages, n_fit, n_held, seed):
    """Disjoint page-level split: held first, then fit; truncate to exact word budgets."""
    rng = random.Random(seed)
    P = list(pages); rng.shuffle(P)
    held, fit, extra = [], [], []
    for p in P:
        if len(held) < n_held:
            held.extend(p)
        elif len(fit) < n_fit:
            fit.extend(p)
        else:
            extra.extend(p)
    return fit[:n_fit], held[:n_held], extra
