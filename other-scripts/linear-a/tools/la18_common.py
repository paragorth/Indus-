#!/usr/bin/env python3
"""LA-18 shared code: WORDS THAT AVOID EACH OTHER.

Idea: alternative forms of one lexeme (cases, number, with / without a suffix) fill the same slot, so inside
one document they should be mutually exclusive, while sharing contexts across the corpus. We ignore what
co-occurs and study what never does.

Objects
  units  = documents with >= 2 distinct word types (>= 2 signs); presence/absence incidence (sets).
  null   = curveball randomisation of the type x document incidence, stratified by site: keeps every word's
           document frequency and every document's size exactly (Strona et al. 2014 algorithm).
  pairs  = word-type pairs related by form only (no sound value):
             SUF1  w2 = w1 + one sign            PRE1  w2 = one sign + w1
             FIN   same length >= 3, differ only in the last sign
             INI   same length >= 3, differ only in the first sign
             RND   frequency-matched random pairs with no form relation (comparison family)
  avoidance of a pair = O (documents holding both) vs its null distribution; lower-tail mid-P.
No sound value enters the search. Sound values are used only to label the Linear B positive control.
"""
import os, sys, json, random, collections, math, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from la5_common import la_docs, lb_docs
OUT = os.path.join(HERE, '..', 'data', 'la18')
os.makedirs(OUT, exist_ok=True)
LOOPS = os.path.join(HERE, '..', 'loops')


# ------------------------------------------------------------------ corpora
def _docs(raw, sitekey='site'):
    out = []
    for d in raw:
        toks = []
        for L in d['lines']:
            toks.append(('NL',))
            toks.extend(L)
        ws = [t[1] for t in toks if t[0] == 'W' and len(t[1]) >= 2]
        out.append(dict(id=d['id'], site=d[sitekey], toks=toks, words=ws,
                        support=d.get('support') or d.get('series', '')))
    return out


def la_corpus():
    return _docs(la_docs(admin_only=False))


def lb_corpus(sites=('KN', 'PY', 'TH', 'MY', 'TI', 'KH')):
    return [d for d in _docs(lb_docs()) if d['site'] in sites]


def units(docs):
    """multiword units -> (list of frozensets of types, site list, doc ids)"""
    S, site, ids = [], [], []
    for d in docs:
        s = frozenset(d['words'])
        if len(s) >= 2:
            S.append(s); site.append(d['site']); ids.append(d['id'])
    return S, site, ids


# ------------------------------------------------------------------ curveball null
class Curveball:
    def __init__(self, sets, strata, seed=0):
        self.cur = [set(s) for s in sets]
        g = collections.defaultdict(list)
        for i, s in enumerate(strata): g[s].append(i)
        self.groups = [v for v in g.values() if len(v) >= 2]
        self.w = np.array([len(v) for v in self.groups], float); self.w /= self.w.sum()
        self.rnd = random.Random(seed)
        self.n = len(sets)

    def step(self, k):
        r = self.rnd; cur = self.cur; G = self.groups
        gi = r.choices(range(len(G)), weights=self.w, k=k)
        for g in gi:
            a, b = r.sample(G[g], 2)
            A, B = cur[a], cur[b]
            ua = A - B; ub = B - A
            if not ua or not ub: continue
            pool = list(ua | ub); r.shuffle(pool)
            na = len(ua)
            sh = A & B
            cur[a] = sh | set(pool[:na]); cur[b] = sh | set(pool[na:])

    def samples(self, nsamp, burn=None, thin=None):
        burn = burn or 20 * self.n; thin = thin or 3 * self.n
        self.step(burn)
        for _ in range(nsamp):
            self.step(thin)
            yield self.cur


def doc_shuffle(sets, strata, seed):
    """Negative control: word TOKENS shuffled across documents within stratum (sizes kept, types redistributed)."""
    r = random.Random(seed)
    g = collections.defaultdict(list)
    for i, s in enumerate(strata): g[s].append(i)
    out = [None] * len(sets)
    for idx in g.values():
        pool = [w for i in idx for w in sets[i]]; r.shuffle(pool)
        k = 0
        for i in idx:
            n = len(sets[i]); out[i] = frozenset(pool[k:k + n]); k += n
    # duplicates collapse; that lowers sizes slightly, acceptable for a negative control
    return out


# ------------------------------------------------------------------ candidate pairs
def form_pairs(types):
    T = set(types); P = {}
    for w in T:
        if len(w) >= 2:
            # SUF1 / PRE1
            pass
    by_prefix = collections.defaultdict(list)
    for w in T:
        if len(w) >= 3: by_prefix[w[:-1]].append(w)
    for w in T:
        if w in by_prefix:
            for v in by_prefix[w]: P[(w, v)] = 'SUF1'
    by_suffix = collections.defaultdict(list)
    for w in T:
        if len(w) >= 3: by_suffix[w[1:]].append(w)
    for w in T:
        if w in by_suffix:
            for v in by_suffix[w]: P.setdefault((w, v), 'PRE1')
    for key, fam in (('FIN', lambda w: w[:-1]), ('INI', lambda w: w[1:])):
        g = collections.defaultdict(list)
        for w in T:
            if len(w) >= 3: g[(len(w), fam(w))].append(w)
        for ws in g.values():
            ws.sort()
            for a, b in itertools.combinations(ws, 2): P.setdefault((a, b), key)
    return P


def random_pairs(types, df, P, k, seed):
    """frequency-matched random pairs: for each form pair draw a pair with the same df pair (nearest bins)."""
    r = random.Random(seed)
    bins = collections.defaultdict(list)
    def b(x): return min(x, 8) if x < 8 else (8 if x < 12 else 12)
    for w in types: bins[b(df[w])].append(w)
    out = {}
    fp = list(P)
    for _ in range(k):
        for (u, v) in fp:
            for _t in range(20):
                a = r.choice(bins[b(df[u])]); c = r.choice(bins[b(df[v])])
                if a == c: continue
                key = (a, c) if a < c else (c, a)
                if key in P or key in out: continue
                if a[:1] == c[:1] or a[-1:] == c[-1:]: continue  # no shared edge sign
                out[key] = 'RND'; break
    return out


# ------------------------------------------------------------------ counting
class Counter2:
    def __init__(self, pairs):
        self.pairs = list(pairs)
        self.ix = {p: i for i, p in enumerate(self.pairs)}
        self.partners = collections.defaultdict(list)
        for i, (a, b) in enumerate(self.pairs):
            self.partners[a].append((b, i))

    def count(self, sets):
        c = np.zeros(len(self.pairs), np.int16)
        P = self.partners
        for s in sets:
            for w in s:
                pl = P.get(w)
                if pl:
                    for (v, i) in pl:
                        if v in s: c[i] += 1
        return c


def null_matrix(sets, strata, pairs, nsamp, seed):
    C = Counter2(pairs); cb = Curveball(sets, strata, seed)
    N = np.zeros((nsamp, len(C.pairs)), np.int16)
    for k, cur in enumerate(cb.samples(nsamp)):
        N[k] = C.count(cur)
    return C.count(sets), N


def lower_p(obs, N):
    """mid-P of O <= obs under null columns; also for each null row (for max-stat FWER)."""
    S = N.shape[0]
    p_obs = ((N < obs).sum(0) + 0.5 * (N == obs).sum(0) + 0.5) / (S + 1)
    # null-row p-values via sorting each column
    p_null = np.empty(N.shape, float)
    for j in range(N.shape[1]):
        col = N[:, j]; srt = np.sort(col)
        lt = np.searchsorted(srt, col, 'left'); le = np.searchsorted(srt, col, 'right')
        p_null[:, j] = (lt + 0.5 * (le - lt) + 0.5) / (S + 1)
    return p_obs, p_null


def fwer(p_obs, p_null, mask=None):
    if mask is not None: p_obs, p_null = p_obs[mask], p_null[:, mask]
    if p_obs.size == 0: return np.array([]), np.array([])
    mn = p_null.min(1)
    return np.array([((mn <= p).sum() + 1) / (len(mn) + 1) for p in p_obs]), mn


def bh(p):
    p = np.asarray(p); n = len(p)
    if n == 0: return p
    o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def family_excess(obs, N, idx):
    """aggregate avoidance of a family: E - O summed, z against the null rows, one-sided P(O_sum <= obs)."""
    if len(idx) == 0: return dict(n=0)
    o = obs[idx].sum(); ns = N[:, idx].sum(1).astype(float)
    E = ns.mean(); sd = ns.std() + 1e-9
    p = ((ns <= o).sum() + 1) / (len(ns) + 1)
    p_hi = ((ns >= o).sum() + 1) / (len(ns) + 1)
    return dict(n=int(len(idx)), O=int(o), E=round(float(E), 2), ratio=round(float(o / E), 3) if E > 0 else None,
                z=round(float((o - E) / sd), 2), p_avoid=round(p, 4), p_attract=round(p_hi, 4))


# ------------------------------------------------------------------ contexts
def context_vectors(docs):
    """bag of context features per type: previous / next token class, line-start, logograms on the doc,
    site, support. No word identity of neighbours (so paradigm partners do not share by co-occurring)."""
    V = collections.defaultdict(collections.Counter)
    def cls(t):
        if t is None: return 'END'
        if t[0] == 'NL': return 'NL'
        if t[0] == 'W': return 'W'
        if t[0] == 'F': return t[1]
        return t[0]
    for d in docs:
        toks = d['toks']
        logos = {t[1] for t in toks if t[0] == 'F' and t[1] != 'NUM'}
        for i, t in enumerate(toks):
            if t[0] != 'W' or len(t[1]) < 2: continue
            w = t[1]; c = V[w]
            c['p:' + cls(toks[i - 1] if i > 0 else None)] += 1
            c['n:' + cls(toks[i + 1] if i + 1 < len(toks) else None)] += 1
            for L in logos: c['L:' + L] += 0.5
            c['site:' + d['site']] += 0.5
            c['sup:' + str(d['support'])] += 0.5
    return V


def cos(a, b):
    if not a or not b: return 0.0
    num = sum(v * b.get(k, 0) for k, v in a.items())
    return num / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def dump(obj, name):
    json.dump(obj, open(os.path.join(OUT, name), 'w'), indent=1, default=str)
