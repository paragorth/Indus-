"""v82 TRY TO KILL THE ONSET LINK, THEN SAY WHAT IT IS: shared helpers.

v81 B: the front of a Voynich word (glyphs 1-3) predicts the next word's onset beyond the last-glyph junction and
beyond page topic (+0.03-0.10 bits held out). v81 C: half line/passage mood, half adjacency. Stated kill: a generator
with line-level onset mood plus a front-glyph agreement rule, fitted to the Voynich, reaching +0.03 bits under the
same junction-preserving null.

This module
  * Mini: a light compiled corpus with the attributes the frozen v81 test functions read (v81_c3.gain, v81_c3.reroll,
    v81_c3b.reroll_page are imported unchanged; the three frozen onset definitions are v81_c3.DEFS);
  * frozen(): the frozen v81 test (train leaf half 0, test leaf half 1; 20 re-rolls);
  * surf_stats(): the Voynich surface statistics a kill generator must also match (line effects, page vocabulary,
    doubling rate, neighbour avoidance, junction, word law);
  * gen_moodagr(): the kill generator family: STACK / copy-and-vary base + drifting line or passage mood that biases
    word onsets + a front-glyph agreement rule (copy or harmony class), every draw reweighted by onset class.
"""
import os, sys, math, random, pickle
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v82_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V
import v81_lib as L81
import v81_c3 as C3
import v81_c3b as C3B
try:
    os.rmdir(L81.CK)          # v81_lib creates its (deleted) checkpoint dir at import; remove it if empty
except OSError:
    pass

DEFS = C3.DEFS               # frozen: e1c_line k1, noq_line k1, e1c_line k2


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def psave(name, obj): pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    p = os.path.join(CK, name)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


# ------------------------------------------------------------------ light compiled corpus
class Mini:
    def __init__(self, pages, variants=('e1c_line', 'noq_line')):
        secs = sorted({p['sec'] for p in pages}); sx = {s: i for i, s in enumerate(secs)}
        W, pg, ln, pos, ps0 = [], [], [], [], []
        Ln = 0
        for pi, p in enumerate(pages):
            for l in p['lines']:
                for i, w in enumerate(l['w']):
                    W.append(w); pg.append(pi); ln.append(Ln); pos.append(i); ps0.append(bool(l['ps']) and i == 0)
                Ln += 1
        self.W = W; self.n = n = len(W)
        self.pg = np.array(pg); self.ln = np.array(ln); self.pos = np.array(pos)
        self.sec = np.array([sx[pages[i]['sec']] for i in self.pg]); self.nsec = len(secs)
        self.leaf = np.array([L81.leaf(pages[i]['id']) for i in self.pg])
        self.end = np.array([L81.AIX.get(w[-1], L81.AIX['_']) for w in W])
        li0 = self.pos == 0
        self.G = {}
        for v in variants:
            A = np.full((n, 3), L81.AIX['$'], np.int16); cache = {}
            for t, w in enumerate(W):
                key = (w, bool(li0[t]), ps0[t]); x = cache.get(key)
                if x is None: x = L81.variant(w, li0[t], ps0[t], v); cache[key] = x
                for j, c in enumerate(x[:3]): A[t, j] = L81.AIX.get(c, L81.AIX['_'])
            self.G[v] = A
        same1 = (self.ln[1:] == self.ln[:-1]) & (self.pos[:-1] >= 1)
        self.p1 = np.nonzero(same1)[0]
        self.tr, self.te = self.leaf == 0, self.leaf == 1


def frozen(c, nrep=20, page=False, seed=813, defs=None):
    """the frozen v81 junction-preserving test, onset slot. Returns list of (gain - null mean, z) per definition."""
    rng = np.random.default_rng(seed); out = []
    rr = C3B.reroll_page if page else C3.reroll
    for d in (defs or DEFS):
        o = L81.onset(c, d)
        r = C3.gain(c, o, o, c.tr, c.te)
        nul = [C3.gain(c, o, rr(c, o, rng), c.tr, c.te) for _ in range(nrep)]
        m, s = float(np.mean(nul)), float(np.std(nul))
        out.append((r - m, (r - m) / (s + 1e-9)))
    return out


def lag_ratio(c, nrep=10, seed=814):
    """v81 c3b lag-1 and lag-2 gains (re-roll corrected) for the three definitions."""
    rng = np.random.default_rng(seed); out = []
    for d in DEFS:
        o = L81.onset(c, d)
        g1 = C3B.gain_lag(c, o, o, c.tr, c.te, 1); g2 = C3B.gain_lag(c, o, o, c.tr, c.te, 2)
        g1r = np.mean([C3B.gain_lag(c, o, C3.reroll(c, o, rng), c.tr, c.te, 1) for _ in range(nrep)])
        g2r = np.mean([C3B.gain_lag(c, o, C3.reroll(c, o, rng), c.tr, c.te, 2) for _ in range(nrep)])
        out.append((g1 - g1r, g2 - g2r))
    return out


# ------------------------------------------------------------------ surface statistics to match
def _mi_xy(x, y):
    return L81._mi(np.asarray(x), np.asarray(y))


def surf_stats(pages, seed=5):
    """statistics a kill generator must reproduce. All on the whole text."""
    rng = random.Random(seed)
    toks = []   # (page, sec, line, pos, nline, word)
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']):
            n = len(l['w'])
            for i, w in enumerate(l['w']): toks.append((pi, p['sec'], li, i, n, w))
    W = [t[5] for t in toks]
    # line effects: MI(position class; first glyph), MI(position class; last glyph)
    pc = [0 if t[3] == 0 else (2 if t[3] == t[4] - 1 else 1) for t in toks]
    f1 = [w[0] for w in W]; lz = [w[-1] for w in W]
    s = {}
    s['line_first'] = _mi_xy(pc, [hash(x) for x in f1])
    s['line_last'] = _mi_xy(pc, [hash(x) for x in lz])
    # page vocabulary: MI(page; word) excess over a within-section page shuffle
    pg = np.array([t[0] for t in toks]); sec = [t[1] for t in toks]
    _, wid = np.unique(W, return_inverse=True)
    perm = np.arange(len(W)); bys = defaultdict(list)
    for i, x in enumerate(sec): bys[x].append(i)
    for ix in bys.values():
        ix = np.array(ix); perm[ix] = ix[np.random.default_rng(seed).permutation(len(ix))]
    s['page_vocab'] = _mi_xy(pg, wid) - _mi_xy(pg, wid[perm])
    # doubling and neighbour avoidance (v78 ratio: same word at lag 1 / mean at lags 2-3, within line)
    lag = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    for p in pages:
        for l in p['lines']:
            w = l['w']
            for k in (1, 2, 3):
                for i in range(len(w) - k):
                    lag[k][0] += w[i] == w[i + k]; lag[k][1] += 1
    r = {k: lag[k][0] / max(1, lag[k][1]) for k in lag}
    s['doubling'] = r[1]
    s['nbr_ratio'] = r[1] / max(1e-9, (r[2] + r[3]) / 2)
    # junction (last glyph -> next first glyph within line), word law
    a, b = [], []
    for p in pages:
        for l in p['lines']:
            for x, y in zip(l['w'], l['w'][1:]): a.append(hash(x[-1])); b.append(hash(y[0]))
    s['junction'] = _mi_xy(a, b)
    s['ttr'] = len(set(W)) / len(W)
    s['wlen'] = float(np.mean([len(w) for w in W]))
    s['hapax'] = sum(1 for v in Counter(W).values() if v == 1) / len(W)
    return s


STAT_KEYS = ['line_first', 'line_last', 'page_vocab', 'doubling', 'nbr_ratio', 'junction', 'ttr', 'wlen', 'hapax']


def bands(sz, si):
    """match band per statistic: ZL value +- max(25% relative, 2x |ZL - IT2a|), with a floor."""
    floor = dict(line_first=0.01, line_last=0.01, page_vocab=0.02, doubling=0.004, nbr_ratio=0.12, junction=0.02,
                 ttr=0.02, wlen=0.2, hapax=0.02)
    return {k: (sz[k], max(0.25 * abs(sz[k]), 2 * abs(sz[k] - si[k]), floor[k])) for k in STAT_KEYS}


def match(s, B):
    bad = [k for k in STAT_KEYS if abs(s[k] - B[k][0]) > B[k][1]]
    return len(bad) == 0, bad


# ------------------------------------------------------------------ the kill generator family
GLYPHS = list('abcdefghijklmnopqrstuvwxyzCSTKPF')


def front(w, k):
    """generator's onset unit: first k glyphs after a leading q (the frozen test uses its own definitions)."""
    if len(w) > 1 and w[0] == 'q': w = w[1:]
    return w[:k]


def sample_params(rng):
    def lu(a, b): return math.exp(rng.uniform(math.log(a), math.log(b)))
    P = {}
    P['base'] = rng.choice(['stack', 'stack', 'selfcit', 'junc'])
    P['p_vert'] = rng.uniform(0, 0.3) if P['base'] == 'stack' else 0.0
    P['p_cite'] = rng.uniform(0, 0.5) if P['base'] != 'junc' else 0.0
    P['p_mod'] = rng.uniform(0.2, 0.7)
    P['k'] = rng.choice([1, 2, 2])                       # onset unit length the mood/agreement act on
    P['M'] = rng.choice([1, 2, 3, 4, 6, 8, 12])            # number of moods (1 = no mood)
    P['mood_src'] = rng.choice(['fit', 'fit', 'rand'])     # profiles fitted to Voynich line onset bags, or random
    P['mood_beta'] = rng.uniform(0, 2.5)
    P['mood_scope'] = rng.choice(['line', 'word', 'passage'])
    P['mood_rate'] = lu(0.05, 1.0) if P['mood_scope'] != 'word' else lu(0.01, 0.3)
    P['agr'] = rng.choice(['none', 'copy', 'harm', 'harm', 'copy2', 'pmi'])   # pmi = fitted harmony table (ceiling)
    P['agr_src'] = rng.choice([0, 1, 1])                   # previous word's glyph 1 or glyph 2 (after q)
    P['agr_lam'] = rng.uniform(-1.0, 3.0)
    P['H'] = rng.randint(2, 7)
    P['hseed'] = rng.randrange(10 ** 9)
    P['bias_copies'] = rng.random() < 0.5                  # copies (vertical / cited) also pass the onset bias
    return P


def fit_moods(pages, k, M, seed):
    """EM mixture of multinomials over the onset bags of line interiors; returns log-ratio profiles [M][onset]."""
    rng = np.random.default_rng(seed)
    bags = []; voc = {}
    for p in pages:
        for l in p['lines']:
            on = [front(w, k) for w in l['w'][1:]]
            if len(on) >= 3:
                bags.append(Counter(on)); [voc.setdefault(x, len(voc)) for x in on]
    X = np.zeros((len(bags), len(voc)))
    for i, b in enumerate(bags):
        for x, c in b.items(): X[i, voc[x]] = c
    g = X.sum(0) + 1; g /= g.sum()
    if M == 1: return voc, np.zeros((1, len(voc)))
    th = rng.dirichlet(np.ones(len(voc)) * 5, size=M) * 0.5 + g * 0.5; pi = np.ones(M) / M
    for _ in range(40):
        ll = X @ np.log(th).T + np.log(pi)
        ll -= ll.max(1, keepdims=True); R = np.exp(ll); R /= R.sum(1, keepdims=True)
        pi = R.mean(0) + 1e-6; pi /= pi.sum()
        th = R.T @ X + 0.5; th /= th.sum(1, keepdims=True)
    return voc, np.log(th) - np.log(g)


class _Pool:
    """a word pool with its words grouped by onset unit."""
    __slots__ = ('ons', 'cnt', 'words')

    def __init__(self, ws, k):
        d = defaultdict(list)
        for w in ws: d[front(w, k)].append(w)
        self.ons = list(d); self.cnt = np.array([len(d[o]) for o in self.ons], float); self.words = [d[o] for o in self.ons]


def gen_moodagr(pages, P, seed, moodcache=None):
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    k = P['k']
    # base pools (STACK structure, grouped by section|language)
    gk = lambda p: '%s|%s' % (p['sec'], p.get('lang', '-'))
    POOLW = defaultdict(list); FOLW = defaultdict(list); GL = defaultdict(Counter); SECW = defaultdict(list)
    for p in pages:
        g = gk(p)
        for l in p['lines']:
            n = len(l['w'])
            for i, w in enumerate(l['w']):
                pc = 'F' if i == 0 else ('L' if i == n - 1 else 'M')
                POOLW[(g, l['ps'], pc)].append(w); GL[g].update(w); SECW[g].append(w)
                if i > 0: FOLW[(g, l['ps'], pc, l['w'][i - 1][-1])].append(w)
    pools = {}

    def pool(key, src):
        if key not in pools: pools[key] = _Pool(src[key], k) if src.get(key) else None
        return pools[key]
    GLc = {g: (list(c), np.cumsum([c[x] for x in c]) / sum(c.values())) for g, c in GL.items()}
    # moods
    if P['M'] > 1:
        if P['mood_src'] == 'fit':
            key = (k, P['M'])
            if moodcache is not None and key in moodcache: voc, prof = moodcache[key]
            else:
                voc, prof = fit_moods(pages, k, P['M'], 82)
                if moodcache is not None: moodcache[key] = (voc, prof)
        else:
            ons = sorted({front(w, k) for ws in SECW.values() for w in ws}); voc = {o: i for i, o in enumerate(ons)}
            prof = nrng.normal(0, 1, size=(P['M'], len(voc)))
        prof = prof * P['mood_beta']
    else:
        voc, prof = {}, None
    # harmony classes
    hr = random.Random(P['hseed']); hcl = {c: hr.randrange(P['H']) for c in GLYPHS}
    PMI = {}
    if P['agr'] == 'pmi':     # fitted: log p(next onset unit | previous word's glyph src) / p(next onset unit)
        cj = Counter(); ca = Counter(); cb = Counter()
        for p in pages:
            for l in p['lines']:
                for x, y in zip(l['w'][1:], l['w'][2:]):
                    fx = front(x, 2)
                    if len(fx) <= P['agr_src']: continue
                    a, b = fx[P['agr_src']], front(y, k); cj[(a, b)] += 1; ca[a] += 1; cb[b] += 1
        N = sum(cj.values())
        PMI = {ab: math.log((c + 0.5) * N / (ca[ab[0]] * cb[ab[1]] + 0.5)) for ab, c in cj.items() if c >= 5}

    def agree_vec(pl, prevw):
        if P['agr'] == 'none' or prevw is None: return None
        pw = front(prevw, 2)
        if len(pw) <= P['agr_src']: return None
        src = pw[P['agr_src']]
        if P['agr'] == 'pmi':
            v = np.array([PMI.get((src, o), 0.0) for o in pl.ons])
        elif P['agr'] == 'copy':
            v = np.array([1.0 if o[:1] == src else 0.0 for o in pl.ons])
        elif P['agr'] == 'copy2':
            v = np.array([1.0 if o[:2] == pw[:2] else 0.0 for o in pl.ons])
        else:
            hs = hcl.get(src, -1); v = np.array([1.0 if o and hcl.get(o[0], -2) == hs else 0.0 for o in pl.ons])
        return v

    def draw(pl, mood, prevw):
        lw = np.zeros(len(pl.ons))
        if prof is not None:
            lw += np.array([prof[mood, voc[o]] if o in voc else 0.0 for o in pl.ons])
        av = agree_vec(pl, prevw)
        if av is not None: lw += P['agr_lam'] * av
        wt = pl.cnt * np.exp(lw - lw.max()); wt /= wt.sum()
        j = int(np.searchsorted(np.cumsum(wt), rng.random())); j = min(j, len(wt) - 1)
        return rng.choice(pl.words[j])

    def bias_ok(w, mood, prevw):
        """rejection step for copies when bias_copies: accept with prob proportional to the onset weight."""
        o = front(w, k); lw = 0.0
        if prof is not None and o in voc: lw += prof[mood, voc[o]]
        if P['agr'] != 'none' and prevw is not None:
            pw = front(prevw, 2)
            if len(pw) > P['agr_src']:
                src = pw[P['agr_src']]
                if P['agr'] == 'pmi': a = PMI.get((src, o), 0.0)
                elif P['agr'] == 'copy': a = o[:1] == src
                elif P['agr'] == 'copy2': a = o[:2] == pw[:2]
                else: a = bool(o) and hcl.get(o[0], -2) == hcl.get(src, -1)
                lw += P['agr_lam'] * a
        return rng.random() < math.exp(min(0.0, lw - 2.5))

    def mutate(w, g):
        ks, cp = GLc[g]; j = rng.randrange(len(w))
        c = ks[min(int(np.searchsorted(cp, rng.random())), len(ks) - 1)]
        return w[:j] + c + w[j + 1:]

    M = max(1, P['M']); out = []; mood = rng.randrange(M)
    for p in pages:
        g = gk(p); hist = []; nl = []; prev = None
        if P['mood_scope'] != 'passage': mood = rng.randrange(M)
        for l in p['lines']:
            if P['mood_scope'] in ('line', 'passage') and rng.random() < P['mood_rate']: mood = rng.randrange(M)
            n = len(l['w']); ws = []
            for i in range(n):
                if P['mood_scope'] == 'word' and rng.random() < P['mood_rate']: mood = rng.randrange(M)
                pc = 'F' if i == 0 else ('L' if i == n - 1 else 'M')
                prevw = ws[-1] if ws else None
                u = rng.random(); w = None
                for _try in range(3):
                    if prev is not None and u < P['p_vert'] and prev:
                        j = min(i, len(prev) - 1) if pc != 'L' else len(prev) - 1
                        w = prev[j]
                        if rng.random() < P['p_mod']: w = mutate(w, g)
                    elif len(hist) >= 3 and u < P['p_vert'] + P['p_cite']:
                        w = rng.choice(hist[-60:])
                        if rng.random() < P['p_mod']: w = mutate(w, g)
                    else:
                        w = None; break
                    if not P['bias_copies'] or bias_ok(w, mood, prevw): break
                    w = None; u = 1.0     # rejected copy: fall through to a fresh draw
                if w is None:
                    pl = None
                    if P['base'] == 'selfcit':
                        pl = pool(g, SECW)
                    else:
                        if i > 0: pl = pool((g, l['ps'], pc, ws[-1][-1]), FOLW)
                        if pl is None: pl = pool((g, l['ps'], pc), POOLW) or pool((g, False, 'M'), POOLW) or pool((g, True, 'M'), POOLW)
                    w = draw(pl, mood, prevw if i > 0 else None)
                ws.append(w); hist.append(w)
            nl.append(dict(l, w=ws)); prev = ws
        out.append(dict(p, lines=nl))
    return out
