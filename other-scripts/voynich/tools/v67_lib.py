"""v67 EACH LINE SAYS ONE THING: shared helpers.

A corpus here = list of lines; each line = dict(words=[glyph strings], folio, sec, truth (plants only)).
Glyph strings use vlib.glyphs (ch, sh, benched gallows merged into one character each).

Line-token stream scoring (within-page line shuffle null, order kept otherwise):
  MI   : bigram MI of consecutive OFF-DIAGONAL line-token pairs (a != b) inside a page, z vs shuffle
  ASYM : directional asymmetry, sum over unordered pairs {a,b} of (n_ab-n_ba)^2/(n_ab+n_ba), z vs shuffle
  REC  : number of distinct ordered pairs a->b (a != b) seen on >= 2 pages, z vs shuffle
  REP  : adjacent identical tokens / shuffle expectation (descriptive)
"""
import os, sys, json, math, random, zlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib

ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v67_ckpt')
os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')


def g(w):
    return ''.join(vlib.glyphs(w))


# ---------------------------------------------------------------- corpora
def voynich(name='ZL3b'):
    out = []
    for r in vlib.load_voynich(name):
        ws = [g(w) for w in r['words'] if w]
        if ws:
            out.append({'words': ws, 'folio': r['folio'], 'sec': r['illus'], 'lang': r['lang'],
                        'para_start': r['para_start']})
    return out


def lev(a, b):
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _vpools(V):
    words = [w for L in V for w in L['words']]
    ends = Counter(w[-k:] for w in words for k in (1, 2, 3) if len(w) > k)
    heads = Counter(w[:1] for w in words)
    glyph = Counter(c for w in words for c in w)
    return words, ends, heads, glyph


def _draw(cnt, rng):
    ks = list(cnt.keys()); ws = list(cnt.values())
    return rng.choices(ks, ws)[0]


def plant(V, lang='Latin-Caesar', style='mid', seed=1, filler=0.25):
    """Planted line-per-word encoding laid into the Voynich line/page skeleton.
    Each real word -> opaque glyph code (letter -> 1-2 random glyphs); each line = len(Voynich line) variants
    of that code, with a share `filler` of random Voynich words.
    style mid : variant = code truncated to >=60% + random Voynich ending (p .5), random head glyph (p .3),
                random glyph insertion (p .08 per position)
    style hard: variant = random contiguous chunk of >= half of the code + random ending, filler 0.35
    """
    rng = random.Random(seed)
    refl = vlib.load_ref(lang, max_words=len(V) + 50, skip_frac=0.05)
    real = [w for L in refl for w in L['words']][:len(V)]
    words, ends, heads, glyph = _vpools(V)
    galph = [c for c, _ in glyph.most_common(20)]
    letters = sorted(set(''.join(real)))
    code = {c: ''.join(rng.choice(galph) for _ in range(rng.choice((1, 1, 2)))) for c in letters}
    enc = lambda w: ''.join(code[c] for c in w)
    if style == 'hard':
        filler = 0.35
    out = []
    for L, wd in zip(V, real):
        e = enc(wd)
        ws = []
        for _ in range(len(L['words'])):
            if rng.random() < filler:
                ws.append(rng.choice(words)); continue
            if style == 'mid':
                k = max(1, int(math.ceil(len(e) * rng.uniform(0.6, 1.0))))
                v = e[:k]
                if rng.random() < 0.5: v += _draw(ends, rng)
                if rng.random() < 0.3: v = _draw(heads, rng) + v
                v = ''.join(c + (rng.choice(galph) if rng.random() < 0.08 else '') for c in v)
            else:
                n = max(1, int(math.ceil(len(e) * rng.uniform(0.5, 0.8))))
                s = rng.randrange(0, len(e) - n + 1)
                v = e[s:s + n]
                if rng.random() < 0.6: v += _draw(ends, rng)
                if rng.random() < 0.4: v = _draw(heads, rng) + v
            ws.append(v)
        out.append({'words': ws, 'folio': L['folio'], 'sec': L['sec'], 'truth': wd})
    return out


def lrg(V, seed=1, beta=1.2, drift=0.0):
    """Line-reuse generator (negative control): each line picks a seed word from its page's pool; the line's
    words are drawn from the page pool with weight exp(-beta*lev(w, seed)). With drift>0 the seed is, with
    that probability, a word of the previous line (gives neighbour-line sharing). No word order beyond that."""
    rng = random.Random(seed)
    pages = defaultdict(list)
    for L in V: pages[L['folio']].extend(L['words'])
    out = []; prev = None; prevf = None
    cache = {}
    for L in V:
        pool = pages[L['folio']]
        if prev and prevf == L['folio'] and rng.random() < drift:
            s = rng.choice(prev)
        else:
            s = rng.choice(pool)
        key = (L['folio'], s)
        if key not in cache:
            uniq = sorted(set(pool)); cnt = Counter(pool)
            cache[key] = (uniq, [cnt[u] * math.exp(-beta * lev(u, s)) for u in uniq])
        uniq, wts = cache[key]
        ws = rng.choices(uniq, wts, k=len(L['words']))
        out.append({'words': ws, 'folio': L['folio'], 'sec': L['sec']})
        prev, prevf = ws, L['folio']
    return out


def cpv(V, seed=1, pcopy=0.5, window=40):
    """copy-and-vary generator (x4 gen_cpv, glyph level), page by page."""
    rng = random.Random(seed)
    words, ends, heads, glyph = _vpools(V)
    out = []; hist = []; pf = None
    for L in V:
        if L['folio'] != pf: hist = []; pf = L['folio']
        ws = []
        for _ in L['words']:
            if hist and rng.random() < pcopy:
                w = rng.choice(hist[-window:]); i = rng.randrange(len(w))
                w = w[:i] + _draw(glyph, rng) + w[i + 1:]
            else:
                w = rng.choice(words)
            hist.append(w); ws.append(w)
        out.append({'words': ws, 'folio': L['folio'], 'sec': L['sec']})
    return out


def within_shuffle_words(V, seed=1):
    """words shuffled across the lines of a page (line unity destroyed, page kept)."""
    rng = random.Random(seed)
    byp = defaultdict(list)
    for i, L in enumerate(V): byp[L['folio']].append(i)
    out = [dict(L) for L in V]
    for f, idx in byp.items():
        ws = [w for i in idx for w in V[i]['words']]; rng.shuffle(ws); k = 0
        for i in idx:
            n = len(V[i]['words']); out[i] = dict(V[i]); out[i]['words'] = ws[k:k + n]; k += n
    return out


# ---------------------------------------------------------------- consensus rules
def alphabet(C):
    return [c for c, _ in Counter(c for L in C for w in L['words'] for c in w).most_common()]


def random_rule(rng, alph):
    r = {}
    t = rng.random()
    if t < 0.35:
        r['red'] = ('id',)
    elif t < 0.7:
        m = rng.randint(3, 12); r['red'] = ('cls', m, rng.randrange(1 << 30))
    else:
        k = rng.randint(1, 6); r['red'] = ('del', tuple(sorted(rng.sample(alph[:18], k))))
    ex = rng.choice(['pre', 'pre', 'suf', 'whole', 'infix', 'bag', 'bag'])
    r['ex'] = ex
    r['k'] = rng.randint(1, 4) if ex in ('pre', 'suf') else (rng.randint(2, 3) if ex == 'bag' else 0)
    r['agg'] = rng.choice(['mode', 'idf', 'medial', 'mode2'])
    r['tie'] = rng.choice(['rare', 'common', 'first'])
    return r


def rule_name(r):
    red = r['red']
    rs = 'id' if red[0] == 'id' else (f'cls{red[1]}:{red[2] % 1000}' if red[0] == 'cls' else 'del' + ''.join(red[1]))
    return f"{rs}|{r['ex']}{r['k'] or ''}|{r['agg']}|{r['tie']}"


def reducer(r, alph):
    red = r['red']
    if red[0] == 'id':
        return lambda w: w
    if red[0] == 'cls':
        rr = random.Random(red[2]); m = red[1]
        mp = {c: chr(65 + rr.randrange(m)) for c in alph}
        return lambda w: ''.join(mp.get(c, '?') for c in w)
    D = set(red[1])
    return lambda w: ''.join(c for c in w if c not in D)


def units(r, w):
    ex, k = r['ex'], r['k']
    if ex == 'pre': return [w[:k]] if w else []
    if ex == 'suf': return [w[-k:]] if w else []
    if ex == 'whole': return [w] if w else []
    if ex == 'infix': return [w[1:-1]] if len(w) > 2 else ([w] if w else [])
    if ex == 'bag':
        return list({w[i:i + k] for i in range(len(w) - k + 1)}) if len(w) >= k else ([w] if w else [])


def line_tokens(C, r, alph):
    """returns list of tokens (str) and agreement share per line"""
    red = reducer(r, alph)
    ucache = {}
    LU = []
    df = Counter()
    for L in C:
        us = []
        for w in L['words']:
            u = ucache.get(w)
            if u is None:
                u = ucache[w] = units(r, red(w))
            us.append(u)
        LU.append(us)
        for u in us: df.update(u)
    N = sum(df.values()) or 1
    lg = {x: math.log(N / v) for x, v in df.items()} if r['agg'] == 'idf' else None
    tie = r['tie']; mode2 = r['agg'] == 'mode2'
    toks, agree = [], []
    for us in LU:
        use = us[1:-1] if (r['agg'] == 'medial' and len(us) >= 3) else us
        cnt = {}; first = {}
        for i, u in enumerate(use):
            for x in u:
                cnt[x] = cnt.get(x, 0) + 1
                if x not in first: first[x] = i
        if not cnt:
            toks.append('<0>'); agree.append(0.0); continue
        if lg is not None:
            sc = {x: c * lg[x] for x, c in cnt.items()}
        else:
            sc = cnt
        if tie == 'common': key = lambda x: (-sc[x], -df[x], x)
        elif tie == 'rare': key = lambda x: (-sc[x], df[x], x)
        else: key = lambda x: (-sc[x], first[x], x)
        if mode2 and len(cnt) > 1:
            best = sorted(cnt, key=key)[:2]; tok = '+'.join(sorted(best)); b0 = best[0]
        else:
            b0 = min(cnt, key=key); tok = b0
        toks.append(tok); agree.append(cnt[b0] / max(1, len(use)))
    return toks, agree


# ---------------------------------------------------------------- scoring
def encode(toks):
    d = {}
    return np.array([d.setdefault(t, len(d)) for t in toks], dtype=np.int64), len(d)


def _stats(T, P, V):
    """T: token ids in reading order, P: page ids (same order). returns MI(off-diag), ASYM, REC, REP"""
    same = P[1:] == P[:-1]
    a = T[:-1][same]; b = T[1:][same]; pg = P[1:][same]
    rep = float(np.mean(a == b)) if len(a) else 0.0
    off = a != b
    a, b, pg = a[off], b[off], pg[off]
    n = len(a)
    if n < 5: return 0.0, 0.0, 0.0, rep
    code = a * V + b
    u, c = np.unique(code, return_counts=True)
    ca = np.bincount(a, minlength=V); cb = np.bincount(b, minlength=V)
    ua, ub = u // V, u % V
    mi = float(np.sum(c / n * np.log2(c * n / (ca[ua] * cb[ub]))))
    rev = ub * V + ua
    pos = np.searchsorted(u, rev); pos[pos >= len(u)] = 0
    crev = np.where(u[pos] == rev, c[pos], 0)
    m = ua < ub
    m2 = (ua > ub) & (crev == 0)
    asym = float(np.sum((c[m] - crev[m]) ** 2 / (c[m] + crev[m])) + np.sum(c[m2]))
    up = np.unique(code * 100000 + pg)
    cc = np.bincount(np.searchsorted(u, up // 100000), minlength=len(u))
    rec = float(np.sum(cc >= 2))
    return mi, asym, rec, rep


def score(toks, pages, nshuf=20, seed=0):
    T, V = encode(toks)
    pd = {}; P = np.array([pd.setdefault(p, len(pd)) for p in pages], dtype=np.int64)
    obs = _stats(T, P, V)
    rng = np.random.default_rng(seed)
    S = []
    for _ in range(nshuf):
        o = np.lexsort((rng.random(len(T)), P))
        # keep page order as in the text: P is non-decreasing in reading order only if pages are contiguous
        S.append(_stats(T[o], P[o], V))
    S = np.array(S)
    mu, sd = S.mean(0), S.std(0) + 1e-9
    z = (np.array(obs) - mu) / sd
    cnt = Counter(toks)
    return {'MI_z': float(z[0]), 'ASYM_z': float(z[1]), 'REC_z': float(z[2]),
            'REP_ratio': float(obs[3] / (mu[3] + 1e-9)), 'REP': obs[3],
            'types': len(cnt), 'ttr': len(cnt) / len(toks),
            'zipf': vlib.zipf_slope(toks, rmax=min(300, len(cnt))) if len(cnt) > 10 else 0.0,
            'LANG': float(z[0] + z[1] + z[2])}


def nmi(x, y):
    n = len(x); cx = Counter(x); cy = Counter(y); cxy = Counter(zip(x, y))
    hx = -sum(v / n * math.log(v / n) for v in cx.values())
    hy = -sum(v / n * math.log(v / n) for v in cy.values())
    mi = sum(v / n * math.log(v * n / (cx[a] * cy[b])) for (a, b), v in cxy.items())
    return mi / max(1e-9, math.sqrt(hx * hy))


def purity(toks, truth):
    """share of lines whose token's majority true word equals the line's true word"""
    by = defaultdict(Counter)
    for t, w in zip(toks, truth): by[t][w] += 1
    return sum(c.most_common(1)[0][1] for c in by.values()) / len(toks)


def row(fn, rid, method, result, verdict):
    with open(fn, 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')
