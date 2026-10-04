"""v23 DOES THE TEXT KNOW WHICH WAY TIME FLOWS?  Shared library.

A corpus is a list of pages; a page is a list of paragraphs; a paragraph a list of lines; a line a
list of words; a word a string of single-character glyph units (Voynich: EVA with ch/sh/benched
gallows merged, vlib.glyphs).  The full time reversal R reverses everything: page order is kept
(pages are the null's exchangeable blocks) but inside a page paragraphs, lines, words and glyphs
are all read backwards.

Signed signatures give one contribution a_p per page, a_p = s(page as written) - s(R page), sign
convention: POSITIVE = the text as written is the 'forward' (more predictable, or more 'X') reading.
Null 1 (exact matching, 'time-reversed text'): each page is read forwards or backwards by coin
flip; equivalently the a_p are given random signs (sign-flip test).  Null 2 (calibration): text
generated from a fitted REVERSIBLE Markov chain (detailed balance by construction) run through the
same pipeline; z-scores there must look N(0,1).

Unsigned signatures (entropy production: KL between the lag-k pair law and its transpose,
ordinal-pattern and visibility-graph irreversibility) are compared with the same statistic on the
random-page-direction surrogate (exact matching) and reported as excess over that null.
"""
import os, sys, math, random, json, re, lzma, zlib
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

ROOT = vlib.ROOT
DATA = vlib.DATA
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'v23_ckpt'); os.makedirs(CK, exist_ok=True)
LN2 = math.log(2)
SEP_W, SEP_L, SEP_P = ' ', '|', '#'


# ------------------------------------------------------------------ corpora
def U(w):
    return ''.join(vlib.glyphs(w))


def voynich(name='ZL3b'):
    lines = vlib.load_voynich(name, drop_uncertain=True)
    pages, order, cur = {}, [], None
    for L in lines:
        f = L['folio']
        if f not in pages:
            pages[f] = []; order.append(f); cur = None
        ws = [U(w) for w in L['words']]
        ws = [w for w in ws if w and '?' not in w and '*' not in w]
        if not ws: continue
        if L['para_start'] or cur is None:
            cur = []; pages[f].append(cur)
        cur.append(ws)
        if L.get('para_end'): cur = None
    return [pages[f] for f in order if sum(len(l) for pa in pages[f] for l in pa) >= 20]


def paginate(lines, page_tok=160, max_tok=None):
    """lines: [{'words','para_start'}] -> pages of ~page_tok tokens, paragraphs kept inside pages."""
    pages, page, para, nt, tot = [], [], None, 0, 0
    for L in lines:
        ws = L['words']
        if not ws: continue
        if L.get('para_start') or para is None:
            para = []; page.append(para)
        para.append(list(ws)); nt += len(ws); tot += len(ws)
        if nt >= page_tok:
            pages.append(page); page, para, nt = [], None, 0
        if max_tok and tot >= max_tok: break
    if page and nt >= 20: pages.append(page)
    return pages


def wrap(words, width=42):
    out, cur, cw = [], [], 0
    for w in words:
        if cur and cw + len(w) + 1 > width:
            out.append(cur); cur, cw = [], 0
        cur.append(w); cw += len(w) + 1
    if cur: out.append(cur)
    return out


def ref(key, max_tok=19000):
    return paginate(vlib.load_ref(key), max_tok=max_tok)


def czech(max_tok=19000):
    txt = open(os.path.join(DATA, 'plain', 'cs.txt'), encoding='utf-8').read().split('\n')
    lines, prev_blank = [], True
    for s in txt:
        ws = re.findall(r'[^\W\d_]+', s.lower())
        if not ws: prev_blank = True; continue
        lines.append({'words': ws, 'para_start': prev_blank}); prev_blank = False
    return paginate(lines, max_tok=max_tok)


def hebrew(max_tok=19000):
    txt = open(os.path.join(CK, 'he_mishneh_torah.txt'), encoding='utf-8').read().split('\n')
    lines = []
    for s in txt:
        ws = re.findall(r'[א-ת]+', s)
        for i, l in enumerate(wrap(ws, 42)):
            lines.append({'words': l, 'para_start': i == 0})
    return paginate(lines, max_tok=max_tok)


def ntok(C):
    return sum(len(l) for p in C for pa in p for l in pa)


def subsample_pages(C, max_tok, seed=0):
    idx = list(range(len(C))); random.Random(seed).shuffle(idx)
    keep, t = set(), 0
    for i in idx:
        if t >= max_tok: break
        keep.add(i); t += sum(len(l) for pa in C[i] for l in pa)
    return [C[i] for i in sorted(keep)]


# ------------------------------------------------------------------ reversal and streams
def rev_page(p):
    return [[[w[::-1] for w in l[::-1]] for l in pa[::-1]] for pa in p[::-1]]


def rev_words_only(p):
    """word order reversed at every level, glyphs inside words untouched"""
    return [[[w for w in l[::-1]] for l in pa[::-1]] for pa in p[::-1]]


def stream(p):
    """page -> glyph stream with separators (string)"""
    return SEP_P.join(SEP_L.join(SEP_W.join(l) for l in pa) for pa in p)


def wstream(p):
    out = []
    for i, pa in enumerate(p):
        if i: out.append('<P>')
        for j, l in enumerate(pa):
            if j: out.append('<L>')
            out.extend(l)
    return out


def tokens(p):
    return [w for pa in p for l in pa for w in l]


def page_lines(p):
    return [l for pa in p for l in pa]


# ------------------------------------------------------------------ surrogates
def _from_stream(s, page_tok):
    lines = []
    for pi, para in enumerate(s.split(SEP_P)):
        for li, l in enumerate(para.split(SEP_L)):
            ws = [w for w in l.split(SEP_W) if w]
            if ws: lines.append({'words': ws, 'para_start': li == 0})
    return paginate(lines, page_tok=page_tok)


def _chain_generate(trans, start, n, rng):
    """trans: dict ctx -> (symbols list, cumprobs array)"""
    out = list(start); k = len(start)
    for _ in range(n):
        ctx = tuple(out[-k:]) if k else ()
        if ctx not in trans:
            ctx = next(iter(trans))
        syms, cp = trans[ctx]
        out.append(syms[int(np.searchsorted(cp, rng.random() * cp[-1], side='right'))])
    return out


def reversible_glyph(C, seed):
    """order-1 reversible chain on the glyph stream (symmetrised bigram counts)."""
    rng = random.Random(seed)
    s = SEP_P.join(stream(p) for p in C)
    cnt = Counter(zip(s, s[1:]))
    sym = Counter()
    for (a, b), c in cnt.items():
        sym[(a, b)] += c; sym[(b, a)] += c
    rows = defaultdict(list)
    for (a, b), c in sym.items(): rows[a].append((b, c))
    trans = {(a,): ([b for b, _ in r], np.cumsum([c for _, c in r]).astype(float)) for a, r in rows.items()}
    out = _chain_generate(trans, [s[0]], len(s), rng)
    ptok = max(40, ntok(C) // len(C))
    return _from_stream(''.join(out), ptok)


def reversible_word(C, seed):
    """order-1 reversible chain on word tokens incl. line/para break tokens."""
    rng = random.Random(seed)
    s = []
    for p in C:
        s += wstream(p) + ['<P>']
    cnt = Counter(zip(s, s[1:]))
    sym = Counter()
    for (a, b), c in cnt.items():
        sym[(a, b)] += c; sym[(b, a)] += c
    rows = defaultdict(list)
    for (a, b), c in sym.items(): rows[a].append((b, c))
    trans = {(a,): ([b for b, _ in r], np.cumsum([c for _, c in r]).astype(float)) for a, r in rows.items()}
    out = _chain_generate(trans, [s[0]], len(s), rng)
    st = ''.join(SEP_P if t == '<P>' else SEP_L if t == '<L>' else (SEP_W + t + SEP_W) for t in out)
    st = re.sub(' +', ' ', st)
    return _from_stream(st, max(40, ntok(C) // len(C)))


def markov_glyph(C, seed, k=3):
    """ordinary (non-reversible) order-k glyph chain resynthesis."""
    rng = random.Random(seed)
    s = SEP_P.join(stream(p) for p in C)
    rows = defaultdict(Counter)
    for i in range(k, len(s)): rows[tuple(s[i - k:i])][s[i]] += 1
    trans = {c: (list(r.keys()), np.cumsum(list(r.values())).astype(float)) for c, r in rows.items()}
    out = _chain_generate(trans, list(s[:k]), len(s), rng)
    return _from_stream(''.join(out), max(40, ntok(C) // len(C)))


def markov_word(C, seed):
    rng = random.Random(seed)
    s = []
    for p in C: s += wstream(p) + ['<P>']
    rows = defaultdict(Counter)
    for a, b in zip(s, s[1:]): rows[(a,)][b] += 1
    trans = {c: (list(r.keys()), np.cumsum(list(r.values())).astype(float)) for c, r in rows.items()}
    out = _chain_generate(trans, [s[0]], len(s), rng)
    st = ''.join(SEP_P if t == '<P>' else SEP_L if t == '<L>' else (SEP_W + t + SEP_W) for t in out)
    st = re.sub(' +', ' ', st)
    return _from_stream(st, max(40, ntok(C) // len(C)))


# ------------------------------------------------------------------ Kneser-Ney n-gram
class KN:
    def __init__(self, n, D=0.75):
        self.n, self.D = n, D

    def fit(self, seqs):
        n = self.n
        self.c = [Counter() for _ in range(n + 1)]     # c[k][(ctx..., w)] k-gram counts (highest uses raw)
        self.ctx = [Counter() for _ in range(n + 1)]
        self.typ = [Counter() for _ in range(n + 1)]   # number of distinct followers per ctx
        V = set()
        for s in seqs:
            t = ['<s>'] * (n - 1) + list(s) + ['</s>']
            V.update(t)
            for i in range(n - 1, len(t)):
                g = tuple(t[i - n + 1:i + 1])
                self.c[n][g] += 1
        # lower order k-gram continuation count = number of distinct left extensions
        for k in range(n - 1, 0, -1):
            src = self.c[k + 1]
            for g in src:
                self.c[k][g[1:]] += 1
        for k in range(1, n + 1):
            for g, v in self.c[k].items():
                self.ctx[k][g[:-1]] += v; self.typ[k][g[:-1]] += 1
        self.V = len(V) + 1
        return self

    def prob(self, hist, w):
        p = 1.0 / self.V
        D = self.D
        for k in range(1, self.n + 1):
            ctx = tuple(hist[len(hist) - (k - 1):]) if k > 1 else ()
            cc = self.ctx[k].get(ctx, 0)
            if cc == 0: continue
            cw = self.c[k].get(ctx + (w,), 0)
            p = max(cw - D, 0) / cc + D * self.typ[k][ctx] / cc * p
        return p

    def bits(self, s):
        n = self.n
        t = ['<s>'] * (n - 1) + list(s) + ['</s>']
        tot = 0.0
        for i in range(n - 1, len(t)):
            tot -= math.log2(self.prob(t[i - n + 1:i], t[i]))
        return tot, len(t) - n + 1


def kn_arrow(seqs_by_page, n, nfold=5, seed=0, rev=lambda s: s[::-1], unk_min=0):
    """seqs_by_page: list (pages) of list of sequences. Returns per-page (bits_fwd, bits_bwd, nsym)."""
    P = len(seqs_by_page)
    idx = list(range(P)); random.Random(seed).shuffle(idx)
    fold = {p: i % nfold for i, p in enumerate(idx)}
    out = [None] * P
    for f in range(nfold):
        tr = [s for p in range(P) if fold[p] != f for s in seqs_by_page[p]]
        if unk_min:
            cnt = Counter(x for s in tr for x in s)
            m = lambda s: [x if cnt[x] >= unk_min else '<unk>' for x in s]
            tr = [m(s) for s in tr]
        else:
            m = lambda s: s
        Mf = KN(n).fit(tr)
        Mb = KN(n).fit([rev(s) for s in tr])
        for p in range(P):
            if fold[p] != f: continue
            bf = bb = ns = 0
            for s in seqs_by_page[p]:
                s2 = m(s)
                a, k = Mf.bits(s2); b, _ = Mb.bits(rev(s2))
                bf += a; bb += b; ns += k
            out[p] = (bf, bb, ns)
    return out


# ------------------------------------------------------------------ helpers for signatures
def gfreq(C):
    return Counter(w for p in C for w in tokens(p))


def ed1(a, b):
    """edit distance exactly 1 (sub/ins/del)"""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or a == b: return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb: a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]: i += 1
    return a[i:] == b[i + 1:]


def kl_asym(cnt, eps=0.5):
    """KL( P(a,b) || P(b,a) ) in bits for a pair Counter, add-eps on the support union"""
    keys = set(cnt) | {(b, a) for a, b in cnt}
    tot = sum(cnt.values()) + eps * len(keys)
    s = 0.0
    for (a, b) in keys:
        p = (cnt.get((a, b), 0) + eps) / tot
        q = (cnt.get((b, a), 0) + eps) / tot
        s += p * math.log2(p / q)
    return s


def sgn(x):
    return (x > 0) - (x < 0)


def ordinal_counts(series_list):
    c = Counter()
    for x in series_list:
        for i in range(len(x) - 2):
            c[(sgn(x[i + 1] - x[i]), sgn(x[i + 2] - x[i + 1]))] += 1
    return c


def ordinal_irrev(c, eps=0.5):
    """KL between pattern law and the law of reversed patterns: (s1,s2)->(-s2,-s1)"""
    keys = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    tot = sum(c.values()) + eps * 9
    s = 0.0
    for k in keys:
        r = (-k[1], -k[0])
        p = (c.get(k, 0) + eps) / tot; q = (c.get(r, 0) + eps) / tot
        s += p * math.log2(p / q)
    return s


def hvg_degrees(x):
    """horizontal visibility graph: out-degree (to future) and in-degree (from past) per node"""
    n = len(x); kout = [0] * n; kin = [0] * n
    for i in range(n):
        mx = -1e18
        for j in range(i + 1, n):
            if mx < min(x[i], x[j]):
                kout[i] += 1; kin[j] += 1
            mx = max(mx, x[j])
            if x[j] >= x[i]: break
    return kout, kin


def hvg_irrev(series_list, kmax=12, eps=0.5):
    co, ci = Counter(), Counter()
    for x in series_list:
        if len(x) < 3: continue
        ko, ki = hvg_degrees(x)
        co.update(min(k, kmax) for k in ko); ci.update(min(k, kmax) for k in ki)
    keys = range(0, kmax + 1)
    to = sum(co.values()) + eps * len(keys); ti = sum(ci.values()) + eps * len(keys)
    return sum(((co[k] + eps) / to) * math.log2(((co[k] + eps) / to) / ((ci[k] + eps) / ti)) for k in keys)


# ------------------------------------------------------------------ statistics
def signflip(a, nperm=20000, seed=0):
    a = np.asarray([x for x in a if x is not None], float)
    obs = a.sum()
    rng = np.random.default_rng(seed)
    S = (rng.choice([-1.0, 1.0], size=(nperm, len(a))) * a).sum(1)
    p = (1 + np.sum(np.abs(S) >= abs(obs) - 1e-12)) / (nperm + 1)
    z = obs / math.sqrt((a ** 2).sum()) if (a ** 2).sum() > 0 else 0.0
    return z, p


def bh(ps, q=0.05):
    ps = np.asarray(ps); n = len(ps); o = np.argsort(ps)
    thr = 0;
    for r, i in enumerate(o, 1):
        if ps[i] <= q * r / n: thr = r
    keep = np.zeros(n, bool); keep[o[:thr]] = True
    return keep


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')
