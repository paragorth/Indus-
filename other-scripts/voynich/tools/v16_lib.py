"""v16 'The spaces are lies': shared library.

A corpus is a set of lines; each line is a glyph string (ints) with the written word starts.
A segmentation is a boolean 'start' mask over the concatenated glyph stream (line starts always True).
Scores: two-part MDL (lexicon spelled with a flat glyph code + unigram token code), Zipf slope / R^2,
junction MI excess (last glyph of a unit -> first glyph of the next unit in the same line, over a
within-line unit shuffle), boundary F against a reference mask.
"""
import os, sys, math, random, re, unicodedata, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

RESDIR = os.path.join(vlib.RES, 'v16')
os.makedirs(RESDIR, exist_ok=True)
LOOPS = os.path.join(vlib.ROOT, 'loops')

P = np.uint64(0x9E3779B97F4A7C15 | 1)


class Corpus:
    def __init__(self, name, lines_syms, lines_starts, alphabet, truth=None):
        """lines_syms: list of lists of ints; lines_starts: list of lists of bools (written starts);
        truth: optional list of lists of bools (true word starts, for planted controls)."""
        self.name = name
        self.alphabet = alphabet
        self.A = len(alphabet)
        self.sym = np.array([s for L in lines_syms for s in L], dtype=np.int64)
        self.n = len(self.sym)
        self.line_id = np.array([i for i, L in enumerate(lines_syms) for _ in L], dtype=np.int64)
        self.pos = np.array([j for L in lines_syms for j in range(len(L))], dtype=np.int64)
        self.ls = self.pos == 0
        self.written = np.array([b for L in lines_starts for b in L], dtype=bool) | self.ls
        self.truth = None if truth is None else (np.array([b for L in truth for b in L], dtype=bool) | self.ls)
        self.nlines = len(lines_syms)
        self.line_len = np.bincount(self.line_id, minlength=self.nlines)
        # polynomial prefix hash (wrapping uint64)
        with np.errstate(over='ignore'):
            pw = np.ones(self.n + 1, dtype=np.uint64)
            pw = np.cumprod(np.full(self.n + 1, P, dtype=np.uint64))
            pw = np.concatenate([[np.uint64(1)], pw[:-1]])
            self.pw = pw
            h = np.zeros(self.n + 1, dtype=np.uint64)
            h[1:] = np.cumsum((self.sym.astype(np.uint64) + np.uint64(1)) * pw[:-1])
            self.h = h

    def subset(self, line_mask):
        """New corpus with only the chosen lines."""
        keep = np.flatnonzero(line_mask)
        syms, sts, tr = [], [], []
        starts = np.r_[0, np.cumsum(self.line_len)]
        for i in keep:
            a, b = starts[i], starts[i + 1]
            syms.append(self.sym[a:b].tolist()); sts.append(self.written[a:b].tolist())
            if self.truth is not None: tr.append(self.truth[a:b].tolist())
        return Corpus(self.name, syms, sts, self.alphabet, tr if self.truth is not None else None)


def tokens(c, st):
    s = np.flatnonzero(st)
    e = np.r_[s[1:], c.n]
    return s, e


def token_hash(c, s, e):
    """Hash of token spelling, position independent: sum_j (sym+1) P^(j - s)  via  (H[e]-H[s]) * P^(-s).
    P^(-s) is avoided by normalising with the hash of the line offset: we use (H[e]-H[s]) and a second
    hash on length; position-independence via multiplying by inverse powers precomputed lazily."""
    if not hasattr(c, 'ipw'):
        inv = pow(int(P), -1, 1 << 64)
        with np.errstate(over='ignore'):
            ip = np.cumprod(np.full(c.n + 1, np.uint64(inv), dtype=np.uint64))
            c.ipw = np.concatenate([[np.uint64(1)], ip[:-1]])
    with np.errstate(over='ignore'):
        hv = (c.h[e] - c.h[s]) * c.ipw[s]
        hv ^= (e - s).astype(np.uint64) * np.uint64(0x2545F4914F6CDD1D)
    return hv


def mdl(c, st, return_parts=False):
    s, e = tokens(c, st)
    hv = token_hash(c, s, e)
    u, idx, cnt = np.unique(hv, return_index=True, return_counts=True)
    L = (e - s)[idx]
    N = len(s)
    text = N * math.log2(N) - float(np.sum(cnt * np.log2(cnt)))
    lex = float(np.sum(L + 1)) * math.log2(c.A + 1)
    tot = text + lex
    if return_parts:
        return dict(total=tot, text=text, lex=lex, bpg=tot / c.n, ntok=int(N), ntype=int(len(u)),
                    mean_len=float(np.mean(e - s)), sd_len=float(np.std(e - s)), counts=cnt, hv=hv, u=u, idx=idx)
    return tot


# ---------- MDL2: adaptive (Chinese-restaurant) token code + lexicon spelled by an adaptive glyph-bigram code ----------
from scipy.special import gammaln as _gl
_LN2 = math.log(2)
_ALPHAS = np.exp(np.linspace(np.log(1.0), np.log(5000.0), 25))


def lex_bigram_bits(c, S, E):
    A = c.A; k = A + 1
    lens = E - S
    tot = int(lens.sum())
    base = np.repeat(S - np.r_[0, np.cumsum(lens)[:-1]], lens)
    pos = base + np.arange(tot)
    first = np.zeros(tot, bool); first[np.r_[0, np.cumsum(lens)[:-1]]] = True
    prev = np.where(first, A, c.sym[np.maximum(pos - 1, 0)])
    cur = c.sym[pos]
    prev = np.r_[prev, c.sym[E - 1]]; cur = np.r_[cur, np.full(len(E), A)]
    t = np.bincount(prev * k + cur, minlength=k * k).reshape(k, k).astype(float)
    row = t.sum(1)
    nats = np.sum(_gl(row + k / 2) - _gl(k / 2) - np.sum(_gl(t + 0.5), 1) + k * _gl(0.5))
    return float(nats / _LN2)


def crp_bits(cnt):
    N = float(cnt.sum()); K = len(cnt)
    sg = float(np.sum(_gl(cnt)))
    best = min(-(K * np.log(a) + _gl(a) - _gl(a + N) + sg) for a in _ALPHAS)
    return float(best / _LN2) + math.log2(len(_ALPHAS))


def mdl2(c, st, return_parts=False):
    s, e = tokens(c, st)
    hv = token_hash(c, s, e)
    u, idx, cnt = np.unique(hv, return_index=True, return_counts=True)
    text = crp_bits(cnt)
    lex = lex_bigram_bits(c, s[idx], e[idx])
    tot = text + lex
    if return_parts:
        return dict(total=tot, text=text, lex=lex, bpg=tot / c.n, ntok=int(len(s)), ntype=int(len(u)),
                    mean_len=float(np.mean(e - s)), sd_len=float(np.std(e - s)), counts=cnt)
    return tot


def zipf(cnt, rmax=1000):
    c = np.sort(cnt)[::-1][:rmax].astype(float)
    if len(c) < 10:
        return float('nan'), float('nan')
    r = np.arange(1, len(c) + 1)
    x, y = np.log(r), np.log(c)
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(b), float(r2)


def _mi(a, b, A):
    t = np.bincount(a * A + b, minlength=A * A).reshape(A, A).astype(float)
    n = t.sum()
    if n == 0: return 0.0
    pa = t.sum(1, keepdims=True) / n; pb = t.sum(0, keepdims=True) / n; p = t / n
    m = p > 0
    return float(np.sum(p[m] * np.log2(p[m] / (pa @ pb)[m])))


def junction(c, st, rng, nshuf=2):
    """Excess MI(last glyph of unit, first glyph of next unit in the same line) over within-line unit shuffle."""
    s, e = tokens(c, st)
    first = c.sym[s]; last = c.sym[e - 1]; lid = c.line_id[s]
    same = lid[1:] == lid[:-1]
    obs = _mi(last[:-1][same], first[1:][same], c.A)
    nul = []
    for _ in range(nshuf):
        key = lid + rng.random(len(s))
        o = np.argsort(key, kind='stable')
        f2, l2, d2 = first[o], last[o], lid[o]
        sm = d2[1:] == d2[:-1]
        nul.append(_mi(l2[:-1][sm], f2[1:][sm], c.A))
    return obs - float(np.mean(nul)), obs


def bf(st, ref, ls):
    """Boundary precision/recall/F over internal positions (line starts excluded)."""
    a = st & ~ls; b = ref & ~ls
    tp = int(np.sum(a & b)); pa = int(a.sum()); pb = int(b.sum())
    p = tp / pa if pa else 0.0; r = tp / pb if pb else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(c, st, rng, full=True):
    d = mdl2(c, st, return_parts=True)
    out = dict(bpg=d['bpg'], bpg_flat=mdl(c, st) / c.n, ntok=d['ntok'], ntype=d['ntype'], mean_len=d['mean_len'], sd_len=d['sd_len'])
    if full:
        out['zipf_slope'], out['zipf_r2'] = zipf(d['counts'])
        out['junc_ex'], out['junc_obs'] = junction(c, st, rng)
        out['bf_written'] = bf(st, c.written, c.ls)[2]
        if c.truth is not None:
            out['bf_truth'] = bf(st, c.truth, c.ls)[2]
    return out


# ---------------- corpora ----------------
def _letters(w):
    w = unicodedata.normalize('NFD', w.lower())
    return ''.join(ch for ch in w if 'a' <= ch <= 'z')


def from_words(name, lines_words, glyph_fn=list):
    alpha = {}
    syms, sts = [], []
    for ws in lines_words:
        s, t = [], []
        for w in ws:
            g = glyph_fn(w)
            for j, ch in enumerate(g):
                if ch not in alpha: alpha[ch] = len(alpha)
                s.append(alpha[ch]); t.append(j == 0)
        if s:
            syms.append(s); sts.append(t)
    inv = [None] * len(alpha)
    for k, v in alpha.items(): inv[v] = k
    return Corpus(name, syms, sts, inv)


def voynich(name='ZL3b'):
    L = vlib.load_voynich(name)
    return from_words('V-' + name, [l['words'] for l in L], vlib.glyphs)


def reference(key, nglyph):
    lines = vlib.load_ref(key, skip_frac=0.05)
    out, tot = [], 0
    for l in lines:
        ws = [x for x in (_letters(w) for w in l['words']) if x]
        if not ws: continue
        out.append(ws); tot += sum(len(w) for w in ws)
        if tot >= nglyph: break
    return from_words(key, out)


def planted(c, rule, seed=0, name=None):
    """Delete spaces of a true-spaced corpus and re-space by a planted mechanical rule.
    The true word starts are kept as c.truth."""
    st = apply_rule(c, rule, np.random.default_rng(seed))
    syms, sts, tr = [], [], []
    starts = np.r_[0, np.cumsum(c.line_len)]
    for i in range(c.nlines):
        a, b = starts[i], starts[i + 1]
        syms.append(c.sym[a:b].tolist()); sts.append(st[a:b].tolist()); tr.append(c.written[a:b].tolist())
    return Corpus(name or (c.name + '|' + rule['kind']), syms, sts, c.alphabet, tr)


def shuffle_within_line(c, seed=0):
    """Null: glyphs shuffled inside each line, written space positions kept."""
    rng = np.random.default_rng(seed)
    syms, sts = [], []
    starts = np.r_[0, np.cumsum(c.line_len)]
    for i in range(c.nlines):
        a, b = starts[i], starts[i + 1]
        s = c.sym[a:b].copy(); rng.shuffle(s)
        syms.append(s.tolist()); sts.append(c.written[a:b].tolist())
    return Corpus(c.name + '|glyphshuf', syms, sts, c.alphabet)


def markov2(c, seed=0):
    """Null: order-2 Markov chain over glyphs + space, trained on whole lines (BOL/EOL), regenerated with
    the same number of lines. Its spaces come from a purely local rule."""
    rng = random.Random(seed)
    SP, EOL, BOL = c.A, c.A + 1, c.A + 2
    from collections import defaultdict, Counter
    tr = defaultdict(Counter)
    starts = np.r_[0, np.cumsum(c.line_len)]
    seqs = []
    for i in range(c.nlines):
        a, b = starts[i], starts[i + 1]
        q = []
        for j in range(a, b):
            if j > a and c.written[j]: q.append(SP)
            q.append(int(c.sym[j]))
        seqs.append(q)
        s = [BOL, BOL] + q + [EOL]
        for k in range(2, len(s)): tr[(s[k - 2], s[k - 1])][s[k]] += 1
    tabs = {k: (list(v.keys()), list(v.values())) for k, v in tr.items()}
    syms, sts = [], []
    for q in seqs:
        while True:
            ctx = (BOL, BOL); out = []
            while len(out) < 200:
                ks, vs = tabs[ctx]; x = rng.choices(ks, vs)[0]
                if x == EOL: break
                out.append(x); ctx = (ctx[1], x)
            s, t, nxt = [], [], True
            for x in out:
                if x == SP: nxt = True; continue
                s.append(x); t.append(nxt); nxt = False
            if s: break
        syms.append(s); sts.append(t)
    return Corpus(c.name + '|markov2', syms, sts, c.alphabet)


# ---------------- rule grammar ----------------
def _enforce_min(c, st, m):
    if m <= 1: return st
    st = st.copy()
    for _ in range(3):
        s = np.flatnonzero(st)
        gap_prev = np.r_[m, np.diff(s)]
        bad = (gap_prev < m) & ~c.ls[s]
        if not bad.any(): break
        # drop every other offending start to avoid cascades
        st[s[bad][::2]] = False
    return st


def apply_rule(c, rule, rng):
    k = rule['kind']
    if k == 'written':
        st = c.written.copy()
    elif k == 'before':
        st = np.isin(c.sym, rule['X']) | c.ls
    elif k == 'after':
        st = np.zeros(c.n, bool); st[1:] = np.isin(c.sym[:-1], rule['Y']); st |= c.ls
    elif k == 'beforeafter':
        st = np.isin(c.sym, rule['X']); st[1:] |= np.isin(c.sym[:-1], rule['Y']); st |= c.ls
    elif k == 'pairs':
        M = np.zeros((c.A, c.A), bool)
        for a, b in rule['S']: M[a, b] = True
        st = np.zeros(c.n, bool); st[1:] = M[c.sym[:-1], c.sym[1:]]; st |= c.ls
    elif k == 'every':
        st = ((c.pos - rule['phase']) % rule['k'] == 0) | c.ls
    elif k == 'shift':
        s = np.flatnonzero(c.written & ~c.ls)
        t = s + rule['d']
        ok = (t >= 0) & (t < c.n)
        t = t[ok]; ok2 = c.line_id[t] == c.line_id[s[ok]]
        st = c.ls.copy(); st[t[ok2]] = True
    elif k == 'jitter':
        s = np.flatnonzero(c.written & ~c.ls)
        d = rng.choice([-1, 1], size=len(s)) * (rng.random(len(s)) < rule['p'])
        t = s + d; t = np.clip(t, 0, c.n - 1)
        ok = c.line_id[t] == c.line_id[s]
        st = c.ls.copy(); st[t[ok]] = True; st[s[~ok]] = True
    elif k == 'dropins':
        st = c.written.copy()
        st &= ~(rng.random(c.n) < rule['q']) | c.ls
        st |= rng.random(c.n) < rule['r']
    elif k == 'random':
        st = (rng.random(c.n) < rule['r']) | c.ls
    elif k == 'keep_after':      # keep a written space only if the glyph before it is in Y
        st = c.written.copy(); st[1:] &= np.isin(c.sym[:-1], rule['Y']); st |= c.ls
    elif k == 'keep_before':     # keep a written space only if the glyph after it is in X
        st = c.written & np.isin(c.sym, rule['X']) | c.ls
    elif k == 'plus_before':     # written spaces plus a break before every glyph in X
        st = c.written | np.isin(c.sym, rule['X'])
    elif k == 'plus_after':
        st = c.written.copy(); st[1:] |= np.isin(c.sym[:-1], rule['Y']); st |= c.ls
    else:
        raise ValueError(k)
    return _enforce_min(c, st, rule.get('m', 1))


def random_rule(c, rng, written_pairs=None):
    kinds = ['before', 'after', 'beforeafter', 'pairs', 'every', 'shift', 'jitter', 'dropins', 'random',
             'keep_after', 'keep_before', 'plus_before', 'plus_after']
    k = kinds[rng.integers(len(kinds))]
    r = {'kind': k, 'm': int(rng.choice([1, 1, 2, 3]))}
    A = c.A
    def subset():
        p = rng.uniform(0.05, 0.4)
        X = [a for a in range(A) if rng.random() < p]
        return X or [int(rng.integers(A))]
    if k in ('before', 'keep_before', 'plus_before'): r['X'] = subset()
    elif k in ('after', 'keep_after', 'plus_after'): r['Y'] = subset()
    elif k == 'beforeafter': r['X'] = subset(); r['Y'] = subset()
    elif k == 'pairs':
        # random set of glyph bigrams, density drawn widely; half the time biased toward bigrams that
        # occur in the text (the plausible mechanical 'break between a and b' rule)
        dens = rng.uniform(0.02, 0.4)
        if written_pairs is not None and rng.random() < 0.5:
            cand = written_pairs
        else:
            cand = [(a, b) for a in range(A) for b in range(A)]
        r['S'] = [p for p in cand if rng.random() < dens]
    elif k == 'every': r['k'] = int(rng.integers(2, 9)); r['phase'] = int(rng.integers(0, r['k']))
    elif k == 'shift': r['d'] = int(rng.choice([-3, -2, -1, 1, 2, 3]))
    elif k == 'jitter': r['p'] = float(rng.uniform(0.05, 0.8))
    elif k == 'dropins': r['q'] = float(rng.uniform(0, 0.5)); r['r'] = float(rng.uniform(0, 0.15))
    elif k == 'random': r['r'] = float(rng.uniform(0.08, 0.4))
    return r


def describe(c, r):
    k = r['kind']; a = c.alphabet
    if k == 'before': return f"before{{{''.join(a[x] for x in r['X'])}}} m{r['m']}"
    if k == 'after': return f"after{{{''.join(a[x] for x in r['Y'])}}} m{r['m']}"
    if k == 'beforeafter': return f"before{{{''.join(a[x] for x in r['X'])}}}+after{{{''.join(a[x] for x in r['Y'])}}} m{r['m']}"
    if k in ('keep_before', 'plus_before'): return f"{k}{{{''.join(a[x] for x in r['X'])}}} m{r['m']}"
    if k in ('keep_after', 'plus_after'): return f"{k}{{{''.join(a[x] for x in r['Y'])}}} m{r['m']}"
    if k == 'pairs': return f"pairs[{len(r['S'])}] m{r['m']}"
    if k == 'every': return f"every{r['k']}@{r['phase']} m{r['m']}"
    if k == 'shift': return f"shift{r['d']:+d} m{r['m']}"
    if k == 'jitter': return f"jitter p{r['p']:.2f} m{r['m']}"
    if k == 'dropins': return f"drop{r['q']:.2f}/ins{r['r']:.2f} m{r['m']}"
    if k == 'random': return f"random r{r['r']:.2f} m{r['m']}"
    return k


def append_rows(fname, rows, header=None):
    path = os.path.join(LOOPS, fname)
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new and header: f.write(header + '\n')
        for r in rows: f.write(r + '\n')
