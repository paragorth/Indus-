"""v73 ONE WORD PER LINE IS THE MESSAGE, THE REST IS SETTING: shared library.

A corpus is a list of pages (v72 format: dict(id, sec, lines=[dict(w=[words], ps=bool)])), words are glyph-unit
strings (vlib units: C=ch S=sh T=cth K=ckh P=cph F=cfh).

A MARKING RULE picks exactly one word per line: per-word score = sum_f w_f * feature_f (+ a tiny
first-position tie-break), argmax within each line. Features are computed from the corpus itself
(position, length, frequency, glyph content, neighbours, line mode, page uniqueness), never from a payload.

The extracted ITEM STREAM E (one token per line) is scored on a page mask against R, the stream of a random
word per line from the same lines (mean of 4 draws), so every statistic is "beyond the line mates":
  PI  within-page cache gain (bits/token): leave-one-line-out page cache vs leave-page-out unigram of the stream
  XP  neighbour recurrence: P(token type occurs in the stream of pages +-1..2) - P(... of 4 far pages, same section)
  VS  separation: Jensen-Shannon divergence (bits) between the stream's type law and the non-selected words' law
  TT  type/token ratio of the stream (equal token counts for E and R)
Each statistic enters as D = stat(E) - stat(R). Nulls are the same rule bank on generator text fitted to the
corpus (MK2 glyph trigram, SELFCIT copy-and-vary, SC10, JUNC) and on the corpus with words shuffled across lines
within the page (LSHUF).
"""
import os, sys, json, math, random, re, zlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v73_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V72


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write('| %s | %s | %s | %s |\n' % (rid, method, result, verdict))


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'))


def jload(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ corpus -> flat arrays
GALL = set('ktpf'); BENCH = set('TKPF')
MODECLS = {}
for c in 'CSkKtTpPfF': MODECLS[c] = 0
for c in 'oel': MODECLS[c] = 1
for c in 'qdsy': MODECLS[c] = 2
for c in 'ai': MODECLS[c] = 3
for c in 'rnm': MODECLS[c] = 4

FEATS = ['pos0', 'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'rpos0', 'rpos1', 'rpos2', 'posn', 'len', 'logf', 'pguniq',
         'hapax', 'has_k', 'has_t', 'has_p', 'has_f', 'bench', 'has_C', 'has_S', 'st_q', 'st_o', 'st_d', 'st_CS',
         'st_gall', 'en_y', 'en_n', 'en_l', 'en_r', 'en_mg', 'modemis', 'glyrar', 'nrare', 'ndist', 'secspec',
         'n_a', 'irun', 'nobigr', 'centre'] + ['aft%d' % i for i in range(10)] + ['bef%d' % i for i in range(10)]
NF = len(FEATS)


class Corpus:
    def __init__(self, pages, name='x'):
        self.name = name; self.pages = pages
        W, PG, LN, POS, RP, LL = [], [], [], [], [], []
        li = 0; self.line_page = []; self.line_start = []
        for pi, p in enumerate(pages):
            for l in p['lines']:
                n = len(l['w'])
                if n == 0: continue
                self.line_start.append(len(W)); self.line_page.append(pi)
                for i, w in enumerate(l['w']):
                    W.append(w); PG.append(pi); LN.append(li); POS.append(i); RP.append(n - 1 - i); LL.append(n)
                li += 1
        self.words = W; self.nl = li; self.npg = len(pages)
        self.page = np.array(PG); self.line = np.array(LN); self.pos = np.array(POS); self.rpos = np.array(RP)
        self.llen = np.array(LL); self.line_start = np.array(self.line_start); self.line_page = np.array(self.line_page)
        voc = {}; self.tok = np.array([voc.setdefault(w, len(voc)) for w in W]); self.voc = voc; self.V = len(voc)
        self.sec = np.array([zlib.crc32(str(p['sec']).encode()) % 1000 for p in pages])
        self.half = np.array([V72.leaf_half(p['id']) for p in pages])
        self._feats()
        self._neigh()

    def _feats(self):
        W = self.words; N = len(W)
        F = np.zeros((N, NF), dtype=np.float32)
        cnt = np.bincount(self.tok, minlength=self.V)
        gl = Counter(c for w in W for c in w); gt = sum(gl.values())
        gfreq = {c: gl[c] / gt for c in gl}
        pgcnt = Counter(zip(self.page.tolist(), self.tok.tolist()))
        seccnt = Counter(zip(self.sec[self.page].tolist(), self.tok.tolist())); secn = Counter(self.sec[self.page].tolist())
        top = [w for w, _ in Counter(W).most_common(10)]
        topi = {w: i for i, w in enumerate(top)}
        # line mode class (majority first-glyph class of the line)
        fcls = np.array([MODECLS.get(w[0], 5) for w in W])
        mode = np.zeros(self.nl, dtype=int)
        for l in range(self.nl):
            s = self.line_start[l]; e = self.line_start[l + 1] if l + 1 < self.nl else N
            mode[l] = Counter(fcls[s:e].tolist()).most_common(1)[0][0]
        for j, w in enumerate(W):
            f = F[j]; p = self.pos[j]; r = self.rpos[j]; n = self.llen[j]
            if p < 6: f[p] = 1
            if r < 3: f[6 + r] = 1
            f[9] = p / max(1, n - 1)
            f[10] = len(w); f[11] = math.log(cnt[self.tok[j]]); f[12] = pgcnt[(self.page[j], self.tok[j])] == 1
            f[13] = cnt[self.tok[j]] <= 2
            f[14] = 'k' in w; f[15] = 't' in w; f[16] = 'p' in w; f[17] = 'f' in w; f[18] = any(c in BENCH for c in w)
            f[19] = 'C' in w; f[20] = 'S' in w; f[21] = w[0] == 'q'; f[22] = w[0] == 'o'; f[23] = w[0] == 'd'
            f[24] = w[0] in 'CS'; f[25] = w[0] in GALL or w[0] in BENCH or (len(w) > 1 and w[0] in 'qo' and (w[1] in GALL or w[1] in BENCH))
            f[26] = w[-1] == 'y'; f[27] = w[-1] == 'n'; f[28] = w[-1] == 'l'; f[29] = w[-1] == 'r'; f[30] = w[-1] in 'mg'
            f[31] = fcls[j] != mode[self.line[j]]
            rr = [-math.log(gfreq[c]) for c in w]; f[32] = max(rr); f[33] = sum(gfreq[c] < 0.005 for c in w)
            f[34] = len(set(w))
            s = self.sec[self.page[j]]
            f[35] = math.log((seccnt[(s, self.tok[j])] + 0.1) / secn[s]) - math.log((cnt[self.tok[j]] + 0.1) / N)
            f[36] = w.count('a'); m = re.findall(r'i+', w); f[37] = max((len(x) for x in m), default=0)
            f[39] = -abs(f[9] - 0.5)
            if p > 0 and W[j - 1] in topi: f[40 + topi[W[j - 1]]] = 1
            if r > 0 and W[j + 1] in topi: f[50 + topi[W[j + 1]]] = 1
        # no shared glyph bigram with line mates
        for l in range(self.nl):
            s = self.line_start[l]; e = self.line_start[l + 1] if l + 1 < self.nl else N
            bg = [set(W[k][i:i + 2] for i in range(len(W[k]) - 1)) for k in range(s, e)]
            for a in range(e - s):
                others = set().union(*[bg[b] for b in range(e - s) if b != a]) if e - s > 1 else set()
                F[s + a, 38] = len(bg[a] & others) == 0
        # standardise (argmax is invariant to per-line constants; this only makes weights comparable)
        mu = F.mean(0); sd = F.std(0); sd[sd == 0] = 1
        self.F = (F - mu) / sd
        self.FT = np.ascontiguousarray(self.F.T.astype(np.float64))

    def _neigh(self):
        """page adjacency (+-1..2 in book order) and far pages (4 of the same section at distance >= 10)."""
        n = self.npg; rng = random.Random(73)
        rn, cn, rf, cf = [], [], [], []
        for p in range(n):
            for d in (-2, -1, 1, 2):
                q = p + d
                if 0 <= q < n: rn.append(p); cn.append(q)
            far = [q for q in range(n) if abs(q - p) >= 10 and self.sec[q] == self.sec[p]]
            if len(far) < 4: far = [q for q in range(n) if abs(q - p) >= 10]
            for q in rng.sample(far, min(4, len(far))): rf.append(p); cf.append(q)
        self.nb = np.full((n, 4), -1); self.far = np.full((n, 4), -1)
        for p, q in zip(rn, cn): self.nb[p, list(self.nb[p]).index(-1)] = q
        for p, q in zip(rf, cf): self.far[p, list(self.far[p]).index(-1)] = q

    # --------------------------------------------------------------- selection
    def select(self, w):
        """w: dense weight vector (NF,) -> index of the selected word in each line."""
        s = -1e-6 * self.pos
        for f in np.nonzero(w)[0]: s = s + float(w[f]) * self.FT[f]
        mx = np.maximum.reduceat(s, self.line_start)
        hit = s >= mx[self.line] - 1e-12
        idx = np.full(self.nl, -1); cand = np.nonzero(hit)[0]
        # first hit per line
        ln = self.line[cand]; first = np.ones(len(cand), bool); first[1:] = ln[1:] != ln[:-1]
        idx[ln[first]] = cand[first]
        return idx

    def random_select(self, seed):
        rng = np.random.default_rng(seed)
        return self.line_start + (rng.random(self.nl) * self.llen[self.line_start]).astype(int)


# ------------------------------------------------------------------ stream statistics
def stream_stats(C, idx, mask_pages):
    """idx: selected word index per line. mask_pages: bool per page. Returns dict PI, XP, VS, TT."""
    lm = mask_pages[C.line_page]
    sel = idx[lm]; t = C.tok[sel]; pg = C.line_page[lm]
    n = len(t)
    # PI: within-page cache gain
    key = pg.astype(np.int64) * C.V + t
    o = np.argsort(key, kind='stable'); ks = key[o]
    st = np.r_[True, ks[1:] != ks[:-1]]; gid = np.cumsum(st) - 1; gc = np.bincount(gid)
    kcount = np.empty(n, np.int64); kcount[o] = gc[gid]
    cpt = kcount - 1
    npg = np.bincount(pg, minlength=C.npg)[pg] - 1
    tc = np.bincount(t, minlength=C.V)[t]
    base = (tc - kcount + 0.5) / (n - npg - 1 + 0.5 * C.V)
    lam = 0.2
    cache = np.where(npg > 0, cpt / np.maximum(npg, 1), 0.0)
    PI = float(np.mean(np.log2(lam * cache + (1 - lam) * base) - np.log2(base)))
    # XP: neighbour vs far recurrence, using the stream on ALL pages as the reference
    H = np.zeros(C.npg * C.V + 1, bool); H[C.line_page.astype(np.int64) * C.V + C.tok[idx]] = True
    a = np.zeros(n, bool); b = np.zeros(n, bool)
    for q in C.nb.T:
        qq = q[pg]; a |= (qq >= 0) & H[np.where(qq >= 0, qq, 0).astype(np.int64) * C.V + t]
    for q in C.far.T:
        qq = q[pg]; b |= (qq >= 0) & H[np.where(qq >= 0, qq, 0).astype(np.int64) * C.V + t]
    XP = float(a.mean() - b.mean())
    # VS: JSD between stream law and non-selected law (mask lines)
    allw = np.nonzero(lm[C.line])[0]
    pe = np.bincount(t, minlength=C.V).astype(float)
    po = np.bincount(C.tok[allw], minlength=C.V).astype(float) - pe
    pe /= pe.sum(); po /= max(po.sum(), 1)
    m = 0.5 * (pe + po)
    def kl(p, q):
        nz = p > 0
        return float(np.sum(p[nz] * np.log2(p[nz] / q[nz])))
    VS = 0.5 * kl(pe, m) + 0.5 * kl(po, m)
    TT = len(np.unique(t)) / max(1, n)
    return dict(PI=PI, XP=XP, VS=VS, TT=TT)


STATS = ['PI', 'XP', 'VS', 'TT']


def baseline(C, mask_pages, ndraw=4):
    acc = defaultdict(float)
    for s in range(ndraw):
        st = stream_stats(C, C.random_select(1000 + s), mask_pages)
        for k in STATS: acc[k] += st[k] / ndraw
    return dict(acc)


def score_bank(C, bank, halves=(0, 1)):
    """bank: array (nrules, NF). Returns array (nrules, len(halves), len(STATS)) of D = stat(E) - stat(R)."""
    out = np.zeros((len(bank), len(halves), len(STATS)))
    masks = [C.half == h for h in halves]
    bases = [baseline(C, m) for m in masks]
    for i, w in enumerate(bank):
        idx = C.select(w)
        for j, m in enumerate(masks):
            st = stream_stats(C, idx, m)
            out[i, j] = [st[k] - bases[j][k] for k in STATS]
    return out


# ------------------------------------------------------------------ rule bank
def make_bank(n_random=3000, seed=73):
    rng = np.random.default_rng(seed)
    bank, names = [], []
    for f in range(NF):
        for sgn in (1, -1):
            w = np.zeros(NF); w[f] = sgn; bank.append(w); names.append(('+' if sgn > 0 else '-') + FEATS[f])
    while len(bank) < 2 * NF + n_random:
        k = rng.integers(1, 4); fs = rng.choice(NF, k, replace=False)
        w = np.zeros(NF); w[fs] = rng.normal(0, 1, k) + np.sign(rng.normal(0, 1, k)) * 0.5
        bank.append(w); names.append('R:' + ','.join('%s%+.2f' % (FEATS[f], w[f]) for f in fs))
    return np.array(bank), names


# ------------------------------------------------------------------ nulls
def lshuf(pages, seed=1):
    rng = random.Random(seed); out = []
    for p in pages:
        ws = [w for l in p['lines'] for w in l['w']]; rng.shuffle(ws); it = iter(ws)
        out.append(dict(p, lines=[dict(l, w=[next(it) for _ in l['w']]) for l in p['lines']]))
    return out


NULLS = dict(MK2=lambda P, s: V72.gen_mk2(P, s), SELFCIT=lambda P, s: V72.gen_selfcit(P, s),
             SC10=lambda P, s: V72.gen_selfcit(P, s, p_copy=0.10), JUNC=lambda P, s: V72.gen_junction(P, s),
             LSHUF=lambda P, s: lshuf(P, s))


# ------------------------------------------------------------------ planted list text
def item_stream(which='antid'):
    import v73_texts as T
    if which == 'antid':
        ents = T.antidotarium_items()
        ents = [[w for w in e if w not in ('recipe', 'iiii', 'ana', 'unciam', 'drachmas')] for e in ents]
    else:
        ents = T.apicius_items()
    return [w for e in ents for w in e]


def code_items(items, skeleton, code='VOY', seed=73):
    """Latin item words -> Voynich-like glyph words through a one-word codebook (nomenclator).
      VOY  : item type of frequency rank r -> the Voynich word type of rank 20 + r (looks exactly like filler words;
             frequent items share their spelling with common filler)
      NOVEL: item type -> a new word from the skeleton's glyph trigram law (Voynich form, but not a filler type),
             shortest words to the most frequent items"""
    rng = random.Random(seed)
    ic = Counter(items); itypes = [w for w, _ in ic.most_common()]
    vc = Counter(w for p in skeleton for l in p['lines'] for w in l['w'])
    if code == 'VOY':
        vt = [w for w, _ in vc.most_common()][20:]
        book = {w: vt[i % len(vt)] for i, w in enumerate(itypes)}
    else:
        gen = V72.gen_mk2(skeleton, seed)
        pool = [w for p in gen for l in p['lines'] for w in l['w'] if w not in vc]
        pool = list(dict.fromkeys(pool)); rng.shuffle(pool)
        pool = sorted(pool[:len(itypes)], key=len)
        book = {w: pool[i] for i, w in enumerate(itypes)}
    return [book[w] for w in items]


def plant(skeleton, scheme, filler='MK2', code='VOY', which='antid', seed=73):
    """skeleton: real corpus pages (only shape is used). filler text: generator fitted to the skeleton.
    One item per line, item order = list order, placed by `scheme`:
      POS2  : third word (last if the line is shorter)
      AFTER : random position >= 1, preceded by the marker word 'Col' (which also occurs as filler)
      FREE  : random position, no positional mark (only the item's own form differs)
    Returns (pages, truth) with truth = list of (page, line, pos) of the items."""
    rng = random.Random(seed)
    fill = NULLS[filler](skeleton, seed)
    items = item_stream(which)
    coded = code_items(items, skeleton, code, seed)
    k = 0; out = []; truth = []
    for pi, p in enumerate(fill):
        nl = []
        for li, l in enumerate(p['lines']):
            ws = list(l['w'])
            if ws and k < len(coded):
                n = len(ws)
                if scheme == 'POS2': j = min(2, n - 1)
                elif scheme == 'AFTER':
                    if n == 1: j = 0
                    else: j = rng.randrange(1, n); ws[j - 1] = 'Col'
                else: j = rng.randrange(n)
                ws[j] = coded[k]; k += 1; truth.append((pi, li, j))
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    # drop pages past the end of the list
    last = truth[-1][0] if truth else 0
    return out[:last + 1], truth


# ------------------------------------------------------------------ cycle 2: model-based features + learned selector
FEATS2 = ['ctxL', 'ctxR', 'formS', 'firstpg', 'firstbk', 'oddone', 'lineS']


def extend(C):
    """append model-based features: context surprisal (left/right bigram, leave-one-out), glyph-trigram form
    surprisal per glyph, first occurrence on the page / in the book so far, odd-one-out (normalised edit distance
    to the nearest line mate), and the word's surprisal under its own line's glyph unigram."""
    W = C.words; N = len(W); t = C.tok
    uni = np.bincount(t, minlength=C.V).astype(float)
    big = Counter(); left = np.full(N, -1); right = np.full(N, -1)
    for j in range(N):
        if C.pos[j] > 0: left[j] = t[j - 1]; big[(t[j - 1], t[j])] += 1
        if C.rpos[j] > 0: right[j] = t[j + 1]
    X = np.zeros((N, len(FEATS2)))
    tri = defaultdict(Counter)
    for w in W:
        x = '^^' + w + '$'
        for i in range(2, len(x)): tri[x[i - 2:i]][x[i]] += 1
    G = len(set(c for w in W for c in w)) + 1
    seenp = set(); seenb = set()
    import difflib
    for j, w in enumerate(W):
        a = t[j]
        pu = (uni[a] - 1 + 0.1) / (N + 0.1 * C.V)
        if left[j] >= 0:
            c = big[(left[j], a)] - 1; X[j, 0] = -math.log((c + 2 * pu) / (uni[left[j]] - 1 + 2))
        else: X[j, 0] = -math.log(pu)
        if right[j] >= 0:
            c = big[(a, right[j])] - 1; pr = (uni[right[j]] + 0.1) / (N + 0.1 * C.V)
            X[j, 1] = -math.log((c + 2 * pr) / (uni[a] - 1 + 2))
        else: X[j, 1] = -math.log(pu)
        x = '^^' + w + '$'; s = 0.0
        for i in range(2, len(x)):
            d = tri[x[i - 2:i]]; s -= math.log((d[x[i]] - 1 + 0.5) / (sum(d.values()) - 1 + 0.5 * G))
        X[j, 2] = s / (len(w) + 1)
        k = (C.page[j], a); X[j, 3] = k not in seenp; seenp.add(k)
        X[j, 4] = a not in seenb; seenb.add(a)
    for l in range(C.nl):
        s0 = C.line_start[l]; e = C.line_start[l + 1] if l + 1 < C.nl else N
        ws = W[s0:e]
        gl = Counter(c for w in ws for c in w); tot = sum(gl.values())
        for i, w in enumerate(ws):
            if len(ws) > 1:
                X[s0 + i, 5] = 1 - max(difflib.SequenceMatcher(None, w, v).ratio() for k2, v in enumerate(ws) if k2 != i)
            X[s0 + i, 6] = -sum(math.log((gl[c] - 1 + 0.5) / (tot - len(w) + 0.5 * 30)) for c in w) / len(w)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1; X = (X - mu) / sd
    C.F = np.hstack([C.F, X.astype(np.float32)])
    C.FT = np.ascontiguousarray(C.F.T.astype(np.float64))
    return C


ALLF = FEATS + FEATS2


def make_bank2(n_random=2500, seed=173):
    """random rules over the extended feature set; every rule uses at least one model-based feature."""
    rng = np.random.default_rng(seed); nf = len(ALLF); bank, names = [], []
    for f in range(NF, nf):
        for sgn in (1, -1):
            w = np.zeros(nf); w[f] = sgn; bank.append(w); names.append(('+' if sgn > 0 else '-') + ALLF[f])
    while len(bank) < 2 * len(FEATS2) + n_random:
        k = rng.integers(1, 4); fs = list(rng.choice(nf, k, replace=False))
        if not any(f >= NF for f in fs): fs[0] = int(rng.integers(NF, nf))
        fs = sorted(set(fs)); w = np.zeros(nf); w[fs] = rng.normal(0, 1, len(fs)) + np.sign(rng.normal(0, 1, len(fs))) * 0.5
        bank.append(w); names.append('R:' + ','.join('%s%+.2f' % (ALLF[f], w[f]) for f in fs))
    return np.array(bank), names


def icm_select(C, mask_pages, iters=6, seed=0):
    """latent item per line chosen to maximise page + neighbour-page concentration (lift over the word's global
    rate) of the selected stream; own page excluded because page-local filler repeats inside a page, by iterated conditional modes on the masked pages."""
    rng = np.random.default_rng(seed)
    idx = C.random_select(seed)
    lines = np.nonzero(mask_pages[C.line_page])[0]
    glob_ = np.bincount(C.tok, minlength=C.V).astype(float)
    for it in range(iters):
        sel = np.zeros((C.npg, C.V), np.float32)
        np.add.at(sel, (C.line_page[lines], C.tok[idx[lines]]), 1)
        for l in rng.permutation(lines):
            p = C.line_page[l]; s0 = C.line_start[l]; e = s0 + C.llen[s0]
            cand = np.arange(s0, e); tk = C.tok[cand]
            sel[p, C.tok[idx[l]]] -= 1
            nb = [q for q in C.nb[p] if q >= 0]
            sc = np.log((sel[nb][:, tk].sum(0) + 0.05) / glob_[tk])   # neighbour-page lift (own page excluded)
            j = cand[int(np.argmax(sc))]; idx[l] = j; sel[p, C.tok[j]] += 1
    return idx


def fit_rule(C, idx, mask_pages, l2=1.0, iters=300, lr=0.5):
    """conditional-logit (softmax over the words of a line) fit of the feature weights to the chosen words."""
    lines = np.nonzero(mask_pages[C.line_page])[0]
    wl = np.nonzero(mask_pages[C.page])[0]
    Fm = C.F[wl].astype(np.float64); lnm = C.line[wl]
    y = np.zeros(len(wl)); pos_of = {g: i for i, g in enumerate(wl)}
    for l in lines: y[pos_of[idx[l]]] = 1
    uniq, linv = np.unique(lnm, return_inverse=True)
    w = np.zeros(C.F.shape[1])
    for _ in range(iters):
        s = Fm @ w; mx = np.zeros(len(uniq)); np.maximum.at(mx, linv, s)
        e = np.exp(s - mx[linv]); z = np.bincount(linv, e); p = e / z[linv]
        g = Fm.T @ (y - p) / len(uniq) - l2 * w / len(uniq)
        w += lr * g
    return w
