"""v20 shared helpers: THE PAGES WERE WRITTEN TO A QUOTA.

Hypothesis: a constructed text (lipogram, balanced exercise, filler written to a target, text produced by
consuming a fixed stock of tokens) can be UNDER-dispersed: some glyph, glyph pair or word class is spread
across lines / paragraphs / pages / bifolios more evenly than chance. Natural text is over-dispersed.

Representation: a corpus is a token table. Each token has a word type id, a slot class
(0 = paragraph-first word, 1 = line-first, 2 = line-last, 3 = medial) and unit ids for the levels
line / para / page / bifol, plus a permutation domain per level (where the null may move it).
Null: inside each (domain, slot) group the word tokens are dealt back at random into the same positions,
so every unit keeps its token count and its slot make-up; only WHICH words land where changes.
Feature matrix T: word type x feature (counts of a glyph, glyph bigram, word class, random glyph set ...).
Unit counts = (unit x type count matrix) @ T.
"""
import os, re, sys, json, math, random
import numpy as np
from collections import Counter, defaultdict
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'results', 'v20'); os.makedirs(CK, exist_ok=True)
LEVELS = ['line', 'para', 'page', 'bifol']


def page_headers(path=os.path.join(vlib.DATA, 'ZL3b-n.txt')):
    hdr = {}
    for L in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'^<(f\d+[rv]\d*)>\s*<!([^>]*)>', L)
        if m:
            hdr[m.group(1)] = dict(re.findall(r'\$(\w)=(\w+)', m.group(2)))
    return hdr


class Corpus:
    """lines: list of dict(words, line, para, page, bifol, stratum). Builds token arrays."""
    def __init__(self, lines, name, glyph_mode):
        self.name = name; self.glyph_mode = glyph_mode
        self.lines = lines
        words, slot = [], []
        keys = {k: [] for k in ['line', 'para', 'page', 'bifol', 'stratum']}
        prev_para = None
        for L in lines:
            for j, w in enumerate(L['words']):
                words.append(w)
                if j == 0 and L['para'] != prev_para: s = 0
                elif j == 0: s = 1
                elif j == len(L['words']) - 1: s = 2
                else: s = 3
                slot.append(s)
                for k in keys: keys[k].append(L[k])
            prev_para = L['para']
        self.words = words
        self.types = sorted(set(words)); ti = {w: i for i, w in enumerate(self.types)}
        self.tid = np.array([ti[w] for w in words], dtype=np.int64)
        self.slot = np.array(slot, dtype=np.int64)
        self.unit, self.nunits = {}, {}
        for k in keys:
            u = {v: i for i, v in enumerate(dict.fromkeys(keys[k]))}
            self.unit[k] = np.array([u[v] for v in keys[k]], dtype=np.int64); self.nunits[k] = len(u)
        self.N = len(words)
        # permutation domains
        self.domain = {'line': self.unit['page'], 'para': self.unit['page'],
                       'page': self.unit['stratum'], 'bifol': self.unit['stratum']}

    def glyphs(self, w):
        return vlib.glyphs(w) if self.glyph_mode else list(w)

    def perm_tid(self, level, rng, tid=None):
        tid = self.tid if tid is None else tid
        g = self.domain[level] * 4 + self.slot
        base = np.argsort(g, kind='stable')
        idx = np.lexsort((rng.random(self.N), g))
        out = np.empty_like(tid); out[base] = tid[idx]
        return out

    def counts(self, level, tid, T):
        X = sp.csr_matrix((np.ones(self.N, dtype=np.float32), (self.unit[level], tid)),
                          shape=(self.nunits[level], len(self.types)))
        return np.asarray(X @ T)

    def expected(self, level, T, tid=None):
        """exact expectation of unit counts under the (domain, slot) permutation null."""
        tid = self.tid if tid is None else tid
        g = self.domain[level] * 4 + self.slot
        gi = {v: i for i, v in enumerate(np.unique(g))}; gg = np.array([gi[v] for v in g])
        G = sp.csr_matrix((np.ones(self.N, dtype=np.float32), (gg, tid)), shape=(len(gi), len(self.types)))
        gsize = np.bincount(gg).astype(np.float32)
        M = np.asarray(G @ T) / gsize[:, None]
        S = sp.csr_matrix((np.ones(self.N, dtype=np.float32), (self.unit[level], gg)),
                          shape=(self.nunits[level], len(gi)))
        return np.asarray(S @ M)


# ---------------- corpora ----------------
STRAT_SECS = None

def voynich_corpus(name='ZL3b', drop_uncertain=True):
    hdr = page_headers()
    recs = vlib.load_voynich(name, ltypes=('P',), drop_uncertain=drop_uncertain)
    lines, para, fol = [], 0, None
    for k, r in enumerate(recs):
        f = r['folio']
        if r['para_start'] or f != fol: para += 1
        fol = f
        h = hdr.get(f, {})
        lines.append(dict(words=[w for w in r['words'] if '?' not in w and '*' not in w] or r['words'][:0],
                          line=k, para=para, page=f, bifol=(h.get('Q', '?') + h.get('B', '?')),
                          stratum=(r.get('illus') or 'x') + (r.get('lang') or 'x') + (r.get('hand') or 'x')))
    lines = [L for L in lines if L['words']]
    return Corpus(lines, 'Voynich-' + name, True)


def _wrap(paras, width):
    out = []
    for p in paras:
        cur, n, first = [], 0, True
        for w in p:
            if cur and n + 1 + len(w) > width:
                out.append((cur, first, False)); cur, n, first = [], 0, False
            cur.append(w); n += len(w) + (1 if n else 0)
        if cur: out.append((cur, first, True))
    return out


def ref_corpus(key, max_tokens=35000, width=46, verse=False, lines_per_page=19, pages_per_bifol=4):
    """Reference text arranged like the Voynich: lines (wrapped to ~Voynich width, or verse lines),
    paragraphs (blank-line blocks), pages (chunks of lines_per_page lines), bifolios (4 pages), and
    strata (chunks of 12 pages ~ a Voynich section/hand block)."""
    if key == 'Latin-Isidore':
        raw = open(os.path.join(vlib.DATA, 'plain', 'la.txt'), encoding='utf-8').read().split('\n')
        paras = []
        for L in raw:
            ws = [vlib._norm_word(w) for w in re.split(r'[\s\-]+', re.sub(r'\(\d+[A-D]?\)', ' ', L))]
            ws = [w for w in ws if w and w.isascii()]
            if len(ws) >= 3: paras.append(ws)
        wl = _wrap(paras, width)
    elif key == 'English-Gadsby-lipogram':
        txt = open(os.path.join(vlib.DATA, 'pg47342.txt'), encoding='utf-8').read()
        m1 = re.search(r'\*\*\* ?START[^\n]*\n', txt); m2 = re.search(r'\*\*\* ?END', txt)
        txt = txt[m1.end():m2.start()]
        i = txt.find('CHAPTER I'); txt = txt[i:] if i > 0 else txt
        paras = []
        for b in re.split(r'\n\s*\n', txt):
            ws = [vlib._norm_word(w) for w in re.split(r'[\s\-]+', b)]
            ws = [w for w in ws if w]
            if len(ws) >= 3: paras.append(ws)
        wl = _wrap(paras, width)
    else:
        rl = vlib.load_ref(key, skip_frac=0.05)
        if verse:
            wl = [(L['words'], L['para_start'], L['para_end']) for L in rl]
        else:
            paras, cur = [], []
            for L in rl:
                if L['para_start'] and cur: paras.append(cur); cur = []
                cur += L['words']
            if cur: paras.append(cur)
            wl = _wrap(paras, width)
    lines, para, tok = [], 0, 0
    for k, (ws, ps, pe) in enumerate(wl):
        if ps or k == 0: para += 1
        pg = k // lines_per_page
        lines.append(dict(words=ws, line=k, para=para, page=pg, bifol=pg // pages_per_bifol, stratum=pg // 12))
        tok += len(ws)
        if tok >= max_tokens: break
    return Corpus(lines, key, False)


# ---------------- features ----------------
def build_features(C, n_rand_glyph=2500, n_rand_word=1000, top_words=150, seed=7, min_bigram=30):
    """Returns names (list) and T (types x F float32)."""
    rng = random.Random(seed)
    gl = [C.glyphs(w) for w in C.types]
    tokc = Counter(C.words)
    gcount = Counter()
    for w, g in zip(C.types, gl):
        for x in g: gcount[x] += tokc[w]
    alph = [g for g, c in gcount.most_common() if c >= 30]
    bc = Counter()
    for w, g in zip(C.types, gl):
        for a, b in zip(g, g[1:]): bc[a + '|' + b] += tokc[w]
    bigr = [b for b, c in bc.most_common() if c >= min_bigram]
    cols, names = [], []
    def add(name, vec): names.append(name); cols.append(vec)
    nT = len(C.types)
    for a in alph:
        add('g:' + a, np.array([g.count(a) for g in gl], dtype=np.float32))
    for b in bigr:
        x, y = b.split('|')
        add('bg:' + x + y, np.array([sum(1 for p, q in zip(g, g[1:]) if p == x and q == y) for g in gl], dtype=np.float32))
    # word classes
    for L in range(1, 11):
        add('len:%d' % L, np.array([1.0 if (len(g) == L or (L == 10 and len(g) >= 10)) else 0.0 for g in gl], dtype=np.float32))
    ini = Counter(); fin = Counter(); ini2 = Counter(); fin2 = Counter()
    for w, g in zip(C.types, gl):
        ini[g[0]] += tokc[w]; fin[g[-1]] += tokc[w]
        if len(g) >= 2: ini2[''.join(g[:2])] += tokc[w]; fin2[''.join(g[-2:])] += tokc[w]
    for a, c in ini.items():
        if c >= 30: add('ini:' + a, np.array([1.0 if g[0] == a else 0.0 for g in gl], dtype=np.float32))
    for a, c in fin.items():
        if c >= 30: add('fin:' + a, np.array([1.0 if g[-1] == a else 0.0 for g in gl], dtype=np.float32))
    for a, c in ini2.most_common(40):
        if c >= 30: add('ini2:' + a, np.array([1.0 if ''.join(g[:2]) == a else 0.0 for g in gl], dtype=np.float32))
    for a, c in fin2.most_common(40):
        if c >= 30: add('fin2:' + a, np.array([1.0 if ''.join(g[-2:]) == a else 0.0 for g in gl], dtype=np.float32))
    ti = {w: i for i, w in enumerate(C.types)}
    for w, c in tokc.most_common(top_words):
        v = np.zeros(nT, dtype=np.float32); v[ti[w]] = 1; add('w:' + w, v)
    add('glyphs', np.array([len(g) for g in gl], dtype=np.float32))
    # random glyph sets (count glyph occurrences of any member)
    G = np.array([[g.count(a) for a in alph] for g in gl], dtype=np.float32)
    for i in range(n_rand_glyph):
        k = rng.randint(2, min(8, len(alph) - 1)); s = rng.sample(range(len(alph)), k)
        add('rg:' + ''.join(alph[j] for j in sorted(s)), G[:, s].sum(1))
    freq_types = [w for w, c in tokc.items() if c >= 5]
    for i in range(n_rand_word):
        s = rng.sample(freq_types, min(20, len(freq_types)))
        v = np.zeros(nT, dtype=np.float32)
        for w in s: v[ti[w]] = 1
        add('rw:%d' % i, v)
    T = np.stack(cols, 1)
    return names, T, alph


# ---------------- statistics ----------------
def disp_stats(X, E):
    """per-feature: chi2 dispersion D, max per unit (ceiling), number of empty units (floor)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.where(E > 1e-9, (X - E) ** 2 / np.maximum(E, 1e-9), 0.0)
    return q.sum(0), X.max(0), (X == 0).sum(0)


def run_level(C, level, T, reps=200, seed=1, tid=None):
    rng = np.random.default_rng(seed)
    tid = C.tid if tid is None else tid
    E = C.expected(level, T, tid)
    X = C.counts(level, tid, T)
    obs = disp_stats(X, E)
    nulls = [[], [], []]
    for r in range(reps):
        Xn = C.counts(level, C.perm_tid(level, rng, tid), T)
        for k, v in enumerate(disp_stats(Xn, E)): nulls[k].append(v)
    nulls = [np.array(n) for n in nulls]
    return obs, nulls


def zscores(obs, nulls):
    """z of observed and of each null rep (leave-one-in), per statistic."""
    out = []
    for o, n in zip(obs, nulls):
        m = n.mean(0); s = n.std(0) + 1e-9
        out.append(((o - m) / s, (n - m) / s, o / np.maximum(m, 1e-9)))
    return out


def fw_p(zobs, znull, tail='low'):
    """family-wise p for each feature from the max-statistic distribution of the null reps."""
    if tail == 'low':
        mins = znull.min(1); return np.array([(mins <= z).mean() for z in zobs])
    maxs = znull.max(1); return np.array([(maxs >= z).mean() for z in zobs])


def markov_resynth(C, order=2, seed=3):
    """Same line/page template; each word replaced by a word generated from an order-k glyph Markov chain
    trained on the corpus's own stratum. Returns new Corpus."""
    rng = random.Random(seed)
    by_str = defaultdict(list)
    for L in C.lines: by_str[L['stratum']].extend(L['words'])
    models = {}
    for s, ws in by_str.items():
        m = defaultdict(Counter)
        for w in ws:
            g = ['^'] * order + C.glyphs(w) + ['$']
            for i in range(order, len(g)): m[tuple(g[i - order:i])][g[i]] += 1
        models[s] = {k: (list(v.keys()), list(v.values())) for k, v in m.items()}
    def gen(s):
        m = models[s]; ctx = ['^'] * order; out = []
        while True:
            ks, vs = m[tuple(ctx[-order:])]
            x = rng.choices(ks, vs)[0]
            if x == '$' or len(out) > 15: break
            out.append(x); ctx.append(x)
        return ''.join(out) if out else 'o'
    lines = [dict(L, words=[gen(L['stratum']) for _ in L['words']]) for L in C.lines]
    return Corpus(lines, C.name + '-markov2', False)


def write_rows(path, rows, header=None):
    with open(path, 'a') as f:
        if header: f.write(header + '\n')
        for r in rows: f.write(r + '\n')
