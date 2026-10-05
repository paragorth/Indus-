"""v65: THE TEXT REMEMBERS WHICH PAGE CAME NEXT.

Page-to-page carry-over (tail of page a -> head of page b), searched over every
physically allowed arrangement of each quire (bifolio nesting order x inside-out
refolds), held out by word class, and checked against physical adjacency evidence.

Physical model (from the ZL3b IVTFF page headers $Q quire, $B bifolio, $F leaf):
  quire = list of bifolios; bifolio = (first-half leaf or None, second-half leaf or None);
  leaf = folio number with sides r, v (foldout panels merged into their side).
  A quire arrangement = nesting order (outermost first) + per-bifolio refold flag
  (refold inside-out: each leaf reads v then r; text stays upright).
  Page sequence = first-half leaves outer->inner, then second-half leaves inner->outer.
  A missing leaf contributes two GAP pages (chain break).

Corpora are dicts {side_key: [list of lines, each a list of word strings]} in a
shared physical structure.
"""
import os, re, sys, json, math, random, itertools, hashlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v65_ckpt')
LOOPS = os.path.join(ROOT, 'loops')
os.makedirs(CK, exist_ok=True)

GAP = -1
MIN_WORDS = 8
# foliation gaps at a quire centre: Q8 lost f59-f64, Q20 lost f109-f110 (voynich.nu quire notes)
LOST_INNER = ('H', 'T')


# ---------------------------------------------------------------- structure
def parse_headers(name='ZL3b'):
    hdr = {}
    pat = re.compile(r'^<(f\d+[rv]\d*)>\s*<!\s*([^>]*)>')
    for line in open(os.path.join(DATA, name + '-n.txt'), encoding='utf-8', errors='replace'):
        m = pat.match(line)
        if not m:
            continue
        kv = dict(re.findall(r'\$(\w)=(\w+)', m.group(2)))
        hdr[m.group(1)] = kv
    return hdr


def side_of(folio):
    if folio == 'fRos':
        return 86, 'r'
    m = re.match(r'f(\d+)([rv])', folio)
    return int(m.group(1)), m.group(2)


def structure(name='ZL3b'):
    """-> quires: OrderedDict quire -> list of bifolios [(B, leafA or None, leafB or None)]
       in current nesting (outer first); leafinfo: folio -> dict(sec, lang, hand)."""
    hdr = parse_headers(name)
    leaves = {}
    for fol, kv in hdr.items():
        n, s = side_of(fol)
        d = leaves.setdefault(n, {'Q': kv.get('Q'), 'B': int(kv.get('B', 0)), 'F': kv.get('F'),
                                  'sec': {}, 'lang': kv.get('L'), 'hand': kv.get('H')})
        d['sec'].setdefault(s, kv.get('I'))
    quires = {}
    for n in sorted(leaves):
        d = leaves[n]
        q = quires.setdefault(d['Q'], {})
        first = d['F'] <= 'm'
        b = q.setdefault(d['B'], [None, None])
        b[0 if first else 1] = n
    out = []
    for qn in sorted(quires):
        bl = [(B, v[0], v[1]) for B, v in sorted(quires[qn].items())]
        if qn in LOST_INNER:   # bifolios known lost from the quire centre (foliation gaps)
            bl.append((99, None, None))
        out.append((qn, bl))
    return out, leaves


def seq_for(bifs, order, flips):
    """bifs: list of (B, leafA, leafB); order: tuple of indices outer->inner; flips: tuple of 0/1.
    returns list of side keys (n, 'r'/'v') or GAP."""
    seq = []
    def leafpages(n, f):
        if n is None:
            return [GAP, GAP]
        return [(n, 'v'), (n, 'r')] if f else [(n, 'r'), (n, 'v')]
    for i in order:
        seq += leafpages(bifs[i][1], flips[i])
    for i in reversed(order):
        seq += leafpages(bifs[i][2], flips[i])
    return seq


def all_configs(nb, empty=()):
    """every (order, flips) for nb bifolios; flips of empty (lost) bifolios fixed at 0."""
    out = []
    free = [i for i in range(nb) if i not in empty]
    for order in itertools.permutations(range(nb)):
        for fl in itertools.product((0, 1), repeat=len(free)):
            flips = [0] * nb
            for i, f in zip(free, fl):
                flips[i] = f
            out.append((order, tuple(flips)))
    return out


# ---------------------------------------------------------------- corpora
def voynich_pages(name='ZL3b'):
    """side key -> list of lines (P paragraph lines) as glyph-unit strings, in page order."""
    L = vlib.load_voynich(name, ltypes=('P',), drop_uncertain=False)
    pages = defaultdict(list)
    meta = {}
    for r in L:
        ws = [''.join(vlib.glyphs(w)) for w in r['words']]
        ws = [w for w in ws if w and '?' not in w and '*' not in w]
        if not ws:
            continue
        k = side_of(r['folio'])
        pages[k].append(ws)
        meta.setdefault(k, {'sec': r['illus'], 'lang': r['lang'], 'hand': r['hand'], 'quire': r['quire']})
    return dict(pages), meta


OPAQUE = list('ABDGHIJMRUVWXZ0123456789')
PAD_I, PAD_M, PAD_F = ['q', 'o'], ['e'], ['y', 'n', 'l']


def opaque_encoder(seed):
    rng = random.Random(seed)
    codes, used = {}, set()
    def code(c):
        if c not in codes:
            while True:
                s = ''.join(rng.choice(OPAQUE) for _ in range(rng.choice([1, 2])))
                if s not in used:
                    break
            used.add(s); codes[c] = s
        return codes[c]
    prng = random.Random(seed + 7)
    def enc(word):
        w = prng.choice(PAD_I) if prng.random() < 0.4 else ''
        for i, c in enumerate(word):
            if i and prng.random() < 0.25:
                w += prng.choice(PAD_M)
            w += code(c)
        if prng.random() < 0.6:
            w += prng.choice(PAD_F)
        return w
    return enc


WRE = re.compile(r"[^\W\d_]+", re.UNICODE)


def norm_words(text):
    import unicodedata
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return [w.lower() for w in WRE.findall(text)]


def isidore_words():
    t = open(os.path.join(DATA, 'plain', 'la.txt'), encoding='utf-8', errors='replace').read()
    return norm_words(t)


def konrad_entries():
    import glob, html
    src = os.path.join(DATA, 'v58_ckpt', 'src', 'konrad_buch_der_natur')
    out = []
    for f in sorted(glob.glob(os.path.join(src, 'kon_[0-9]*.html'))):
        b = os.path.basename(f)
        if b[:5] not in ('kon_4', 'kon_5'):
            continue
        s = open(f, encoding='utf-8', errors='replace').read()
        i = s.find('contentus'); s = s[i if i > 0 else 0:]
        k = s.find('____'); s = s[k if k > 0 else 0:]
        k = s.find('&lt;&lt;&lt;'); s = s[:k] if k > 0 else s
        t = html.unescape(re.sub(r'<[^>]+>', ' ', s))
        t = re.sub(r'\[\d+\]|\{[^}]*\}|_+', ' ', t)
        ws = norm_words(t)
        if len(ws) >= 5:
            out.append(ws)
    return out


def macer_entries():
    t = open(os.path.join(DATA, 'v58_ckpt', 'src', 'macer_floridus.wiki'), encoding='utf-8').read()
    t = re.sub(r'\{\{Versus\|\d+\}\}', ' ', t)
    parts = re.split(r'\n==+([^=\n]+)==+\n', t)
    out = []
    for k in range(1, len(parts), 2):
        body = re.sub(r'<[^>]+>|\{\{[^}]*\}\}', ' ', parts[k + 1])
        ws = norm_words(body)
        if len(ws) >= 5:
            out.append(ws)
    return out


def binding_order(quires):
    """current binding page order (identity configs)."""
    seq = []
    for qn, bifs in quires:
        nb = len(bifs)
        seq += seq_for(bifs, tuple(range(nb)), (0,) * nb)
    return [s for s in seq if s != GAP]


def pour(template, words_or_entries, mode, seed, quires, encode=True, order=None):
    """Fill the template's page layout (lines x words per line, in binding order) with a
    control text. mode 'flow': one continuous text; 'entry': one entry per page (truncated,
    or the page cut short if the entry is shorter)."""
    enc = opaque_encoder(seed) if encode else (lambda w: w)
    order = [k for k in (order or binding_order(quires)) if k in template]
    pages = {}
    if mode == 'flow':
        ws = words_or_entries; i = 0
        for k in order:
            lines = []
            for L in template[k]:
                lines.append([enc(w) for w in ws[i:i + len(L)]]); i += len(L)
            pages[k] = lines
    else:
        ents = words_or_entries
        for j, k in enumerate(order):
            e = ents[j % len(ents)]; i = 0; lines = []
            for L in template[k]:
                if i >= len(e):
                    break
                lines.append([enc(w) for w in e[i:i + len(L)]]); i += len(L)
            pages[k] = lines
    return pages


def markov_pages(pages, meta, seed, order=1):
    """Word-bigram Markov resynthesis trained per section x language stratum; each page
    regenerated independently (no memory across pages), same line/word counts."""
    rng = random.Random(seed)
    strat = defaultdict(list)
    for k, L in pages.items():
        strat[(meta[k]['sec'], meta[k]['lang'])].append(k)
    out = {}
    for s, ks in strat.items():
        big = defaultdict(Counter); uni = Counter()
        for k in ks:
            for L in pages[k]:
                prev = '^'
                for w in L:
                    big[prev][w] += 1; uni[w] += 1; prev = w
        bt = {a: (list(c.keys()), list(c.values())) for a, c in big.items()}
        ut = (list(uni.keys()), list(uni.values()))
        for k in ks:
            lines = []
            for L in pages[k]:
                prev = '^'; nl = []
                for _ in L:
                    keys, vals = bt.get(prev, ut)
                    w = rng.choices(keys, vals)[0]; nl.append(w); prev = w
                lines.append(nl)
            out[k] = lines
    return out


def selfcit_pages(pages, quires, seed):
    """Copy-and-modify generator written through the book in binding order (it HAS memory)."""
    import gen
    order = [k for k in binding_order(quires) if k in pages]
    lines = [{'words': L, 'k': k} for k in order for L in pages[k]]
    g = gen.self_citation(lines, seed=seed)
    out = defaultdict(list)
    for L in g:
        out[L['k']].append(L['words'])
    return dict(out)


# ---------------------------------------------------------------- features
SKEL_DROP = set('qedCS')


def skeleton(w):
    s = ''.join(c for c in w if c not in SKEL_DROP)
    return s or '_'


def word_class(w, salt):
    h = hashlib.md5((salt + w).encode()).digest()[0]
    return h & 1


def carry_matrices(pages, keys, nl=3, cls=None, salt='a', feat='word', dfmax=0.2):
    """Returns J_seam[a,b] = weighted overlap of tail(a) (last nl lines) and head(b) (first nl
    lines), and J_page[a,b] = whole-page overlap; restricted to word types of class cls."""
    def toks(L):
        out = []
        for line in L:
            for w in line:
                t = skeleton(w) if feat == 'skel' else w
                if cls is None or word_class(t, salt) == cls:
                    out.append(t)
        return out
    heads = [set(toks(pages[k][:nl])) for k in keys]
    tails = [set(toks(pages[k][-nl:])) for k in keys]
    whole = [set(toks(pages[k])) for k in keys]
    N = len(keys)
    df = Counter(t for s in whole for t in s)
    vocab = {t: i for i, t in enumerate(t for t, c in df.items() if 2 <= c <= max(3, dfmax * N))}
    idf = np.zeros(len(vocab))
    for t, i in vocab.items():
        idf[i] = math.log(N / df[t])
    def mat(sets):
        M = np.zeros((N, len(vocab)))
        for a, s in enumerate(sets):
            for t in s:
                j = vocab.get(t)
                if j is not None:
                    M[a, j] = 1.0
        return M
    H, T, W = mat(heads), mat(tails), mat(whole)
    def sim(A, B):
        num = (A * idf) @ B.T
        na = np.sqrt((A * idf).sum(1) + 1e-9); nb = np.sqrt((B * idf).sum(1) + 1e-9)
        return num / na[:, None] / nb[None, :]
    Js = sim(T, H); Jp = sim(W, W)
    np.fill_diagonal(Js, np.nan); np.fill_diagonal(Jp, np.nan)
    return Js, Jp


def centre(J, groups):
    """double-centre J within row-group / column-group (section x language)."""
    N = J.shape[0]
    g = np.array(groups)
    W = np.full_like(J, np.nan)
    rmean = np.zeros(N); cmean = np.zeros(N)
    for a in range(N):
        m = g == g[a]
        rmean[a] = np.nanmean(J[a, m]) if m.sum() > 1 else np.nanmean(J[a])
        cmean[a] = np.nanmean(J[m, a]) if m.sum() > 1 else np.nanmean(J[:, a])
    gm = {}
    for x in set(groups):
        m = g == x
        gm[x] = np.nanmean(J[np.ix_(m, m)]) if m.sum() > 1 else np.nanmean(J)
    for a in range(N):
        W[a] = J[a] - rmean[a] - cmean + 0.5 * (gm[g[a]] + np.array([gm[x] for x in groups]))
    np.fill_diagonal(W, 0.0)
    return np.nan_to_num(W)


# ---------------------------------------------------------------- search
class QuireSpace:
    """All arrangements of one quire, as index arrays into the page list (-1 = gap)."""

    def __init__(self, bifs, kidx):
        self.bifs = bifs
        nb = len(bifs)
        empty = tuple(i for i, b in enumerate(bifs) if b[1] is None and b[2] is None)
        self.configs = all_configs(nb, empty)
        seqs = []
        for o, f in self.configs:
            s = seq_for(bifs, o, f)
            seqs.append([kidx.get(x, -1) if x != GAP else -1 for x in s])
        self.S = np.array(seqs, dtype=np.int32)
        self.ident = self.configs.index((tuple(range(nb)), (0,) * nb))
        a, b = self.S[:, :-1], self.S[:, 1:]
        self.valid = (a >= 0) & (b >= 0)
        self.a = np.where(self.valid, a, 0); self.b = np.where(self.valid, b, 0)

    def scores(self, W):
        return (W[self.a, self.b] * self.valid).sum(1)

    def pairs(self, ci):
        return [(int(x), int(y)) for x, y, v in zip(self.a[ci], self.b[ci], self.valid[ci]) if v]


def adjacency_set(space, ci):
    return set(space.pairs(ci))
