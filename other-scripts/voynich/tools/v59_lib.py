"""v59 The book contains its own Rosetta stone: unsupervised A-to-B translation.

A corpus side is a list of pages: {'id', 'sec', 'lines': [[word, ...], ...]}.
A transducer T is a list of rules applied in order to every word type of side 1.
Score of T: translate side 1 by T, then ask whether each translated word is used in
the same line contexts as the same word on side 2 (context retrieval in side-2 space).
Data sources used as raw symbols only (ZL3b, IT2a, Isidore la.txt, Dalimil cs.txt,
ReF German via v30 corpora.json).
"""
import json, os, re, random, collections, unicodedata, hashlib
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v59_ckpt')
os.makedirs(CK, exist_ok=True)


# ---------------------------------------------------------------- corpora
def voynich(name='ZL3b', lang='A'):
    L = json.load(open(os.path.join(ROOT, 'data', 'derived', name + '_lines.json')))
    pages = collections.OrderedDict()
    for l in L:
        if l['ltype'] != 'P' or l['lang'] != lang:
            continue
        ws = [w for w in l['words'] if w and '?' not in w and re.fullmatch(r'[a-z]+', w)]
        if not ws:
            continue
        p = pages.setdefault(l['folio'], {'id': l['folio'], 'sec': str(l['illus']), 'hand': str(l['hand']),
                                          'quire': l['quire'], 'lines': []})
        p['lines'].append(ws)
    return [p for p in pages.values() if sum(map(len, p['lines'])) >= 10]


def _norm(w):
    w = unicodedata.normalize('NFD', w.lower())
    return ''.join(ch for ch in w if unicodedata.category(ch)[0] == 'L')


def isidore():
    """Etymologiae (Migne): books are sections; chapters chunked into ~160-word pages, 8-word lines."""
    txt = open(os.path.join(ROOT, 'data', 'plain', 'la.txt'), encoding='utf-8').read().split('\n')
    book, chap, chaps = -1, None, []
    for line in txt:
        if line.startswith('CAPUT'):
            if line.startswith('CAPUT PRIMUM'):
                book += 1
            chap = {'book': book, 'words': []}
            chaps.append(chap)
            continue
        if chap is None:
            continue
        line = re.sub(r'\(\d+[A-D]\)', ' ', line)
        line = re.sub(r'^\d+\.', ' ', line)
        ws = [_norm(x) for x in line.split()]
        chap['words'] += [w for w in ws if w and w.isascii() and not w.isdigit()]
    pages = []
    for ci, c in enumerate(chaps):
        ws = c['words']
        for i in range(0, len(ws), 160):
            seg = ws[i:i + 160]
            if len(seg) < 40:
                continue
            pages.append({'id': f'c{ci}_{i}', 'chap': ci, 'sec': str(c['book']),
                          'lines': [seg[j:j + 8] for j in range(0, len(seg), 8)]})
    return pages


def czech():
    txt = open(os.path.join(ROOT, 'data', 'plain', 'cs.txt'), encoding='utf-8').read().split('\n')
    lines = [[w for w in (_norm(x) for x in l.split()) if w] for l in txt]
    lines = [l for l in lines if l]
    pages = []
    for i in range(0, len(lines), 30):
        pages.append({'id': f'cz{i}', 'sec': str(i // 600), 'lines': lines[i:i + 30]})
    return pages


def v30_corpus(key):
    C = json.load(open(os.path.join(ROOT, 'data', 'v30_ckpt', 'corpora.json')))['corpora'][key]
    return [{'id': f'{key}{i}', 'sec': str(i * 6 // len(C)), 'lines': [p[j:j + 8] for j in range(0, len(p), 8)]}
            for i, p in enumerate(C)]


# ------------------------------------------------------- planted spellings
LAT_SCRIBAL = [('sub', 'ae', 'e'), ('sub', 'oe', 'e'), ('sub', 'ti', 'ci'), ('pre', 'h', ''),
               ('sub', 'y', 'i'), ('sub', 'ph', 'f'), ('suf', 'um', 'ũ'), ('suf', 'que', 'q'), ('sub', 'mn', 'mpn')]
VOY_PLANT = [('suf', 'chy', 'chedy'), ('pre', 'cth', 't'), ('suf', 'or', 'ar'), ('sub', 'ol', 'eol')]
OLD_CZ = [('č', 'cz'), ('ř', 'rz'), ('š', 'ss'), ('ž', 'z'), ('ě', 'ie'), ('ů', 'o'), ('ň', 'n'), ('ť', 't'),
          ('ď', 'd'), ('j', 'g'), ('v', 'w'), ('c', 'cz')]


def old_czech(w):
    w = unicodedata.normalize('NFC', w)
    out, i = [], 0
    while i < len(w):
        for a, b in OLD_CZ:
            if w.startswith(a, i):
                out.append(b); i += len(a); break
        else:
            out.append(w[i]); i += 1
    return unicodedata.normalize('NFD', ''.join(out))


def opaque_verbose(pages, seed):
    """Each letter -> fixed random string of 1-3 private symbols; spaces and lines kept."""
    rng = random.Random(seed)
    al = sorted({c for p in pages for l in p['lines'] for w in l for c in w})
    sym = [chr(0x4e00 + i) for i in range(40)]
    key = {c: ''.join(rng.choice(sym) for _ in range(rng.choice([1, 1, 2, 2, 3]))) for c in al}
    return [dict(p, lines=[[''.join(key[c] for c in w) for w in l] for l in p['lines']]) for p in pages], key


# ------------------------------------------------------------- transducers
def apply_rule(w, r):
    kind, a, b = r
    if kind == 'sub':
        return w.replace(a, b) if a else w
    if kind == 'pre':
        return b + w[len(a):] if w.startswith(a) else w
    if kind == 'suf':
        return w[:len(w) - len(a)] + b if (w.endswith(a) and len(a) > 0) else w
    if kind == 'ins':           # insert b after every a
        return w.replace(a, a + b)
    if kind == 'del':
        return w.replace(a, '')
    return w


def apply_T(w, T):
    for r in T:
        w = apply_rule(w, r)
        if not w:
            return w
    return w


def map_pages(pages, T):
    cache = {}
    def f(w):
        if w not in cache:
            cache[w] = apply_T(w, T)
        return cache[w]
    return [dict(p, lines=[[f(w) for w in l if f(w)] for l in p['lines']]) for p in pages]


def ngram_stats(pages, n_top=40):
    sub, pre, suf = collections.Counter(), collections.Counter(), collections.Counter()
    for p in pages:
        for l in p['lines']:
            for w in l:
                for k in (1, 2, 3):
                    for i in range(len(w) - k + 1):
                        sub[w[i:i + k]] += 1
                    if len(w) > k:
                        pre[w[:k]] += 1; suf[w[-k:]] += 1
    top = lambda c: [x for x, _ in c.most_common(n_top)]
    return top(sub), top(pre), top(suf)


class RuleSampler:
    def __init__(self, side1, side2, seed):
        self.rng = random.Random(seed)
        s1, p1, f1 = ngram_stats(side1)
        s2, p2, f2 = ngram_stats(side2)
        self.s1, self.s2, self.p1, self.p2, self.f1, self.f2 = s1, s2, p1, p2, f1, f2
        self.g = sorted({c for x in s1 + s2 for c in x})

    def rule(self):
        r = self.rng
        k = r.random()
        if k < 0.30:
            return ('sub', r.choice(self.s1), r.choice(self.s2 + ['']))
        if k < 0.50:
            return ('suf', r.choice(self.f1), r.choice(self.f2 + ['']))
        if k < 0.70:
            return ('pre', r.choice(self.p1), r.choice(self.p2 + ['']))
        if k < 0.85:
            return ('ins', r.choice(self.g), r.choice(self.g))
        return ('del', r.choice(self.g), '')

    def T(self, nmax=4):
        return [self.rule() for _ in range(self.rng.randint(1, nmax))]


# --------------------------------------------------------------- scoring
class Scorer:
    """Side-2 space: top-K side-2 types serve as targets and as context basis."""

    def __init__(self, side2, K=300, minc=5):
        cnt = collections.Counter(w for p in side2 for l in p['lines'] for w in l)
        self.vocab = [w for w, _ in cnt.most_common(K)]
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self.K = len(self.vocab)
        self.minc = minc
        X = self._inc(side2)
        self.cB = np.asarray(X.sum(0)).ravel()
        self.NB = self.cB.sum()
        self.RB = self._ppmi(X)

    def _inc(self, pages):
        rows, cols = [], []
        r = 0
        for p in pages:
            for l in p['lines']:
                for w in l:
                    j = self.idx.get(w)
                    if j is not None:
                        rows.append(r); cols.append(j)
                r += 1
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(max(r, 1), self.K))

    def _ppmi(self, X):
        C = (X.T @ X).toarray()
        np.fill_diagonal(C, 0)
        tot = C.sum() + 1e-9
        rs = C.sum(1, keepdims=True) + 1e-9
        cs = C.sum(0, keepdims=True) ** 0.75
        cs = cs / (cs.sum() + 1e-9)
        P = np.log((C / tot) / ((rs / tot) * cs + 1e-12) + 1e-12)
        P[C == 0] = 0
        P = np.maximum(P, 0)
        n = np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
        return P / n

    def score(self, pages1, detail=False):
        """pages1 already in side-2 spelling. Returns weighted retrieval score and coverage."""
        X = self._inc(pages1)
        c1 = np.asarray(X.sum(0)).ravel()
        N1 = sum(len(l) for p in pages1 for l in p['lines'])
        R1 = self._ppmi(X)
        S = R1 @ self.RB.T
        ok = (c1 >= self.minc) & (self.cB >= self.minc)
        diag = np.diag(S)
        rank = (S > diag[:, None]).sum(1)
        w = np.minimum(c1, self.cB * (N1 / self.NB))
        hit = ok & (rank < 3)
        sc = (w * hit).sum() / max(N1, 1)
        cov = (w * ok).sum() / max(N1, 1)
        if detail:
            return sc, cov, {self.vocab[i]: int(rank[i]) for i in np.where(ok)[0]}
        return sc, cov


def split_pages(pages, seed):
    """Stratified by section: alternate pages within each section after a seeded shuffle."""
    rng = random.Random(seed)
    bys = collections.defaultdict(list)
    for p in pages:
        bys[p['sec']].append(p)
    a, b = [], []
    for s in sorted(bys):
        L = bys[s][:]
        rng.shuffle(L)
        a += L[0::2]; b += L[1::2]
    return a, b


def markov_resynth(pages, seed, order=2):
    rng = random.Random(seed)
    tri = collections.defaultdict(collections.Counter)
    for p in pages:
        for l in p['lines']:
            for w in l:
                s = '^' * order + w + '$'
                for i in range(order, len(s)):
                    tri[s[i - order:i]][s[i]] += 1
    tab = {k: (list(v.keys()), list(v.values())) for k, v in tri.items()}
    def gen():
        ctx, w = '^' * order, ''
        while True:
            ks, vs = tab[ctx]
            c = rng.choices(ks, vs)[0]
            if c == '$' or len(w) > 20:
                return w or 'o'
            w += c; ctx = (ctx + c)[-order:]
    return [dict(p, lines=[[gen() for _ in l] for l in p['lines']]) for p in pages]


def word_shuffle(pages, seed):
    """Shuffle tokens across lines within each section: keeps frequencies, kills line contexts."""
    rng = random.Random(seed)
    bys = collections.defaultdict(list)
    for p in pages:
        for l in p['lines']:
            bys[p['sec']] += l
    for s in bys:
        rng.shuffle(bys[s])
    out, pos = [], collections.Counter()
    for p in pages:
        nl = []
        for l in p['lines']:
            nl.append(bys[p['sec']][pos[p['sec']]:pos[p['sec']] + len(l)]); pos[p['sec']] += len(l)
        out.append(dict(p, lines=nl))
    return out


def word_bigram_resynth(pages, seed):
    """Line-wise word-bigram Markov chain: keeps adjacency syntax, kills longer line context."""
    rng = random.Random(seed)
    big = collections.defaultdict(collections.Counter)
    for p in pages:
        for l in p['lines']:
            s = ['<s>'] + l + ['</s>']
            for a, b in zip(s, s[1:]):
                big[a][b] += 1
    tab = {k: (list(v.keys()), list(v.values())) for k, v in big.items()}
    out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            cur, line = '<s>', []
            while len(line) < len(l):
                ks, vs = tab[cur]
                nx = rng.choices(ks, vs)[0]
                if nx == '</s>':
                    cur = '<s>'; continue
                line.append(nx); cur = nx
            nl.append(line)
        out.append(dict(p, lines=nl))
    return out
