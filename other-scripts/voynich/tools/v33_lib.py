"""v33 THE MORPHOLOGY IS AN ECOSYSTEM.

Words are split into stem + ending by a segmentation rule; the stem x ending presence matrix (stems = species,
endings = sites) is measured with community-ecology tools against null matrices with fixed row and column
totals (curveball algorithm):
  conn   connectance (fill)
  NODF   nestedness (overlap and decreasing fill), raw and z
  C      checkerboard score (Stone-Roberts, mean over ending pairs, normalised), raw and z
  Q      Barber bipartite modularity (BRIM from several random starts, best of), raw and z
  degree heterogeneity: coefficient of variation of stem degrees, Gini of ending degrees
Segmentation rule families (corpus-relative, so one recipe applies to any corpus):
  fix-k      ending = last k units
  pos-t      ending starts at the first unit whose mean relative in-word position >= t (generalised slot rule)
  slot-c     Voynich v7 slot order q o [gallows ch sh] e [d s] a i [n l r] [m g] y; ending = units of slot >= c
  freq-K     K most frequent suffixes (by type), longest match
  harris     successor-variety peak
  rand-s     random K (5..150) suffixes drawn from the corpus' type-weighted suffix distribution, random min stem
"""
import os, sys, re, json, random, math
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

ROOT = vlib.ROOT
DATA = vlib.DATA
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'v33_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
NTOK = 10000
R_STEMS, E_ENDS = 250, 30


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    with open(os.path.join(CK, name), 'w') as f: json.dump(obj, f, default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ corpora: token lists (lines kept for generators)
def U(w):
    return ''.join(vlib.glyphs(w))


def voy_lines(src='ZL3b', lang=None):
    d = json.load(open(os.path.join(DATA, 'derived', f'{src}_lines.json')))
    out = []
    for L in d:
        if L['ltype'] != 'P' or (lang and L['lang'] != lang): continue
        ws = [U(w) for w, u in zip(L['words'], L['uncertain']) if not u]
        ws = [w for w in ws if w and '?' not in w and re.fullmatch(r'[a-zA-Z]+', w)]
        if ws: out.append(ws)
    return out


def _wordlines(lines_iter, rx=r'[^\W\d_]+'):
    out = []
    for s in lines_iter:
        s = s.lower().replace('̇', '').replace("'", '').replace('’', '')
        ws = re.findall(rx, s)
        if ws: out.append(ws)
    return out


def lang_lines(code):
    if code == 'la':
        return _wordlines(open(os.path.join(DATA, 'plain', 'la.txt'), encoding='utf-8'))
    if code == 'cs':
        return _wordlines(open(os.path.join(DATA, 'plain', 'cs.txt'), encoding='utf-8'))
    if code == 'de':
        return [L['words'] for L in vlib.load_ref('German-Kafka')]
    if code == 'it':
        return [L['words'] for L in vlib.load_ref('Italian-Manzoni')]
    if code in ('hu', 'tr'):
        n = {'hu': 'hun', 'tr': 'tur'}[code]
        f = os.path.join(SCR, 'harm', f'{n}_news_2020_10K', f'{n}_news_2020_10K-sentences.txt')
        return _wordlines(s.split('\t', 1)[-1] for s in open(f, encoding='utf-8'))
    if code == 'he':
        f = os.path.join(DATA, 'v23_ckpt', 'he_mishneh_torah.txt')
        return _wordlines(open(f, encoding='utf-8'), rx=r'[א-ת]+')
    raise KeyError(code)


def blocks(lines, n=NTOK, nb=2, seed=0):
    """nb contiguous token blocks of n tokens (random starts; overlapping allowed when the corpus is short)."""
    toks = [w for l in lines for w in l]
    rng = random.Random(seed)
    if len(toks) <= n: return [toks]
    return [toks[s:s + n] for s in [rng.randrange(0, len(toks) - n + 1) for _ in range(nb)]]


# ------------------------------------------------------------------ generators (all produce lines of words)
class Trigram:
    """Unit trigram over the running text (word boundary = ' '): Markov resynthesis."""
    def __init__(self, lines):
        T = defaultdict(Counter)
        for l in lines:
            x = '  ' + ' '.join(l) + ' '
            for i in range(2, len(x)): T[x[i - 2:i]][x[i]] += 1
        self.T = {k: (list(v), np.cumsum(list(v.values())) / sum(v.values())) for k, v in T.items()}

    def gen(self, ntok, rng):
        out, cur, ctx = [], '', '  '
        while len(out) < ntok:
            ks, cp = self.T.get(ctx) or self.T['  ']
            c = ks[int(np.searchsorted(cp, rng.random()))]
            if c == ' ':
                if cur: out.append(cur); cur = ''
            else:
                cur += c
            ctx = ctx[1] + c
        return out


class SlotGen:
    """In-word unit trigram (word built glyph by glyph, no word-to-word dependency) = v21 slot generator."""
    def __init__(self, lines):
        T = defaultdict(Counter)
        for l in lines:
            for w in l:
                x = '^^' + w + '$'
                for i in range(2, len(x)): T[x[i - 2:i]][x[i]] += 1
        self.T = {k: (list(v), np.cumsum(list(v.values())) / sum(v.values())) for k, v in T.items()}

    def word(self, rng):
        x = '^^'
        while len(x) < 22:
            ks, cp = self.T[x[-2:]]
            c = ks[int(np.searchsorted(cp, rng.random()))]
            if c == '$': break
            x += c
        return x[2:] or 'o'

    def gen(self, ntok, rng):
        return [self.word(rng) for _ in range(ntok)]


class TableGrille:
    """Rugg-style table and grille: a 3-column table (prefix | middle | suffix) of chunks drawn from the source's
    own pos-0.5 segmentation pieces; a grille with 3 holes slides down the table; each word = the 3 cells under
    the holes (empty cells allowed); the grille offset moves by a die each word."""
    def __init__(self, lines, rows=40, seed=0):
        rng = random.Random(seed)
        toks = [w for l in lines for w in l]
        mp = mean_pos(toks)
        pre, mid, suf = Counter(), Counter(), Counter()
        for w in toks:
            a = [i for i, c in enumerate(w) if mp.get(c, 0) < 0.34]
            b = [i for i, c in enumerate(w) if 0.34 <= mp.get(c, 0) < 0.67]
            i1 = max(a) + 1 if a else 0
            i2 = max([i for i in b if i >= i1], default=i1 - 1) + 1
            pre[w[:i1]] += 1; mid[w[i1:i2]] += 1; suf[w[i2:]] += 1
        def draw(c):
            ks = list(c); p = np.array([c[k] for k in ks], float); p /= p.sum()
            return [ks[i] for i in np.random.default_rng(rng.randrange(1 << 30)).choice(len(ks), rows, p=p)]
        self.cols = [draw(pre), draw(mid), draw(suf)]
        self.rows = rows
        self.holes = [0, rng.randrange(1, 7), rng.randrange(1, 13)]

    def gen(self, ntok, rng):
        out, pos = [], 0
        while len(out) < ntok:
            w = ''.join(self.cols[j][(pos + self.holes[j]) % self.rows] for j in range(3))
            if w: out.append(w)
            pos += rng.randint(1, 6)
        return out


def v26_tokens(src='ZL3b', seed=0, ntok=None):
    """Output of the v26 minimal genome (rich junction key + width + line-initial chain + paragraph drift)."""
    import v26_lib
    from v21_lib import voynich_pages, tokens
    ev = json.load(open(os.path.join(DATA, 'v26_ckpt', 'eval_V.json')))
    g = ev['minimal']
    C = voynich_pages(src)
    S = v26_lib.Scribe(C, ('C', 'S'))
    F = S.forge(C, g, random.Random(seed))
    return [w for p in F for w in tokens(p)]


def plant_paradigm(toks, frac=0.3, nstem=150, seed=0, nclass=3, nend=6):
    """Hide an inflecting paradigm in Voynich-like text. Stems: novel words from the slot generator (Voynich-like);
    each stem belongs to one of nclass declension classes; each class has nend endings (Voynich-like suffixes
    of 1-3 units drawn from the text's own frequent word ends, disjoint across classes where possible);
    stem frequency Zipf(1), ending choice Zipf(1) within class. A fraction frac of the tokens is replaced."""
    rng = random.Random(seed)
    sg = SlotGen([toks])
    ends = Counter(w[-k:] for w in toks for k in (1, 2, 3) if len(w) > k)
    pool = [e for e, _ in ends.most_common(80)]
    rng.shuffle(pool)
    cls_ends = [pool[i * nend:(i + 1) * nend] for i in range(nclass)]
    stems = []
    while len(stems) < nstem:
        w = sg.word(rng)
        s = w[:max(1, len(w) - 2)]
        if 2 <= len(s) <= 5 and s not in stems: stems.append(s)
    scls = [rng.randrange(nclass) for _ in stems]
    sw = np.array([1 / (i + 1) for i in range(nstem)]); sw /= sw.sum()
    ew = np.array([1 / (i + 1) for i in range(nend)]); ew /= ew.sum()
    nr = np.random.default_rng(seed)
    out = list(toks)
    for i in range(len(out)):
        if rng.random() < frac:
            s = int(nr.choice(nstem, p=sw)); e = int(nr.choice(nend, p=ew))
            out[i] = stems[s] + cls_ends[scls[s]][e]
    return out, dict(stems=stems, cls=scls, ends=cls_ends)


# ------------------------------------------------------------------ segmentation rules
SLOT = {}
for k, gs in enumerate(['q', 'o', 'ktpfKTPFCS', 'e', 'ds', 'a', 'i', 'nlr', 'mg', 'y']):
    for c in gs: SLOT[c] = k


def mean_pos(toks):
    s, n = Counter(), Counter()
    for w in set(toks):
        L = len(w)
        if L < 2: continue
        for i, c in enumerate(w): s[c] += i / (L - 1); n[c] += 1
    return {c: s[c] / n[c] for c in n}


def suffix_counts(types, maxlen=4, minstem=1):
    c = Counter()
    for w in types:
        for k in range(1, maxlen + 1):
            if len(w) - k >= minstem: c[w[-k:]] += 1
    return c


class Rule:
    def __init__(self, kind, param, seed=0):
        self.kind, self.param, self.seed = kind, param, seed
        self.name = f'{kind}-{param}' + (f's{seed}' if kind == 'rand' else '')

    def fit(self, toks):
        types = set(toks)
        if self.kind == 'pos':
            self.mp = mean_pos(toks)
        elif self.kind == 'freq':
            self.suf = set(s for s, _ in suffix_counts(types).most_common(self.param))
            self.minstem = 1
        elif self.kind == 'rand':
            rng = random.Random(self.seed * 7919 + 13)
            K = rng.randint(5, 150); self.minstem = rng.randint(1, 3); maxlen = rng.randint(1, 4)
            sc = suffix_counts(types, maxlen, self.minstem)
            ks = list(sc); p = np.array([sc[k] for k in ks], float) ** rng.uniform(0.3, 1.2); p /= p.sum()
            K = min(K, len(ks))
            self.suf = set(ks[i] for i in np.random.default_rng(self.seed).choice(len(ks), K, replace=False, p=p))
        elif self.kind == 'harris':
            sv = defaultdict(set)
            for w in types:
                for i in range(1, len(w)): sv[w[:i]].add(w[i])
            self.sv = {k: len(v) for k, v in sv.items()}
        return self

    def split(self, w):
        L = len(w)
        if L < 2: return w, '#'
        k = self.kind
        if k == 'fix':
            j = max(1, L - self.param)
        elif k == 'pos':
            j = next((i for i in range(1, L) if self.mp.get(w[i], 0) >= self.param), L)
        elif k == 'slot':
            j = next((i for i in range(1, L) if SLOT.get(w[i], 0) >= self.param), L)
        elif k in ('freq', 'rand'):
            j = L
            for i in range(max(self.minstem, 1), L):
                if w[i:] in self.suf: j = i; break
        elif k == 'harris':
            j = max(range(1, L), key=lambda i: (self.sv.get(w[:i], 0), -i))
        return w[:j], (w[j:] or '#')


def rule_set(n_rand=20, voynich_alpha=False):
    R = [Rule('fix', k) for k in (1, 2, 3)] + [Rule('pos', t) for t in (0.5, 0.6, 0.7, 0.8)] + \
        [Rule('freq', K) for K in (10, 25, 50, 100)] + [Rule('harris', 0)] + \
        [Rule('rand', 0, seed=s) for s in range(n_rand)]
    if voynich_alpha: R += [Rule('slot', c) for c in (3, 4, 5, 6)]
    return R


# ------------------------------------------------------------------ matrix
def matrix(toks, rule, R=R_STEMS, E=E_ENDS, pairing_shuffle=None):
    """Presence matrix of the R most frequent stems x E most frequent endings (token frequencies).
    pairing_shuffle=rng: endings re-paired to stems at random over tokens (keeps both token marginals)."""
    tc = Counter(toks)
    pairs = [(rule.split(w), n) for w, n in tc.items()]
    if pairing_shuffle is not None:
        st = [s for (s, e), n in pairs for _ in range(n)]
        en = [e for (s, e), n in pairs for _ in range(n)]
        pairing_shuffle.shuffle(en)
        c = Counter(zip(st, en)); pairs = [((s, e), n) for (s, e), n in c.items()]
    sf, ef = Counter(), Counter()
    for (s, e), n in pairs: sf[s] += n; ef[e] += n
    stems = [s for s, _ in sf.most_common(R)]; ends = [e for e, _ in ef.most_common(E)]
    si = {s: i for i, s in enumerate(stems)}; ei = {e: i for i, e in enumerate(ends)}
    M = np.zeros((len(stems), len(ends)), np.int8)
    for (s, e), n in pairs:
        if s in si and e in ei: M[si[s], ei[e]] = 1
    M = M[M.sum(1) > 0][:, M.sum(0) > 0]
    return M


# ------------------------------------------------------------------ ecology metrics
def nodf(M):
    def half(A):
        d = A.sum(1).astype(float)
        if len(d) < 2: return 0.0, 0
        O = A.astype(np.float32) @ A.T.astype(np.float32)
        Di, Dj = d[:, None], d[None, :]
        valid = Di > Dj
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.where(valid & (Dj > 0), O / np.maximum(Dj, 1), 0.0)
        n = len(d)
        return v.sum() * 100.0, n * (n - 1) / 2
    a, na = half(M); b, nb = half(M.T)
    return (a + b) / max(na + nb, 1)


def cscore(M):
    c = M.T.astype(np.float32)
    r = c.sum(1)
    S = c @ c.T
    X = (r[:, None] - S) * (r[None, :] - S)
    n = len(r); iu = np.triu_indices(n, 1)
    denom = (r[:, None] * r[None, :])[iu]
    v = X[iu] / np.maximum(denom, 1)
    return float(v.mean()) if len(v) else 0.0


def barber_Q(M, rng, starts=6, maxc=8, iters=30):
    A = M.astype(float); m = A.sum()
    if m == 0: return 0.0
    k = A.sum(1); d = A.sum(0)
    B = A - np.outer(k, d) / m
    best = -1
    nr, nc = A.shape
    for s in range(starts):
        c = rng.integers(2, maxc + 1)
        col = rng.integers(0, c, nc)
        prevQ = -9
        for _ in range(iters):
            T = np.zeros((nc, c)); T[np.arange(nc), col] = 1
            rowl = (B @ T).argmax(1)
            R = np.zeros((nr, c)); R[np.arange(nr), rowl] = 1
            col = (B.T @ R).argmax(1)
            T = np.zeros((nc, c)); T[np.arange(nc), col] = 1
            Q = float(np.trace(R.T @ B @ T)) / m
            if Q <= prevQ + 1e-9: break
            prevQ = Q
        best = max(best, prevQ)
    return best


def curveball(M, rng, ntrade=None):
    rows = [set(np.flatnonzero(r)) for r in M]
    n = len(rows)
    ntrade = ntrade or 5 * n
    pi = rng.integers(0, n, (ntrade, 2))
    for a, b in pi:
        if a == b: continue
        A, Bs = rows[a], rows[b]
        oa = A - Bs; ob = Bs - A
        if not oa or not ob: continue
        pool = list(oa | ob); k = len(oa)
        rng.shuffle(pool)
        sh = A & Bs
        rows[a] = sh | set(pool[:k]); rows[b] = sh | set(pool[k:])
    N = np.zeros_like(M)
    for i, r in enumerate(rows):
        if r: N[i, list(r)] = 1
    return N


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0: return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def profile(M, nnull=30, seed=0):
    rng = np.random.default_rng(seed)
    obs = dict(conn=float(M.mean()), NODF=nodf(M), C=cscore(M), Q=barber_Q(M, rng),
               cvrow=float(M.sum(1).std() / max(M.sum(1).mean(), 1e-9)), ginicol=gini(M.sum(0)),
               nrow=int(M.shape[0]), ncol=int(M.shape[1]))
    nul = {'NODF': [], 'C': [], 'Q': []}
    for _ in range(nnull):
        N = curveball(M, rng)
        nul['NODF'].append(nodf(N)); nul['C'].append(cscore(N)); nul['Q'].append(barber_Q(N, rng))
    for k, v in nul.items():
        v = np.array(v); sd = v.std() + 1e-9
        obs['z' + k] = float((obs[k] - v.mean()) / sd)
        obs['d' + k] = float(obs[k] - v.mean())
    return obs


FEATS = ['conn', 'zNODF', 'zC', 'zQ', 'cvrow', 'ginicol']


def fvec(p):
    return np.array([p['conn'], *(np.sign(p[k]) * np.log1p(abs(p[k])) for k in ('zNODF', 'zC', 'zQ')),
                     p['cvrow'], p['ginicol']])
