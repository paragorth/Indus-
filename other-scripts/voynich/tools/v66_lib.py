"""v66: THE BOOK'S MIND MAP MATCHES A REAL ONE.

Concept network (classes of medical/herbal meaning, co-occurrence within entries,
averaged over real herbals in several languages) vs keyword network of a text
(page-bursty words, co-occurrence within pages), matched by many-restart simulated
annealing over injective maps concepts -> keywords (C code, v66_qap.c).
"""
import os, sys, json, math, random, ctypes, hashlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v66_lexicon import classify, CLASS_NAMES

VD = os.path.dirname(HERE)
DATA = os.path.join(VD, 'data')
CK = os.path.join(DATA, 'v66_ckpt')
LOOPS = os.path.join(VD, 'loops')

_lib = ctypes.CDLL(os.path.join(CK, 'v66_qap.so'))
_lib.qap_sa.restype = ctypes.c_double
_lib.qap_sa.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
                        ctypes.c_ulonglong, ctypes.c_double, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p]

REF = ['culpeper', 'konrad_plants', 'konrad_body', 'macer', 'circa_fr', 'celsus_lat', 'celsus_eng', 'v21_IT']
TWIN = {'celsus_lat': 'celsus_eng', 'celsus_eng': 'celsus_lat', 'konrad_plants': 'konrad_body', 'konrad_body': 'konrad_plants'}
UNRELATED = ['caesar', 'dalimil', 'manzoni', 'kafka', 'descartes', 'psalms_en']


def load_texts():
    return json.load(open(os.path.join(CK, 'texts.json')))


def voynich_pages(name='ZL3b', lang=None):
    L = json.load(open(os.path.join(DATA, 'derived', '%s_lines.json' % name)))
    pages = defaultdict(list); meta = {}
    for l in L:
        if l['ltype'] != 'P': continue
        if lang and l.get('lang') != lang: continue
        ws = [w for w, u in zip(l['words'], l.get('uncertain', [False] * len(l['words']))) if not u and '?' not in w and w]
        pages[l['folio']] += ws
        meta[l['folio']] = (l.get('illus'), l.get('lang'), l.get('quire'))
    keys = [k for k in pages if len(pages[k]) >= 20]
    return keys, [pages[k] for k in keys], meta


# ---------------------------------------------------------------- keywords
def keywords(units, K=80, min_count=10, min_df=4, skip_top=40):
    """page-bursty words: concentrated on fewer units than random placement predicts."""
    N = sum(len(u) for u in units)
    lens = np.array([len(u) for u in units], float)
    c = Counter(w for u in units for w in u)
    df = Counter(w for u in units for w in set(u))
    top = set(w for w, _ in c.most_common(skip_top))
    sc = []
    for w, n in c.items():
        if n < min_count or df[w] < min_df or w in top: continue
        exp_df = float(np.sum(1 - (1 - lens / N) ** n))
        sc.append((exp_df / df[w], w))
    sc.sort(reverse=True)
    return [w for _, w in sc[:K]]


def incidence(units, vocab):
    ix = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(units), len(vocab)), np.uint8)
    for r, u in enumerate(units):
        for w in u:
            j = ix.get(w)
            if j is not None: X[r, j] = 1
    return X


def phi(X):
    X = X.astype(float)
    n = X.shape[0]
    p = X.mean(0)
    cov = X.T @ X / n - np.outer(p, p)
    sd = np.sqrt(np.maximum(p * (1 - p), 1e-12))
    R = cov / np.outer(sd, sd)
    np.fill_diagonal(R, 0)
    return R


def zoff(M):
    M = M.copy(); n = M.shape[0]
    iu = np.triu_indices(n, 1)
    v = M[iu]; mu, sd = v.mean(), v.std() + 1e-12
    M = (M - mu) / sd
    np.fill_diagonal(M, 0)
    return M


# ---------------------------------------------------------------- concept graph
def class_units(units):
    X = np.zeros((len(units), len(CLASS_NAMES)), np.uint8)
    for r, u in enumerate(units):
        for w in set(u):
            c = classify(w)
            if c: X[r, CLASS_NAMES.index(c)] = 1
    return X


def concept_graph(T, names, min_units=25):
    Ms = []
    for k in names:
        X = class_units(T[k]['units'])
        Ms.append(np.arctanh(np.clip(phi(X), -0.99, 0.99)))
    M = np.tanh(np.mean(Ms, 0))
    np.fill_diagonal(M, 0)
    return M


# ---------------------------------------------------------------- opaque control
def opaque(units, seed, pad=0.25, nvar=3, nfill=300):
    """real text -> opaque tokens: each word type gets 1..nvar spelling variants (chosen per token),
    plus Zipfian filler tokens inserted at rate pad. Returns units and token->plain word map."""
    rng = random.Random(seed)
    var = {}
    fill = ['F%03d' % i for i in range(nfill)]
    fw = [1 / (i + 1) for i in range(nfill)]
    out, back = [], {}
    for u in units:
        o = []
        for w in u:
            if w not in var:
                h = hashlib.md5((w + str(seed)).encode()).hexdigest()[:6]
                var[w] = ['g%s%d' % (h, v) for v in range(rng.randint(1, nvar))]
                for t in var[w]: back[t] = w
            o.append(rng.choice(var[w]))
            if rng.random() < pad: o.append(rng.choices(fill, fw)[0])
        out.append(o)
    return out, back


def sample_units(units, n, seed):
    if len(units) <= n: return list(units)
    rng = random.Random(seed)
    ix = sorted(rng.sample(range(len(units)), n))
    return [units[i] for i in ix]


# ---------------------------------------------------------------- nulls
def curveball(X, seed, rounds=None):
    """degree-preserving randomisation of a binary unit x keyword matrix (row and column sums kept)."""
    rng = np.random.default_rng(seed)
    rows = [set(np.nonzero(X[r])[0]) for r in range(X.shape[0])]
    R = len(rows); rounds = rounds or 5 * R
    for _ in range(rounds):
        a, b = rng.choice(R, 2, replace=False)
        A, B = rows[a], rows[b]
        sa, sb = A - B, B - A
        if not sa and not sb: continue
        pool = list(sa | sb); rng.shuffle(pool)
        na = len(sa)
        A2 = (A & B) | set(pool[:na]); B2 = (A & B) | set(pool[na:])
        rows[a], rows[b] = A2, B2
    Y = np.zeros_like(X)
    for r, s in enumerate(rows):
        for j in s: Y[r, j] = 1
    return Y


def markov_units(units, seed, order=1):
    """bigram word resynthesis of the whole text, cut into units of the same lengths."""
    rng = random.Random(seed)
    flat = [w for u in units for w in u]
    nxt = defaultdict(list)
    for a, b in zip(flat, flat[1:]): nxt[a].append(b)
    w = rng.choice(flat); out = []
    for u in units:
        o = []
        for _ in range(len(u)):
            o.append(w)
            w = rng.choice(nxt[w]) if nxt[w] else rng.choice(flat)
        out.append(o)
    return out


# ---------------------------------------------------------------- matching
def qap(C, B, iters=200000, restarts=64, seed=1, T0=0.05, T1=0.0005, init=None, keep_maps=False):
    n, K = C.shape[0], B.shape[0]
    C = np.ascontiguousarray(C, np.float64); B = np.ascontiguousarray(B, np.float64)
    bm = np.zeros(n, np.int32); rsc = np.zeros(restarts, np.float64)
    im = np.ascontiguousarray(init, np.int32) if init is not None else None
    rm = np.zeros((restarts, n), np.int32) if keep_maps else None
    best = _lib.qap_sa(n, K, C.ctypes.data, B.ctypes.data, iters, restarts, seed, T0, T1, bm.ctypes.data, rsc.ctypes.data,
                       im.ctypes.data if im is not None else None, rm.ctypes.data if rm is not None else None)
    return best, bm, rsc, rm


def score(C, B, m):
    n = C.shape[0]; iu = np.triu_indices(n, 1)
    Bm = B[np.ix_(m, m)]
    return float(np.sum(C[iu] * Bm[iu]) / len(iu[0]))


def keyword_graph(units, K=80, kw=None):
    kw = kw or keywords(units, K)
    X = incidence(units, kw)
    return kw, X, zoff(phi(X))


def recovery(m, kw, back):
    """concepts whose assigned keyword's plain word has that class; and chance expectation."""
    cls = [classify(back.get(w, w)) for w in kw]
    hit = sum(1 for i, k in enumerate(m) if cls[k] == CLASS_NAMES[i])
    cnt = Counter(c for c in cls if c)
    exp = sum(cnt[c] for c in CLASS_NAMES) / len(kw)   # E[hits] under a random injective map = sum_i n_i / K
    return hit, exp, sum(cnt.values())


def row(f, cols):
    with open(f, 'a') as fh:
        fh.write('| ' + ' | '.join(str(c) for c in cols) + ' |\n')
