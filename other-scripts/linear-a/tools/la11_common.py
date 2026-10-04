#!/usr/bin/env python3
"""LA-11 shared code: blind compression contest of Linear A against real languages.

Every input (a list of word types written in opaque symbols) is scored against every language's syllable bigram model
under the BEST one-to-one map from the input's top-K symbols to the language's top-M syllables (simulated annealing,
C core la11_core.so).  No sound value is ever assumed: the Linear A and Linear B sign labels are used only as IDs.
"""
import os, sys, json, math, random, ctypes, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = os.path.join(HERE, '..', 'data', 'la11')
LANGD = os.path.join(D, 'lang')
_lib = ctypes.CDLL(os.path.join(HERE, 'la11_core.so'))
_lib.anneal.restype = ctypes.c_double
_lib.anneal.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_uint64,
                        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]

# ---------------------------------------------------------------- languages
_cache = {}
def lang_codes(min_train=1000, scheme='MIXED'):
    out = []
    for fn in sorted(os.listdir(LANGD)):
        if not fn.endswith('.json'): continue
        r = load_lang(fn[:-5])
        if r['schemes'][scheme]['n_train_types'] >= min_train: out.append(fn[:-5])
    return out

def load_lang(code):
    if code not in _cache: _cache[code] = json.load(open(os.path.join(LANGD, code + '.json')))
    return _cache[code]

def lang_model(code, scheme, M, D_=0.75, max_train=None, seed=0):
    """-> L ((M+3)x(M+3) log2 P(y|x)), syllable list. Interpolated absolute discounting over types.
    max_train: if set, the model is rebuilt from the stored bigram table thinned to this many types (approximation:
    counts scaled), to equalise training sizes."""
    s = load_lang(code)['schemes'][scheme]
    uni = s['uni']; top = [a for a, _ in sorted(uni.items(), key=lambda kv: (-kv[1], kv[0]))[:M]]
    if len(top) < M: raise ValueError('inventory too small')
    ix = {a: i for i, a in enumerate(top)}
    OTH, BOS, EOS = M, M + 1, M + 2
    n = M + 3
    cnt = np.zeros((n, n))
    for k, v in s['bi'].items():
        a, b = k.split('\t')
        i = BOS if a == '^' else ix.get(a, OTH)
        j = EOS if b == '$' else ix.get(b, OTH)
        cnt[i, j] += v
    if max_train and s['n_train_types'] > max_train: cnt *= max_train / s['n_train_types']
    succ = cnt.sum(0); succ[BOS] = 0
    Y = [j for j in range(n) if j != BOS]
    pu = np.zeros(n); pu[Y] = (succ[Y] + 0.5) / (succ[Y].sum() + 0.5 * len(Y))
    L = np.full((n, n), -60.0)
    for i in range(n):
        if i == EOS: continue
        row = cnt[i]; c = row.sum()
        if c <= 0:
            p = pu.copy()
        else:
            n1 = (row[Y] > 0).sum()
            p = np.maximum(row - D_, 0) / c + (D_ * n1 / c) * pu
        p[BOS] = 0
        with np.errstate(divide='ignore'):
            L[i] = np.where(p > 0, np.log2(np.maximum(p, 1e-300)), -60.0)
    return np.ascontiguousarray(L), top

def lang_test(code, scheme):
    return [tuple(x) for x in load_lang(code)['schemes'][scheme]['test']]

# ---------------------------------------------------------------- inputs
def count_matrix(types, K):
    """types: list of tuples of opaque symbols -> C ((K+3)^2), top-K symbol list, n predicted symbols."""
    u = collections.Counter(a for t in types for a in t)
    top = [a for a, _ in sorted(u.items(), key=lambda kv: (-kv[1], str(kv[0])))[:K]]
    if len(top) < K: raise ValueError('too few symbols')
    ix = {a: i for i, a in enumerate(top)}
    OTH, BOS, EOS = K, K + 1, K + 2
    C = np.zeros((K + 3, K + 3))
    for t in types:
        seq = [BOS] + [ix.get(a, OTH) for a in t] + [EOS]
        for a, b in zip(seq, seq[1:]): C[a, b] += 1
    return np.ascontiguousarray(C), top, int(C.sum())

def anneal(C, L, K, M, R=12, S=60000, T0=3.0, T1=0.02, seed=1):
    m = (ctypes.c_int * (K + 3))(); rest = (ctypes.c_double * R)()
    best = _lib.anneal(C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       K, M, R, S, T0, T1, seed, m, rest)
    return best, list(m), list(rest)

def la_types():
    import la5_common as c5
    return sorted(set(x[2] for x in c5.words_of(c5.la_docs())))

def lb_types():
    import la5_common as c5
    return sorted(set(x[2] for x in c5.words_of(c5.lb_docs())))

def len_hist(types, cap=6):
    return collections.Counter(min(len(t), cap) for t in types)

def length_matched(pool, hist, rnd, cap=6):
    """sample without replacement from pool to reproduce the length histogram (lengths >= cap pooled)."""
    by = collections.defaultdict(list)
    for t in pool: by[min(len(t), cap)].append(t)
    for v in by.values(): rnd.shuffle(v)
    out = []; short = 0
    for L, n in hist.items():
        take = by[L][:n]; out += take; short += n - len(take); by[L] = by[L][n:]
    rest = [t for v in by.values() for t in v]; rnd.shuffle(rest)
    out += rest[:short]
    return out

def shuffle_types(types, rnd):
    pool = [a for t in types for a in t]; rnd.shuffle(pool); out = []; k = 0
    for t in types: out.append(tuple(pool[k:k + len(t)])); k += len(t)
    return out

def markov_types(hist_types, K, rnd, alpha=0.3, n_sym=60):
    """random structured strings: a random first-order chain (Dirichlet(alpha) rows) over n_sym symbols, with the
    unigram roughly Zipfian; lengths copied from hist_types."""
    w0 = np.array([1.0 / (r + 1) for r in range(n_sym)])
    rng = np.random.default_rng(rnd.randrange(1 << 30))
    P = rng.dirichlet([alpha] * n_sym, size=n_sym) * 0.5 + 0.5 * w0 / w0.sum()
    P /= P.sum(1, keepdims=True)
    start = w0 / w0.sum()
    out = []
    for t in hist_types:
        s = [rng.choice(n_sym, p=start)]
        for _ in range(len(t) - 1): s.append(rng.choice(n_sym, p=P[s[-1]]))
        out.append(tuple('r%d' % x for x in s))
    return out
