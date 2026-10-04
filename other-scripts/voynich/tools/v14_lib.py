"""v14: 'the words were rolled, not written' -- shared machinery.

Dice fingerprint of the slot grammar:
  * a DEVICE is a multiset of elementary outcome probabilities ('atoms'):
    one d6 (6 x 1/6), coin + d6 (12 x 1/12), two dice read through a lookup
    table (36 x 1/36), the sum of two dice (11 atoms 1..6..1 /36), the sum of
    three dice (16 atoms /216), the product of two dice (18 atoms /36), one
    astragalus (faces 1,3,4,6 at .1,.4,.4,.1), two astragali summed (atoms
    of the 4x4 product grouped by sum).
  * a slot is RANDOMISED BY A DEVICE if its filler distribution is obtained
    by assigning every atom to one filler (several atoms may give the same
    filler = merged faces / table cells).  Fillers below the top k are pooled
    as OTHER, which is itself a filler that must receive atoms.
  * the best assignment maximises the multinomial likelihood; exact greedy
    for equal atoms (concave integer allocation), local search with
    restarts for unequal atoms.  Fit = G statistic vs the best device law.
  * search-corrected null: the real filler distribution, log-jittered
    (sigma) and resampled to the same N, put through the identical search.
Independence: MI between slot fillers (top-9 + OTHER) vs permutation null
(one column shuffled within a stratum)."""
import sys, os, math, random, json, pickle, itertools
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from vlib import DATA, RES, glyphs, load_voynich
from test_c_slots import learn_order

OUT = os.path.join(RES, 'v14'); os.makedirs(OUT, exist_ok=True)
LOOPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'loops')

# ------------------------------------------------------------------ devices
def _sum_atoms(nd):
    c = Counter(sum(t) for t in itertools.product(range(1, 7), repeat=nd))
    tot = 6 ** nd
    return np.array([c[s] / tot for s in sorted(c)])

def _prod_atoms():
    c = Counter(a * b for a in range(1, 7) for b in range(1, 7))
    return np.array([c[s] / 36 for s in sorted(c)])

AST = {1: 0.1, 3: 0.4, 4: 0.4, 6: 0.1}
def _ast2():
    c = Counter()
    for a, pa in AST.items():
        for b, pb in AST.items():
            c[a + b] += pa * pb
    return np.array([c[s] for s in sorted(c)])

# ordered from least to most flexible (used by the parsimony rule)
DEVICES = {
    'astragalus':   np.array([.1, .4, .4, .1]),
    'd6':           np.full(6, 1 / 6),
    'astragali2sum': _ast2(),
    'coin+d6':      np.full(12, 1 / 12),
    '2d6sum':       _sum_atoms(2),
    '2d6prod':      _prod_atoms(),
    '3d6sum':       _sum_atoms(3),
    'd6+table36':   np.full(36, 1 / 36),
}
DEV_ORDER = list(DEVICES)

def gstat(c, q):
    c = np.asarray(c, float); N = c.sum()
    m = c > 0
    if np.any(q[m] <= 0): return np.inf
    return 2 * float(np.sum(c[m] * np.log(c[m] / (N * q[m]))))

def fit_equal(c, m):
    """m equal atoms -> integer allocation n_f>=1 maximising sum c log n."""
    k = len(c)
    if k > m: return np.inf, None
    n = np.ones(k, int)
    for _ in range(m - k):
        gain = np.where(c > 0, c * np.log((n + 1) / n), 0.0)
        n[np.argmax(gain)] += 1
    q = n / m
    return gstat(c, q), n

def fit_unequal(c, atoms, restarts=6, rng=None):
    k, m = len(c), len(atoms)
    if k > m: return np.inf, None
    rng = rng or random.Random(0)
    c = np.asarray(c, float)
    order = np.argsort(-atoms)
    best = (np.inf, None)
    def obj(assign):
        S = np.bincount(assign, weights=atoms, minlength=k)
        if np.any(S[c > 0] <= 0): return -np.inf
        return float(np.sum(c[c > 0] * np.log(S[c > 0])))
    for r in range(restarts):
        assign = np.zeros(m, int)
        S = np.zeros(k); tgt = c / c.sum()
        idx = list(order) if r == 0 else rng.sample(list(order), m)
        # first make sure every filler gets one atom: largest fillers first get largest atoms
        fl = list(np.argsort(-c))
        for j, a in enumerate(idx[:k]):
            assign[a] = fl[j]
            S[assign[a]] += atoms[a]
        for a in idx[k:]:
            f = int(np.argmax(tgt - S)) if r == 0 else (int(np.argmax(tgt - S)) if rng.random() < .7 else rng.randrange(k))
            assign[a] = f; S[f] += atoms[a]
        cur = obj(assign)
        improved = True
        while improved:
            improved = False
            for a in range(m):
                fa = assign[a]
                for f in range(k):
                    if f == fa: continue
                    assign[a] = f; v = obj(assign)
                    if v > cur + 1e-9: cur, fa, improved = v, f, True
                    else: assign[a] = fa
            for a in range(m):
                for b in range(a + 1, m):
                    if assign[a] == assign[b] or atoms[a] == atoms[b]: continue
                    assign[a], assign[b] = assign[b], assign[a]; v = obj(assign)
                    if v > cur + 1e-9: cur, improved = v, True
                    else: assign[a], assign[b] = assign[b], assign[a]
        q = np.bincount(assign, weights=atoms, minlength=k)
        g = gstat(c, q)
        if g < best[0]: best = (g, assign.copy())
    return best

def fit_device(c, dev, rng=None):
    atoms = DEVICES[dev]
    if np.allclose(atoms, atoms[0]):
        return fit_equal(np.asarray(c, float), len(atoms))
    return fit_unequal(c, atoms, rng=rng)

def collapse(counts, k):
    """counts: Counter filler->n. -> top-k counts + OTHER (if non-empty)."""
    top = counts.most_common()
    c = [n for _, n in top[:k]]
    other = sum(n for _, n in top[k:])
    if other > 0: c.append(other)
    return np.array(c, float)

KS = range(3, 10)
def slot_scores(counts, rng=None, devices=DEV_ORDER, ks=None):
    """-> {dev: {'z': min_k standardised G, 'k': argmin, 'G': {k: G}}}"""
    out = {}
    for d in devices:
        Gs = {}
        best = (np.inf, None)
        n = len(counts); m = len(DEVICES[d])
        for k in (ks or sorted({min(9, n), max(2, min(n, m - 1, 9))})):
            if k > len(counts): continue
            c = collapse(counts, k)
            g, _ = fit_device(c, d, rng)
            df = len(c) - 1
            Gs[k] = (g, len(c))
            z = (g - df) / math.sqrt(2 * df)
            if z < best[0]: best = (z, k)
        out[d] = {'z': best[0], 'k': best[1], 'G': Gs}
    return out

def jitter_counts(counts, sigma, N, rng):
    keys = list(counts); p = np.array([counts[x] for x in keys], float); p /= p.sum()
    if sigma > 0:
        p = p * np.exp(rng.normal(0, sigma, len(p))); p /= p.sum()
    s = rng.multinomial(N, p)
    return Counter({k: int(v) for k, v in zip(keys, s) if v > 0})

def parsimony(scores, alpha=0.001):
    """least flexible device whose best G at the finest resolution (largest k)
    is not rejected at alpha (chi2 with ncat-1 df)."""
    from scipy.stats import chi2
    for d in DEV_ORDER:
        G = scores[d]['G']
        if not G: continue
        k = max(G); g, nc = G[k]
        if np.isfinite(g) and g <= chi2.ppf(1 - alpha, nc - 1): return d
    return None

# ------------------------------------------------------------------ corpora / parsing
def voy_lines(name='ZL3b'):
    L = load_voynich(name, ('P',), True)
    out = []
    for l in L:
        nl = dict(l); nl['words'] = [''.join(glyphs(w)) for w in l['words']]; out.append(nl)
    return out

def chop(tokens, rng, lmin=8, lmax=10):
    out, i = [], 0
    while i < len(tokens):
        k = rng.randint(lmin, lmax)
        out.append({'words': tokens[i:i + k], 'illus': 'X', 'lang': 'X'}); i += k
    return out

def latin_verbose(n_words=35000, seed=3):
    sys.path.insert(0, os.path.dirname(__file__))
    from v7_numlib import latin_verbose as lv
    L = lv(n_words, seed)
    for l in L: l['illus'] = 'X'; l['lang'] = 'X'
    return L

def italian_verbose(n_words=35000, seed=5):
    """Manzoni Italian, same verbose substitution scheme as the Latin control."""
    from vlib import REFS
    rng = random.Random(seed)
    txt = open(os.path.join(DATA, REFS['Italian-Manzoni']), encoding='utf-8', errors='ignore').read().lower()
    words = [''.join(c for c in w if 'a' <= c <= 'z') for w in txt.split()]
    words = [w for w in words if w]
    s = len(words) // 3
    words = words[s:s + n_words]
    units = list('qoktpfCSKTPFedsainlrmgy')
    code, used = {}, set()
    for c in sorted(set(''.join(words))):
        while True:
            k = 1 if c in 'aeiou' else rng.choice([1, 2])
            x = ''.join(rng.choice(units) for _ in range(k))
            if x not in used: used.add(x); code[c] = x; break
    return chop([''.join(code[c] for c in w) for w in words], rng)

def learn_model(lines, K=4, cache_key=None):
    cache = os.path.join(OUT, 'models.json')
    d = json.load(open(cache)) if os.path.exists(cache) else {}
    if cache_key and cache_key in d: return d[cache_key]
    words = [w for L in lines for w in L['words']]
    order = learn_order(words)
    rank = {c: i for i, c in enumerate(order)}; U = len(order)
    tc = Counter(words); types = list(tc)
    rk = [[rank[c] for c in w] for w in types]
    best = None
    for cuts in itertools.combinations(range(1, U), K - 1):
        b = [0] + list(cuts) + [U]
        binr = np.searchsorted(np.array(b[1:]), np.arange(U), side='right')
        cnt = [Counter() for _ in range(K)]; fill = []
        for w, r in zip(types, rk):
            f = [''] * K
            for ch, x in zip(w, r): f[binr[x]] += ch
            fill.append(f)
            for k in range(K): cnt[k][f[k]] += tc[w]
        top = [set(x for x, _ in cnt[k].most_common(10)) for k in range(K)]
        cov = sum(tc[w] for w, f in zip(types, fill) if all(f[k] in top[k] for k in range(K)))
        if best is None or cov > best[0]: best = (cov, b)
    m = {'order': order, 'cuts': best[1], 'coverage': best[0] / len(words)}
    if cache_key:
        d[cache_key] = m; json.dump(d, open(cache, 'w'))
    return m

def parse(lines, model):
    """-> list of tokens: dict(f=tuple of K filler strings, line, pos, illus, lang)."""
    rank = {c: i for i, c in enumerate(model['order'])}; U = len(model['order'])
    b = model['cuts']; K = len(b) - 1
    binr = np.searchsorted(np.array(b[1:]), np.arange(U + 1), side='right')
    toks = []
    for li, L in enumerate(lines):
        for p, w in enumerate(L['words']):
            f = [''] * K
            for ch in w: f[min(K - 1, binr[rank.get(ch, U - 1)])] += ch
            toks.append({'f': tuple(f), 'line': li, 'pos': p, 'illus': L.get('illus'), 'lang': L.get('lang')})
    return toks

def slot_counts(toks, K):
    return [Counter(t['f'][k] for t in toks) for k in range(K)]

# ------------------------------------------------------------------ independence
def code_cols(toks, K, top=9):
    cnt = slot_counts(toks, K)
    cols = []
    for k in range(K):
        idx = {x: i for i, (x, _) in enumerate(cnt[k].most_common(top))}
        cols.append(np.array([idx.get(t['f'][k], top) for t in toks]))
    return np.array(cols)  # K x n

def mi(a, b, na=10, nb=10):
    j = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(float)
    n = j.sum(); pj = j / n
    pa = pj.sum(1, keepdims=True); pb = pj.sum(0, keepdims=True)
    m = pj > 0
    return float(np.sum(pj[m] * np.log2(pj[m] / (pa @ pb)[m])))

def perm_within(x, strata, rng):
    y = x.copy()
    for s in np.unique(strata):
        ix = np.where(strata == s)[0]
        y[ix] = x[rng.permutation(ix)]
    return y

def independence(toks, K, nperm=200, strata=None, seed=0):
    """MI per slot pair; null: shuffle the second column within strata."""
    rng = np.random.default_rng(seed)
    C = code_cols(toks, K)
    st = np.zeros(len(toks), int) if strata is None else np.array(strata)
    res = {}
    for i, j in itertools.combinations(range(K), 2):
        real = mi(C[i], C[j])
        null = np.array([mi(C[i], perm_within(C[j], st, rng)) for _ in range(nperm)])
        res[(i, j)] = {'mi': real, 'null_mean': float(null.mean()), 'null_max': float(null.max()),
                       'z': float((real - null.mean()) / (null.std() + 1e-12)),
                       'p': float((1 + np.sum(null >= real)) / (nperm + 1))}
    return res

def save(name, obj): pickle.dump(obj, open(os.path.join(OUT, name + '.pkl'), 'wb'))
def load(name):
    p = os.path.join(OUT, name + '.pkl')
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None

def zof(score, d, which):
    """standardised G of device d at 'full' (largest k) or 'core' (smallest k) resolution"""
    G = score[d]['G']
    if not G: return np.inf
    k = max(G) if which == 'full' else min(G)
    g, nc = G[k]
    return (g - (nc - 1)) / math.sqrt(2 * (nc - 1))

# ------------------------------------------------------------------ planted dice corpora
P_SLOTS = [['', 'q', 'o', 'qo', 'qoo'], ['', 'k', 't', 'p', 'f', 'C', 'S', 'K', 'T', 'P'],
           ['', 'e', 'ee', 'd', 's', 'ed', 'es', 'eee'], ['y', 'l', 'r', 'ain', 'aiin', 'ar', 'al', 'm', 'iin', 'n']]

def roll(dev, rng, n):
    at = DEVICES[dev]
    return rng.choice(len(at), size=n, p=at / at.sum())

def planted_dice(n=35000, seed=11, design='mixed'):
    """design 'mixed': slot1 d6 merged faces, slot2 2d6 sum, slot3 astragalus, slot4 coin+d6.
       design 'table': slot1 d6, slot2 two dice + 36-cell lookup table, slot3 d6, slot4 3d6 sum."""
    rng = np.random.default_rng(seed)
    if design == 'mixed':
        spec = [('d6', ['', '', 'q', 'o', 'qo', '']),
                ('2d6sum', ['P', 'f', 'k', 't', 'C', '', 'S', 'K', 'T', 'p', 'f']),
                ('astragalus', ['ee', '', 'e', 'd']),
                ('coin+d6', ['y', 'y', 'l', 'r', 'aiin', 'ain', 'y', 'ar', 'al', 'l', 'm', 'r'])]
    else:
        r2 = random.Random(seed)
        tab = [r2.choice(P_SLOTS[1]) for _ in range(36)]
        spec = [('d6', ['', 'q', 'o', 'qo', '', 'qoo']), ('d6+table36', tab),
                ('d6', ['', 'e', 'ee', 'd', 'ed', 's']),
                ('3d6sum', [r2.choice(P_SLOTS[3]) for _ in range(16)])]
    cols = [[fl[i] for i in roll(dev, rng, n)] for dev, fl in spec]
    words = [''.join(t) for t in zip(*cols)]
    words = [w if w else 'y' for w in words]
    return chop(words, random.Random(seed)), [d for d, _ in spec]

def shuffle_slots(toks, K, seed):
    """Voynich with each slot's fillers shuffled across tokens independently."""
    rng = random.Random(seed)
    cols = [[t['f'][k] for t in toks] for k in range(K)]
    for c in cols: rng.shuffle(c)
    out = []
    for i, t in enumerate(toks):
        nt = dict(t); nt['f'] = tuple(cols[k][i] for k in range(K)); out.append(nt)
    return out
