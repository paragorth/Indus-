"""v31 THE TEXTURE OF A SPELL: alphabet-agnostic feature battery, sampling, classifiers.

A sample is N consecutive tokens (default 150) of one corpus, with its line breaks. Every feature that depends
on the alphabet is either a ratio to its own value under a within-sample word shuffle (excess) or normalised by
the sample's unigram unit entropy, so EVA units and letters can be compared.

LINE-FREE features (used by the classifiers) and LINE features (only reported; line breaks are real in the
Voynich and the gibberish samples but editorial or synthetic in most other corpora).
"""
import os, sys, json, math, random, zlib
import numpy as np
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v31_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
CLASSES = ['LANG', 'MAGIC', 'INVENT', 'GIBB', 'GEN']


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ sampling
def samples(docs, N=150, maxs=12, seed=0, minN=None):
    """docs: list of documents (list of lines of words). Returns up to maxs samples of exactly N tokens,
    each a list of lines (lines cut at the sample edges), spread evenly over the corpus."""
    minN = minN or N
    cands = []
    for d in docs:
        flat = [(li, w) for li, l in enumerate(d) for w in l]
        for s in range(0, len(flat) - minN + 1, N):
            seg = flat[s:s + N]
            lines = []; cur = None; last = None
            for li, w in seg:
                if li != last: lines.append([]); last = li
                lines[-1].append(w)
            cands.append(lines)
    if len(cands) <= maxs: return cands
    idx = np.linspace(0, len(cands) - 1, maxs).round().astype(int)
    return [cands[i] for i in idx]


# ------------------------------------------------------------------ helpers
def near1(a, b):
    if a == b: return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or min(la, lb) < 3: return False
    if la == lb: return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb: a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]: i += 1
    return a[i:] == b[i + 1:]


def H(c):
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values() if v) if n else 0.0


def mi(pairs):
    j = Counter(pairs); a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs)
    return H(a) + H(b) - H(j)


def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def seq_stats(t):
    """Order-dependent statistics of a token sequence (computed on real order and on shuffles)."""
    n = len(t); S = {}
    S['junc_mi'] = mi([(t[i][-1], t[i + 1][0]) for i in range(n - 1)])
    S['junc_same'] = np.mean([t[i][-1] == t[i + 1][0] for i in range(n - 1)])
    S['first_eq'] = np.mean([t[i][0] == t[i + 1][0] for i in range(n - 1)])
    S['last_eq'] = np.mean([t[i][-1] == t[i + 1][-1] for i in range(n - 1)])
    S['last_eq2'] = np.mean([t[i][-1] == t[i + 2][-1] for i in range(n - 2)])
    S['rep_adj'] = np.mean([t[i] == t[i + 1] for i in range(n - 1)])
    S['rep_win'] = np.mean([t[i] in t[max(0, i - 10):i - 1] for i in range(2, n)])
    nl = nf = 0
    for i in range(n):
        for j in range(i - 1, max(-1, i - 41), -1):
            if near1(t[i], t[j]):
                if i - j <= 15: nl += 1
                else: nf += 1
                break
    S['near_loc'] = nl / n; S['near_far'] = nf / n
    L = [len(w) for w in t]
    S['wl_ac1'] = _corr(L[:-1], L[1:]); S['wl_ac2'] = _corr(L[:-2], L[2:])
    seen = set(); br = 0
    for i in range(n - 1):
        b = (t[i], t[i + 1])
        br += b in seen; seen.add(b)
    S['bigram_rep'] = br / (n - 1)
    # burstiness: CV of gaps of types with >= 3 tokens
    pos = defaultdict(list)
    for i, w in enumerate(t): pos[w].append(i)
    cvs = []
    for w, p in pos.items():
        if len(p) >= 3:
            g = np.diff(p); cvs.append(g.std() / g.mean())
    S['gap_cv'] = float(np.mean(cvs)) if cvs else 0.0
    h = n // 2
    A, B = set(t[:h]), set(t[h:])
    S['halves_j'] = len(A & B) / max(1, len(A | B))
    return S


SHUF_KEYS = ['junc_mi', 'junc_same', 'first_eq', 'last_eq', 'last_eq2', 'rep_adj', 'rep_win', 'near_loc', 'near_far',
             'bigram_rep', 'gap_cv', 'halves_j']


def variant_runs(t):
    """Generalised ch/sh alternation: among edit-1 same-length type pairs in the sample, the most frequent unit
    substitution x<->y; tokens carrying exactly one of x,y form a binary sequence; returns observed switch rate
    divided by its expectation 2pq (< 1 = the variant choice comes in runs)."""
    types = sorted(set(t)); sub = Counter()
    byl = defaultdict(list)
    for w in types: byl[len(w)].append(w)
    for L, ws in byl.items():
        if L < 3: continue
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = ws[i], ws[j]
                d = [(x, y) for x, y in zip(a, b) if x != y]
                if len(d) == 1: sub[tuple(sorted(d[0]))] += 1
    out = []
    for (x, y), c in sub.most_common(3):
        seq = [(x in w) for w in t if (x in w) != (y in w)]
        if len(seq) < 8: continue
        p = np.mean(seq); e = 2 * p * (1 - p)
        if e == 0: continue
        sw = np.mean([a != b for a, b in zip(seq, seq[1:])])
        out.append(sw / e)
    return float(np.mean(out)) if out else 1.0, float(sum(sub.values()) / max(1, len(types)))


def order_rigidity(types):
    """Share of within-word unit-pair co-occurrences (types) whose order is the majority order: 1 = every pair
    of units always comes in the same order (a slot template)."""
    o = Counter()
    for w in types:
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if w[i] != w[j]: o[(w[i], w[j])] += 1
    tot = maj = 0
    done = set()
    for (a, b), c in o.items():
        if (b, a) in done or (a, b) in done: continue
        d = o.get((b, a), 0); done.add((a, b))
        tot += c + d; maj += max(c, d)
    return maj / tot if tot else 1.0


def features(lines, rng, nshuf=12):
    t = [w for l in lines for w in l]
    n = len(t); F = {}
    cnt = Counter(t)
    F['ttr'] = len(cnt) / n
    F['hapax'] = sum(1 for c in cnt.values() if c == 1) / n
    F['top10'] = sum(c for _, c in cnt.most_common(10)) / n
    L = np.array([len(w) for w in t], float)
    F['wl_mean'] = L.mean(); F['wl_cv'] = L.std() / L.mean()
    F['wl_H'] = H(Counter(L.tolist()))
    u = Counter(c for w in t for c in w)
    h1 = H(u); F['h1'] = h1; F['alph_eff'] = 2 ** h1
    # conditional unit entropy inside words (with boundary)
    bg = Counter(); ctx = Counter()
    for w in t:
        x = '^' + w + '$'
        for i in range(1, len(x)): bg[(x[i - 1], x[i])] += 1; ctx[x[i - 1]] += 1
    hc = H(bg) - H(ctx)
    F['h2_ratio'] = hc / h1 if h1 else 0
    # positional
    pos = []
    for w in t:
        if len(w) == 1: pos.append((w, 's')); continue
        pos.append((w[0], 'f')); pos.append((w[-1], 'l'))
        pos += [(c, 'm') for c in w[1:-1]]
    F['pos_mi'] = mi(pos) / h1 if h1 else 0
    F['first_H'] = H(Counter(w[0] for w in t)) / h1 if h1 else 0
    F['last_H'] = H(Counter(w[-1] for w in t)) / h1 if h1 else 0
    F['order_rig'] = order_rigidity(list(cnt))
    F['dbl'] = np.mean([any(w[i] == w[i + 1] for i in range(len(w) - 1)) for w in t])
    # word-family density (types with an edit-1 neighbour)
    ty = list(cnt); fam = 0
    S = set(ty)
    for w in ty:
        ok = False
        for i in range(len(w)):
            if w[:i] + w[i + 1:] in S and len(w) > 3: ok = True; break
        if not ok:
            for v in ty:
                if near1(w, v): ok = True; break
        fam += ok
    F['family'] = fam / len(ty)
    F['var_runs'], F['var_pairs'] = variant_runs(t)
    s = ' '.join(t).encode()
    F['zlib'] = len(zlib.compress(s, 9)) / len(s)
    # order-dependent: observed / shuffled
    obs = seq_stats(t)
    sh = defaultdict(list)
    for _ in range(nshuf):
        q = t[:]; rng.shuffle(q)
        for k, v in seq_stats(q).items(): sh[k].append(v)
    for k in SHUF_KEYS:
        m = float(np.mean(sh[k]))
        F['x_' + k] = (obs[k] + 1e-3) / (m + 1e-3) if k not in ('junc_mi',) else obs[k] - m
    F['x_wl_ac1'] = obs['wl_ac1']; F['x_wl_ac2'] = obs['wl_ac2']
    return F


def line_features(lines, rng, nshuf=20):
    """Line-dependent signatures; only meaningful where line breaks are real."""
    F = {}
    if len(lines) < 4: return None
    def chain(ls):
        return np.mean([ls[i][0][0] == ls[i - 1][0][0] for i in range(1, len(ls)) if ls[i] and ls[i - 1]])
    o = chain(lines); sh = []
    for _ in range(nshuf):
        q = lines[:]; rng.shuffle(q); sh.append(chain(q))
    F['l_chain'] = (o + 1e-3) / (np.mean(sh) + 1e-3)
    allw = [w for l in lines for w in l]
    mw = np.mean([len(w) for w in allw])
    F['l_first_len'] = np.mean([len(l[0]) for l in lines]) / mw
    F['l_last_len'] = np.mean([len(l[-1]) for l in lines]) / mw
    lastu = Counter(l[-1][-1] for l in lines); allu = Counter(w[-1] for w in allw)
    F['l_last_H'] = H(lastu) / max(1e-9, H(allu))
    return F


FEATS = None


def feature_names():
    rng = random.Random(0)
    lines = [['abc', 'abd', 'bcd', 'cde'] * 10] * 4
    return sorted(features(lines, rng).keys())
