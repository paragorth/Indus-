"""LA-79 claim engines: each returns statistics for the real key AND its decoy family on a doc subset."""
import collections, math, re
import numpy as np

from la79_common import norm

BIG = ['GRA', 'OLE', 'CYP', 'VIN', 'OLIV', 'VIR', 'QA2', 'AROM', 'HIDE', '*401']


def types_of(docs, minlen=2):
    return {w['s'] for d in docs for w in d['words'] if len(w['s']) >= minlen}


# ---------------------------------------------------------------- A: affixes (hypergeometric, all signs at once)
def affix_z(docs, side='pre'):
    T = types_of(docs)
    long = [t for t in T if len(t) >= 3]
    N = len(long)
    if N < 20:
        return {}
    key = (lambda t: t[0]) if side == 'pre' else (lambda t: t[-1])
    rest = (lambda t: t[1:]) if side == 'pre' else (lambda t: t[:-1])
    hit = [rest(t) in T for t in long]
    m = sum(hit)
    nX = collections.Counter(key(t) for t in long)
    oX = collections.Counter(key(t) for t, h in zip(long, hit) if h)
    out = {}
    p = m / N
    for X, n in nX.items():
        if n < 3:
            continue
        mu = n * p
        var = n * p * (1 - p) * (N - n) / max(N - 1, 1)
        out[X] = (oX.get(X, 0) - mu) / math.sqrt(var) if var > 0 else 0.0
    return out


# ---------------------------------------------------------------- B: consecutive entries share a feature
def make_feature_decoys(signs, n=300, K=60, seed=1):
    rng = np.random.default_rng(seed)
    dec = []
    for i in range(n):
        pos = ['first', 'second', 'last'][i % 3]
        h = {s: int(rng.integers(K)) for s in signs}
        dec.append((pos, h))
    return dec


def _feat(w, pos, h):
    s = w['s']
    if pos == 'first':
        x = s[0]
    elif pos == 'second':
        if len(s) < 2:
            return None
        x = s[1]
    else:
        x = s[-1]
    return x if h is None else h.get(x, -1)


def consec_z(docs, feats):
    """feats: list of (pos, h or None). z of adjacent same-feature pairs vs within-doc permutation."""
    out = []
    for pos, h in feats:
        obs = E = 0.0
        for d in docs:
            f = [_feat(w, pos, h) for w in d['words'] if len(w['s']) >= 2]
            f = [x for x in f if x is not None]
            n = len(f)
            if n < 3:
                continue
            c = collections.Counter(f)
            obs += sum(1 for a, b in zip(f, f[1:]) if a == b)
            E += (n - 1) * sum(k * (k - 1) for k in c.values()) / (n * (n - 1))
        out.append((obs - E) / math.sqrt(E) if E > 0 else 0.0)
    return np.array(out)


# ---------------------------------------------------------------- C: consonant-group avoidance inside words
def consonant(s):
    if s.startswith('*') or s in ('VS', 'VAS'):
        return None
    m = re.match(r'([^AEIOU]*)', re.sub(r'\d', '', s))
    return m.group(1) or 'V'


def cons_avoid_z(docs, maps):
    """maps: list of dict sign->group. z (negative = avoidance) of adjacent different-sign same-group pairs."""
    pairs = []
    freq = collections.Counter()
    for d in docs:
        for w in d['words']:
            s = [x for x in w['s'] if consonant(x) is not None]
            freq.update(s)
            pairs += [(a, b) for a, b in zip(s, s[1:]) if a != b]
    if len(pairs) < 50:
        return None
    tot = sum(freq.values())
    p = {s: c / tot for s, c in freq.items()}
    signs = list(p)
    pv = np.array([p[s] for s in signs])
    idx = {s: i for i, s in enumerate(signs)}
    A = np.array([idx[a] for a, b in pairs])
    B = np.array([idx[b] for a, b in pairs])
    out = []
    for g in maps:
        gv = np.array([g.get(s, -1) for s in signs])
        # mass of each group
        gm = collections.defaultdict(float)
        for s, gi in zip(signs, gv):
            gm[gi] += p[s]
        same_mass = np.array([gm[gi] - p[s] for s, gi in zip(signs, gv)])
        q = same_mass[A] / (1 - pv[A])
        obs = float(np.sum(gv[A] == gv[B]))
        E = q.sum()
        V = (q * (1 - q)).sum()
        out.append((obs - E) / math.sqrt(V) if V > 0 else 0.0)
    return np.array(out)


# ---------------------------------------------------------------- D: Linear B shared vocabulary
def lb_lexicon():
    from la15_common import load_lb
    lex = set()
    for d in load_lb():
        for w in d['words']:
            t = tuple(norm(x) for x in w.split('-'))
            if len(t) >= 3:
                lex.add(t)
    return lex


def lb_z(docs, lex, nperm=100, seed=3):
    T = [t for t in types_of(docs, 3)]
    if len(T) < 20:
        return None, 0
    obs = sum(t in lex for t in T)
    rng = np.random.default_rng(seed)
    flat = [s for t in T for s in t]
    L = [len(t) for t in T]
    null = []
    for _ in range(nperm):
        rng.shuffle(flat)
        k = 0
        c = 0
        for l in L:
            c += tuple(flat[k:k + l]) in lex
            k += l
        null.append(c)
    null = np.array(null)
    sd = null.std() if null.std() > 0 else 0.5
    return (obs - null.mean()) / sd, obs


# ---------------------------------------------------------------- E/F: learned orders
def first_order(seq, items):
    o = []
    for x in seq:
        if x in items and x not in o:
            o.append(x)
    return o


def pair_counts(seqs):
    c = collections.Counter()
    for o in seqs:
        for i in range(len(o)):
            for j in range(i + 1, len(o)):
                if o[i] != o[j]:
                    c[(o[i], o[j])] += 1
    return c


def copeland(c, items):
    sc = {x: 0 for x in items}
    for a in items:
        for b in items:
            if a < b:
                ab, ba = c.get((a, b), 0), c.get((b, a), 0)
                if ab > ba:
                    sc[a] += 1; sc[b] -= 1
                elif ba > ab:
                    sc[b] += 1; sc[a] -= 1
    return sorted(items, key=lambda x: -sc[x])


def order_eval(train_seqs, test_seqs, items, n_dec=5000, seed=4):
    ct = pair_counts(train_seqs)
    order = copeland(ct, items)
    te = pair_counts(test_seqs)
    pairs = [(a, b, n) for (a, b), n in te.items()]
    ntot = sum(n for _, _, n in pairs)
    if ntot < 5:
        return None
    rk = {x: i for i, x in enumerate(order)}
    agree = sum(n for a, b, n in pairs if rk[a] < rk[b]) / ntot
    rng = np.random.default_rng(seed)
    dec = []
    for _ in range(n_dec):
        p = rng.permutation(len(items))
        r = {x: p[i] for i, x in enumerate(items)}
        dec.append(sum(n for a, b, n in pairs if r[a] < r[b]) / ntot)
    dec = np.array(dec)
    return dict(agree=agree, n=ntot, pct=float((dec < agree).mean() + 0.5 * (dec == agree).mean()), order=order)


# ---------------------------------------------------------------- G: fraction letter depends on commodity
def frac_comm_eval(train, test, n_dec=200, seed=5):
    def rows(docs):
        r = []
        for d in docs:
            for fr, c in d['fracs']:
                c = c if c in BIG else 'other'
                for x in fr:
                    r.append((x, c))
        return r
    tr, te = rows(train), rows(test)
    if len(te) < 15 or len(tr) < 15:
        return None
    letters = sorted({x for x, _ in tr + te})
    comms = sorted({c for _, c in tr + te})

    def gain(tr_rows):
        marg = collections.Counter(x for x, _ in tr_rows)
        cond = collections.defaultdict(collections.Counter)
        for x, c in tr_rows:
            cond[c][x] += 1
        V = len(letters)
        tm = sum(marg.values())
        g = 0.0
        for x, c in te:
            pm = (marg[x] + 0.5) / (tm + 0.5 * V)
            cc = cond[c]
            n = sum(cc.values())
            pc = (cc[x] + 2 * pm) / (n + 2)   # shrink toward marginal
            g += math.log2(pc / pm)
        return g / len(te)
    real = gain(tr)
    rng = np.random.default_rng(seed)
    xs = [x for x, _ in tr]
    cs = [c for _, c in tr]
    dec = []
    for _ in range(n_dec):
        p = rng.permutation(len(cs))
        dec.append(gain(list(zip(xs, [cs[i] for i in p]))))
    dec = np.array(dec)
    return dict(gain=real, n=len(te), pct=float((dec < real).mean()))


def affix_all_z(docs, side='pre', nperm=40, seed=6):
    """Aggregate: # len>=3 types whose tail (pre) / head (suf) is itself a type, vs types rebuilt by
    shuffling first, middle and last signs separately across types (the original 2f control)."""
    T = sorted(types_of(docs))
    if len([t for t in T if len(t) >= 3]) < 20:
        return None

    def count(TT):
        S = set(TT)
        return sum(((t[1:] if side == 'pre' else t[:-1]) in S) for t in TT if len(t) >= 3)
    obs = count(T)
    rng = np.random.default_rng(seed)
    firsts = [t[0] for t in T]; lasts = [t[-1] for t in T]
    mids = [s for t in T for s in t[1:-1]]
    null = []
    for _ in range(nperm):
        f = list(rng.permutation(firsts)); l = list(rng.permutation(lasts)); m = list(rng.permutation(mids))
        TT, k = [], 0
        for i, t in enumerate(T):
            nm = len(t) - 2
            TT.append(tuple([f[i]] + m[k:k + nm] + [l[i]]))
            k += nm
        null.append(count(TT))
    null = np.array(null)
    return float((obs - null.mean()) / max(null.std(), 0.5))
