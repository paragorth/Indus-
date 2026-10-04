"""v9: 'the text was dialled out of a volvelle' -- shared machinery.

Corpus -> lines of filler tuples.  Each word is cut into K slots (prefix /
core / suffix rings) by the corpus-learned slot grammar of v7 (v7_numlib.
slot_model: glyph order learned on the corpus itself, cut into K contiguous
glyph bins).  Per slot the M commonest fillers get their own symbol, the
rest share one symbol OTHER (index M).

Volvelle model for one ring:
  ring = cyclic list of n cells, each carrying one filler symbol (duplicates
  allowed, so a frequent filler can sit on several cells);
  hidden state = which cell is under the reading pointer;
  between two words of a line the ring turns by d cells, d ~ q(d | ctx)
  (the turning rule; ctx = class of the previous word's last-slot filler for
  rule R2, a single class for rule R1);
  at a line start the ring is (re)set: s0 ~ pi  (reset)  -- or, for the
  no-reset variant, the ring continues from the previous line's last cell.
  Emission is deterministic: the word shows the filler on the current cell.
Rings are independent given the observed sequence (R1, R2), so the word
likelihood is the product of K one-dimensional HMMs, each run batched over
lines.  q and pi are fitted by EM; ring contents by simulated annealing.
"""
import sys, os, json, math, random, itertools
from collections import Counter, defaultdict
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from vlib import DATA, RES

LOOPS = os.path.join(os.path.dirname(DATA), 'loops')
CACHE = os.path.join(DATA, 'derived', 'v9_slotmodels.json')


# ------------------------------------------------------------------ corpora
def corpus(name):
    import v7_numlib as N
    if name == 'voynich':
        return N.voynich('ZL3b')
    if name == 'voynich_IT':
        return N.voynich('IT2a')
    if name == 'vs_latin':
        return N.latin_verbose(35000, seed=3)
    if name == 'vs_italian':
        import v5_corpora as C
        units = C.verbose_lang('Italian-Manzoni', max_words=35000)
        # tuples of letters A..R -> strings; reuse v7 chop for a common layout
        toks = [''.join(w) for u in units for L in u['lines'] for w in L]
        return N.chop(toks, random.Random(5))
    raise KeyError(name)


def slot_model_cached(name, lines, K):
    import v7_numlib as N
    c = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = '%s_K%d' % (name, K)
    if key not in c:
        order = c.get('%s_order' % name)
        m = N.slot_model(lines, K, order=order)
        c[key] = m
        c['%s_order' % name] = m['order']
        json.dump(c, open(CACHE, 'w'))
    return c[key]


def parse_word(w, model):
    order, b = model['order'], model['cuts']
    rank = {c: i for i, c in enumerate(order)}
    K = len(b) - 1
    f = [''] * K
    for ch in w:
        r = rank.get(ch, len(order) - 1)
        for k in range(K):
            if r < b[k + 1]:
                f[k] += ch; break
    return f


def encode(lines, model, M=12, vocab=None):
    """-> (O: int array (L,T,K) filler symbols, -1 pad; lens; vocab per slot)."""
    K = len(model['cuts']) - 1
    P = [[parse_word(w, model) for w in L['words']] for L in lines]
    if vocab is None:
        cnt = [Counter() for _ in range(K)]
        for L in P:
            for f in L:
                for k in range(K):
                    cnt[k][f[k]] += 1
        vocab = [[x for x, _ in cnt[k].most_common(M)] for k in range(K)]
    idx = [{x: i for i, x in enumerate(v)} for v in vocab]
    T = max(len(L) for L in P)
    O = -np.ones((len(P), T, K), dtype=np.int64)
    for i, L in enumerate(P):
        for t, f in enumerate(L):
            for k in range(K):
                O[i, t, k] = idx[k].get(f[k], len(vocab[k]))
    lens = np.array([len(L) for L in P])
    return O, lens, vocab


def split_pages(lines, seed=0):
    """Hold out alternate pages (folio); returns train idx, test idx."""
    fol = sorted({L.get('folio') for L in lines}, key=lambda f: [l.get('folio') for l in lines].index(f))
    hold = set(fol[1::2])
    tr = [i for i, L in enumerate(lines) if L.get('folio') not in hold]
    te = [i for i, L in enumerate(lines) if L.get('folio') in hold]
    return np.array(tr), np.array(te)


# ------------------------------------------------------------------ ring HMM
def circulant(q):
    n = len(q)
    # T[i, j] = q[(j - i) mod n]
    I = np.arange(n)
    return q[(I[None, :] - I[:, None]) % n]


class Ring:
    """One ring: labels (n,), symbols 0..S-1; rule contexts C."""

    def __init__(self, labels, S, C=1, eps=1e-6):
        self.lab = np.array(labels)
        self.n = len(labels)
        self.S = S
        self.C = C
        self.q = np.ones((C, self.n)) / self.n
        self.pi = np.ones(self.n) / self.n
        self.eps = eps

    def E(self):
        E = np.zeros((self.S, self.n))
        E[self.lab, np.arange(self.n)] = 1.0
        # every symbol must be emittable: tiny floor so missing labels are not -inf
        E += self.eps
        return E


def _apply(prev, Tm, c):
    if Tm.shape[0] == 1:
        return prev @ Tm[0]
    out = np.empty_like(prev)
    for k in range(Tm.shape[0]):
        m = c == k
        if m.any(): out[m] = prev[m] @ Tm[k]
    return out


def _apply_T(eb, Tm, c):
    if Tm.shape[0] == 1:
        return eb @ Tm[0].T
    out = np.empty_like(eb)
    for k in range(Tm.shape[0]):
        m = c == k
        if m.any(): out[m] = eb[m] @ Tm[k].T
    return out


def forward_backward(ring, obs, ctx, lens, reset=True, need_post=False):
    """obs (L,T) symbols (-1 pad); ctx (L,T) context of the transition INTO t.
    reset=False: lines are concatenated per corpus order (carry the state).
    Returns loglik, and if need_post: expected step counts (C,n), start counts (n)."""
    L, T = obs.shape
    n = ring.n
    E = ring.E()
    Tm = np.stack([circulant(ring.q[c]) for c in range(ring.C)])  # (C,n,n)
    if not reset:
        # flatten all lines into one long sequence
        o = obs[obs >= 0][None, :]
        cx = ctx[obs >= 0][None, :]
        obs, ctx = o, cx
        L, T = obs.shape
        lens = np.array([T])
    alpha = np.zeros((L, T, n))
    scale = np.zeros((L, T))
    a = ring.pi[None, :] * E[obs[:, 0]]
    s = a.sum(1); scale[:, 0] = s; alpha[:, 0] = a / s[:, None]
    for t in range(1, T):
        valid = obs[:, t] >= 0
        prev = alpha[:, t - 1]
        nxt = _apply(prev, Tm, ctx[:, t])
        a = nxt * E[np.maximum(obs[:, t], 0)]
        s = a.sum(1)
        s = np.where(valid, s, 1.0)
        a = np.where(valid[:, None], a / s[:, None], prev)
        alpha[:, t] = a; scale[:, t] = s
    ll = np.log(scale).sum()
    if not need_post:
        return ll, None, None
    beta = np.ones((L, T, n))
    stepc = np.zeros((ring.C, n))
    I = np.arange(n)
    D = (I[None, :] - I[:, None]) % n
    for t in range(T - 1, 0, -1):
        valid = obs[:, t] >= 0
        eb = E[np.maximum(obs[:, t], 0)] * beta[:, t] / scale[:, t][:, None]  # (L,n)
        # xi(i,j) = alpha_{t-1}(i) T(i,j) eb(j)
        for c in range(ring.C):
            sel = valid & (ctx[:, t] == c)
            if sel.any():
                X = alpha[sel, t - 1].T @ eb[sel]  # (n,n)
                X *= Tm[c]
                stepc[c] += np.bincount(D.ravel(), weights=X.ravel(), minlength=n)
        b = _apply_T(eb, Tm, ctx[:, t])
        beta[:, t - 1] = np.where(valid[:, None], b, beta[:, t])
    post0 = alpha[:, 0] * beta[:, 0]
    post0 /= post0.sum(1, keepdims=True)
    return ll, stepc, post0.sum(0)


def em(ring, obs, ctx, lens, reset=True, iters=8, alpha_q=0.5, alpha_pi=0.5, fix_uniform=False):
    ll = None
    for _ in range(iters):
        ll, sc, st = forward_backward(ring, obs, ctx, lens, reset, need_post=True)
        if fix_uniform:
            ring.pi = (st + alpha_pi) / (st + alpha_pi).sum()
            continue
        ring.q = (sc + alpha_q) / (sc + alpha_q).sum(1, keepdims=True)
        if reset:
            ring.pi = (st + alpha_pi) / (st + alpha_pi).sum()
    ll, _, _ = forward_backward(ring, obs, ctx, lens, reset)
    return ll


def init_labels(counts, n, rng):
    """Cells per symbol proportional to frequency (>=1 for every symbol), random order."""
    S = len(counts)
    p = np.asarray(counts, float) + 0.5
    k = np.ones(S, int)
    rest = n - S
    if rest > 0:
        extra = np.floor(p / p.sum() * rest).astype(int)
        k += extra
        while k.sum() < n:
            k[np.argmax(p / k)] += 1
    lab = np.repeat(np.arange(S), k)[:n]
    rng.shuffle(lab)
    return list(lab)


def anneal_ring(obs, ctx, lens, S, n, C=1, reset=True, steps=800, seed=0, T0=30.0, refit_every=40):
    """Simulated annealing over ring contents (swap two cells / relabel one cell).
    Proposals are scored by the forward likelihood under the current turning rule;
    the rule (q, pi) is refitted by EM every refit_every steps."""
    rng = np.random.RandomState(seed)
    counts = np.bincount(obs[obs >= 0], minlength=S)
    lab = init_labels(counts, n, rng)
    r = Ring(lab, S, C)
    cur = em(r, obs, ctx, lens, reset, iters=4)
    best = (cur, list(r.lab), r.q.copy(), r.pi.copy())
    for it in range(steps):
        Tt = T0 * (1 - it / steps) ** 2 + 0.05
        lab2 = r.lab.copy()
        u = rng.rand()
        if u < 0.6:
            i, j = rng.choice(n, 2, replace=False); lab2[i], lab2[j] = lab2[j], lab2[i]
        elif u < 0.8:   # move a block (segment reversal, 2-opt style)
            i, j = sorted(rng.choice(n, 2, replace=False)); lab2[i:j + 1] = lab2[i:j + 1][::-1].copy()
        else:
            i = rng.randint(n); lab2[i] = rng.randint(S)
        r2 = Ring(lab2, S, C); r2.q = r.q; r2.pi = r.pi
        new = forward_backward(r2, obs, ctx, lens, reset)[0]
        if new > cur or rng.rand() < math.exp((new - cur) / Tt):
            r, cur = r2, new
        if (it + 1) % refit_every == 0:
            r.q = r.q.copy(); r.pi = r.pi.copy()
            cur = em(r, obs, ctx, lens, reset, iters=3)
        if cur > best[0]:
            best = (cur, r.lab.copy(), r.q.copy(), r.pi.copy())
    r = Ring(best[1], S, C); r.q, r.pi = best[2].copy(), best[3].copy()
    best_ll = em(r, obs, ctx, lens, reset, iters=12)
    return r, best_ll


def kernel_summary(q):
    """Entropy (bits) of a step kernel and its top steps."""
    q = np.asarray(q); H = -(q * np.log2(q + 1e-300)).sum()
    top = np.argsort(-q)[:3]
    return H, [(int(d), round(float(q[d]), 3)) for d in top]


# ------------------------------------------------------------------ baselines
def ctx_array(O, k_ctx, C, vocab_sizes):
    """Context of the transition into t: class of previous token's filler in slot k_ctx.
    Classes: the C-1 commonest symbols get their own class, rest share class C-1."""
    L, T, K = O.shape
    ctx = np.zeros((L, T), dtype=np.int64)
    if C == 1:
        return ctx, None
    prev = O[:, :-1, k_ctx]
    vals = prev[prev >= 0]
    top = [s for s, _ in Counter(vals.tolist()).most_common(C - 1)]
    m = {s: i for i, s in enumerate(top)}
    f = np.vectorize(lambda s: m.get(s, C - 1))
    ctx[:, 1:] = np.where(prev >= 0, f(np.maximum(prev, 0)), 0)
    return ctx, top


def markov_ll(train, test, S, order=1, add=0.5, reset=True, ctx_train=None, ctx_test=None):
    """Within-slot Markov chain (order 0/1/2) with line-start distribution; additive smoothing.
    Optional extra context (ctx arrays) to condition on as well."""
    def grams(obs, ctx):
        for l in range(obs.shape[0]):
            h = []
            for t in range(obs.shape[1]):
                s = obs[l, t]
                if s < 0: break
                c = ctx[l, t] if ctx is not None else 0
                key = (('S',) if t == 0 else tuple(h[-order:]) if order else ()) + (c,)
                if t == 0 and order: key = ('S', c)
                yield key, s
                h.append(s)
    cnt = defaultdict(Counter)
    for key, s in grams(train, ctx_train):
        cnt[key][s] += 1
    uni = Counter(); [uni.update(c) for c in cnt.values()]
    tot = sum(uni.values())
    ll = 0.0; n = 0
    for key, s in grams(test, ctx_test):
        pu = (uni[s] + add) / (tot + add * S)
        c = cnt.get(key)
        if c:
            N = sum(c.values()); lam = N / (N + 5.0)
            p = lam * (c[s] + add * pu * S) / (N + add * S) + (1 - lam) * pu
        else:
            p = pu
        ll += math.log(p); n += 1
    nparams = sum(len(c) for c in cnt.values())
    return ll, n, nparams


def word_markov2(trainT, testT, d=0.75, prune=0):
    """Interpolated absolute-discount word trigram on filler tuples, line-reset (BOS tokens)."""
    c3, c2, c1 = defaultdict(Counter), defaultdict(Counter), Counter()
    for L in trainT:
        h = ['<s>', '<s>'] + L
        for i in range(2, len(h)):
            c3[(h[i - 2], h[i - 1])][h[i]] += 1
            c2[h[i - 1]][h[i]] += 1
            c1[h[i]] += 1
    if prune:
        for dct in (c3, c2):
            for k in list(dct):
                for w in list(dct[k]):
                    if dct[k][w] <= prune: del dct[k][w]
                if not dct[k]: del dct[k]
    V = len(set(c1) | {w for L in testT for w in L}) + 1
    N1 = sum(c1.values())
    def p1(w): return (c1[w] + 0.5) / (N1 + 0.5 * V)
    def pk(c, w, lower):
        if not c: return lower
        N = sum(c.values()); u = len(c)
        return max(c[w] - d, 0) / N + d * u / N * lower
    ll = 0.0; n = 0
    for L in testT:
        h = ['<s>', '<s>'] + L
        for i in range(2, len(h)):
            w = h[i]
            p = pk(c3.get((h[i - 2], h[i - 1])), w, pk(c2.get(h[i - 1]), w, p1(w)))
            ll += math.log(p); n += 1
    nparams = sum(len(c) for c in c3.values()) + sum(len(c) for c in c2.values()) + len(c1)
    return ll, n, nparams


def tuples(O):
    out = []
    for l in range(O.shape[0]):
        L = []
        for t in range(O.shape[1]):
            if O[l, t, 0] < 0: break
            L.append(tuple(O[l, t]))
        out.append(L)
    return out


def write_rows(path, header, rows):
    with open(path, 'w') as f:
        f.write(header.rstrip() + '\n\n| ID | Method and control | Result | Verdict |\n|---|---|---|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')
