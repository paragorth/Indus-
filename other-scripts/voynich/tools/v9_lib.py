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
        if getattr(self, 'B', None) is not None:
            return self.B
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


def forward_backward(ring, obs, ctx, lens, reset=True, need_post=False, need_emit=False):
    """obs (L,T) symbols (-1 pad, suffix only); ctx (L,T) context of the transition INTO t.
    reset=False: all lines are concatenated in corpus order (state carried over).
    Returns loglik, and if need_post: expected step counts (C,n), start counts (n)."""
    n = ring.n
    E = ring.E()
    Tm = np.stack([circulant(ring.q[c]) for c in range(ring.C)])  # (C,n,n)
    if not reset:
        m = obs >= 0
        obs, ctx = obs[m][None, :], ctx[m][None, :]
    ln = (obs >= 0).sum(1)
    order = np.argsort(-ln, kind='stable')
    obs, ctx, ln = obs[order], ctx[order], ln[order]
    L, T = obs.shape
    T = int(ln.max())
    act = np.array([(ln > t).sum() for t in range(T)])
    alpha = np.zeros((L, T, n))
    scale = np.ones((L, T))
    a0 = ring.pi[None, :] * E[obs[:, 0]]
    s = a0.sum(1); scale[:, 0] = s; alpha[:, 0] = a0 / s[:, None]
    for t in range(1, T):
        k = act[t]
        nxt = _apply(alpha[:k, t - 1], Tm, ctx[:k, t])
        a1 = nxt * E[obs[:k, t]]
        s = a1.sum(1)
        alpha[:k, t] = a1 / s[:, None]; scale[:k, t] = s
    ll = np.log(scale).sum()
    if not need_post:
        return ll, None, None
    beta = np.ones((L, T, n))
    stepc = np.zeros((ring.C, n))
    I = np.arange(n)
    D = ((I[None, :] - I[:, None]) % n).ravel()
    for t in range(T - 1, 0, -1):
        k = act[t]
        eb = E[obs[:k, t]] * beta[:k, t] / scale[:k, t][:, None]
        cc = ctx[:k, t]
        for c in range(ring.C):
            sel = cc == c
            if sel.any():
                X = alpha[:k, t - 1][sel].T @ eb[sel]
                X *= Tm[c]
                stepc[c] += np.bincount(D, weights=X.ravel(), minlength=n)
        beta[:k, t - 1] = _apply_T(eb, Tm, cc)
    post0 = alpha[:, 0] * beta[:, 0]
    post0 /= post0.sum(1, keepdims=True)
    if need_emit:
        G = alpha * beta
        G /= np.maximum(G.sum(2, keepdims=True), 1e-300)
        valid = obs[:, :T] >= 0
        em_c = np.zeros((ring.S, n))
        np.add.at(em_c, obs[:, :T][valid], G[valid])
        return ll, stepc, post0.sum(0), em_c
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


def chain_labels(obs, S, n, rng):
    """Greedy initial ring: walk the commonest within-line transitions (step +1), respecting
    per-symbol cell quotas proportional to frequency."""
    counts = np.bincount(obs[obs >= 0], minlength=S)
    quota = Counter(init_labels(counts, n, rng))
    Tr = np.ones((S, S)) * 0.01
    a, b = obs[:, :-1].ravel(), obs[:, 1:].ravel(); m = (a >= 0) & (b >= 0)
    np.add.at(Tr, (a[m], b[m]), 1)
    Tr /= Tr.sum(1, keepdims=True)
    cur = int(np.argmax(counts)); lab = [cur]; quota[cur] -= 1
    while len(lab) < n:
        cand = [s for s in range(S) if quota[s] > 0]
        cur = max(cand, key=lambda s: Tr[cur, s] / max(counts[s], 1) ** 0.5)
        lab.append(cur); quota[cur] -= 1
    return lab


def anneal_ring(obs, ctx, lens, S, n, C=1, reset=True, steps=800, seed=0, T0=30.0, refit_every=40):
    """Simulated annealing over ring contents (swap two cells / relabel one cell).
    Proposals are scored by the forward likelihood under the current turning rule;
    the rule (q, pi) is refitted by EM every refit_every steps."""
    rng = np.random.RandomState(seed)
    counts = np.bincount(obs[obs >= 0], minlength=S)
    lab = chain_labels(obs, S, n, rng) if seed == 0 else init_labels(counts, n, rng)
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


def harden(B):
    """One symbol per cell: argmax, then give every symbol absent from the ring the cell
    (among cells whose symbol is duplicated) where it is most probable."""
    B = np.asarray(B)
    lab = B.argmax(0)
    S = B.shape[0]
    for s in np.argsort(-B.sum(1)):
        if (lab == s).any():
            continue
        cnt = np.bincount(lab, minlength=S)
        cand = [c for c in range(len(lab)) if cnt[lab[c]] > 1]
        if not cand:
            break
        c = max(cand, key=lambda c: B[s, c])
        lab[c] = s
    return lab


def soft_em(obs, ctx, S, n, C=1, iters=80, seed=0, reset=True, trace=None):
    """Volvelle as an HMM: n cells on a cycle, circulant (rule) transitions, FREE emission
    distribution per cell; Baum-Welch.  Then harden: each cell keeps its argmax symbol."""
    rng = np.random.RandomState(seed)
    r = Ring(np.zeros(n, int), S, C)
    freq = np.bincount(obs[obs >= 0], minlength=S) + 1.0
    B = rng.dirichlet(np.ones(S), size=n).T * freq[:, None]
    r.B = B / B.sum(0, keepdims=True)
    q = rng.dirichlet(np.ones(n) * 0.5, size=C)
    r.q = q
    ll = None
    for it in range(iters):
        ll, sc, st, ec = forward_backward(r, obs, ctx, None, reset, need_post=True, need_emit=True)
        r.q = (sc + 0.1) / (sc + 0.1).sum(1, keepdims=True)
        if reset:
            r.pi = (st + 0.1) / (st + 0.1).sum()
        r.B = (ec + 1e-3) / (ec + 1e-3).sum(0, keepdims=True)
        if trace is not None and it % 20 == 0:
            trace.append((it, round(float(ll))))
    lab = harden(r.B)
    h = Ring(lab, S, C); h.q = r.q.copy(); h.pi = r.pi.copy()
    hll = em(h, obs, ctx, None, reset, iters=6)
    h.soft = r
    return h, hll, ll


def soft_em_multi(obs, ctx, S, n, C=1, seeds=6, burn=40, keep=2, more=120, reset=True, seed0=0):
    """Population search: `seeds` random Baum-Welch starts run `burn` iterations; the best `keep`
    continue `more` iterations; the best is hardened.  Returns (hard ring, hard ll, soft ll)."""
    pop = []
    for sd in range(seed0, seed0 + seeds):
        h, hll, sll = soft_em(obs, ctx, S, n, C=C, iters=burn, seed=sd, reset=reset)
        pop.append((sll, h.soft))
    pop.sort(key=lambda x: -x[0])
    best = None
    for sll, r in pop[:keep]:
        for it in range(more):
            ll, sc, st, ec = forward_backward(r, obs, ctx, None, reset, need_post=True, need_emit=True)
            r.q = (sc + 0.1) / (sc + 0.1).sum(1, keepdims=True)
            if reset:
                r.pi = (st + 0.1) / (st + 0.1).sum()
            r.B = (ec + 1e-3) / (ec + 1e-3).sum(0, keepdims=True)
        lab = harden(r.B)
        h = Ring(lab, S, C); h.q = r.q.copy(); h.pi = r.pi.copy()
        hll = em(h, obs, ctx, None, reset, iters=6)
        h.soft = r
        if best is None or hll > best[1]:
            best = (h, hll, ll)
    return best


def local_search(r, obs, ctx, reset=True, sweeps=3):
    """Steepest-ascent polish: all single relabels, then all swaps; refit rule after each sweep."""
    cur = forward_backward(r, obs, ctx, None, reset)[0]
    for _ in range(sweeps):
        improved = False
        for i in range(r.n):
            for s in range(r.S):
                if s == r.lab[i]: continue
                lab2 = r.lab.copy(); lab2[i] = s
                r2 = Ring(lab2, r.S, r.C); r2.q = r.q; r2.pi = r.pi
                v = forward_backward(r2, obs, ctx, None, reset)[0]
                if v > cur + 1e-6:
                    r.lab, cur, improved = lab2, v, True
        for i in range(r.n):
            for j in range(i + 1, r.n):
                if r.lab[i] == r.lab[j]: continue
                lab2 = r.lab.copy(); lab2[i], lab2[j] = lab2[j], lab2[i]
                r2 = Ring(lab2, r.S, r.C); r2.q = r.q; r2.pi = r.pi
                v = forward_backward(r2, obs, ctx, None, reset)[0]
                if v > cur + 1e-6:
                    r.lab, cur, improved = lab2, v, True
        cur = em(r, obs, ctx, None, reset, iters=4)
        if not improved:
            break
    return r, cur


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
