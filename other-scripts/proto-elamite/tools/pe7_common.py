"""pe7 shared machinery: names treated as DNA.

* EditModel  : stochastic string-edit model with match / substitution / insertion /
               deletion rates (insert and substitute draw the new element from the
               corpus element frequencies).  Fitted by hard EM: every name is
               Viterbi-aligned to its nearest neighbour, op counts re-estimate rates.
               Distance d(a,b) = (C(a->b) + C(b->a) - C(a->a) - C(b->b)) / 2, in bits.
* nj         : neighbour joining (numpy), returns a rooted binary tree (arrays).
* delta      : mean quartet delta score (Holland et al. 2002) on sampled quartets.
* Parsimony  : Fitch parsimony on binary element-presence characters, characters
               packed as Python big-int bitsets; NNI + SPR hill climbing from NJ and
               random-addition trees (many restarts); CI and RI.
* mcmc       : Metropolis tree search over SPR moves with pseudo-likelihood
               exp(-L / T) (T = 1); clade support of the best tree.
* label_ps   : Fitch parsimony score of a (set-valued) label on the tree vs
               leaf-label permutations (trait-association test).
* pair_test  : mean distance of label-sharing pairs vs permutation.
No sign readings are used anywhere.
"""
import json, math, os, random, sys
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(DATA, 'pe7_ckpt')
os.makedirs(CKPT, exist_ok=True)
LG2 = math.log(2)


def corpora():
    return json.load(open(os.path.join(DATA, 'pe7_corpora.json')))


# ------------------------------------------------------------------ edit model
class EditModel:
    def __init__(self, names, rates=(0.85, 0.05, 0.05, 0.05)):
        f = Counter(s for w in names for s in w)
        tot = sum(f.values())
        self.lf = {s: -math.log2(c / tot) for s, c in f.items()}
        self.lf_default = -math.log2(0.5 / tot)
        self.set_rates(*rates)

    def set_rates(self, pm, ps, pi, pd):
        z = pm + ps + pi + pd
        self.rates = (pm / z, ps / z, pi / z, pd / z)
        self.cm, self.cs, self.ci, self.cd = (-math.log2(x / z) for x in (pm, ps, pi, pd))

    def align(self, a, b, ops=False):
        """Viterbi cost of a -> b (bits); optionally op counts (m, s, i, d)."""
        n, m = len(a), len(b)
        lf, df = self.lf, self.lf_default
        cm, cs, ci, cd = self.cm, self.cs, self.ci, self.cd
        INF = 1e18
        D = [[INF] * (m + 1) for _ in range(n + 1)]
        B = [[0] * (m + 1) for _ in range(n + 1)] if ops else None
        D[0][0] = 0.0
        for i in range(n + 1):
            Di = D[i]
            for j in range(m + 1):
                if i == 0 and j == 0:
                    continue
                best, bk = INF, 0
                if i and j:
                    if a[i - 1] == b[j - 1]:
                        c = D[i - 1][j - 1] + cm
                        k = 1
                    else:
                        c = D[i - 1][j - 1] + cs + lf.get(b[j - 1], df)
                        k = 2
                    if c < best:
                        best, bk = c, k
                if j:
                    c = Di[j - 1] + ci + lf.get(b[j - 1], df)
                    if c < best:
                        best, bk = c, 3
                if i:
                    c = D[i - 1][j] + cd
                    if c < best:
                        best, bk = c, 4
                Di[j] = best
                if ops:
                    B[i][j] = bk
        if not ops:
            return D[n][m]
        cnt = [0, 0, 0, 0]
        i, j = n, m
        while i or j:
            k = B[i][j]
            cnt[k - 1] += 1
            if k in (1, 2):
                i, j = i - 1, j - 1
            elif k == 3:
                j -= 1
            else:
                i -= 1
        return D[n][m], cnt

    def dist_matrix(self, names):
        n = len(names)
        self_c = [self.align(w, w) for w in names]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    C[i, j] = self.align(names[i], names[j])
        D = (C + C.T - np.add.outer(self_c, self_c)) / 2.0
        np.fill_diagonal(D, 0.0)
        return np.maximum(D, 0.0)

    def fit(self, names, iters=4):
        D = None
        for _ in range(iters):
            D = self.dist_matrix(names)
            Dn = D + np.eye(len(names)) * 1e18
            nn = Dn.argmin(1)
            tot = np.zeros(4)
            for i, j in enumerate(nn):
                _, c = self.align(names[j], names[i], ops=True)   # neighbour -> name
                tot += c
            tot += 0.5
            self.set_rates(*tot)
        return self.dist_matrix(names)


# ------------------------------------------------------------------ trees
class Tree:
    """Rooted binary tree: leaves 0..n-1, internal n..2n-2; L, R, P arrays."""

    def __init__(self, n):
        self.n = n
        N = 2 * n - 1
        self.L = [-1] * N
        self.R = [-1] * N
        self.P = [-1] * N
        self.root = -1

    def copy(self):
        t = Tree.__new__(Tree)
        t.n, t.L, t.R, t.P, t.root = self.n, self.L[:], self.R[:], self.P[:], self.root
        return t

    def postorder(self):
        out, st = [], [(self.root, 0)]
        while st:
            v, s = st.pop()
            if v < self.n:
                out.append(v)
            elif s == 0:
                st.append((v, 1))
                st.append((self.R[v], 0))
                st.append((self.L[v], 0))
            else:
                out.append(v)
        return out

    def clades(self):
        """bitmask of leaves under every internal node except the root."""
        m = {}
        res = []
        for v in self.postorder():
            if v < self.n:
                m[v] = 1 << v
            else:
                m[v] = m[self.L[v]] | m[self.R[v]]
                if v != self.root:
                    res.append(m[v])
        return res

    def leafsets(self):
        m = {}
        for v in self.postorder():
            m[v] = (1 << v) if v < self.n else m[self.L[v]] | m[self.R[v]]
        return m


def random_tree(n, rng):
    t = Tree(n)
    nodes = list(range(n))
    nxt = n
    while len(nodes) > 1:
        i, j = rng.sample(range(len(nodes)), 2)
        a, b = nodes[i], nodes[j]
        t.L[nxt], t.R[nxt] = a, b
        t.P[a] = t.P[b] = nxt
        for k in sorted((i, j), reverse=True):
            nodes.pop(k)
        nodes.append(nxt)
        nxt += 1
    t.root = nodes[0]
    return t


def nj(D):
    D = np.array(D, float)
    n = len(D)
    t = Tree(n)
    active = list(range(n))
    ids = list(range(n))
    nxt = n
    M = D.copy()
    while len(active) > 2:
        k = len(active)
        sub = M[np.ix_(active, active)]
        r = sub.sum(1)
        Q = (k - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        a, b = active[i], active[j]
        na, nb = ids[a], ids[b]
        t.L[nxt], t.R[nxt] = na, nb
        t.P[na] = t.P[nb] = nxt
        # new distances stored in slot a
        newd = 0.5 * (M[a, :] + M[b, :] - M[a, b])
        M[a, :] = newd
        M[:, a] = newd
        M[a, a] = 0
        ids[a] = nxt
        active.remove(b)
        nxt += 1
    a, b = active
    t.L[nxt], t.R[nxt] = ids[a], ids[b]
    t.P[ids[a]] = t.P[ids[b]] = nxt
    t.root = nxt
    return t


def delta_score(D, nq=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(D)
    Q = np.array([rng.choice(n, 4, replace=False) for _ in range(nq)])
    i, j, k, l = Q.T
    s = np.stack([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]], 1)
    s.sort(1)
    s3, s2, s1 = s[:, 0], s[:, 1], s[:, 2]
    ok = s1 - s3 > 1e-9
    return float(np.mean((s1[ok] - s2[ok]) / (s1[ok] - s3[ok])))


# ------------------------------------------------------------------ parsimony
class Parsimony:
    def __init__(self, names, min_count=2):
        n = len(names)
        c = Counter(s for w in names for s in set(w))
        self.chars = [s for s, k in c.items() if min_count <= k <= n - 2]
        idx = {s: i for i, s in enumerate(self.chars)}
        self.m = len(self.chars)
        self.full = (1 << self.m) - 1
        self.n = n
        self.O0 = [0] * n
        for v, w in enumerate(names):
            b = 0
            for s in set(w):
                if s in idx:
                    b |= 1 << idx[s]
            self.O0[v] = b
        cnt = [c[s] for s in self.chars]
        self.minlen = self.m
        self.maxlen = sum(min(k, n - k) for k in cnt)

    def score(self, t):
        n, full = self.n, self.full
        Z = {}
        O = {}
        L = 0
        for v in t.postorder():
            if v < n:
                O[v] = self.O0[v]
                Z[v] = full & ~self.O0[v]
            else:
                a, b = t.L[v], t.R[v]
                i0 = Z[a] & Z[b]
                i1 = O[a] & O[b]
                e = full & ~(i0 | i1)
                if e:
                    L += e.bit_count()
                    i0 |= e & (Z[a] | Z[b])
                    i1 |= e & (O[a] | O[b])
                Z[v], O[v] = i0, i1
        return L

    def ci_ri(self, L):
        ci = self.minlen / L if L else 1.0
        ri = (self.maxlen - L) / (self.maxlen - self.minlen) if self.maxlen > self.minlen else 0.0
        return ci, ri

    def per_char(self, t):
        """per-character Fitch steps on tree t."""
        n, full = self.n, self.full
        Z, O = {}, {}
        steps = [0] * self.m
        for v in t.postorder():
            if v < n:
                O[v] = self.O0[v]
                Z[v] = full & ~self.O0[v]
            else:
                a, b = t.L[v], t.R[v]
                i0 = Z[a] & Z[b]
                i1 = O[a] & O[b]
                e = full & ~(i0 | i1)
                x = e
                while x:
                    lb = x & -x
                    steps[lb.bit_length() - 1] += 1
                    x ^= lb
                i0 |= e & (Z[a] | Z[b])
                i1 |= e & (O[a] | O[b])
                Z[v], O[v] = i0, i1
        return steps


def spr(t, s, tgt):
    """prune subtree s and regraft it on the edge above tgt; returns new tree or None."""
    if s == t.root:
        return None
    p = t.P[s]
    q = t.R[p] if t.L[p] == s else t.L[p]
    if tgt in (s, p, q) and tgt != q:
        return None
    if tgt == q:
        return None
    # tgt must not be inside subtree s
    x = tgt
    while x != -1:
        if x == s:
            return None
        x = t.P[x]
    u = t.copy()
    g = u.P[p]
    if g == -1:
        u.root = q
        u.P[q] = -1
    else:
        if u.L[g] == p:
            u.L[g] = q
        else:
            u.R[g] = q
        u.P[q] = g
    tp = u.P[tgt]
    u.L[p], u.R[p] = s, tgt
    u.P[s] = p
    u.P[tgt] = p
    if tp == -1:
        u.root = p
        u.P[p] = -1
    else:
        if u.L[tp] == tgt:
            u.L[tp] = p
        else:
            u.R[tp] = p
        u.P[p] = tp
    return u


def climb(P, t, rng, max_fail=None):
    """random SPR hill climbing (ties accepted) until max_fail consecutive failures."""
    N = 2 * P.n - 1
    max_fail = max_fail or 6 * N
    L = P.score(t)
    fail = 0
    while fail < max_fail:
        s = rng.randrange(N)
        tg = rng.randrange(N)
        u = spr(t, s, tg)
        if u is None:
            continue
        Lu = P.score(u)
        if Lu < L:
            t, L, fail = u, Lu, 0
        else:
            if Lu == L:
                t = u
            fail += 1
    return t, L


def best_tree(P, D, rng, restarts=6):
    starts = [nj(D)] + [random_tree(P.n, rng) for _ in range(restarts - 1)]
    best = None
    lens = []
    for t0 in starts:
        t, L = climb(P, t0, rng)
        lens.append(L)
        if best is None or L < best[1]:
            best = (t, L)
    return best[0], best[1], lens


def mcmc(P, t, rng, T=1.0, steps=None, nsamp=100):
    N = 2 * P.n - 1
    steps = steps or 40 * N
    L = P.score(t)
    burn = steps // 4
    every = max(1, (steps - burn) // nsamp)
    sup = Counter()
    ns = 0
    acc = 0
    for it in range(steps):
        u = spr(t, rng.randrange(N), rng.randrange(N))
        if u is not None:
            Lu = P.score(u)
            if Lu <= L or rng.random() < math.exp(-(Lu - L) / T):
                t, L = u, Lu
                acc += 1
        if it >= burn and (it - burn) % every == 0:
            ns += 1
            for c in set(_canon(x, P.n) for x in t.clades()):
                sup[c] += 1
    return sup, ns, acc / steps


def _canon(mask, n):
    full = (1 << n) - 1
    c = full ^ mask
    return min(mask, c)


def clade_support(best, sup, ns, n, minsize=2):
    vals = []
    for c in best.clades():
        cc = _canon(c, n)
        k = bin(cc).count('1')
        if k < minsize or k > n - minsize:
            continue
        vals.append(sup[cc] / ns)
    vals = np.array(vals) if vals else np.zeros(1)
    return float(vals.mean()), float((vals >= 0.5).mean()), float((vals >= 0.9).mean())


# ------------------------------------------------------------------ labels
def label_ps(t, labsets, n):
    """Fitch score of set-valued leaf labels (list of int bitmasks; 0 = unknown)."""
    st = {}
    L = 0
    for v in t.postorder():
        if v < n:
            st[v] = labsets[v] if labsets[v] else -1
        else:
            a, b = st[t.L[v]], st[t.R[v]]
            if a == -1:
                st[v] = b
            elif b == -1:
                st[v] = a
            else:
                i = a & b
                if i:
                    st[v] = i
                else:
                    st[v] = a | b
                    L += 1
    return L


def label_assoc(t, labs, rng, nperm=499):
    """labs: list of sets (labels per leaf).  Returns obs PS, null mean, p (lower = clustered)."""
    n = len(labs)
    vocab = {x: i for i, x in enumerate(sorted({x for s in labs for x in s}, key=str))}
    masks = [sum(1 << vocab[x] for x in s) for s in labs]
    obs = label_ps(t, masks, n)
    null = []
    for _ in range(nperm):
        m = masks[:]
        rng.shuffle(m)
        null.append(label_ps(t, m, n))
    null = np.array(null)
    p = (1 + (null <= obs).sum()) / (nperm + 1)
    return {'ps': obs, 'null': float(null.mean()), 'ratio': float(null.mean() / obs) if obs else None,
            'z': float((null.mean() - obs) / (null.std() + 1e-9)), 'p': float(p)}


def pair_test(D, labs, rng, nperm=999, exclude=None, groups=None):
    """mean distance of pairs sharing >=1 label minus mean of other pairs.
    exclude: boolean n x n mask of pairs to ignore (e.g. same tablet).
    groups : if given, permute labels among groups (list of group id per leaf;
             all leaves of a group carry the group's label set)."""
    n = len(labs)
    vocab = sorted({x for s in labs for x in s}, key=str)
    if not vocab:
        return None
    vi = {x: i for i, x in enumerate(vocab)}
    M = np.zeros((n, len(vocab)), bool)
    for i, s in enumerate(labs):
        for x in s:
            M[i, vi[x]] = True
    iu = np.triu_indices(n, 1)
    ok = np.ones(len(iu[0]), bool)
    if exclude is not None:
        ok = ~exclude[iu]
    d = D[iu][ok]

    def stat(Mx):
        S = (Mx.astype(np.int32) @ Mx.T.astype(np.int32)) > 0
        s = S[iu][ok]
        if s.sum() == 0 or (~s).sum() == 0:
            return np.nan, 0
        return d[s].mean() - d[~s].mean(), int(s.sum())

    obs, npairs = stat(M)
    null = []
    if groups is None:
        for _ in range(nperm):
            null.append(stat(M[rng.permutation(n)])[0])
    else:
        g = np.array(groups)
        ug = np.unique(g)
        rows = {x: np.where(g == x)[0] for x in ug}
        rep = {x: M[rows[x][0]] for x in ug}
        for _ in range(nperm):
            pg = rng.permutation(ug)
            Mx = np.zeros_like(M)
            for a, b in zip(ug, pg):
                Mx[rows[a]] = rep[b]
            null.append(stat(Mx)[0])
    null = np.array([x for x in null if not np.isnan(x)])
    if np.isnan(obs) or len(null) == 0:
        return None
    p = (1 + (null <= obs).sum()) / (len(null) + 1)
    return {'diff': float(obs), 'null': float(null.mean()), 'sd': float(null.std()),
            'z': float((null.mean() - obs) / (null.std() + 1e-9)), 'p': float(p), 'pairs': npairs}


# ------------------------------------------------------------------ helpers
def shuffle_elements(names, rng):
    toks = [s for w in names for s in w]
    rng.shuffle(toks)
    out, k = [], 0
    for w in names:
        out.append(tuple(toks[k:k + len(w)]))
        k += len(w)
    return out


def random_strings(names, rng):
    f = Counter(s for w in names for s in w)
    el, wt = zip(*f.items())
    return [tuple(rng.choices(el, wt, k=len(w))) for w in names]


def ckpt(key):
    p = os.path.join(CKPT, key + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def save_ckpt(key, obj):
    p = os.path.join(CKPT, key + '.json')
    json.dump(obj, open(p + '.tmp', 'w'))
    os.replace(p + '.tmp', p)


def tree_stats(names, rng, restarts=6, do_mcmc=True, mcmc_steps=None):
    """fit the edit model, NJ + parsimony search + MCMC; return stats, tree, D, model."""
    em = EditModel(names)
    D = em.fit(names)
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    dl = delta_score(D, rng=nrng)
    P = Parsimony(names)
    t, L, lens = best_tree(P, D, rng, restarts)
    ci, ri = P.ci_ri(L)
    out = {'n': len(names), 'chars': P.m, 'rates': em.rates, 'delta': dl, 'L': L,
           'ci': ci, 'ri': ri, 'restart_lens': lens, 'L_nj': P.score(nj(D))}
    if do_mcmc:
        sup, ns, acc = mcmc(P, t.copy(), rng, steps=mcmc_steps)
        ms, f50, f90 = clade_support(t, sup, ns, P.n)
        out.update({'support_mean': ms, 'support_ge50': f50, 'support_ge90': f90, 'acc': acc})
    return out, t, D, em, P
