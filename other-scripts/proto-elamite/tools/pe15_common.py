"""pe15 shared code: bipartite 'food web' tools.

Network = list of events (consumer, resource, tablet).  Tools:
  matrix()        weighted consumer x resource count matrix (min degree filters)
  nodf()          binary NODF (Almeida-Neto 2008), 0-100
  curveball()     binary null with fixed row and column degrees (Strona 2014)
  h2()            Bluethgen H2' with exact max and greedy min bounds
  brim()          bipartite Barber modularity Q, many random restarts
  sbm_fit()       bipartite degree-corrected Poisson SBM (Karrer-Newman), Metropolis
                  annealing with restarts (numba); returns group labels and logL
  cond_prob()     P(resource | consumer) from a fitted SBM (smoothed); K=1 = degree only
  hist_prob()     P(resource | consumer) from the consumer's own history (Dirichlet)
"""
import json, math, os, random
from collections import Counter, defaultdict
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe15_ckpt')


def load_nets():
    return json.load(open(os.path.join(CK, 'nets.json')))


def subsample_tablets(ev, target, seed):
    """Random whole tablets until >= target events (size matching)."""
    by = defaultdict(list)
    for e in ev:
        by[e[2]].append(e)
    tabs = sorted(by)
    random.Random(seed).shuffle(tabs)
    out = []
    for t in tabs:
        out += by[t]
        if len(out) >= target:
            break
    return out


def matrix(ev, min_c=2, min_r=1, iters=3):
    """Weighted matrix; drop consumers with < min_c events and resources < min_r (iterated)."""
    ev = [tuple(e[:2]) for e in ev]
    for _ in range(iters):
        cc = Counter(e[0] for e in ev)
        rc = Counter(e[1] for e in ev)
        ev = [e for e in ev if cc[e[0]] >= min_c and rc[e[1]] >= min_r]
    C = sorted({e[0] for e in ev})
    R = sorted({e[1] for e in ev})
    ci = {c: i for i, c in enumerate(C)}
    ri = {r: i for i, r in enumerate(R)}
    W = np.zeros((len(C), len(R)), dtype=np.int64)
    for c, r in ev:
        W[ci[c], ri[r]] += 1
    return W, C, R


def nodf(B):
    B = (np.asarray(B) > 0).astype(np.float64)
    B = B[B.sum(1) > 0][:, B.sum(0) > 0]

    def side(M):
        d = M.sum(1)
        O = M @ M.T
        n = len(d)
        tot, cnt = 0.0, 0
        di = d[:, None]
        dj = d[None, :]
        mask = np.triu(np.ones((n, n), bool), 1)
        gt = (di > dj) & mask
        lt = (di < dj) & mask
        val = np.where(gt, O / np.maximum(dj, 1), 0) + np.where(lt, O / np.maximum(di, 1), 0)
        return val.sum(), mask.sum()
    a, na = side(B)
    b, nb = side(B.T)
    return 100.0 * (a + b) / max(na + nb, 1)


def curveball(B, seed, nswap=None):
    B = np.asarray(B) > 0
    rows = [set(np.nonzero(B[i])[0].tolist()) for i in range(B.shape[0])]
    rng = random.Random(seed)
    n = len(rows)
    nswap = nswap or 5 * n
    for _ in range(nswap):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        a, b = rows[i], rows[j]
        oa, ob = a - b, b - a
        if not oa or not ob:
            continue
        both = a & b
        pool = list(oa | ob)
        rng.shuffle(pool)
        rows[i] = both | set(pool[:len(oa)])
        rows[j] = both | set(pool[len(oa):])
    out = np.zeros(B.shape, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, list(r)] = 1
    return out


def shuffle_events(W, rng):
    """Weighted null: pair consumer tokens and resource tokens at random (fixed weighted marginals)."""
    r, c = np.nonzero(W)
    ci = np.repeat(r, W[r, c])
    ri = np.repeat(c, W[r, c])
    rng.shuffle(ri)
    M = np.zeros_like(W)
    np.add.at(M, (ci, ri), 1)
    return M


def _ent(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def h2(W):
    W = np.asarray(W, dtype=np.float64)
    N = W.sum()
    H = _ent((W / N).ravel())
    rs, cs = W.sum(1), W.sum(0)
    Hmax = _ent(np.outer(rs, cs).ravel() / N ** 2)   # independence = maximum entropy given marginals
    # greedy minimum: fill the largest remaining margins first (Bluethgen-style)
    r, c = rs.copy(), cs.copy()
    cells = []
    while r.sum() > 0.5:
        i, j = np.argmax(r), np.argmax(c)
        v = min(r[i], c[j])
        cells.append(v)
        r[i] -= v
        c[j] -= v
    Hmin = _ent(np.array(cells) / N)
    return (Hmax - H) / max(Hmax - Hmin, 1e-12), H


def brim(W, restarts=50, K=None, seed=0):
    """Barber bipartite modularity, BRIM from random column labels; best of restarts."""
    A = (np.asarray(W) > 0).astype(np.float64)
    m = A.sum()
    k = A.sum(1)
    d = A.sum(0)
    Bm = A - np.outer(k, d) / m
    rng = np.random.default_rng(seed)
    best, bl = -1, None
    nr, nc = A.shape
    for r in range(restarts):
        KK = K or rng.integers(2, max(3, min(nc, 25)))
        cl = rng.integers(0, KK, nc)
        prevQ = -9
        for it in range(100):
            T = np.zeros((nc, KK)); T[np.arange(nc), cl] = 1
            rl = np.argmax(Bm @ T, 1)
            R = np.zeros((nr, KK)); R[np.arange(nr), rl] = 1
            cl = np.argmax(Bm.T @ R, 1)
            T = np.zeros((nc, KK)); T[np.arange(nc), cl] = 1
            Q = (R * (Bm @ T)).sum() / m
            if Q <= prevQ + 1e-10:
                break
            prevQ = Q
        if prevQ > best:
            best, bl = prevQ, (rl.copy(), cl.copy())
    return best, bl


# ----------------------------------------------------------------------------- DC-SBM
@njit(cache=True)
def _xlogx(x):
    return x * np.log(x) if x > 0 else 0.0


@njit(cache=True)
def _sbm_run(ei, ej, ew, nC, nR, Kc, Kr, sweeps, seed, bc0, br0, use_init):
    np.random.seed(seed)
    bc = np.empty(nC, np.int64)
    br = np.empty(nR, np.int64)
    for i in range(nC):
        bc[i] = bc0[i] if use_init else np.random.randint(Kc)
    for j in range(nR):
        br[j] = br0[j] if use_init else np.random.randint(Kr)
    # adjacency lists
    cdeg = np.zeros(nC); rdeg = np.zeros(nR)
    for e in range(len(ei)):
        cdeg[ei[e]] += ew[e]; rdeg[ej[e]] += ew[e]
    cptr = np.zeros(nC + 1, np.int64); rptr = np.zeros(nR + 1, np.int64)
    for e in range(len(ei)):
        cptr[ei[e] + 1] += 1; rptr[ej[e] + 1] += 1
    for i in range(nC):
        cptr[i + 1] += cptr[i]
    for j in range(nR):
        rptr[j + 1] += rptr[j]
    cadj = np.empty(len(ei), np.int64); cw = np.empty(len(ei)); radj = np.empty(len(ei), np.int64); rw = np.empty(len(ei))
    cf = cptr.copy(); rf = rptr.copy()
    for e in range(len(ei)):
        cadj[cf[ei[e]]] = ej[e]; cw[cf[ei[e]]] = ew[e]; cf[ei[e]] += 1
        radj[rf[ej[e]]] = ei[e]; rw[rf[ej[e]]] = ew[e]; rf[ej[e]] += 1
    M = np.zeros((Kc, Kr)); kc = np.zeros(Kc); kr = np.zeros(Kr)
    for e in range(len(ei)):
        M[bc[ei[e]], br[ej[e]]] += ew[e]
    for a in range(Kc):
        for s in range(Kr):
            kc[a] += M[a, s]; kr[s] += M[a, s]

    L = 0.0
    for a in range(Kc):
        for s in range(Kr):
            if M[a, s] > 0:
                L += M[a, s] * np.log(M[a, s] / (kc[a] * kr[s]))
    bestL = L
    bbc = bc.copy(); bbr = br.copy()
    dv = np.zeros(max(Kc, Kr))
    for sw in range(sweeps):
        beta = 1.0 + 30.0 * (sw / max(sweeps - 1, 1)) ** 2  # anneal from beta 1 towards greedy
        for side in range(2):
            n = nC if side == 0 else nR
            for _ in range(n):
                if side == 0:
                    i = np.random.randint(nC); a = bc[i]; b = np.random.randint(Kc)
                    if a == b:
                        continue
                    for s in range(Kr):
                        dv[s] = 0.0
                    for p in range(cptr[i], cptr[i + 1]):
                        dv[br[cadj[p]]] += cw[p]
                    k = cdeg[i]
                    old = 0.0; new = 0.0
                    for s in range(Kr):
                        old += _xlogx(M[a, s]) + _xlogx(M[b, s])
                        new += _xlogx(M[a, s] - dv[s]) + _xlogx(M[b, s] + dv[s])
                    old -= _xlogx(kc[a]) + _xlogx(kc[b])
                    new -= _xlogx(kc[a] - k) + _xlogx(kc[b] + k)
                    dL = new - old
                    if dL >= 0 or np.random.random() < np.exp(beta * dL):
                        for s in range(Kr):
                            M[a, s] -= dv[s]; M[b, s] += dv[s]
                        kc[a] -= k; kc[b] += k; bc[i] = b; L += dL
                else:
                    j = np.random.randint(nR); a = br[j]; b = np.random.randint(Kr)
                    if a == b:
                        continue
                    for s in range(Kc):
                        dv[s] = 0.0
                    for p in range(rptr[j], rptr[j + 1]):
                        dv[bc[radj[p]]] += rw[p]
                    k = rdeg[j]
                    old = 0.0; new = 0.0
                    for s in range(Kc):
                        old += _xlogx(M[s, a]) + _xlogx(M[s, b])
                        new += _xlogx(M[s, a] - dv[s]) + _xlogx(M[s, b] + dv[s])
                    old -= _xlogx(kr[a]) + _xlogx(kr[b])
                    new -= _xlogx(kr[a] - k) + _xlogx(kr[b] + k)
                    dL = new - old
                    if dL >= 0 or np.random.random() < np.exp(beta * dL):
                        for s in range(Kc):
                            M[s, a] -= dv[s]; M[s, b] += dv[s]
                        kr[a] -= k; kr[b] += k; br[j] = b; L += dL
            if L > bestL + 1e-9:
                bestL = L; bbc[:] = bc; bbr[:] = br
    return bbc, bbr, bestL


def sbm_fit(W, Kc, Kr, restarts=20, sweeps=200, seed=0):
    """Best-of-restarts DC-SBM.  Returns (bc, br, logL)."""
    ei, ej = np.nonzero(W)
    ew = W[ei, ej].astype(np.float64)
    nC, nR = W.shape
    best = None
    z0c = np.zeros(nC, np.int64); z0r = np.zeros(nR, np.int64)
    for r in range(restarts):
        bc, br, L = _sbm_run(ei.astype(np.int64), ej.astype(np.int64), ew, nC, nR, Kc, Kr, sweeps, seed * 1000 + r, z0c, z0r, False)
        if best is None or L > best[2]:
            best = (bc, br, L)
    return best


def cond_prob(W, bc, br, Kc, Kr, alpha=0.5):
    """P(resource j | consumer i) = (k_j / kr[h_j]) * (M[g_i,h_j]+a)/(kc[g_i]+a Kr); rows sum to 1."""
    W = np.asarray(W, dtype=np.float64)
    M = np.zeros((Kc, Kr))
    np.add.at(M, (bc[:, None].repeat(W.shape[1], 1), br[None, :].repeat(W.shape[0], 0)), W)
    kr = M.sum(0)
    rd = W.sum(0) + 0.1
    krs = np.zeros(Kr); np.add.at(krs, br, rd)
    within = rd / krs[br]                      # share of resource j in its group
    G = (M + alpha) / (M.sum(1, keepdims=True) + alpha * Kr)   # Kc x Kr
    P = G[bc][:, br] * within[None, :]
    return P / P.sum(1, keepdims=True)


def hist_prob(W, c=1.0):
    W = np.asarray(W, dtype=np.float64)
    pr = (W.sum(0) + 0.1) / (W.sum() + 0.1 * W.shape[1])
    P = W + c * pr[None, :]
    return P / P.sum(1, keepdims=True)


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    rk = rankdata(allv)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def nmi(a, b):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(a, b)


def planted(nC, nR, nev, K, seed, mix=0.15, gamma=1.6):
    """Planted modular network: K consumer and K resource modules, heavy-tailed degrees,
    fraction `mix` of events off-module.  Returns events and truth labels."""
    rng = np.random.default_rng(seed)
    gc = rng.integers(0, K, nC); gr = rng.integers(0, K, nR)
    gr[:K] = np.arange(K)
    tc = rng.pareto(gamma, nC) + 1; tr = rng.pareto(gamma, nR) + 1
    ev = []
    ci = rng.choice(nC, nev, p=tc / tc.sum())
    for i in ci:
        if rng.random() < mix:
            cand = np.arange(nR)
        else:
            cand = np.nonzero(gr == gc[i])[0]
        p = tr[cand] / tr[cand].sum()
        j = rng.choice(cand, p=p)
        ev.append(('c%d' % i, 'r%d' % j, 't%d' % rng.integers(0, nev // 4)))
    return ev, {('c%d' % i): int(gc[i]) for i in range(nC)}
