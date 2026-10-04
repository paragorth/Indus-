"""pe40 THE TABLETS ARE A DUTY ROTA.  Shared code.
Players = name-like units; rounds = tablets.  Circular seriation of the tablet graph, then held-out
prediction of hidden player occurrences by RING / LINE / BLOCK / KNN scorers."""
import os as _o
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): _o.environ[_v]='1'
import os, sys, json, collections, math, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa
CK = os.path.join(HERE, '..', 'data', 'pe40_ckpt')
os.makedirs(CK, exist_ok=True)
CLASS = {'M288', 'M297', 'M263', 'M346', 'M264', 'M072', 'M003', 'M354', 'M371', 'M096', 'M376', 'M036',
         'M002', 'M243'}


def pe_rounds(kind='str'):
    """kind 'str': entry name strings (multi-sign entries, final class sign dropped, >=2 signs left, base forms).
       kind 'sign': rare non-final signs of multi-sign entries (base form).  Returns list of (tid, [players in order])."""
    T = load(); out = []
    for t in T:
        if 'Susa' not in (t['provenience'] or ''):
            continue
        seq = []
        for l in t['lines']:
            if not l['numerals'] or l.get('header_comment'):
                continue
            s = [base(x) for x in l['signs'] if is_sign(x)]
            if len(s) != len([x for x in l['signs']]) or len(s) < 2:
                continue  # skip lines with x / broken signs
            if s[-1] in CLASS:
                s = s[:-1]
            if kind == 'str':
                if len(s) >= 2:
                    seq.append(' '.join(s))
            else:
                seq.extend([x for x in s if x not in CLASS])
        if seq:
            out.append((t['id'], seq))
    return out


def filter_players(rounds, lo=2, hi=None, fixed_hi_frac=0.05):
    """keep players found on >= lo tablets and <= hi tablets; drop rounds left empty."""
    df = collections.Counter(p for _, s in rounds for p in set(s))
    n = len(rounds)
    hi = hi or max(3, int(fixed_hi_frac * n))
    R = []
    for tid, s in rounds:
        s2 = [p for p in s if lo <= df[p] <= hi]
        if s2:
            R.append((tid, s2))
    return R


def incidence(R):
    players = sorted({p for _, s in R for p in s})
    ix = {p: i for i, p in enumerate(players)}
    X = np.zeros((len(R), len(players)), dtype=np.float64)
    for i, (_, s) in enumerate(R):
        for p in set(s):
            X[i, ix[p]] = 1
    return X, players


def giant(X):
    """indices of the largest connected component of the tablet graph."""
    n = X.shape[0]
    A = (X @ X.T) > 0
    seen = -np.ones(n, int); comp = 0
    for s in range(n):
        if seen[s] >= 0:
            continue
        st = [s]; seen[s] = comp
        while st:
            u = st.pop()
            for v in np.nonzero(A[u])[0]:
                if seen[v] < 0:
                    seen[v] = comp; st.append(v)
        comp += 1
    c = collections.Counter(seen)
    g = c.most_common(1)[0][0]
    return np.nonzero(seen == g)[0]


def weights(X):
    df = X.sum(0)
    idf = np.where(df > 1, 1.0 / np.maximum(df - 1, 1), 0.0)
    W = (X * idf) @ X.T
    np.fill_diagonal(W, 0)
    return W


# ------------------------------------------------------------------ seriation
def spectral(W, circular=True, rng=None):
    d = W.sum(1) + 1e-9
    Dm = 1 / np.sqrt(d)
    L = np.eye(len(W)) - (Dm[:, None] * W * Dm[None, :])
    L[np.abs(L) < 1e-12] = 0.0
    ev, V = np.linalg.eigh(L)
    if circular:
        ang = np.arctan2(V[:, 2] * Dm, V[:, 1] * Dm)
        return np.argsort(ang)
    return np.argsort(V[:, 1] * Dm)


def tour_score(W, order, h=3, circular=True):
    n = len(order); pos = np.empty(n, int); pos[order] = np.arange(n)
    s = 0.0
    for k in range(1, h + 1):
        a = order; b = np.roll(order, -k) if circular else order[k:]
        a = a if circular else order[:-k]
        s += W[a, b].sum() / k
    return s


def anneal(W, order, h=3, circular=True, iters=20000, rng=None, T0=None):
    """2-opt / segment-reversal local search maximising band weight within h."""
    rng = rng or np.random.default_rng(0)
    n = len(order); order = order.copy()
    cur = tour_score(W, order, h, circular)
    T0 = T0 or (cur / max(n, 1)) * 0.05 + 1e-9
    best = cur; bo = order.copy()
    for it in range(iters):
        i, j = sorted(rng.integers(0, n, 2))
        if j - i < 1:
            continue
        new = order.copy(); new[i:j + 1] = new[i:j + 1][::-1]
        sc = tour_score(W, new, h, circular)
        T = T0 * (1 - it / iters) + 1e-12
        if sc >= cur or rng.random() < math.exp((sc - cur) / T):
            order, cur = new, sc
            if cur > best:
                best, bo = cur, order.copy()
    return bo, best


def seriate(W, circular=True, restarts=4, iters=6000, h=3, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(W)
    o0 = spectral(W, circular)
    best = (None, -1)
    starts = [o0] + [rng.permutation(n) for _ in range(restarts - 1)]
    for o in starts:
        o2, s = anneal(W, o, h, circular, iters, rng)
        if s > best[1]:
            best = (o2, s)
    return best


# ------------------------------------------------------------------ held-out prediction
def split_hidden(X, rng, frac=0.2):
    """hide frac of occurrences of players with df>=3 (so >=2 remain), keeping each tablet >= 1 shared player."""
    Xtr = X.copy(); hid = []
    df = X.sum(0)
    occ = [(i, j) for i, j in zip(*np.nonzero(X)) if df[j] >= 3]
    rng.shuffle(occ)
    k = int(frac * len(occ))
    for i, j in occ:
        if len(hid) >= k:
            break
        if Xtr[i].sum() <= 1 or Xtr[:, j].sum() <= 2:
            continue
        Xtr[i, j] = 0; hid.append((i, j))
    return Xtr, hid


def ranks(score, Xtr, hid):
    """mean reciprocal rank of hidden player among players not already on that tablet (training)."""
    rr = []
    for i, j in hid:
        s = score[i].copy(); s[Xtr[i] > 0] = -np.inf
        s = s + 1e-9 * np.random.default_rng(i * 7919 + j).random(len(s))
        r = (s > s[j]).sum() + 1
        rr.append(1.0 / r)
    return float(np.mean(rr))


def kernel_score(order, Xtr, circular, bw, df_pow=0.0):
    n = len(order); pos = np.empty(n, int); pos[order] = np.arange(n)
    D = np.abs(pos[:, None] - pos[None, :])
    if circular:
        D = np.minimum(D, n - D)
    K = np.exp(-D / bw); np.fill_diagonal(K, 0)
    S = K @ Xtr
    return S * (Xtr.sum(0) + 1) ** df_pow


def knn_score(Xtr):
    W = weights(Xtr)
    return W @ Xtr


def block_score(Xtr, k, rng):
    W = weights(Xtr)
    d = W.sum(1) + 1e-9; Dm = 1 / np.sqrt(d)
    L = np.eye(len(W)) - (Dm[:, None] * W * Dm[None, :])
    L[np.abs(L) < 1e-12] = 0.0
    ev, V = np.linalg.eigh(L)
    E = V[:, :k] * Dm[:, None]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    # k-means
    best = None
    for r in range(5):
        C = E[rng.choice(len(E), k, replace=False)]
        for _ in range(50):
            lab = np.argmax(E @ C.T, 1)
            C2 = np.array([E[lab == c].mean(0) if (lab == c).any() else C[c] for c in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        inert = (E * C[lab]).sum()
        if best is None or inert > best[0]:
            best = (inert, lab)
    lab = best[1]
    M = (lab[:, None] == lab[None, :]).astype(float); np.fill_diagonal(M, 0)
    return M @ Xtr


def curveball(X, rng, nswap=None):
    """bipartite degree-preserving shuffle (names shuffled across tablets, sizes and frequencies kept)."""
    rows = [set(np.nonzero(r)[0]) for r in X]
    n = len(rows); nswap = nswap or 5 * n
    for _ in range(nswap):
        a, b = rng.integers(0, n, 2)
        if a == b:
            continue
        A, B = rows[a], rows[b]
        sa = list(A - B); sb = list(B - A)
        if not sa or not sb:
            continue
        pool = sa + sb; rng.shuffle(pool)
        na = pool[:len(sa)]; nb = pool[len(sa):]
        rows[a] = (A & B) | set(na); rows[b] = (A & B) | set(nb)
    Y = np.zeros_like(X)
    for i, r in enumerate(rows):
        Y[i, list(r)] = 1
    return Y


def plant_rota(n_tab, sizes, n_players, period, overlap=1, noise=0.3, rng=None, ngroups=None):
    """planted rota: players split into `period` groups in a cycle; tablet at step s draws from group s
    (and s+1 with prob for handover); `noise` share of each tablet replaced by random players."""
    rng = rng or np.random.default_rng(0)
    G = np.array_split(rng.permutation(n_players), period)
    steps = rng.integers(0, period, n_tab)
    X = np.zeros((n_tab, n_players))
    for i in range(n_tab):
        k = sizes[i]; s = steps[i]
        pool = list(G[s]) + (list(G[(s + 1) % period]) if overlap else [])
        for _ in range(k):
            p = rng.integers(0, n_players) if rng.random() < noise else pool[rng.integers(0, len(pool))]
            X[i, p] = 1
    return X, steps


def plant_block(n_tab, sizes, n_players, k, noise=0.3, rng=None):
    rng = rng or np.random.default_rng(0)
    G = np.array_split(rng.permutation(n_players), k)
    steps = rng.integers(0, k, n_tab)
    X = np.zeros((n_tab, n_players))
    for i in range(n_tab):
        pool = G[steps[i]]
        for _ in range(sizes[i]):
            p = rng.integers(0, n_players) if rng.random() < noise else pool[rng.integers(0, len(pool))]
            X[i, p] = 1
    return X, steps


def circ_corr(order, truth, period):
    """|circular-circular correlation| between tour position angle and truth angle (Fisher-Lee type)."""
    n = len(order); pos = np.empty(n, int); pos[order] = np.arange(n)
    a = 2 * np.pi * pos / n; b = 2 * np.pi * np.asarray(truth) / period
    best = 0
    for sgn in (1, -1):
        z = np.exp(1j * (sgn * a - b))
        best = max(best, abs(z.mean()))
    return best
