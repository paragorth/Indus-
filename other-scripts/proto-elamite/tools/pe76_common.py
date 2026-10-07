"""pe76: how long did the archive last, in scribal careers?

Pivot from seriation (done in pe19 / pe43): no tablet order is ever built.  Instead the archive's
TIME DEPTH (span S, in units of one clerk's career) is inferred from the geometry of how
content-neutral habit markers (variant forms ~x of signs) are shared between tablets, by ABC over
tens of thousands of random simulated archives (scribes with careers, fashions with lifetimes,
offices that never change, random noise).  The fitting code never reads provenience, museum
numbers, volumes or levels.
"""
import os, sys, json, collections, hashlib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(DATA, 'pe76_ckpt')
os.makedirs(CKPT, exist_ok=True)


def tablet_sizes(T):
    return np.array([max(1, sum(1 for l in t['lines'] for s in l['signs'] if common.is_sign(s))) for t in T], float)


def variant_markers(T, fmin=2, fmax=60, keep_base=None):
    """Variant forms (sign~x, not compounds) on fmin..fmax tablets.  Returns (names, list of tablet-index sets)."""
    tab = collections.defaultdict(set)
    for i, t in enumerate(T):
        for l in t['lines']:
            for s in l['signs']:
                if common.is_sign(s) and '~' in s and not s.startswith('|'):
                    if keep_base is None or common.base(s) in keep_base:
                        tab[s].add(i)
    names = sorted(k for k, v in tab.items() if fmin <= len(v) <= fmax)
    return names, [tab[k] for k in names]


def numeral_markers(T, fmin=2, fmax=60):
    """Second, independent marker class: numeral sign variants (codes with @ or letter suffix forms)
    and comma habit per numeral code are NOT used; only written numeral variant codes."""
    tab = collections.defaultdict(set)
    for i, t in enumerate(T):
        for l in t['lines']:
            for n, c in l['numerals']:
                if '@' in c or c in ('N39A', 'N39B', 'N39C', 'N30C', 'N30D', 'N29B', 'N8A', 'N8B', 'N08A', 'N51G', 'N54G'):
                    tab['num:' + c].add(i)
    names = sorted(k for k, v in tab.items() if fmin <= len(v) <= fmax)
    return names, [tab[k] for k in names]


def to_matrix(sets, n):
    rows, cols = [], []
    for j, s in enumerate(sets):
        for i in s:
            rows.append(i); cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, len(sets)))


# ---------------------------------------------------------------- summary statistics
def stats(X, rng=None):
    """X: tablets x markers binary csr.  Returns a fixed-length vector of geometry statistics."""
    rng = rng or np.random.default_rng(0)
    X = X.tocsc()
    f = np.asarray(X.sum(0)).ravel()
    keep = f >= 2
    X = X[:, keep].tocsr()
    f = f[keep]
    C = (X.T @ X).toarray()                       # marker co-occurrence
    np.fill_diagonal(C, 0)
    m = C.shape[0]
    A1 = C >= 1
    A2 = C >= 2
    npair = m * (m - 1)
    out = []
    out.append(A1.sum() / npair)                   # share >=1 tablet
    out.append(A2.sum() / npair)
    # Jaccard of co-occurring pairs
    U = f[:, None] + f[None, :] - C
    J = np.where(A1, C / np.maximum(U, 1), 0)
    out.append(J[A1].mean() if A1.any() else 0)
    out.append(np.quantile(J[A1], 0.9) if A1.any() else 0)
    # transitivity of marker graph (>=1 and >=2)
    for A in (A1, A2):
        Af = A.astype(np.float32)
        tri = float(((Af @ Af) * Af).sum())
        d = Af.sum(1)
        trip = (d * (d - 1)).sum()
        out.append(tri / trip if trip > 0 else 0)
    # spectral (CA-like) eigenvalue profile of the marker co-occurrence with diagonal
    G0 = sp.csr_matrix(A1)
    _, lab0 = connected_components(G0, directed=False)
    gi = np.where(lab0 == np.bincount(lab0).argmax())[0]
    Cd = (C + np.diag(f))[np.ix_(gi, gi)]
    dd = 1 / np.sqrt(np.maximum(Cd.sum(1), 1e-9))
    Ms = Cd * dd[:, None] * dd[None, :]
    ev = np.sort(np.linalg.eigvalsh(Ms))[::-1]
    ev = ev[1:13] / max(ev[0], 1e-9)
    out.extend(list(ev))
    # path lengths in the >=1 marker graph (giant component)
    G = sp.csr_matrix(A1)
    ncomp, lab = connected_components(G, directed=False)
    big = np.bincount(lab).argmax()
    idx = np.where(lab == big)[0]
    out.append(len(idx) / m)
    src = rng.choice(idx, size=min(40, len(idx)), replace=False)
    D = shortest_path(G, unweighted=True, indices=src)[:, idx]
    D = D[np.isfinite(D)]
    out.append(D.mean())
    out.append(D.max())
    # tablet side: markers per tablet, tablets sharing
    r = np.asarray(X.sum(1)).ravel()
    out.append((r >= 1).mean())
    out.append((r >= 2).mean())
    out.append((r ** 2).mean())
    TT = (X @ X.T)
    TT.setdiag(0); TT.eliminate_zeros()
    out.append(TT.nnz / (X.shape[0] * (X.shape[0] - 1)))
    B = (TT >= 1).astype(np.float32)
    tri = (B @ B).multiply(B).sum()
    d = np.asarray(B.sum(1)).ravel()
    trip = (d * (d - 1)).sum()
    out.append(tri / trip if trip > 0 else 0)
    return np.array(out, float)


STAT_NAMES = (['p_share1', 'p_share2', 'jac_mean', 'jac_q90', 'trans1', 'trans2'] +
              ['ev%d' % k for k in range(2, 14)] +
              ['giant', 'path_mean', 'path_max', 'tab_ge1', 'tab_ge2', 'tab_r2', 'tab_share', 'tab_trans'])


# ---------------------------------------------------------------- simulator
def draw_theta(rng):
    w = rng.dirichlet([1, 1, 1, 1])
    return dict(logS=rng.uniform(np.log(0.1), np.log(30)),
                M=float(np.exp(rng.uniform(np.log(1), np.log(30)))),
                K=int(np.round(np.exp(rng.uniform(0, np.log(40))))),
                w_id=w[0], w_fa=w[1], w_of=w[2], w_rn=w[3],
                logF=rng.uniform(np.log(0.05), np.log(5)),
                eps=rng.uniform(0, 0.3))


THETA_KEYS = ['logS', 'M', 'K', 'w_id', 'w_fa', 'w_of', 'w_rn', 'logF', 'eps']


def simulate(theta, size, freqs, rng, return_truth=False):
    """size: per-tablet weight (sign count); freqs: marker tablet frequencies."""
    n = len(size)
    S = float(np.exp(theta['logS']))
    K = max(1, int(theta['K']))
    t = rng.uniform(0, S, n)
    wk = rng.dirichlet(np.ones(K))
    off = rng.choice(K, n, p=wk)
    # scribes per office
    scr = np.full(n, -1)
    nscr = 0
    sc_of = []
    for k in range(K):
        ti = np.where(off == k)[0]
        if not len(ti):
            continue
        Mk = max(0.3, theta['M'] * wk[k])
        ns = max(1, rng.poisson(Mk * (S + 1)))
        st = rng.uniform(-1, S, ns)
        act = (st[None, :] <= t[ti, None]) & (t[ti, None] < st[None, :] + 1)
        r = rng.random(act.shape) * act - (~act)
        j = r.argmax(1)
        none = ~act.any(1)
        if none.any():
            j[none] = np.abs(st[None, :] + 0.5 - t[ti[none], None]).argmin(1)
        scr[ti] = nscr + j
        nscr += ns
    tw = size / size.sum()
    scr_tabs = collections.defaultdict(list)
    for i, s in enumerate(scr):
        scr_tabs[s].append(i)
    scr_ids = np.array(list(scr_tabs.keys()))
    scr_w = np.array([size[scr_tabs[s]].sum() for s in scr_ids]); scr_w /= scr_w.sum()
    of_ids = np.unique(off)
    of_w = np.array([size[off == k].sum() for k in of_ids]); of_w /= of_w.sum()
    F = float(np.exp(theta['logF']))
    pw = np.array([theta['w_id'], theta['w_fa'], theta['w_of'], theta['w_rn']]); pw /= pw.sum()
    sets = []
    types = []
    for fq in freqs:
        ty = rng.choice(4, p=pw)
        if ty == 0:
            C = np.array(scr_tabs[rng.choice(scr_ids, p=scr_w)])
        elif ty == 1:
            c = rng.uniform(-F / 2, S + F / 2)
            C = np.where(np.abs(t - c) < F / 2)[0]
        elif ty == 2:
            C = np.where(off == rng.choice(of_ids, p=of_w))[0]
        else:
            C = np.arange(n)
        nn = int(rng.binomial(fq, theta['eps']))
        k = min(fq - nn, len(C))
        chosen = set()
        if k > 0:
            p = size[C] / size[C].sum()
            chosen |= set(rng.choice(C, size=k, replace=False, p=p).tolist())
        while len(chosen) < fq:
            chosen.add(int(rng.choice(n, p=tw)))
        sets.append(chosen)
        types.append(ty)
    if return_truth:
        return sets, dict(t=t, off=off, scr=scr, types=types)
    return sets


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
