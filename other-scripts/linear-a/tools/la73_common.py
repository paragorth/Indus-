"""la73 common: a simulator of writers in time (persons with careers, a stable term vocabulary,
offices, retention skew) that writes a corpus with a target's own per-tablet word counts, plus the
summary statistics used for ABC.

Units: mean career length L = 1. W = 10**logR is the archive window (time over which the surviving
tablets were written). Everything is value-free: words are integer ids.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

PRIOR = dict(logR=(-1.5, 1.7), logNp=(np.log10(20), 3.0), sig=(0.0, 2.0), phi=(0.0, 0.9),
             logVt=(1.0, 3.5), zt=(0.6, 1.6), logK=(0.0, np.log10(12)), loc=(0.0, 1.0),
             rep=(0.0, 0.3), kappa=(0.0, 5.0), logm=(0.0, 2.0))
PNAMES = list(PRIOR)
# v2 (cycle 2): officials have bursts of office (a term of length tau, activity x H); wider clock prior
PRIOR2 = dict(PRIOR, logR=(-2.5, 1.7), b=(0.0, 1.0), logTau=(-2.0, 0.0), logH=(0.0, 2.0))
PNAMES2 = list(PRIOR2)
VER = {'v': 1}


def set_version(v):
    VER['v'] = v
    global PNAMES
    PNAMES = PNAMES2 if v == 2 else list(PRIOR)


def draw_prior(rng, n):
    pr = PRIOR2 if VER['v'] == 2 else PRIOR
    return np.array([[rng.uniform(*pr[k]) for k in PNAMES] for _ in range(n)])


def simulate(th, lengths, rng, extra=0):
    """th: parameter vector (PNAMES order). lengths: per-tablet token counts. extra: number of
    further tablets written by the same world after the surviving set (for predictions), drawn from
    the same time density, with lengths resampled from `lengths`. Returns (docs, derived)."""
    p = dict(zip(PNAMES, th))
    W = 10 ** p['logR']; Np = 10 ** p['logNp']; K = max(1, int(round(10 ** p['logK'])))
    lengths = np.asarray(lengths)
    n = len(lengths) + extra
    L = np.concatenate([lengths, rng.choice(lengths, extra)]) if extra else lengths
    # tablet times with retention skew: density ~ exp(kappa (t/W - 1)) on [0, W]
    u = rng.random(n); k = p['kappa']
    t = W * (np.log1p(u * np.expm1(k)) / k) if k > 1e-6 else W * u
    off = rng.integers(0, K, n)
    # persons: arrival rate Np per unit time over [-6, W], career ~ Exp(1)
    npers = rng.poisson(Np * (W + 6.0))
    npers = max(npers, 5)
    s = rng.uniform(-6.0, W, npers); e = s + rng.exponential(1.0, npers)
    poff = rng.integers(0, K, npers)
    pop = np.exp(p['sig'] * rng.standard_normal(npers))
    if p.get('b', 0) > 0:
        tau = 10 ** p['logTau']; H = 10 ** p['logH']
        hasb = rng.random(npers) < p['b']
        bs = s + rng.random(npers) * np.maximum(e - s - tau, 0); be = bs + tau
    else:
        hasb = np.zeros(npers, bool); bs = be = s; H = 1.0
    Vt = int(10 ** p['logVt'])
    tw = 1.0 / np.arange(1, Vt + 1) ** p['zt']; tw /= tw.sum(); tcum = np.cumsum(tw)
    docs = []
    for i in range(n):
        act = np.flatnonzero((s <= t[i]) & (e >= t[i]))
        if len(act) == 0:
            act = np.array([np.argmin(np.abs(s - t[i]))])
        same = act[poff[act] == off[i]]
        burst = hasb[act] & (bs[act] <= t[i]) & (be[act] >= t[i])
        wact = pop[act] * np.where(burst, H, 1.0)

        li = int(L[i])
        isterm = rng.random(li) < p['phi']
        nt = int(isterm.sum()); npn = li - nt
        out = np.empty(li, dtype=np.int64)
        if nt:
            out[isterm] = -1 - np.searchsorted(tcum, rng.random(nt))
        if npn:
            uloc = rng.random(npn) < p['loc'] if len(same) else np.zeros(npn, bool)
            vals = np.empty(npn, dtype=np.int64)
            for msk, pool in ((uloc, same), (~uloc, act)):
                k2 = int(msk.sum())
                if k2:
                    ww = wact if pool is act else wact[poff[act] == off[i]]
                    c = np.cumsum(ww); vals[msk] = pool[np.searchsorted(c, rng.random(k2) * c[-1])]
            out[~isterm] = vals
        out = out.tolist()
        for j in range(1, li):
            if rng.random() < p['rep']:
                out[j] = out[rng.integers(j)]
        docs.append(out)
    person_tok = sum(1 for d in docs[:len(lengths)] for x in d if x >= 0)
    ever = Np * (W + 1.0)                       # distinct persons whose careers overlap the window
    m = 10 ** p['logm']                          # mentions per person-career in the full written archive
    f = person_tok / max(1.0, m * ever)          # surviving fraction implied
    q = np.array([0.1, 0.9])
    tq = W * (np.log1p(q * np.expm1(k)) / k) if k > 1e-6 else W * q
    der = dict(R_eff=float(np.log10(max(tq[1] - tq[0], 1e-6))), persons_ever=ever, surv_frac=f, tablets_ever=len(lengths) / max(f, 1e-9),
               distinct_persons_seen=len({x for d in docs[:len(lengths)] for x in d if x >= 0}))
    return docs, der


SNAMES = ['ttr', 'rep_in', 'f1', 'f2', 'f3p', 'maxdf', 'linked', 'gc', 'trans', 'path', 'l3l2',
          'contig', 'pairs2', 'gini', 'deg']
TRAIN = ['ttr', 'rep_in', 'f1', 'f2', 'linked', 'gc', 'trans', 'l3l2', 'contig', 'pairs2']
HELD = ['f3p', 'maxdf', 'path', 'gini', 'deg']


def stats(docs, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(docs)
    ids = {}; rows = []; cols = []; ntok = 0; rep = 0
    for i, d in enumerate(docs):
        seen = set()
        for x in d:
            ntok += 1
            if x in seen:
                rep += 1; continue
            seen.add(x); rows.append(i); cols.append(ids.setdefault(x, len(ids)))
    V = len(ids)
    B = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, V))
    df = np.asarray(B.sum(0)).ravel()
    out = dict(ttr=V / max(ntok, 1), rep_in=rep / max(ntok, 1), f1=np.mean(df == 1), f2=np.mean(df == 2),
               f3p=np.mean(df >= 3), maxdf=df.max() / n)
    sd = np.sort(df); cum = np.cumsum(sd)
    out['gini'] = 1 - 2 * np.sum(cum) / (V * cum[-1]) + 1 / V
    cap = max(3, int(np.ceil(0.05 * n)))
    keep = np.flatnonzero((df >= 2) & (df <= cap))
    Bf = B[:, keep]
    A = (Bf @ Bf.T).tocsr(); A.setdiag(0); A.eliminate_zeros(); A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel()
    out['linked'] = np.mean(deg > 0); out['deg'] = deg.mean()
    # type pairs co-occurring on >= 2 tablets, per filtered type
    C = (Bf.T @ Bf).tocsr(); C.setdiag(0); C.eliminate_zeros()
    out['pairs2'] = (np.sum(C.data >= 2) / 2) / max(1, len(keep))
    tri = (A @ A).multiply(A).sum() ; den = np.sum(deg * (deg - 1))
    out['trans'] = tri / den if den > 0 else 0.0
    nc, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax(); gcn = np.flatnonzero(lab == big)
    out['gc'] = len(gcn) / n
    if len(gcn) >= 6:
        G = A[gcn][:, gcn]
        src = rng.choice(len(gcn), min(25, len(gcn)), replace=False)
        D = shortest_path(G, unweighted=True, directed=False, indices=src)
        out['path'] = D[np.isfinite(D) & (D > 0)].mean() / np.log(len(gcn))
        g = np.asarray(G.sum(1)).ravel(); Dm = 1 / np.sqrt(g)
        Lm = np.eye(len(gcn)) - (Dm[:, None] * G.toarray() * Dm[None, :])
        ev, evec = np.linalg.eigh(Lm)
        out['l3l2'] = np.log(ev[2] / max(ev[1], 1e-9)) if ev[1] > 1e-9 else 0.0
        order = np.argsort(np.argsort(evec[:, 1] * Dm))
        Bg = Bf[gcn].tocsc(); cs = []
        for j in range(Bg.shape[1]):
            r = Bg.indices[Bg.indptr[j]:Bg.indptr[j + 1]]
            if len(r) >= 2:
                pos = order[r]; span = pos.max() - pos.min() + 1
                cs.append((len(r) - 1) / max(span - 1, 1))
        out['contig'] = np.mean(cs) if cs else 0.0
    else:
        out['path'] = 0.0; out['l3l2'] = 0.0; out['contig'] = 0.0
    return np.array([out[k] for k in SNAMES], float)


def to_int_docs(word_docs):
    ids = {}
    return [[ids.setdefault(w, len(ids)) for w in d] for d in word_docs]
