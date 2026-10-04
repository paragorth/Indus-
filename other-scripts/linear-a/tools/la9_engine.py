#!/usr/bin/env python3
"""LA-9 engine: likelihood of a written total under scribal error mechanisms, and a Gibbs sampler
over (hidden letter values on GRID, mechanism weights).

Mechanisms (P(total | entries, values) for each):
  exact    total = sum
  skip     one line left out            total = sum - e_i        (i uniform)
  double   one line counted twice       total = sum + e_i
  carry    written column addition, one carry dropped: total = sum - c_k * place_k, for a column
           that actually produced a carry (k = 0: carry from the fractions into the units)
  slip     counting-board bead slip: total = sum +- place_k (k uniform over places <= magnitude)
  tally    tally/finger miscount: total = sum +- j units, P(j) = 0.5^j (j >= 1)
  round    fraction reckoning: total = floor / ceil / round(sum) or the sum of integer parts only
  misread  one fraction sign in one entry taken as another sign (or ignored)
  bg       anything else (lost lines, wrong section cut): constant density BG
"""
import numpy as np
from la9_common import U, GRID

MECH = ['exact', 'skip', 'double', 'carry', 'slip', 'tally', 'round', 'misread', 'bg']
NM = len(MECH)


def prep(sec, nletters):
    a = sec['a']; places = sec['places']
    # integer column carries (value independent)
    carries = []
    c = 0
    for k in range(len(places) - 1):
        r = places[k + 1] // places[k]
        d = (a // places[k]) % r
        c = (int(d.sum()) + c) // int(r)
        if c > 0: carries.append((c, int(places[k + 1])))
    sec['carries'] = carries
    occ = []
    for i in range(sec['C'].shape[0]):
        for l in range(nletters):
            if sec['C'][i, l] > 0: occ.append((i, l, int(sec['C'][i, l])))
    sec['occ'] = occ
    sec['asum'] = int(a.sum())
    return sec


def mech_probs(sec, V, nletters, BG=1e-3):
    """V: (N, L) int64 values (units of 1/U). Returns (N, NM) probabilities."""
    N = V.shape[0]
    a = sec['a']; Cm = sec['C']
    e = a[:, None] * U + Cm @ V.T                       # (n, N)
    S = e.sum(0)
    T = sec['ta'] * U + V @ sec['ct']
    Dd = T - S
    out = np.zeros((N, NM))
    out[:, 0] = (Dd == 0)
    out[:, 1] = (Dd[None, :] == -e).mean(0)
    out[:, 2] = (Dd[None, :] == e).mean(0)
    # carry
    asum = sec['asum'] * U
    c0 = (S - asum) // U
    cnt = np.full(N, len(sec['carries']), dtype=float) + (c0 > 0)
    hit = np.zeros(N)
    for c, p in sec['carries']:
        hit += (Dd == -c * p * U)
    hit += (c0 > 0) & (Dd == -c0 * U)
    out[:, 3] = np.where(cnt > 0, hit / np.maximum(cnt, 1), 0)
    # slip
    mag = np.maximum(np.abs(S), np.abs(T)) // U
    hit = np.zeros(N); K = np.zeros(N)
    for p in sec['places']:
        ok = (p <= np.maximum(mag, 1))
        K += ok
        hit += ok * ((Dd == p * U).astype(float) + (Dd == -p * U))
    out[:, 4] = hit / (2 * np.maximum(K, 1))
    # tally
    j = np.abs(Dd) // U
    isint = (Dd % U == 0) & (Dd != 0) & (j <= 30)
    out[:, 5] = np.where(isint, 0.25 * 0.5 ** np.clip(j - 1, 0, 60), 0)
    # round
    nonint = (S % U) != 0
    fl = (S // U) * U
    r = (fl + U * ((S - fl) * 2 >= U))
    rr = (T == fl).astype(float) + (T == fl + U) + (T == r) + (T == asum)
    out[:, 6] = np.where(nonint, rr / 4, 0)
    # misread
    if sec['occ']:
        hit = np.zeros(N); tot = 0
        for i, l, w in sec['occ']:
            for l2 in range(nletters + 1):
                if l2 == l: continue
                alt = V[:, l2] if l2 < nletters else 0
                hit += w * (Dd == alt - V[:, l])
            tot += w * nletters
        out[:, 7] = hit / tot
    out[:, 8] = BG
    return out


def gibbs(P, nletters, sweeps=400, burn=100, seed=0, BG=1e-3, alpha=1.0, init=None, fixed=None):
    """P: packed sections (prep() applied). Returns dict with value posterior (L, G) and p mean."""
    rng = np.random.default_rng(seed)
    G = len(GRID)
    vidx = rng.integers(0, G, nletters) if init is None else np.array(init)
    p = rng.dirichlet(np.full(NM, alpha))
    by_letter = [[s for s, sec in enumerate(P) if l in sec['letters_used']] for l in range(nletters)]
    Vcur = GRID[vidx][None, :]
    Pm = np.vstack([mech_probs(sec, Vcur, nletters, BG) for sec in P])     # (S, NM)
    post = np.zeros((nletters, G)); pacc = np.zeros(NM); zacc = np.zeros((len(P), NM)); n = 0
    ll_trace = []
    for it in range(sweeps):
        w = Pm * p[None, :]
        w /= w.sum(1, keepdims=True)
        u = rng.random(len(P))[:, None]
        z = (w.cumsum(1) < u).sum(1)
        z = np.minimum(z, NM - 1)
        p = rng.dirichlet(alpha + np.bincount(z, minlength=NM))
        for l in rng.permutation(nletters):
            if fixed is not None and l in fixed: continue
            secs = by_letter[l]
            if not secs: vidx[l] = rng.integers(0, G); continue
            Vc = np.repeat(GRID[vidx][None, :], G, 0); Vc[:, l] = GRID
            lp = np.zeros(G); cache = {}
            for s in secs:
                m = mech_probs(P[s], Vc, nletters, BG)
                cache[s] = m
                lp += np.log(m @ p)
            lp -= lp.max(); pr = np.exp(lp); pr /= pr.sum()
            g = rng.choice(G, p=pr); vidx[l] = g
            for s in secs: Pm[s] = cache[s][g]
        ll_trace.append(float(np.log(Pm @ p).sum()))
        if it >= burn:
            post[np.arange(nletters), vidx] += 1; pacc += p; zacc[np.arange(len(P)), z] += 1; n += 1
    return {'post': post / n, 'p': pacc / n, 'z': zacc / n, 'll': ll_trace}


def loglik(P, nletters, vidx, p, BG=1e-3):
    V = GRID[np.array(vidx)][None, :]
    return float(sum(np.log(mech_probs(sec, V, nletters, BG) @ p)[0] for sec in P))
