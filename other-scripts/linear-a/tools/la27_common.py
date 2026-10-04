#!/usr/bin/env python3
"""LA-27 shared code: heaping of written amounts on a weight ladder.

A ladder is a tuple of step ratios (r1, r2, ...); its cumulative steps S = (r1, r1 r2, ...)
are the amounts (in the base unit) that one heavier weight makes. If goods were weighed with
such weights and the amount written in the base unit, written integers should heap on
multiples of the steps beyond the decimal heaping that any count shows.

Model for written integers n in 1..NMAX (exponential family):
  base   log p(n) = log LN(n; mu, sd) + a5 [5|n] + a10 [10|n] + a100 [100|n] - log Z
  ladder base + sum_s b_s [s|n], b_s >= 0, s in S
Fitted on training documents; scored by held-out log-likelihood gain (nats per number).
"""
import json, math, os, re
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la27_ckpt')
NMAX = 2000
N = np.arange(1, NMAX + 1)
LOGN = np.log(N)
DEC = np.stack([(N % 5 == 0), (N % 10 == 0), (N % 100 == 0)], 1).astype(float)

def la_numbers():
    """(doc id, label, integer, frac letters) for every LA number token; label = base of the
    last logogram on the same line, else 'W'."""
    c = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for d in c:
        cur = None
        for t in d['tokens']:
            if t['t'] == 'nl': cur = None
            elif t['t'] == 'logo': cur = t['v'].split('+')[0].strip("*[]'")
            elif t['t'] == 'num':
                out.append((d['id'], cur or 'W', int(t['v']), tuple(t.get('frac') or ())))
    return out

def counts(vals):
    c = np.zeros(NMAX)
    for v in vals:
        if 1 <= v <= NMAX: c[v - 1] += 1
    return c

def fit_base(c):
    """ML fit of (mu, sd, a5, a10, a100) to count vector c."""
    tot = c.sum()
    def nll(p):
        mu, ls, a5, a10, a100 = p
        sd = math.exp(ls)
        lp = -0.5 * ((LOGN - mu) / sd) ** 2 - LOGN + DEC @ np.array([a5, a10, a100])
        m = lp.max(); lz = m + math.log(np.exp(lp - m).sum())
        return -(c @ lp - tot * lz)
    mean = (c @ LOGN) / tot
    r = minimize(nll, [mean, 0.0, 0.5, 0.5, 0.5], method='L-BFGS-B',
                 bounds=[(-2, 8), (-3, 2), (-3, 5), (-3, 5), (-3, 5)])
    mu, ls, a5, a10, a100 = r.x
    sd = math.exp(ls)
    return -0.5 * ((LOGN - mu) / sd) ** 2 - LOGN + DEC @ np.array([a5, a10, a100])

def steps(ladder):
    s, out = 1, []
    for r in ladder:
        s *= r
        if s <= NMAX: out.append(s)
    return out

def feat(S):
    return np.stack([(N % s == 0) for s in S], 1).astype(float) if S else np.zeros((NMAX, 0))

def fit_ladder(c, lb, F, iters=25):
    """Newton (projected, b >= 0) for the ladder weights b given base log-prob lb."""
    k = F.shape[1]; b = np.zeros(k); tot = c.sum()
    if k == 0 or tot == 0: return b
    for _ in range(iters):
        lp = lb + F @ b; lp -= lp.max(); p = np.exp(lp); p /= p.sum()
        Ef = p @ F
        g = c @ F - tot * Ef
        H = tot * ((F * p[:, None]).T @ F - np.outer(Ef, Ef)) + 1e-6 * np.eye(k)
        step = np.linalg.solve(H, g)
        b = np.clip(b + step, 0, 6)
        if np.abs(step).max() < 1e-5: break
    return b

def ll(c, lp):
    lp = lp - lp.max(); lz = math.log(np.exp(lp).sum())
    return float(c @ lp - c.sum() * lz)

def folds_by_doc(recs, k=5, seed=0):
    docs = sorted({r[0] for r in recs})
    rng = np.random.default_rng(seed); rng.shuffle(docs)
    f = {d: i % k for i, d in enumerate(docs)}
    return [f[r[0]] for r in recs]

class Scorer:
    """Pre-fits the base model per fold; scores any ladder by held-out LL gain per number."""
    def __init__(self, vals, fold, k=5):
        vals = np.asarray(vals); fold = np.asarray(fold); self.vmax = int(vals.max())
        self.tr, self.te, self.lb = [], [], []
        for i in range(k):
            ctr = counts(vals[fold != i]); cte = counts(vals[fold == i])
            self.tr.append(ctr); self.te.append(cte); self.lb.append(fit_base(ctr))
        self.n = sum(c.sum() for c in self.te)
    def gain(self, ladder):
        S = steps(ladder); F = feat(S); g = 0.0
        for ctr, cte, lb in zip(self.tr, self.te, self.lb):
            b = fit_ladder(ctr, lb, F)
            g += ll(cte, lb + F @ b) - ll(cte, lb)
        return g / self.n

def random_ladders(n, nsteps=2, top=(120, 2000), rmax=30, seed=1):
    """n draws (with repetition) of random ladders: r1 log-uniform on 2..rmax, cumulative top
    log-uniform on `top` (the physical 1:8:60 ladder tops at 480)."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        r1 = int(round(math.exp(rng.uniform(math.log(2), math.log(rmax)))))
        if nsteps == 1: out.append((r1,)); continue
        t = math.exp(rng.uniform(math.log(top[0]), math.log(top[1])))
        out.append((r1, max(2, int(round(t / r1)))))
    return out
