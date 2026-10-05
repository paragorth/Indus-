"""pe50 library: capture-recapture and species-accumulation estimators on tablet incidence data.

Incidence data = list of sets (one per tablet = capture event; set = entities seen on it).
Estimators (all return (N_hat, lo, hi)):
  chao2     bias-corrected Chao2 with Chao (1987) log-normal interval
  ichao2    improved Chao2 (Chiu et al. 2014), same interval form on the Chao2 part
  jack1/2   incidence jackknife (Burnham & Overton), normal interval
  chapman   Lincoln-Petersen (Chapman) between two random halves of the tablets
  ll3       three-list log-linear model (Poisson GLM on the 7 observed cells), model
            chosen by AIC among: independence, one 2-way term, all 2-way terms, and the
            heterogeneity (Darroch Mh) quasi-symmetry model; profile-free delta interval
            via parametric bootstrap of the chosen model (cheap: 60 reps).
"""
import math, random, collections
import numpy as np


def freq(inc):
    c = collections.Counter(e for s in inc for e in s)
    q = collections.Counter(c.values())
    return c, q


def chao2(inc, T=None):
    T = T or len(inc)
    c, q = freq(inc)
    S = len(c)
    Q1, Q2 = q.get(1, 0), q.get(2, 0)
    k = (T - 1) / T
    f0 = k * Q1 * (Q1 - 1) / (2 * (Q2 + 1))
    N = S + f0
    if f0 <= 0:
        return N, S, S
    # variance (Chao 1987, bias-corrected form)
    var = k * Q1 * (Q1 - 1) / (2 * (Q2 + 1)) + k * k * Q1 * (2 * Q1 - 1) ** 2 / (4 * (Q2 + 1) ** 2) \
        + k * k * Q1 * Q1 * Q2 * (Q1 - 1) ** 2 / (4 * (Q2 + 1) ** 4)
    C = math.exp(1.96 * math.sqrt(math.log(1 + var / f0 ** 2)))
    return N, S + f0 / C, S + f0 * C


def ichao2(inc):
    T = len(inc)
    c, q = freq(inc)
    N, lo, hi = chao2(inc)
    Q1, Q2, Q3, Q4 = (q.get(i, 0) for i in (1, 2, 3, 4))
    add = 0.0
    if Q4 > 0:
        add = (T - 3) / (4 * T) * Q3 / Q4 * max(Q1 - (T - 3) / (T - 1) * Q2 * Q3 / (2 * Q4), 0)
    return N + add, lo + add, hi + add


def jack(inc, order=1):
    T = len(inc)
    c, q = freq(inc)
    S = len(c)
    Q1, Q2 = q.get(1, 0), q.get(2, 0)
    if order == 1:
        N = S + Q1 * (T - 1) / T
        se = math.sqrt(max(Q1 * (T - 1) / T, 1))
    else:
        N = S + Q1 * (2 * T - 3) / T - Q2 * (T - 2) ** 2 / (T * (T - 1))
        se = math.sqrt(max(Q1 * 4 + Q2, 1))
    return N, N - 1.96 * se, N + 1.96 * se


def chapman(inc, rng):
    idx = list(range(len(inc)))
    rng.shuffle(idx)
    h = len(idx) // 2
    A = set().union(*[inc[i] for i in idx[:h]]) if h else set()
    B = set().union(*[inc[i] for i in idx[h:]])
    n1, n2, m = len(A), len(B), len(A & B)
    N = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    se = math.sqrt(max(var, 0))
    return N, max(len(A | B), N - 1.96 * se), N + 1.96 * se


# ---------- three-list log-linear ----------
CELLS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def _design(model):
    X = []
    for a, b, c in CELLS:
        row = [1, a, b, c]
        if model == 'ab':
            row += [a * b]
        elif model == 'ac':
            row += [a * c]
        elif model == 'bc':
            row += [b * c]
        elif model == 'all2':
            row += [a * b, a * c, b * c]
        elif model == 'mh':
            row += [a * b + a * c + b * c]
        X.append(row)
    return np.array(X, float)


def _glm(X, y, it=60):
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(y.mean(), 0.5))
    for _ in range(it):
        mu = np.exp(np.clip(X @ beta, -30, 30))
        W = mu
        z = X @ beta + (y - mu) / np.maximum(mu, 1e-9)
        XtW = X.T * W
        try:
            beta = np.linalg.solve(XtW @ X + 1e-8 * np.eye(len(beta)), XtW @ z)
        except np.linalg.LinAlgError:
            return None, None
    mu = np.exp(np.clip(X @ beta, -30, 30))
    ll = float(np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu))
    return beta, ll


def ll3_fit(y, models=('indep', 'ab', 'ac', 'bc', 'all2', 'mh')):
    best = None
    for m in models:
        X = _design(m)
        beta, ll = _glm(X, y)
        if beta is None:
            continue
        aic = -2 * ll + 2 * X.shape[1]
        n0 = math.exp(min(beta[0], 25))
        if best is None or aic < best[0]:
            best = (aic, m, n0)
    return best


def ll3(inc, rng, models=('indep', 'ab', 'ac', 'bc', 'all2', 'mh'), groups=None, boot=40):
    if groups is None:
        groups = [rng.randrange(3) for _ in inc]
    L = [set(), set(), set()]
    for s, g in zip(inc, groups):
        L[g] |= s
    allE = L[0] | L[1] | L[2]
    cnt = collections.Counter((int(e in L[0]), int(e in L[1]), int(e in L[2])) for e in allE)
    y = np.array([cnt.get(c, 0) for c in CELLS], float)
    best = ll3_fit(y, models)
    if best is None:
        return None
    S = len(allE)
    n0 = best[2]
    # parametric bootstrap of observed cells (multinomial on fitted proportions, Poisson total)
    p = y / y.sum()
    bs = []
    nr = np.random.default_rng(rng.randrange(1 << 30))
    for _ in range(boot):
        yb = nr.multinomial(int(y.sum()), p).astype(float)
        b = ll3_fit(yb, (best[1],))
        if b:
            bs.append(S + min(b[2], 1e7))
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (S, S + n0 * 3))
    return S + min(n0, 1e7), float(lo), float(hi), best[1]


def accumulate(inc, rng, steps=20):
    idx = list(range(len(inc)))
    rng.shuffle(idx)
    seen = set()
    out = []
    marks = set(int(len(idx) * (i + 1) / steps) for i in range(steps))
    for i, j in enumerate(idx, 1):
        seen |= inc[j]
        if i in marks:
            out.append((i, len(seen)))
    return out


def thin(inc, q, rng):
    return [s for s in inc if rng.random() < q]


def estimate_all(inc, rng):
    out = {}
    out['chao2'] = chao2(inc)
    out['ichao2'] = ichao2(inc)
    out['jack1'] = jack(inc, 1)
    out['jack2'] = jack(inc, 2)
    out['chapman'] = chapman(inc, rng)
    r = ll3(inc, rng)
    if r:
        out['ll3'] = r[:3]
    return out


# ---------- binomial-thinning mixture ("how many tablets were written") ----------
# Full-archive tablet count per entity m >= 1 ~ discretised log-normal(mu, sigma) (or zeta(a));
# each written tablet survives with probability q; observed k ~ Bin(m, q), k >= 1 seen.
# Fit (shape, q) by maximum likelihood on the zero-truncated spectrum.  T_hat = n_obs / q_hat,
# N_hat = S_obs / P(k >= 1).
from scipy.stats import binom as _binom, norm as _norm
_MG = np.unique(np.concatenate([np.arange(1, 60), np.round(np.logspace(np.log10(60), 5, 160))])).astype(int)
_KB = [1, 2, 3, 4, 5, 6, 8, 11, 16, 26, 51, 10 ** 9]   # k bins [1],[2],...,[51,inf)


def _mweights(fam, a, b):
    m = _MG.astype(float)
    # bin widths for the log part
    edges = np.concatenate([[0.5], (m[1:] + m[:-1]) / 2, [m[-1] * 1.03]])
    if fam == 'lnorm':
        cdf = _norm.cdf((np.log(edges) - a) / b)
    else:  # zeta-like: continuous power law with exponent a, cutoff exp(b)
        x = edges
        cdf = 1 - (x / 0.5) ** (1 - a) * np.exp(-(x - 0.5) / math.exp(b))
    w = np.diff(cdf)
    w = np.maximum(w, 0)
    return w / w.sum() if w.sum() > 0 else None


_PKcache = {}


def _pk(q):
    key = round(q, 6)
    if key not in _PKcache:
        m = _MG
        cols = []
        for lo, hi in zip(_KB[:-1], _KB[1:]):
            cols.append(_binom.cdf(hi - 1, m, q) - _binom.cdf(lo - 1, m, q))
        P = np.array(cols).T          # (M, nbins)
        P0 = _binom.pmf(0, m, q)
        if len(_PKcache) > 400:
            _PKcache.clear()
        _PKcache[key] = (P, P0)
    return _PKcache[key]


def spectrum(inc):
    c = collections.Counter(e for s in inc for e in s)
    k = np.array(list(c.values()))
    f = np.array([np.sum((k >= lo) & (k < hi)) for lo, hi in zip(_KB[:-1], _KB[1:])], float)
    return f, len(c)


QGRID = np.logspace(-3, 0, 37)


def thinfit(inc, fam='lnorm', qgrid=QGRID, f=None):
    if f is None:
        f, S = spectrum(inc)
    else:
        S = f.sum()
    if fam == 'lnorm':
        A = np.linspace(-1, 6, 22)
        B = np.linspace(0.2, 3.5, 16)
    else:
        A = np.linspace(1.05, 3.0, 20)
        B = np.linspace(0, 11, 12)
    best = (-1e300, None)
    prof = []
    for q in qgrid:
        P, P0 = _pk(q)
        bq = -1e300
        for a in A:
            for b in B:
                w = _mweights(fam, a, b)
                if w is None:
                    continue
                pk = w @ P
                pseen = 1 - w @ P0
                if pseen <= 1e-12:
                    continue
                pr = np.maximum(pk / pseen, 1e-300)
                ll = float(f @ np.log(pr))
                if ll > bq:
                    bq = ll
                    if ll > best[0]:
                        best = (ll, (q, a, b, pseen))
        prof.append(bq)
    prof = np.array(prof)
    ok = qgrid[prof >= prof.max() - 1.92]
    q, a, b, pseen = best[1]
    return {'q': q, 'q_lo': float(ok.min()), 'q_hi': float(ok.max()), 'N': S / pseen, 'S': int(S),
            'a': a, 'b': b, 'fam': fam}
