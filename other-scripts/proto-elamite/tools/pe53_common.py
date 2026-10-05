"""pe53: THE LAMBS TELL THE MONTH.  Shared code.

Idea: sheep and goats give birth in a short season, so the young-to-adult ratio of a
herd rises and falls through the year in a shape fixed by biology, not by the corpus.
Every assignment of candidate signs to classes (Y young, F adult female, O other) is
scored by how much better a BIRTH-PULSE mixture explains the per-record young share
than a timeless (non-seasonal) model.  Under the best assignment each record gets a
position in the year, and other signs are tested for seasonal drift.

Outside curve (biology / animal science, not the corpus):
  * births spread over w = 2-4 months (Balasse et al. 2017, Animal 11: 'births occurred
    over a period of 3 to 4 months' in sheep; restricted breeding season inherited from
    the Oriental mouflon);
  * one pulse per year; in western Iran extensive flocks mate in July and lamb in
    December (Kurdi sheep, Ilam: JAST 2011, 13(5): 701-708); Arabi sheep of Khuzestan
    lamb in autumn-winter and are weaned at ~100 days (Ramin Agric. Univ. studies);
  * young per breeding female at the end of the pulse b = 0.7-1.3 (singletons with
    some twins; goats twin more);
  * monthly loss of young (death + offtake) h = 0.02-0.10 (10-70% in the first year);
  * a scribe keeps an animal in the young class for L = 4-12 months (weaning to yearling)
    -- unknown, so it is a free parameter of the curve family.
Model: tablets are written at a uniform random time t in the year (phase unknown, so
only the SHAPE of the curve matters); young share p(t) = Y(t) / (Y(t) + 1) with
Y(t) = b * int_0^min(t,w) (1/w) exp(-h (t-s)) 1[t-s < L] ds.
Observation per record: y ~ BetaBinomial(n = y + f, p, kappa).
Scores (log marginal likelihoods, kappa and curve parameters integrated on grids):
  S1 = seasonal - constant-p          (does the pulse beat a timeless herd?)
  S2 = seasonal - 2-component mixture (does the BIOLOGICAL shape beat any bimodality?)
"""
import json, os, sys, itertools, re
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from scipy.special import gammaln, logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe53_ckpt')
os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(HERE, '..', 'loops')

PGRID = np.linspace(0.005, 0.995, 45)
KAPPAS = np.array([2.0, 6.0, 20.0, 80.0])
TGRID = np.arange(0, 12, 0.25)            # 48 time points (months since pulse start)


def curve(w, b, h, L, c=0.0, t=TGRID):
    # mean over birth times s in [0, min(t, w)]; fraction already born = min(t, w) / w
    born = np.clip(t / w, 0, 1)
    Ys = []
    for i, ti in enumerate(t):
        ss = np.linspace(0, min(ti, w), 41)
        if ti <= 0:
            Ys.append(0.0)
            continue
        a = ti - ss
        v = np.where(a < L, np.exp(-h * a), 0.0)
        Ys.append(b * born[i] * v.mean())
    Y = np.array(Ys)
    # p = Y / (Y + 1) shifted by c on the logit scale: logit p = log Y + c
    return np.where(Y > 0, 1.0 / (1.0 + np.exp(-(np.log(np.maximum(Y, 1e-12)) + c))), 0.0)


OFFSETS = (-1.5, 0.0, 1.5, 3.0)   # scribal level shift on the logit scale (what share of lambs is booked)
THETAS = [(w, b, h, L, c) for w in (2.0, 3.0, 4.0) for b in (0.7, 1.0, 1.3)
          for h in (0.02, 0.05, 0.10) for L in (4, 6, 9, 12) for c in OFFSETS]


def theta_weights():
    """For each curve, weights over PGRID (histogram of p(t) over uniform t)."""
    W = np.zeros((len(THETAS), len(PGRID)))
    C = np.zeros((len(THETAS), len(TGRID)))
    for i, th in enumerate(THETAS):
        p = curve(*th)
        C[i] = p
        j = np.abs(PGRID[None, :] - np.clip(p, PGRID[0], PGRID[-1])[:, None]).argmin(1)
        np.add.at(W[i], j, 1.0)
    W /= W.sum(1, keepdims=True)
    return W, C


W_TH, C_TH = theta_weights()


def bb_logpmf(y, n, p, k):
    """y, n arrays (R,), p (P,), k scalar -> (R, P)."""
    a = p[None, :] * k
    bb = (1 - p[None, :]) * k
    y = y[:, None]
    n = n[:, None]
    return (gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
            + gammaln(y + a) + gammaln(n - y + bb) - gammaln(n + a + bb)
            + gammaln(a + bb) - gammaln(a) - gammaln(bb))


def rec_loglik(y, n):
    """(K, R, P) log-likelihood over kappa x record x p grid, computed on unique pairs."""
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    pairs, inv = np.unique(np.stack([y, n], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    out = np.stack([bb_logpmf(pairs[:, 0], pairs[:, 1], PGRID, k) for k in KAPPAS])
    return out[:, inv, :]


MIXP = [(i, j, m) for i in range(0, 45, 4) for j in range(0, 45, 4) if i < j for m in (0.25, 0.5, 0.75)]
MIX_I = np.array([x[0] for x in MIXP]); MIX_J = np.array([x[1] for x in MIXP]); MIX_M = np.array([x[2] for x in MIXP])


def scores(y, n, want_post=False):
    """Return dict with log marginal likelihoods and S1, S2.  Records with n == 0 dropped."""
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    keep = n > 0
    y, n = y[keep], n[keep]
    R = len(y)
    if R < 3:
        return None
    LL = rec_loglik(y, n)                           # K, R, P
    Lmax = LL.max(axis=2, keepdims=True)
    E = np.exp(LL - Lmax)                            # K, R, P
    # seasonal: per kappa, per theta: sum_r log sum_p W[th,p] E
    mix = np.einsum('krp,tp->ktr', E, W_TH)          # K, T, R
    ls = np.log(mix + 1e-300).sum(2) + Lmax[:, None, :, 0].sum(2)
    Lseas = logsumexp(ls) - np.log(ls.size)
    # constant p: per kappa per p: sum_r LL
    lc = LL.sum(1)                                   # K, P
    Lconst = logsumexp(lc) - np.log(lc.size)
    # 2-component mixture over coarse grid
    v = MIX_M[None, None, :] * E[:, :, MIX_I] + (1 - MIX_M)[None, None, :] * E[:, :, MIX_J]
    lm = (np.log(v + 1e-300).sum(1) + Lmax[:, :, 0].sum(1)[:, None]).T
    Lmix = logsumexp(lm) - np.log(lm.size)
    res = dict(R=int(R), Lseas=float(Lseas), Lconst=float(Lconst), Lmix=float(Lmix),
               S1=float(Lseas - Lconst), S2=float(Lseas - Lmix))
    if want_post:
        k, t = np.unravel_index(np.argmax(ls), ls.shape)
        res['theta'] = THETAS[t]
        res['kappa'] = float(KAPPAS[k])
        # posterior over time for each record under MAP curve
        pc = np.clip(C_TH[t], PGRID[0], PGRID[-1])
        Lt = bb_logpmf(y, n, pc, KAPPAS[k])          # R, T
        Pt = np.exp(Lt - Lt.max(1, keepdims=True))
        Pt /= Pt.sum(1, keepdims=True)
        res['post_t'] = Pt
        res['keep'] = keep
    return res


def all_assignments(K, classes=3, need=(1, 2)):
    """All vectors in {0=O,1=Y,2=F}^K with at least one of each class in need."""
    A = np.array(list(itertools.product(range(classes), repeat=K)), dtype=np.int8)
    ok = np.ones(len(A), bool)
    for c in need:
        ok &= (A == c).any(1)
    return A[ok]


def yn_from(M, a):
    """M (R, K) counts, a assignment -> y, n (young vs young+F)."""
    y = M[:, a == 1].sum(1)
    f = M[:, a == 2].sum(1)
    return y, y + f


def score_all(M, A, key='S1'):
    out = np.empty((len(A), 2))
    for i, a in enumerate(A):
        y, n = yn_from(M, a)
        r = scores(y, n)
        out[i] = (np.nan, np.nan) if r is None else (r['S1'], r['S2'])
    return out


def circ_phase(post_t):
    ang = TGRID / 12 * 2 * np.pi
    z = post_t @ np.exp(1j * ang)
    return (np.angle(z) % (2 * np.pi)) / (2 * np.pi) * 12, np.abs(z)


def best_rotation_hits(phase, month, tol=1.0):
    """Fraction of records whose predicted phase, after the best rotation (time runs
    forward), falls within tol months of the true month (1-12)."""
    best = 0
    m0 = (np.asarray(month) - 1) % 12
    for r in np.arange(0, 12, 0.5):
        d = np.abs(((phase + r) - m0 + 6) % 12 - 6)
        h = np.mean(d <= tol)
        if h > best:
            best, br = h, r
    return best, br


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    json.dump(obj, open(path, 'w'), default=conv, indent=1)


# ------------------------------------------------------------------ data
PE_SIGNS = ['M362', 'M367', 'M346', 'M006', 'M362~a', 'M367~a', 'M346~a', 'M006@g']


def pe_matrix(none_as_zero=True):
    from pe20_common import pe_records
    R = [r for r in pe_records() if r[0] != 'P008389']   # chain copy of P008294 o3
    M = np.array([[(r[2][s] if r[2][s] is not None else (0.0 if none_as_zero else np.nan))
                   for s in PE_SIGNS] for r in R], float)
    return R, M


DR_TERMS = ['udu', 'sila4', 'masz2', 'masz2-gal', 'u8', 'ud5', 'kir11', '{munus}asz2-gar3', 'gukkal']
DR_TRUTH = np.array([0, 1, 1, 0, 2, 2, 1, 1, 0], dtype=np.int8)   # Y = lambs/kids, F = ewes/nannies


def drehem_matrix(min_n=1):
    d = json.load(open(os.path.join(DATA, 'pe47_ckpt', 'ur3_drehem.json')))
    rows, months, ids = [], [], []
    idx = {t: i for i, t in enumerate(DR_TERMS)}
    for x in d:
        m = x['meta']['month']
        if not m:
            continue
        v = np.zeros(len(DR_TERMS))
        for e in x['ents']:
            h = e[0][0]
            if h in idx and e[1] is not None:
                v[idx[h]] += e[1]
        if v.sum() >= min_n:
            rows.append(v)
            months.append(m)
            ids.append(x['id'])
    return np.array(rows), np.array(months), ids
