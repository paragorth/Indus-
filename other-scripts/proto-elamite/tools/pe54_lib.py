"""pe54 FOLLOW ONE FLOCK THROUGH THE ARCHIVE: shared library.

Every herd-like entry (a string followed by counts in several animal classes) is a SIGHTING.
Sightings are linked into flock histories by composition only; the meaning of the class signs
is never used. A LINKING HYPOTHESIS = (role of each class, demographic rates, observation noise):
  role per class: X (unconstrained) or F/M/Y of species 0 or 1 (F adult females, M other adults,
  Y young of the year); annual step per species:
      F' = sa (1 - of) F + sy qf Y
      M' = sa (1 - om) M + sy (1 - qf) (1 - om) Y
      Y' = b F'
  rates drawn from ethnographic / husbandry ranges (see RATE RANGES below), dt in {0,1,2,3} years
  (dt 0 = the same census copied), marginalised with equal prior.
Pair score S[i, j] = log P(x_j | x_i moved forward dt years) - log P(x_j | value of a random other
entry), summed over non-X classes; counts ~ NegBin(mean + 0.5, kappa). Classes with the same role
share the role total in the source's proportions. Links = maximum-weight matching (each sighting
at most one successor and one predecessor, never on the same tablet, only positive S).

RATE RANGES (annual; open sources):
  b  young per adult female 0.5-1.2 (FAO/ILCA traditional flocks; lambing 30-112% in Jordanian herds,
     Utupub 'Sustainable Sheep and Goat Farming in Arid Regions of Jordan'; Redding 1981 Iranian
     herding model uses ~0.8-1.0)
  sy first-year survival 0.50-0.80 (lamb/kid mortality to 1 yr 44-48%, Cote d'Ivoire traditional
     flocks, CGSpace 66812; 20-50% in FAO pastoral surveys)
  sa adult survival 0.75-0.92 (adult mortality 21-23%, same source; 8-15% in managed flocks)
  of offtake of adult females 0-0.15; om offtake of other adults 0.1-0.6 (offtake 20-24%,
     NW Cameroon pastoral systems; males sold or slaughtered first)
  qf share of surviving young that join the female class 0.4-0.6
"""
import os, sys, json, math
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe54_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)

EPS = 0.5
ZERO_MISSING = False     # True: a class written as 0 / not written is treated as unobserved
KAPPAS = (2.0, 6.0, 20.0)
DTS = (0, 1, 2, 3)
NROLE = 7          # 0 X ; 1 F0 2 M0 3 Y0 ; 4 F1 5 M1 6 Y1


def nb_logpmf(x, mu, k):
    mu = mu + EPS
    return (gammaln(x + k) - gammaln(k) - gammaln(x + 1)
            + k * np.log(k / (k + mu)) + x * np.log(mu / (k + mu)))


def sample_rates(rng):
    return dict(b=rng.uniform(0.5, 1.2), sy=rng.uniform(0.5, 0.8), sa=rng.uniform(0.75, 0.92),
                of=rng.uniform(0, 0.15), om=rng.uniform(0.1, 0.6), qf=rng.uniform(0.4, 0.6),
                k=KAPPAS[rng.integers(3)])


def sample_roles(K, rng, two_species=True):
    while True:
        r = rng.integers(0, NROLE if two_species else 4, size=K)
        sp_ok = False
        for s in (0, 1):
            if ((r == 1 + 3 * s).any() and ((r == 3 + 3 * s).any() or (r == 2 + 3 * s).any())):
                sp_ok = True
        if sp_ok:
            return r


def step_matrix(th):
    """3x3 annual projection on (F, M, Y)."""
    b, sy, sa, of, om, qf = th['b'], th['sy'], th['sa'], th['of'], th['om'], th['qf']
    A = np.array([[sa * (1 - of), 0, sy * qf],
                  [0, sa * (1 - om), sy * (1 - qf) * (1 - om)],
                  [0, 0, 0]])
    A[2] = b * A[0]
    return A


def project(X, roles, th, dt):
    """X (n, K) -> predicted (n, K) after dt years; X classes predicted as NaN (not scored)."""
    n, K = X.shape
    P = np.full((n, K), np.nan)
    A = np.linalg.matrix_power(step_matrix(th), dt) if dt > 0 else np.eye(3)
    for s in (0, 1):
        idx = [np.where(roles == 1 + 3 * s + c)[0] for c in range(3)]
        if not any(len(i) for i in idx):
            continue
        st = np.stack([X[:, i].sum(1) if len(i) else np.zeros(n) for i in idx], 1)   # n, 3
        nx = st @ A.T
        for c in range(3):
            i = idx[c]
            if not len(i):
                continue
            sub = X[:, i] + 0.5
            share = sub / sub.sum(1, keepdims=True)
            P[:, i] = nx[:, c:c + 1] * share
    return P


class Corpus:
    def __init__(self, X, tabs, strings=None, years=None, truth=None):
        self.X = np.asarray(X, float)
        self.n, self.K = self.X.shape
        self.tabs = np.asarray(tabs)
        self.strings = strings
        self.years = years
        self.truth = truth
        self.same_tab = self.tabs[:, None] == self.tabs[None, :]
        # background: random other entry of the same class, same noise, per kappa
        self.bg = {}
        for k in KAPPAS:
            B = np.zeros((self.n, self.K))
            for c in range(self.K):
                v = self.X[:, c]
                L = nb_logpmf(v[:, None], v[None, :], k)     # target x, source v
                np.fill_diagonal(L, -np.inf)
                if ZERO_MISSING:
                    nz = v > 0
                    L[:, ~nz] = -np.inf
                    B[:, c] = logsumexp(L, 1) - np.log(max(nz.sum() - 1, 1))
                else:
                    B[:, c] = logsumexp(L, 1) - np.log(self.n - 1)
            self.bg[k] = B


def pair_scores(C, roles, th, dts=DTS, Xsrc=None, src_bg=None):
    """S[i, j] (n_src, n_tgt) log-likelihood ratio that target j is source i moved forward."""
    X = C.X if Xsrc is None else Xsrc
    k = th['k']
    mask = roles > 0
    if not mask.any():
        return np.zeros((len(X), C.n))
    xt = C.X[:, mask]                                  # n_t, m
    bg = C.bg[k][:, mask].sum(1)                       # n_t
    Ls = []
    for dt in dts:
        P = project(X, roles, th, dt)[:, mask]        # n_s, m
        if ZERO_MISSING:
            obs = (X[:, mask] > 0)[:, None, :] & (xt > 0)[None, :, :]
            ll = nb_logpmf(xt[None, :, :], P[:, None, :], k) - C.bg[k][:, mask][None, :, :]
            L = np.where(obs, ll, 0.0).sum(2)
        else:
            L = nb_logpmf(xt[None, :, :], P[:, None, :], k).sum(2)    # n_s, n_t
        Ls.append(L)
    L = logsumexp(np.stack(Ls), 0) - np.log(len(dts))
    if ZERO_MISSING:
        return L
    return L - bg[None, :]


def match(S, forbid):
    """Max-weight matching on positive scores; returns list of (i, j, s) with i -> j."""
    W = np.where(forbid, 0.0, np.maximum(S, 0.0))
    r, c = linear_sum_assignment(-W)
    out = [(i, j, S[i, j]) for i, j in zip(r, c) if W[i, j] > 0]
    return out


def sym_forbid(C):
    f = C.same_tab.copy()
    np.fill_diagonal(f, True)
    return f


def hyp_score(C, roles, th, forbid=None):
    S = pair_scores(C, roles, th)
    if forbid is None:
        forbid = sym_forbid(C)
    S2 = np.where(forbid, -np.inf, S)
    # undirected use: each unordered pair counted once by taking best direction in matching
    M = match(S2, forbid)
    return sum(s for _, _, s in M), M, S


def search(C, n_hyp, rng, keep=50, climb=0, two_species=True):
    """Random hypotheses; returns sorted top list of (score, roles, th)."""
    top = []
    forbid = sym_forbid(C)
    allsc = np.empty(n_hyp)
    for h in range(n_hyp):
        r = sample_roles(C.K, rng, two_species)
        th = sample_rates(rng)
        s, _, _ = hyp_score(C, r, th, forbid)
        allsc[h] = s
        if len(top) < keep or s > top[-1][0]:
            top.append((s, r.copy(), th))
            top.sort(key=lambda x: -x[0])
            top = top[:keep]
    if climb:
        new = []
        for s, r, th in top[:climb]:
            s, r, th = hill_climb(C, r, th, rng, forbid, two_species)
            new.append((s, r, th))
        top = sorted(new + top[climb:], key=lambda x: -x[0])
    return top, allsc


def hill_climb(C, r, th, rng, forbid, two_species=True, sweeps=3):
    best, _, _ = hyp_score(C, r, th, forbid)
    nr = NROLE if two_species else 4
    for _ in range(sweeps):
        improved = False
        for c in rng.permutation(C.K):
            for v in range(nr):
                if v == r[c]:
                    continue
                r2 = r.copy(); r2[c] = v
                s, _, _ = hyp_score(C, r2, th, forbid)
                if s > best:
                    best, r, improved = s, r2, True
        for _ in range(6):
            th2 = dict(th)
            key = ['b', 'sy', 'sa', 'of', 'om', 'qf', 'k'][rng.integers(7)]
            th2.update({kk: vv for kk, vv in sample_rates(rng).items() if kk == key})
            s, _, _ = hyp_score(C, r, th2, forbid)
            if s > best:
                best, th, improved = s, th2, True
        if not improved:
            break
    return best, r, th


# ----------------------------------------------------------------- nulls and plants
def shuffle_within_class(X, rng):
    Y = X.copy()
    for c in range(X.shape[1]):
        Y[:, c] = rng.permutation(Y[:, c])
    return Y


def synthetic_unrelated(X, rng):
    """Unrelated herds: each entry gets a fresh herd of the same total size with the composition
    of another random entry (Dirichlet-multinomial around it), so no entry is a later state of another."""
    n, K = X.shape
    Y = np.zeros_like(X)
    tot = X.sum(1)
    for i in range(n):
        j = rng.integers(n)
        p = X[j] + 0.3
        p = rng.dirichlet(p * 3)
        Y[i] = rng.multinomial(int(max(tot[i], 1)), p)
        Y[i, X[j] == 0] = 0
    return Y


def plant_archive(rng, n_flocks=150, years=8, K=8, survive=1.0, obs_classes=(2, 8), roles=None, th=None, het=5.0):
    """Evolving flocks with true roles; returns X, tab, owner string, year, roles, truth rates."""
    if roles is None:
        roles = np.array([1, 2, 3, 0, 4, 5, 6, 0])[:K]
    if th is None:
        th = dict(b=0.85, sy=0.65, sa=0.85, of=0.07, om=0.35, qf=0.5, k=20.0)
    rows, tabs, owners, yrs = [], [], [], []
    A = step_matrix(th)
    for f in range(n_flocks):
        size = np.exp(rng.uniform(np.log(8), np.log(200)))
        st = {}
        for s in (0, 1):
            w = rng.uniform(0.2, 1.0) if s == 1 else 1.0
            st[s] = rng.dirichlet(np.array([0.55, 0.15, 0.30]) * het * 3) * size * w
        split = {c: rng.dirichlet(np.ones(int((roles == c).sum())) * het) for c in range(1, 7) if (roles == c).any()}
        xvals = rng.uniform(1, 15, size=K)
        for y in range(years):
            # each flock-year has its own rates noise (good and bad years)
            for s in (0, 1):
                g = rng.lognormal(0, 0.15)
                st[s] = (A @ st[s]) * g if y > 0 else st[s]
            if rng.random() > survive:
                continue
            x = np.zeros(K)
            for c in range(1, 7):
                idx = np.where(roles == c)[0]
                if not len(idx):
                    continue
                s, k = (c - 1) // 3, (c - 1) % 3
                mu = st[s][k] * split[c]
                x[idx] = rng.negative_binomial(th['k'], th['k'] / (th['k'] + mu))
            for c in np.where(roles == 0)[0]:
                x[c] = rng.poisson(xvals[c])
            m = rng.integers(obs_classes[0], obs_classes[1] + 1)
            keepc = rng.choice(K, size=m, replace=False)
            xx = np.zeros(K); xx[keepc] = x[keepc]
            rows.append(xx); owners.append('flock%03d' % f); yrs.append(y)
            tabs.append('T%d_%d' % (y, rng.integers(0, 40)))
    return np.array(rows), np.array(tabs), owners, np.array(yrs), roles, th


# ----------------------------------------------------------------- evaluation
def link_eval(M, strings, years=None):
    same = [strings[i] == strings[j] for i, j, _ in M]
    n = len(strings)
    from collections import Counter
    c = Counter(strings)
    p_chance = sum(v * (v - 1) for v in c.values()) / (n * (n - 1))
    order = None
    if years is not None:
        ok = [(years[j] > years[i]) for (i, j, _), s in zip(M, same)
              if s and years[i] is not None and years[j] is not None and years[i] != years[j]]
        order = (int(sum(ok)), len(ok))
    return dict(n_links=len(M), same=int(sum(same)), prec=float(np.mean(same)) if same else 0.0,
                chance=p_chance, order=order)


def pair_auc(S, strings, forbid):
    Ssym = np.maximum(S, S.T)
    iu = np.triu_indices(len(strings), 1)
    pos, neg = [], []
    for i, j in zip(*iu):
        if forbid[i, j]:
            continue
        (pos if strings[i] == strings[j] else neg).append(Ssym[i, j])
    if not pos or not neg:
        return float('nan'), len(pos)
    pos, neg = np.array(pos), np.array(neg)
    auc = (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    return float(auc), len(pos)


def heldout_bits(C, roles, th, train, test, rng=None, random_partner=False):
    """For each test entry j and each non-zero-or-scored class c: choose the train partner that best
    predicts j's OTHER classes, predict x_jc from it, score log2 ratio vs background. Mean bits per cell."""
    mask = roles > 0
    cls = np.where(mask)[0]
    if len(cls) < 2:
        return 0.0, 0
    k = th['k']
    Xtr = C.X[train]
    preds = {dt: project(Xtr, roles, th, dt) for dt in DTS}
    tot, cells = 0.0, 0
    for j in test:
        ok_tr = np.array([C.tabs[i] != C.tabs[j] for i in train])
        if not ok_tr.any():
            continue
        for c in cls:
            other = [d for d in cls if d != c]
            # likelihood of j's other classes from each train source, per dt
            Lo = np.stack([nb_logpmf(C.X[j, other][None, :], preds[dt][:, other], k).sum(1) for dt in DTS])
            Lo = np.where(ok_tr[None, :], Lo, -np.inf)
            if random_partner:
                cand = np.where(ok_tr)[0]
                i = cand[rng.integers(len(cand))]
                w = np.exp(Lo[:, i] - Lo[:, i].max()); w /= w.sum()
                lp = logsumexp(np.log(w + 1e-300) + np.array([nb_logpmf(C.X[j, c], preds[dt][i, c], k) for dt in DTS]))
            else:
                # posterior over (dt, partner)
                W = Lo - logsumexp(Lo)
                Lc = np.stack([nb_logpmf(C.X[j, c], preds[dt][:, c], k) for dt in DTS])
                lp = logsumexp(W + Lc)
            # background from train entries of class c
            v = Xtr[:, c]
            bg = logsumexp(nb_logpmf(C.X[j, c], v, k)) - np.log(len(v))
            tot += (lp - bg) / np.log(2)
            cells += 1
    return (tot / cells if cells else 0.0), cells


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    json.dump(obj, open(path, 'w'), default=conv, indent=1)


# ----------------------------------------------------------------- corpora
PE_SIGNS = ['M362', 'M367', 'M346', 'M006', 'M362~a', 'M367~a', 'M346~a', 'M006@g']


def pe_corpus(drop_copy=True):
    from pe20_common import pe_records
    T = {t['id']: t for t in json.load(open(os.path.join(DATA, 'pe_corpus.json')))}
    R = pe_records()
    if drop_copy:
        R = [r for r in R if r[0] != 'P008389']
    X = np.array([[r[2][s] if r[2][s] is not None else 0.0 for s in PE_SIGNS] for r in R], float)
    strings = []
    for tid, lab, _ in R:
        lines = T[tid]['lines']
        s = None
        for i, l in enumerate(lines):
            if (l['surface'][0] + l['label']) == lab:
                sg = [x for x in l['signs'] if x not in PE_SIGNS]
                if sg and any('M362' in x for x in sg):
                    s = ' '.join(sg)
                elif i > 0:
                    s = ' '.join(lines[i - 1]['signs']) or None
                break
        strings.append(s or '?%s%s' % (tid, lab))
    return R, X, np.array([r[0] for r in R]), strings


def ur3_corpus():
    d = json.load(open(os.path.join(CK, 'ur3_sightings.json')))
    X = np.array([o['v'] for o in d], float)
    return d, X, np.array([o['id'] for o in d]), [o['herd'] for o in d], [o['year'] for o in d]


def retrieval(S, strings, forbid, sizes=None):
    """For each entry with a same-string partner: rank of the nearest true partner among all
    allowed candidates (1 = best) by symmetric score. Returns mean reciprocal rank, hits@1, n, chance MRR."""
    Ssym = np.maximum(S, S.T)
    n = len(strings)
    rr, h1, ch = [], 0, []
    for i in range(n):
        cand = [j for j in range(n) if not forbid[i, j]]
        pos = [j for j in cand if strings[j] == strings[i]]
        if not pos:
            continue
        sc = np.array([Ssym[i, j] for j in cand])
        best = max(Ssym[i, j] for j in pos)
        rank = 1 + np.sum(sc > best)
        rr.append(1.0 / rank)
        h1 += rank == 1
        m = len(cand)
        ch.append(np.mean([1.0 / r for r in range(1, m + 1)]) if len(pos) == 1 else min(1, len(pos) / m * 2))
    return dict(mrr=float(np.mean(rr)) if rr else float('nan'), hit1=int(h1), n=len(rr),
                mrr_chance=float(np.mean(ch)) if ch else float('nan'))


def size_scores(C):
    t = np.log1p(C.X.sum(1))
    return -np.abs(t[:, None] - t[None, :])


def order_eval(S, strings, years, forbid):
    ok, n = 0, 0
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            if forbid[i, j] or strings[i] != strings[j]:
                continue
            if years[i] is None or years[j] is None or years[i] == years[j]:
                continue
            pred_i_first = S[i, j] > S[j, i]
            ok += pred_i_first == (years[i] < years[j])
            n += 1
    return ok, n
