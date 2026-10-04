"""pe11: REBUILD THE CLOCK FROM THE ANIMALS.  Shared code.

Undated tablets are treated as shuffled frames of a film.  Each tablet becomes a
frame: a vector of economic features (commodity shares, log amounts).  A latent
phase z_t in {0..S-1} (S = 12 'months') is searched for every tablet by annealed
Gibbs sampling + hard EM with many restarts, under two models with the SAME number
of parameters per feature:
    circ : mu_k(s) = a + b cos(2 pi s/S) + c sin(2 pi s/S)   (closed loop = a year)
    line : mu_k(s) = a + b t + c t^2, t = s scaled to [-1, 1] (open drift, no year)
Score R2 = 1 - SSE / SST on standardized features.  The seasonal signature is the
LOOP EXCESS  D = R2_circ - R2_line : a yearly cycle with phase-lagged commodities
(lambing, culling, harvest) puts frames on a closed ring; a static mixture or a
secular drift does not.
Nulls: column shuffle across tablets, Gaussian-copula surrogate (same marginals and
rank correlations, no ring), random-walk surrogate (secular drift, no season).
"""
import json, os, re, sys, csv
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa: E402
from pe5_common import pe_system, pe_value, VSETS, CSETS, FRACV, parse_ur_line  # noqa: E402

PEDATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(PEDATA, 'pe11_ckpt')
os.makedirs(CKPT, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
S = 12


# ------------------------------------------------------------------ frames
def pe_tablets(min_entries=3):
    """Tablet -> dict(hdr, entries=[(final, sys, value, nsigns, frac)], signs=set)."""
    T = load()
    out = {}
    vs, cs = VSETS['D3'], CSETS['NOT']
    for t in T:
        lines = t['lines']
        num = [(i, l) for i, l in enumerate(lines) if l['numerals']]
        obv = [x for x in num if x[1]['surface'] == 'obverse']
        off = [x for x in num if x[1]['surface'] == 'reverse']
        tot_idx = off[0][0] if len(off) == 1 and len(obv) >= 2 else None
        hdr = None
        if lines and not lines[0]['numerals']:
            s = [base(x) for x in lines[0]['signs'] if is_sign(x)]
            hdr = s[0] if s else None
        ents = []
        for i, l in num:
            if l['surface'] == 'top' or i == tot_idx:
                continue
            sysn = pe_system(l['numerals'])
            sg = [base(x) for x in l['signs'] if is_sign(x)]
            if sysn is None or not sg:
                continue
            val = pe_value(l['numerals'], sysn, vs, cs)
            if val is None or val <= 0:
                continue
            frac = any(c in FRACV for _, c in l['numerals'])
            ents.append((sg[-1], sysn, float(val), len(sg), frac))
        if len(ents) >= min_entries:
            allsg = set(base(x) for l in lines for x in l['signs'] if is_sign(x))
            out[t['id']] = {'hdr': hdr, 'entries': ents, 'signs': allsg,
                            'design': t.get('designation', '')}
    return out


def pe_pool(tabs, top=16):
    """Final signs used for share features (by number of tablets)."""
    c = Counter()
    for t in tabs.values():
        for f in set(e[0] for e in t['entries']):
            c[f] += 1
    return [s for s, _ in c.most_common(top)]


def pe_features(tabs, pool, ids=None):
    """Feature matrix (complete).  Returns X, names, ids."""
    ids = ids or sorted(tabs)
    names = ['sh_' + s for s in pool] + ['shC', 'shFrac', 'logS', 'logC', 'mlogS', 'sh1', 'logN', 'shLong']
    X = np.zeros((len(ids), len(names)))
    for r, tid in enumerate(ids):
        E = tabs[tid]['entries']
        n = len(E)
        fc = Counter(e[0] for e in E)
        row = [fc[s] / n for s in pool]
        sv = [e[2] for e in E if e[1] == 'S']
        cv = [e[2] for e in E if e[1] == 'C']
        row += [len(cv) / n, sum(e[4] for e in E) / n,
                np.log10(1 + sum(sv)), np.log10(1 + sum(cv)),
                np.mean(np.log10(sv)) if sv else 0.0,
                (sum(1 for v in sv if v == 1) / len(sv)) if sv else 0.0,
                np.log10(n), sum(1 for e in E if e[3] >= 3) / n]
        X[r] = row
    return X, names, ids


# Ur III Drehem (Puzrish-Dagan): animals with known month
UR_TERMS = [('fawn', lambda w: 'amar' in w and 'masz-da3' in w), ('gazelle', lambda w: 'masz-da3' in w),
            ('kir11', lambda w: 'kir11' in w), ('sila4', lambda w: 'sila4' in w),
            ('calf', lambda w: 'amar' in w), ('u8', lambda w: 'u8' in w), ('ud5', lambda w: 'ud5' in w),
            ('masz2', lambda w: 'masz2' in w), ('udu', lambda w: 'udu' in w),
            ('ab2', lambda w: 'ab2' in w), ('gu4', lambda w: 'gu4' in w),
            ('anszi', lambda w: 'anszi' in w or 'dusu2' in w)]


def ur3_tablets(prov='Puzri', min_entries=2):
    """Drehem tablets with month known: id -> dict(month, entries=[(term, value, niga)])."""
    csv.field_size_limit(10 ** 9)
    keep = {}
    for r in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if not (r['period'].startswith('Ur III') and r['provenience'].startswith(prov)):
            continue
        m = re.match(r'^([^.]+)\.(\d\d)\.(\d\d)', r['date_of_origin'].strip())
        if not m or m.group(2) == '00' or int(m.group(3)) > 12 or int(m.group(3)) == 0:
            continue
        keep['P%06d' % int(r['id_text'])] = (m.group(1) + '.' + m.group(2), int(m.group(3)))
    out = {}
    cur, ents = None, []

    def flush():
        if cur and len(ents) >= min_entries:
            out[cur] = {'year': keep[cur][0], 'month': keep[cur][1] - 1, 'entries': list(ents)}
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            flush()
            pid = raw[1:8]
            cur = pid if pid in keep else None
            ents = []
            continue
        if cur is None:
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.strip())
        if not m:
            continue
        p = parse_ur_line(m.group(1))
        if not p or p[1]:
            continue
        w = set(re.split(r'[\s]+', p[2]))
        ww = ' '.join(w)
        term = None
        for name, f in UR_TERMS:
            if f(ww):
                term = name
                break
        if term:
            ents.append((term, float(p[0]), 'niga' in ww))
    flush()
    return out


def ur3_features(tabs, ids=None):
    ids = ids or sorted(tabs)
    terms = [t for t, _ in UR_TERMS]
    names = ['sh_' + t for t in terms] + ['shNiga', 'logTot', 'logN']
    X = np.zeros((len(ids), len(names)))
    for r, tid in enumerate(ids):
        E = tabs[tid]['entries']
        tot = sum(e[1] for e in E)
        c = Counter()
        for e in E:
            c[e[0]] += e[1]
        X[r] = [c[t] / tot for t in terms] + [sum(e[1] for e in E if e[2]) / tot,
                                               np.log10(tot), np.log10(len(E))]
    return X, names, ids


# ------------------------------------------------------------------ model
def basis(model, H=1):
    s = np.arange(S)
    if model == 'circ':
        cols = [np.ones(S)]
        for h in range(1, H + 1):
            cols += [np.cos(2 * np.pi * h * s / S), np.sin(2 * np.pi * h * s / S)]
    else:
        t = s / (S - 1) * 2 - 1
        cols = [t ** k for k in range(2 * H + 1)]
    return np.stack(cols, 1)


def standardize(X):
    X = np.asarray(X, float)
    sd = X.std(0)
    keep = sd > 1e-9
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep]


def fit(X, model, rng, H=1, restarts=8, sweeps=30, T0=None):
    """Annealed Gibbs + hard EM over phase assignments.  Returns (R2, z)."""
    n, p = X.shape
    B = basis(model, H)
    x2 = (X ** 2).sum(1)
    best = (-np.inf, None)
    T0 = T0 or 0.5 * p
    temps = list(T0 * (1e-3 / 1.0) ** (np.arange(sweeps) / max(1, sweeps - 1))) + [0] * 15
    for r in range(restarts):
        z = rng.integers(0, S, n)
        last = None
        for T in temps:
            D = B[z]
            coef, *_ = np.linalg.lstsq(D, X, rcond=None)
            mu = B @ coef                                 # S x p
            cost = x2[:, None] - 2 * X @ mu.T + (mu ** 2).sum(1)[None, :]
            if T > 0:
                lp = -(cost - cost.min(1, keepdims=True)) / T
                pr = np.exp(lp)
                pr /= pr.sum(1, keepdims=True)
                u = rng.random(n)[:, None]
                z = (pr.cumsum(1) < u).sum(1)
                z = np.minimum(z, S - 1)
            else:
                znew = cost.argmin(1)
                if last is not None and np.array_equal(znew, z):
                    break
                last = z = znew
        D = B[z]
        coef, *_ = np.linalg.lstsq(D, X, rcond=None)
        sse = ((X - D @ coef) ** 2).sum()
        r2 = 1 - sse / (n * p)
        if r2 > best[0]:
            best = (r2, z.copy())
    return best


def loop_score(X, rng, H=1, restarts=8, sweeps=30):
    X = standardize(X)
    rc, zc = fit(X, 'circ', rng, H, restarts, sweeps)
    rl, zl = fit(X, 'line', rng, H, restarts, sweeps)
    return {'R2c': rc, 'R2l': rl, 'D': rc - rl, 'z': zc}


# ------------------------------------------------------------------ nulls
def null_shuffle(X, rng):
    Y = X.copy()
    for k in range(X.shape[1]):
        Y[:, k] = rng.permutation(Y[:, k])
    return Y


def _ranks(v):
    from scipy.stats import rankdata
    return rankdata(v)


def null_copula(X, rng):
    """Gaussian copula: same marginals (exact values), same normal-score correlation."""
    from scipy.stats import norm
    n, p = X.shape
    Zs = np.stack([norm.ppf((_ranks(X[:, k]) - 0.5) / n) for k in range(p)], 1)
    C = np.corrcoef(Zs.T) if p > 1 else np.ones((1, 1))
    C = np.nan_to_num(C) + 1e-9 * np.eye(p)
    G = rng.multivariate_normal(np.zeros(p), C, n)
    Y = np.zeros_like(X)
    for k in range(p):
        srt = np.sort(X[:, k])
        Y[np.argsort(G[:, k]), k] = srt
    return Y


def null_rw(X, rng):
    """Independent random walks along one hidden order, mapped to the real marginals."""
    n, p = X.shape
    Y = np.zeros_like(X)
    for k in range(p):
        w = np.cumsum(rng.normal(size=n))
        Y[np.argsort(w), k] = np.sort(X[:, k])
    return Y


NULLS = {'shuffle': null_shuffle, 'copula': null_copula, 'rw': null_rw}


# ------------------------------------------------------------------ order recovery
def align_acc(z, m, tol=1):
    """Best fraction within +-tol months over 12 rotations x 2 reflections, and the map."""
    best = (-1, None)
    for f in (1, -1):
        for r in range(S):
            zz = (f * z + r) % S
            d = np.abs(zz - m) % S
            d = np.minimum(d, S - d)
            a = float(np.mean(d <= tol))
            if a > best[0]:
                best = (a, (f, r))
    return best


def circ_corr(z, m):
    """Fisher-Lee style |circular correlation| maximised over reflection."""
    a = 2 * np.pi * z / S
    b = 2 * np.pi * m / S
    best = 0
    for f in (1, -1):
        c = np.abs(np.mean(np.exp(1j * (f * a - b))))
        best = max(best, c)
    return float(best)


def save(name, obj):
    json.dump(obj, open(os.path.join(PEDATA, name), 'w'), indent=1, default=float)
