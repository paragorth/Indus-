"""pe58 THE WEIGHTS ARE EXCHANGE RATES: shared library.

Arrow in the dark: a sign that multiplies the quantity next to it by a fixed factor behaves like a conversion
(worker grade -> ration, grain -> flour/beer, adult -> young animal). Conversion factors are not arbitrary:
they sit on simple ratios and near physical constants measurable outside the corpus.

Engine
  factor(s): many random estimator specifications x tablet bootstraps of the within-tablet log-quantity shift
             of entries carrying sign s against same-system siblings without s.
             estimators: dmean, dmed, matched (most similar sibling string), all-pairs mode (KDE), corpus-wide
             exact-remainder match (string w vs w+s anywhere); options: system filter, drop q=1, trimming, bandwidth.
  pair(a,b): direct exchange rate between two signs on tablets carrying both (entries with a and not b vs b and not a).
  spike(s):  concentration of all pairwise log ratios at their mode, against quantities shuffled within tablet.
  lattice:   mean distance of |log f| to the nearest simple ratio p/q (p,q<=6, p!=q).
  physics:   CI overlap with external conversion constants (data/pe58_constants.json) vs random constant sets.
Corpora: list of dict(tab, w tuple, q>0, sys).
"""
import os, sys, json, math, random
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe58_ckpt'); os.makedirs(CK, exist_ok=True)


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


# ------------------------------------------------------------------ corpora
def pe_corpus():
    from pe52_lib import pe_corpus as pc
    return pc()


def ur3_ta(total=False, seed=0, opaque=True):
    """Ur III distributive ration/fodder lines: tokens = grade words + up to 3 following words; q = rate per
    head (or n x rate if total). Words become opaque ids; key returned for scoring only."""
    T = json.load(open(os.path.join(CK, 'ur3ta.json')))
    vocab = {}
    rng = random.Random(seed)
    words = sorted({w for t in T for w in t['grade'] + t['after'][:3]})
    ids = list(range(len(words))); rng.shuffle(ids)
    vocab = {w: 'U%05d' % i for w, i in zip(words, ids)}
    out = []
    for t in T:
        w = t['grade'] + [x for x in t['after'][:3] if not x[0].isdigit()]
        w = tuple(dict.fromkeys(vocab[x] if opaque else x for x in w))
        q = t['rate'] * (t['n'] if total else 1)
        if q > 0:
            out.append(dict(tab=t['tab'], w=w, q=float(q), sys='C'))
    return out, vocab


def planted(C, seed, kind, k=10, exclude=()):
    """PE skeleton with k planted multiplicative signs. kind: 'ladder' (simple ratios), 'arb' (arbitrary),
    'phys' (centres of external constants). Count quantities are rounded to integers >= 1."""
    rng = np.random.default_rng(seed)
    G = groups(C)
    cnt = Counter();
    for r in C:
        for s in set(r['w']):
            cnt[s] += 1
    ok = [s for s in cnt if 20 <= cnt[s] <= 250 and s not in exclude and len(sign_groups(C, G, s)) >= 8]
    chosen = list(rng.choice(sorted(ok), size=min(k, len(ok)), replace=False))
    if kind == 'ladder':
        pool = [2, 3, 1.5, 4 / 3, 0.5, 1 / 3, 2 / 3, 0.75, 4, 0.25]
        f = {s: float(rng.choice(pool)) for s in chosen}
    elif kind == 'phys':
        K = constants()
        cs = [math.sqrt(v['lo'] * v['hi']) for v in K.values() if abs(math.log(v['lo'])) > 0.1]
        f = {s: float(rng.choice(cs)) ** float(rng.choice([1, -1])) for s in chosen}
    else:
        f = {s: float(math.exp(rng.choice([-1, 1]) * rng.uniform(0.25, 1.1))) for s in chosen}
    out = []
    for r in C:
        m = 1.0
        for s in set(r['w']):
            m *= f.get(s, 1.0)
        q = r['q'] * m
        if r['sys'] == 'K':
            q = max(1.0, round(q))
        out.append(dict(r, q=float(q)))
    return out, f


def constants():
    return json.load(open(os.path.join(DATA, 'pe58_constants.json')))['constants']


# ------------------------------------------------------------------ nulls
def qshuf_tab(C, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(C):
        by[(r['tab'], r['sys'])].append(i)
    out = [dict(r) for r in C]
    for idx in by.values():
        qs = [C[i]['q'] for i in idx]; rng.shuffle(qs)
        for i, q in zip(idx, qs):
            out[i]['q'] = q
    return out


# ------------------------------------------------------------------ engine
def groups(C):
    G = defaultdict(list)
    for i, r in enumerate(C):
        G[(r['tab'], r['sys'])].append(i)
    return G


def sign_groups(C, G, s):
    out = []
    for g, idx in G.items():
        P = [i for i in idx if s in C[i]['w']]
        if P and len(P) < len(idx):
            out.append((g, P, [i for i in idx if s not in C[i]['w']]))
    return out


def draw_spec(rng):
    return dict(est=rng.choice(['dmean', 'dmed', 'match', 'pairs', 'xmatch']), sys='any', ref=rng.choice(['all', 'plain']),
                drop1=rng.random() < 0.3, agg=rng.choice(['mean', 'median', 'trim', 'mode']),
                bw=rng.uniform(0.05, 0.2), jmin=rng.choice([0.0, 0.2, 0.34, 0.5]), boot=rng.random() < 0.85)


def draw_spec2(rng):
    """calibrated family (cycle 1b: lowest planted error): mean-type estimators, no KDE mode, no corpus-wide match"""
    return dict(est=rng.choice(['dmean', 'match', 'pairs']), sys='any', ref=rng.choice(['all', 'plain']),
                drop1=rng.random() < 0.2, agg=rng.choice(['mean', 'mean', 'trim', 'median']), bw=0.1,
                jmin=rng.choice([0.0, 0.2]), boot=True)


def _jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def _agg(d, wt, how, bw):
    d = np.asarray(d); wt = np.asarray(wt, float)
    if len(d) == 0:
        return None
    if how == 'mean':
        return float(np.sum(d * wt) / np.sum(wt))
    if how == 'median' or len(d) < 4:
        o = np.argsort(d); cw = np.cumsum(wt[o]); return float(d[o][np.searchsorted(cw, cw[-1] / 2)])
    if how == 'trim':
        lo, hi = np.percentile(d, [20, 80]); k = (d >= lo) & (d <= hi)
        return float(np.sum(d[k] * wt[k]) / np.sum(wt[k])) if k.any() else float(np.median(d))
    grid = np.linspace(d.min() - 0.1, d.max() + 0.1, 400)
    dens = (wt[None, :] * np.exp(-0.5 * ((grid[:, None] - d[None, :]) / bw) ** 2)).sum(1)
    return float(grid[np.argmax(dens)])


class Est:
    def __init__(self, C, W=()):
        self.C = C
        self.W = set(W)   # weighted-sign set: 'plain' reference = siblings carrying none of them
        self.G = groups(C)
        self.lq = np.array([math.log(r['q']) for r in C])
        self.cache = {}
        self.xidx = defaultdict(list)   # remainder string -> entry idx (corpus-wide)
        for i, r in enumerate(C):
            self.xidx[(r['sys'], tuple(sorted(set(r['w']))))].append(i)

    def sg(self, s):
        if s not in self.cache:
            self.cache[s] = sign_groups(self.C, self.G, s)
        return self.cache[s]

    def one(self, s, spec, rng):
        C, lq = self.C, self.lq
        if spec['est'] == 'xmatch':
            d = []; wt = []
            seen = set()
            for i, r in enumerate(C):
                if s not in r['w'] or (spec['sys'] != 'any' and r['sys'] != spec['sys']):
                    continue
                rem = tuple(sorted(set(r['w']) - {s}))
                if not rem:
                    continue
                J = self.xidx.get((r['sys'], rem), [])
                if J:
                    d.append(lq[i] - float(np.mean(lq[J]))); wt.append(1.0); seen.add(r['tab'])
            if len(d) < 3:
                return None
            if spec['boot']:
                k = rng.integers(0, len(d), len(d)); d = list(np.array(d)[k]); wt = list(np.array(wt)[k])
            return _agg(d, wt, spec['agg'], spec['bw'])
        gs = [g for g in self.sg(s) if spec['sys'] == 'any' or g[0][1] == spec['sys']]
        if spec.get('ref') == 'plain' and self.W:
            gs = [(g, P, [j for j in M if not (set(C[j]['w']) & self.W)]) for g, P, M in gs]
            gs = [x for x in gs if x[2]]
        if len(gs) < 3:
            return None
        if spec['boot']:
            gs = [gs[j] for j in rng.integers(0, len(gs), len(gs))]
        d = []; wt = []
        for g, P, M in gs:
            if spec['drop1']:
                P = [i for i in P if C[i]['q'] != 1] or P
                M = [i for i in M if C[i]['q'] != 1] or M
            e = spec['est']
            if e == 'dmean':
                d.append(lq[P].mean() - lq[M].mean()); wt.append(math.sqrt(len(P) * len(M)))
            elif e == 'dmed':
                d.append(float(np.median(lq[P]) - np.median(lq[M]))); wt.append(math.sqrt(len(P) * len(M)))
            elif e == 'match':
                for i in P:
                    rem = set(C[i]['w']) - {s}
                    js = [_jac(rem, C[j]['w']) for j in M]
                    b = max(js)
                    if b < spec['jmin']:
                        continue
                    best = [j for j, x in zip(M, js) if x == b]
                    d.append(lq[i] - lq[best].mean()); wt.append(1.0)
            else:
                w0 = 1.0 / (len(P) * len(M))
                for i in P:
                    for j in M:
                        d.append(lq[i] - lq[j]); wt.append(w0)
        if len(d) < 3:
            return None
        return _agg(d, wt, spec['agg'], spec['bw'])

    def factor(self, s, n=200, seed=0, spec2=False):
        rng = np.random.default_rng(seed); r2 = random.Random(seed)
        v = []
        for _ in range(n):
            x = self.one(s, (draw_spec2 if spec2 else draw_spec)(r2), rng)
            if x is not None:
                v.append(x)
        if len(v) < n // 4:
            return None
        v = np.array(v)
        return dict(med=float(np.median(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                    iqr=float(np.subtract(*np.percentile(v, [75, 25]))), n=len(v))

    def pair(self, a, b, nboot=200, seed=0):
        """direct exchange rate a:b on tablets carrying both (entries with a not b vs b not a)"""
        C, lq = self.C, self.lq
        ds = []
        for g, idx in self.G.items():
            A = [i for i in idx if a in C[i]['w'] and b not in C[i]['w']]
            B = [i for i in idx if b in C[i]['w'] and a not in C[i]['w']]
            if A and B:
                ds.append(lq[A].mean() - lq[B].mean())
        if len(ds) < 3:
            return None
        ds = np.array(ds); rng = np.random.default_rng(seed)
        one = np.ones(len(ds))
        bs = []; bm = []
        for _ in range(nboot):
            k = ds[rng.integers(0, len(ds), len(ds))]
            bs.append(float(np.median(k))); bm.append(_agg(k, one, 'mode', 0.08))
        return dict(med=float(np.median(ds)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                    mode=_agg(ds, one, 'mode', 0.08), mlo=float(np.percentile(bm, 2.5)), mhi=float(np.percentile(bm, 97.5)),
                    ntab=len(ds), sd=float(np.std(ds)))

    def spike(self, s, h=0.07):
        d = []; wt = []
        for g, P, M in self.sg(s):
            w0 = 1.0 / (len(P) * len(M))
            for i in P:
                for j in M:
                    d.append(self.lq[i] - self.lq[j]); wt.append(w0)
        if len(d) < 5:
            return None
        d = np.array(d); wt = np.array(wt)
        grid = np.linspace(d.min(), d.max(), 300)
        conc = np.array([wt[np.abs(d - x) <= h].sum() for x in grid]) / wt.sum()
        return float(conc.max()), float(grid[np.argmax(conc)])


# ------------------------------------------------------------------ lattice / physics
def lattice_points(pmax=6):
    pts = sorted({math.log(p / q) for p in range(1, pmax + 1) for q in range(1, pmax + 1) if p != q})
    return np.array(pts)


LAT = lattice_points()
LATS = {6: lattice_points(6), 4: lattice_points(4)}


def lat_dist(x, pmax=6):
    x = np.atleast_1d(np.asarray(x, float))
    L = LATS.get(pmax, LAT)
    return np.min(np.abs(x[:, None] - L[None, :]), axis=1)


def lattice_test(logs, null_pool=None, n=20000, jit=0.2, seed=0, pmax=6):
    """mean distance of the factors to the simple-ratio lattice vs (a) uniform jitter +-jit and (b) an
    empirical pool of factors (same |log| floor) drawn in sets of the same size."""
    logs = np.asarray(logs, float)
    if len(logs) == 0:
        return dict(obs=None, k=0)
    lat_d = lambda x: lat_dist(x, pmax)
    obs = float(lat_d(logs).mean())
    rng = np.random.default_rng(seed)
    J = logs[None, :] + rng.uniform(-jit, jit, (n, len(logs)))
    nj = lat_d(J.ravel()).reshape(J.shape).mean(1)
    out = dict(obs=obs, k=len(logs), jit_mean=float(nj.mean()), p_jit=float((np.sum(nj <= obs) + 1) / (n + 1)))
    if null_pool is not None and len(null_pool) >= len(logs):
        pool = np.asarray(null_pool, float)
        dp = lat_d(pool)
        ne = np.array([dp[rng.choice(len(pool), len(logs), replace=False)].mean() for _ in range(n)])
        out.update(emp_mean=float(ne.mean()), p_emp=float((np.sum(ne <= obs) + 1) / (n + 1)), pool=len(pool))
    return out


def phys_hits(F, K=None, floor=0.1):
    """F: {sign: (lo, hi) of log factor}. A sign hits a constant when its |log| CI overlaps the constant's
    |log| interval. Constants within `floor` of 1 are skipped (no information)."""
    K = K if K is not None else {k: (abs(math.log(v['lo'])), abs(math.log(v['hi']))) for k, v in constants().items()}
    K = {k: (min(a, b), max(a, b)) for k, (a, b) in K.items() if max(a, b) > floor}
    hits = {}
    for s, (lo, hi) in F.items():
        a, b = (lo, hi) if lo >= 0 else ((-hi, -lo) if hi <= 0 else (0.0, max(-lo, hi)))
        hits[s] = [k for k, (c, d) in K.items() if not (b < c or a > d)]
    return hits


def phys_test(F, n=10000, seed=0, floor=0.1, shift=0.4):
    """null: every constant moved by U(-shift, +shift) in log (same widths, similar spread)."""
    K0 = {k: (abs(math.log(v['lo'])), abs(math.log(v['hi']))) for k, v in constants().items()}
    K0 = {k: (min(a, b), max(a, b)) for k, (a, b) in K0.items() if max(a, b) > floor}
    h = phys_hits(F, K0, floor)
    obs = sum(1 for v in h.values() if v)
    nk = sum(1 for k in K0 if any(k in v for v in h.values()))
    rng = np.random.default_rng(seed)
    wid = [b - a for a, b in K0.values()]
    span = max(b for a, b in K0.values())
    nul = []; nulk = []
    for _ in range(n):
        Kr = {}
        for j, (a, b) in enumerate(K0.values()):
            u = abs((a + b) / 2 + rng.uniform(-shift, shift)); w = b - a
            Kr[j] = (max(0.0, u - w / 2), u + w / 2)
        hr = phys_hits(F, Kr, floor)
        nul.append(sum(1 for v in hr.values() if v)); nulk.append(sum(1 for k in Kr if any(k in v for v in hr.values())))
    nul = np.array(nul); nulk = np.array(nulk)
    return dict(obs_signs=obs, null_signs=float(nul.mean()), p_signs=float((np.sum(nul >= obs) + 1) / (n + 1)),
                obs_consts=nk, null_consts=float(nulk.mean()), p_consts=float((np.sum(nulk >= nk) + 1) / (n + 1)),
                hits=h)


def phys_dist_test(est, n=10000, seed=0, floor=0.1, shift=0.4):
    """est: {sign: log factor point estimate}. Statistic: mean distance of |log f| to the nearest constant
    interval (0 inside). Null: constants moved by U(-shift, shift) in log."""
    K0 = [(abs(math.log(v['lo'])), abs(math.log(v['hi']))) for v in constants().values()]
    K0 = [(min(a, b), max(a, b)) for a, b in K0 if max(a, b) > floor]
    x = np.abs(np.array(list(est.values()), float))
    def dist(K):
        lo = np.array([a for a, b in K]); hi = np.array([b for a, b in K])
        d = np.maximum(0, np.maximum(lo[None, :] - x[:, None], x[:, None] - hi[None, :]))
        return d.min(1)
    obs = float(dist(K0).mean())
    rng = np.random.default_rng(seed)
    nul = []
    for _ in range(n):
        Kr = []
        for a, b in K0:
            u = abs((a + b) / 2 + rng.uniform(-shift, shift)); w = b - a
            Kr.append((max(0.0, u - w / 2), u + w / 2))
        nul.append(dist(Kr).mean())
    nul = np.array(nul)
    return dict(obs=obs, null=float(nul.mean()), p=float((np.sum(nul <= obs) + 1) / (n + 1)), k=len(x))


def qshuf_sys(C, seed):
    """quantities shuffled across the whole corpus within number system (keeps roundness, breaks signs)"""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(C):
        by[r['sys']].append(i)
    out = [dict(r) for r in C]
    for idx in by.values():
        qs = [C[i]['q'] for i in idx]; rng.shuffle(qs)
        for i, q in zip(idx, qs):
            out[i]['q'] = q
    return out


def frequent(C, mino=15, mintab=5):
    G = groups(C)
    occ = Counter(s for r in C for s in set(r['w']))
    return sorted(s for s in occ if occ[s] >= mino and len(sign_groups(C, G, s)) >= mintab)


def pair_table(C, signs, minpair=5, floor=0.12, maxw=0.8):
    E = Est(C)
    out = {}
    for i, a in enumerate(signs):
        for b in signs[i + 1:]:
            p = E.pair(a, b)
            if p and p['ntab'] >= minpair:
                out[(a, b)] = p
    use = [p['mode'] for p in out.values() if abs(p['mode']) >= floor and p['mhi'] - p['mlo'] <= maxw]
    return out, use
