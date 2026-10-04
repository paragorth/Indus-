"""LA-17 'words spread like epidemics': shared data, simulator and summaries.

Each word type (2+ syllabic signs) is a pathogen lineage; each site is a host population.
The fitted data are ONLY the site x word incidence matrix and the number of word tokens per
site (sampling effort) plus site coordinates. Archaeological dates are NOT inputs; they are
loaded separately (DATES) and used only to score the inferred order afterwards.
"""
import os, sys, math, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la15_common import load_la, load_lb, write_rows  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'la17')
os.makedirs(OUT, exist_ok=True)

# ---- sites (>= 10 word tokens). Coordinates (lat, lon), approximate, from site gazetteers.
LA_SITES = {
    'Haghia Triada': ('HT', 35.059, 24.792), 'Zakros': ('ZA', 35.098, 26.261),
    'Khania': ('KH', 35.517, 24.018), 'Knossos': ('KN', 35.298, 25.163),
    'Phaistos': ('PH', 35.051, 24.814), 'Palaikastro': ('PK', 35.195, 26.276),
    'Iouktas': ('IO', 35.226, 25.116), 'Arkhalkhori': ('AR', 35.130, 25.270),
    'Syme': ('SY', 35.050, 25.470), 'Tylissos': ('TY', 35.300, 25.020),
    'Petras': ('PE', 35.200, 26.110), 'Malia': ('MA', 35.293, 25.492),
    'Thera': ('TH', 36.352, 25.404),
}
LB_SITES = {'KN': ('KN', 35.298, 25.163), 'PY': ('PY', 37.028, 21.695),
            'TH': ('TH', 38.320, 23.320), 'MY': ('MY', 37.731, 22.756),
            'TI': ('TI', 37.600, 22.800), 'KH': ('KH', 35.517, 24.018)}

# ---- context code -> ordinal phase (used ONLY for scoring, never in fitting)
PHASE = {'MMII': 1, 'MMIII': 2, 'MMIIIA': 2, 'MMIIIB': 2.25, 'LMIA': 3, 'LMI': 3.5,
         'LMIB': 4, 'MMIA': 0}


def la_dates():
    """Median deposit phase per site from the find-context field of the corpus (GORILA /
    site reports as transcribed by lineara.xyz). Kept apart from the fitted data."""
    C = json.load(open(os.path.join(HERE, '..', 'data', 'corpus.json')))
    ph = collections.defaultdict(list)
    for d in C:
        if d['site'] in LA_SITES and d.get('context') in PHASE:
            ph[LA_SITES[d['site']][0]].append(PHASE[d['context']])
    return {k: float(np.median(v)) for k, v in ph.items()}, {k: len(v) for k, v in ph.items()}


LB_DATES = {'KN': 1, 'PY': 2, 'TH': 2, 'MY': 2, 'TI': 2, 'KH': 2}  # KN LM II-IIIA1; rest LH/LM IIIB


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def build(docs, sites):
    codes = [sites[s][0] for s in sites]
    idx = {s: i for i, s in enumerate(sites)}
    E = np.zeros(len(sites))
    words = collections.defaultdict(set)
    for d in docs:
        if d['site'] in idx:
            j = idx[d['site']]
            E[j] += len(d['words'])
            for w in d['words']:
                words[w].add(j)
    W = sorted(words)
    inc = np.zeros((len(W), len(sites)), bool)
    for i, w in enumerate(W):
        inc[i, list(words[w])] = True
    xy = [sites[s][1:] for s in sites]
    D = np.array([[hav(a, b) for b in xy] for a in xy])
    return dict(codes=codes, E=E, inc=inc, D=D, words=W)


def la_data(docs=None):
    return build(docs if docs is not None else load_la(), LA_SITES)


def lb_data(docs=None):
    return build(docs if docs is not None else load_lb(), LB_SITES)


def doc_shuffle(docs, sites, rng):
    """Control: permute site labels over the documents of the analysed sites."""
    sel = [d for d in docs if d['site'] in sites]
    lab = [d['site'] for d in sel]
    rng.shuffle(lab)
    out = []
    for d, s in zip(sel, lab):
        e = dict(d); e['site'] = s; out.append(e)
    return out

# ------------------------------------------------------------------ summaries


def summaries(inc):
    """log richness per site, log shared types per pair, occupancy histogram (1,2,3,4+),
    per-site share of its words seen elsewhere."""
    inc = inc[inc.any(1)]
    K = inc.shape[1]
    f = inc.astype(np.float32)
    r = f.sum(0)
    S = f.T @ f
    iu = np.triu_indices(K, 1)
    occ = inc.sum(1)
    h = [np.sum(occ == 1), np.sum(occ == 2), np.sum(occ == 3), np.sum(occ >= 4)]
    shared = (f * (occ[:, None] > 1)).sum(0) / np.maximum(r, 1)
    tot = max(r.sum(), 1)
    return np.concatenate([np.log1p(r), np.log1p(S[iu]), np.log1p(h), shared,
                           r / tot]).astype(np.float32)

# ------------------------------------------------------------------ simulator

NSTEP = 12


def draw_prior(K, rng, deposit=False):
    p = dict(src=int(rng.integers(K)))
    t = rng.uniform(0, 1, K); t[p['src']] = 0.0
    p['t'] = t
    p['alpha'] = rng.uniform(0, 1.5)       # word births per site ~ effort^alpha (size)
    p['logbeta'] = rng.uniform(math.log(0.05), math.log(50))
    p['logL'] = rng.uniform(math.log(20), math.log(800))   # contact distance scale, km
    p['g'] = rng.uniform(0, 1)             # gravity exponent on site size
    p['phi'] = rng.uniform(0, 1)           # founder copying fraction
    p['gamma'] = rng.uniform(0, 3)         # loss (recovery) rate
    p['s'] = rng.uniform(0.5, 3)           # sd of log word popularity
    if deposit:  # cycle 2: each site is sampled at its own deposit time tau >= t
        p['tau'] = t + (1 - t) * rng.uniform(0, 1, K)
    return p


import ctypes
_lib = ctypes.CDLL(os.path.join(HERE, 'la17_core.so'))
_P = np.ctypeslib.ndpointer
_lib.sim.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _P(np.uint8), _P(np.int32), _P(np.int32),
                     _P(np.int32), _P(np.float64), ctypes.c_double, ctypes.c_double, ctypes.c_double,
                     ctypes.c_double, _P(np.int32), _P(np.int32), _P(np.uint8)]
_lib.seed.argtypes = [ctypes.c_uint64]


def simulate(p, E, D, M, rng, target_types):
    """One outbreak; returns observed incidence (M x K bool)."""
    K = len(E)
    dt = 1.0 / NSTEP
    t = p['t']
    tau = p.get('tau', np.ones(K))
    beta = math.exp(p['logbeta']); L = math.exp(p['logL'])
    En = E / E.mean()
    Wn = np.exp(-D / L) * np.outer(En, En) ** p['g']
    np.fill_diagonal(Wn, 0)
    Wn = Wn / Wn.sum(0).mean()
    steps = np.arange(NSTEP) * dt
    lit = (steps[None, :] >= t[:, None] - 1e-9) & (steps[None, :] < tau[:, None] + 1e-9)
    for j in range(K):
        if not lit[j].any():
            lit[j, min(int(t[j] / dt + 1e-9), NSTEP - 1)] = True
    start = np.array([np.argmax(lit[j]) for j in range(K)], np.int32)
    last = np.array([NSTEP - 1 - np.argmax(lit[j][::-1]) for j in range(K)], np.int32)
    parent = np.full(K, -1, np.int32)
    for j in range(K):
        par = [q for q in range(K) if q != j and lit[q, start[j]] and start[q] < start[j]]
        if par:
            w = Wn[par, j] + 1e-12
            parent[j] = par[rng.choice(len(par), p=w / w.sum())]
    birthw = (En[:, None] ** p['alpha']) * lit
    pr = birthw.ravel() / birthw.sum()
    bi = rng.choice(K * NSTEP, size=M, p=pr)
    bsite = (bi // NSTEP).astype(np.int32); bstep = (bi % NSTEP).astype(np.int32)
    snap = np.zeros(M * K, np.uint8)
    _lib.seed(int(rng.integers(1 << 62)))
    _lib.sim(M, K, NSTEP, np.ascontiguousarray(lit.astype(np.uint8).ravel()), start, parent, last,
             np.ascontiguousarray(Wn.ravel()), beta, dt, p['phi'], p['gamma'], bsite, bstep, snap)
    snap = snap.reshape(M, K).astype(bool)
    wi, sj = np.nonzero(snap)
    f = np.exp(p['s'] * rng.standard_normal(M))
    base = f[wi] * En[sj]
    lo, hi = -25.0, 15.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (1 - np.exp(-math.exp(mid) * base)).sum() < target_types: lo = mid
        else: hi = mid
    pobs = 1 - np.exp(-math.exp((lo + hi) / 2) * base)
    keep = rng.random(len(wi)) < pobs
    out = np.zeros((M, K), bool)
    out[wi[keep], sj[keep]] = True
    return out


def theta_vec(p, K):
    keys = ['alpha', 'logbeta', 'logL', 'g', 'phi', 'gamma', 's']
    return np.concatenate([[p['src']], p['t'], p.get('tau', np.ones(K)), [p[k] for k in keys]])
