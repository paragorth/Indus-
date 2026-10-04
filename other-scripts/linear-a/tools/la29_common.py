#!/usr/bin/env python3
"""la29 shared: per-site commodity and sign profiles, land table, spatial random-field null,
leave-one-site-out (LOSO) multinomial land models. No readings; logogram NAMES and sign identities only."""
import json, math, os, re, sys, collections as C
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la29_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
from la29_land import LA_NAME, SITES

CATS = ['GRA', 'VIN', 'OLE', 'OLIV', 'CYP', 'LIV', 'VIR', 'S301', 'S304', 'S401', 'OTHER']


def cat_la(v):
    v = v.strip("*[]'")
    b = v.split('+')[0].strip("*[]'")
    if b in ('GRA', 'VIN', 'OLE', 'OLIV', 'CYP', 'VIR'): return b
    if b in ('CAP', 'CAPm', 'OVIS', 'SUS', 'BOS', 'HIDE'): return 'LIV'
    if b == '301': return 'S301'
    if b == '304': return 'S304'
    if b == '401': return 'S401'
    return 'OTHER'


def corpus():
    return json.load(open(os.path.join(DATA, 'corpus.json')))


def la_commodity(unit='doc'):
    """{site_code: Counter(cat)}; unit='doc' counts each (document, category) once."""
    out = C.defaultdict(C.Counter)
    for d in corpus():
        s = LA_NAME.get(d['site'])
        if s is None: continue
        seen = set()
        for t in d['tokens']:
            if t['t'] != 'logo': continue
            c = cat_la(t['v'])
            if unit == 'doc':
                if c in seen: continue
                seen.add(c)
            out[s][c] += 1
    return out


ADMIN_SUPPORTS = {'Tablet', 'Nodule', 'Roundel', 'Lames (short thin tablet)', 'Sealing', '3-sided bar',
                  '4-sided bar', 'Label'}


def la_signs():
    """{site: Counter(sign)} over syllabic signs inside words + logogram bases (prefixed 'L:'),
    and site document counts."""
    out = C.defaultdict(C.Counter); nd = C.Counter()
    adm = os.environ.get('ADMIN') == '1'
    for d in corpus():
        s = LA_NAME.get(d['site'])
        if s is None: continue
        if adm and d['support'] not in ADMIN_SUPPORTS: continue
        nd[s] += 1
        for t in d['tokens']:
            if t['t'] == 'word':
                for g in t['s']: out[s][g] += 1
            elif t['t'] == 'logo':
                out[s]['L:' + cat_la(t['v']) if cat_la(t['v']) != 'OTHER' else 'L:' + t['v'].split('+')[0].strip("*[]'")] += 1
    return out, nd


def lb_commodity():
    from la25_common import lb_entries
    cats = ['GRA', 'VIN', 'OLE', 'OLIV', 'FIC', 'LIV', 'LANA', 'OTHER']
    out = C.defaultdict(C.Counter); seen = set()
    code = {'KN': 'KN', 'PY': 'PYL', 'TH': 'THB', 'MY': 'MYC'}
    for s, h, ser, cat, amt in lb_entries():
        k = (h, cat)
        if k in seen: continue
        seen.add(k)
        out[code[s]][cat] += 1
    return out, cats


def land_table():
    return json.load(open(os.path.join(DATA, 'la29_land.json')))


def land_matrix(sites, land=None, exclude=('lat', 'lon')):
    land = land or land_table()
    keys = sorted(k for k in land[sites[0]] if k not in exclude and isinstance(land[sites[0]][k], (int, float))
                  and all(land[s].get(k) is not None for s in sites))
    X = np.array([[land[s][k] for k in keys] for s in sites], float)
    ok = X.std(0) > 1e-9
    return X[:, ok], [k for k, o in zip(keys, ok) if o]


def km_coords(sites):
    lat0 = np.mean([SITES[s][0] for s in sites])
    return np.array([[SITES[s][1] * 111.32 * math.cos(math.radians(lat0)), SITES[s][0] * 110.57] for s in sites])


def grf(sites, L_km, n, rng, kernel='gauss'):
    """n draws of a zero-mean unit-variance spatial random field at site coordinates."""
    P = km_coords(sites)
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    L = np.atleast_1d(L_km)
    out = np.empty((n, len(sites)))
    Ls = rng.choice(L, n) if L.size > 1 else np.full(n, L[0])
    for Lv in np.unique(Ls):
        idx = np.where(Ls == Lv)[0]
        K = np.exp(-(D / Lv) ** 2) if kernel == 'gauss' else np.exp(-D / Lv)
        K += 1e-6 * np.eye(len(sites))
        w, V = np.linalg.eigh(K)
        A = V * np.sqrt(np.clip(w, 0, None))
        out[idx] = rng.standard_normal((idx.size, len(sites))) @ A.T
    return out


# ------------------------------------------------------------------ LOSO multinomial land model
def fit_softmax(Y, x, lam=1.0, alpha=0.5, iters=200):
    """Y[S, C] counts, x[S] standardized covariate (or None). Each site weighted equally
    (its shares, Dirichlet-smoothed with alpha). Returns a[C], b[C]. Ridge lam on b."""
    S, Cn = Y.shape
    P = (Y + alpha) / (Y + alpha).sum(1, keepdims=True)
    a = np.log(P.mean(0)); b = np.zeros(Cn)
    if x is None:
        return a, b
    lr = 0.5
    for _ in range(iters):
        eta = a[None] + x[:, None] * b[None]
        eta -= eta.max(1, keepdims=True)
        Q = np.exp(eta); Q /= Q.sum(1, keepdims=True)
        G = P - Q                                   # gradient of mean cross-entropy wrt eta (neg)
        ga = G.mean(0); gb = (G * x[:, None]).mean(0) - lam * b / S
        a += lr * ga; b += lr * gb
    return a, b


def predict(a, b, x):
    eta = a + x * b
    eta = eta - eta.max()
    q = np.exp(eta); return q / q.sum()


def loso_skill(Y, x, lam=1.0, alpha=0.5):
    """Mean over held-out sites of per-entry log-likelihood gain of land model vs pooled baseline.
    x: [S] covariate (standardized inside each fold on training sites)."""
    S = Y.shape[0]
    g = np.zeros(S)
    for i in range(S):
        tr = np.arange(S) != i
        xt = x[tr]; mu, sd = xt.mean(), xt.std() + 1e-12
        a0, _ = fit_softmax(Y[tr], None, alpha=alpha)
        a, b = fit_softmax(Y[tr], (xt - mu) / sd, lam=lam, alpha=alpha)
        q1 = predict(a, b, np.clip((x[i] - mu) / sd, -3, 3)); q0 = predict(a0, np.zeros_like(a0), 0.0)
        n = Y[i].sum()
        g[i] = (Y[i] * (np.log(q1) - np.log(q0))).sum() / max(n, 1)
    return g.mean(), g


def loso_skill_batch(Y, Xn, lam=1.0, alpha=0.5, iters=150):
    """Vectorised over many covariates: Xn[N, S]. Returns skill[N]."""
    N, S = Xn.shape
    Cn = Y.shape[1]
    P = (Y + alpha) / (Y + alpha).sum(1, keepdims=True)
    gains = np.zeros((N, S))
    for i in range(S):
        tr = np.arange(S) != i
        Pt = P[tr]
        xt = Xn[:, tr]
        mu = xt.mean(1, keepdims=True); sd = xt.std(1, keepdims=True) + 1e-12
        z = (xt - mu) / sd                                     # [N, S-1]
        a = np.tile(np.log(Pt.mean(0)), (N, 1)); b = np.zeros((N, Cn))
        a0 = np.log(Pt.mean(0))
        for _ in range(iters):
            eta = a[:, None, :] + z[:, :, None] * b[:, None, :]
            eta -= eta.max(2, keepdims=True)
            Q = np.exp(eta); Q /= Q.sum(2, keepdims=True)
            G = Pt[None] - Q
            a += 0.5 * G.mean(1)
            b += 0.5 * ((G * z[:, :, None]).mean(1) - lam * b / (S - 1))
        zi = np.clip((Xn[:, i] - mu[:, 0]) / sd[:, 0], -3, 3)
        eta = a + zi[:, None] * b
        eta -= eta.max(1, keepdims=True)
        q1 = np.exp(eta); q1 /= q1.sum(1, keepdims=True)
        q0 = np.exp(a0 - a0.max()); q0 /= q0.sum()
        n = Y[i].sum()
        gains[:, i] = (Y[i][None] * (np.log(q1) - np.log(q0)[None])).sum(1) / max(n, 1)
    return gains.mean(1), gains
