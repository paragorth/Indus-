#!/usr/bin/env python3
"""la29 cycle 1: does land predict a held-out site's commodity mix?

LOSO multinomial (one land variable at a time, ridge, equal site weights) scored as per-entry
log-likelihood gain over the pooled mix. Nulls: 10^4 Gaussian random fields at the site
coordinates with the measured correlation lengths (max over M_eff fields for the best-variable
statistic), site size alone, shuffled site labels. Controls: planted land-driven commodities;
'modern ledger' (CLC 2018 land-cover shares drawn at the LA document counts, predicted from DEM
and climate only); Linear B KN/PY/TH/MY.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la29_common import *

rng = np.random.default_rng(29)
NNULL = int(os.environ.get('NNULL', 10000))
MIN_DOCS = int(os.environ.get("MIN_DOCS", 4))
out_lines = []


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out_lines.append(s)


land = land_table()
corr = json.load(open(os.path.join(CK, 'crete_corr.json')))
LS = sorted({max(3.0, v['efold_km']) for v in corr.values()}) + [10.0, 20.0, 40.0]
LS = np.array(sorted(set(LS)))
log('GRF correlation lengths (km):', LS.tolist())

com = la_commodity('doc')
nd_all = {}
for d in corpus():
    s = LA_NAME.get(d['site'])
    if s: nd_all[s] = nd_all.get(s, 0) + 1
CRETE = [s for s in com if sum(com[s].values()) >= MIN_DOCS and s not in ('THE', 'KEA', 'MI')]
CRETE.sort(key=lambda s: -sum(com[s].values()))
Y = np.array([[com[s][c] for c in CATS] for s in CRETE], float)
keep = Y.sum(0) >= 3
cats = [c for c, k in zip(CATS, keep) if k]
Y = Y[:, keep]
log('sites', CRETE, 'n', Y.sum(1).astype(int).tolist())
log('cats', cats)
for s, row in zip(CRETE, Y):
    log('  ', s, dict(zip(cats, row.astype(int).tolist())))

X, keys = land_matrix(CRETE, land)
S = len(CRETE)
real, gains = loso_skill_batch(Y, X.T)
order = np.argsort(-real)
# effective number of independent variables
R = np.corrcoef(X.T); lam = np.linalg.eigvalsh(np.nan_to_num(R))
meff = int(round(lam.sum() ** 2 / (lam ** 2).sum()))
log(f'{len(keys)} land variables, M_eff = {meff}')

# null fields
F = grf(CRETE, LS, NNULL, rng)
nul, _ = loso_skill_batch(Y, F)
np.save(os.path.join(CK, 'c1_null.npy'), nul)
# max over meff fields
nmax = np.array([nul[rng.choice(NNULL, meff, replace=False)].max() for _ in range(5000)])
size = np.log(np.array([nd_all[s] for s in CRETE], float))
sk_size, g_size = loso_skill_batch(Y, size[None])
log(f'size-only (log docs) skill {sk_size[0]:+.4f}  P_field {(nul >= sk_size[0]).mean():.3f}')
log('top variables: skill, P(single field), P(max of M_eff fields)')
for j in order[:12]:
    log(f'  {keys[j]:16s} {real[j]:+.4f}  P1 {(nul >= real[j]).mean():.4f}  Pmax {(nmax >= real[j]).mean():.4f}  '
        f'r(size) {np.corrcoef(X[:, j], size)[0, 1]:+.2f}  per-site ' +
        ' '.join(f'{s}:{g:+.2f}' for s, g in zip(CRETE, gains[j])))
log(f'null skill quantiles 50/95/99: {np.quantile(nul, [.5, .95, .99]).round(4).tolist()}; max-of-Meff 95%: {np.quantile(nmax, .95):.4f}')
best = real[order[0]]
log(f'share of land variables beating 95% single-field null: {(real > np.quantile(nul, .95)).mean():.2f} (expected 0.05)')

# residual on size: does the best land variable add to size?
xs = np.column_stack([size, X[:, order[0]]])

# shuffled site labels: best-variable skill
sh = []
for k in range(200):
    p = rng.permutation(S)
    r_, _ = loso_skill_batch(Y[p], X.T)
    sh.append(r_.max())
sh = np.array(sh)
log(f'shuffled site labels: best-variable skill median {np.median(sh):+.4f}, 95% {np.quantile(sh, .95):+.4f}; '
    f'real best {best:+.4f} rank P {(sh >= best).mean():.3f}')

# planted: two categories driven by a land variable at real counts
pv = [k for k in ('olive_5', 'upland_5', 'rain_10', 'coast_km') if k in keys]
base = (Y + 0.5) / (Y + 0.5).sum(1, keepdims=True)
pooled = base.mean(0)
log('planted land-driven mixes (LOSO best-variable skill vs max-of-Meff null 95%):')
for v in pv:
    z = X[:, keys.index(v)]; z = (z - z.mean()) / z.std()
    for beta in (0.5, 1.0, 2.0):
        hits = 0; hv = 0
        for rep in range(20):
            eta = np.log(pooled)[None].repeat(S, 0)
            eta[:, cats.index('OLE')] += beta * z
            eta[:, cats.index('GRA')] -= beta * z
            P = np.exp(eta); P /= P.sum(1, keepdims=True)
            Yp = np.array([rng.multinomial(int(n), p) for n, p in zip(Y.sum(1), P)], float)
            r_, _ = loso_skill_batch(Yp, X.T)
            hits += r_.max() > np.quantile(nmax, .95)
            hv += keys[int(np.argmax(r_))] == v or abs(np.corrcoef(X[:, int(np.argmax(r_))], z)[0, 1]) > 0.8
        log(f'   {v:10s} beta {beta}: detected {hits}/20, right variable (|r|>0.8) {hv}/20')

# modern ledger control: CLC shares as 'commodities', DEM+climate predictors
cl = sorted(k for k in land[CRETE[0]] if k.startswith('cl_'))
Pcl = np.array([[land[s][k] for k in cl] for s in CRETE]) + 1e-3
Pcl /= Pcl.sum(1, keepdims=True)
nocl = [j for j, k in enumerate(keys) if not k.startswith('cl_')]
for mult, lab in ((1, 'LA counts'), (10, '10x LA counts')):
    hits = 0; sk = []
    for rep in range(20):
        Yc = np.array([rng.multinomial(int(n) * mult, p) for n, p in zip(Y.sum(1), Pcl)], float)
        r_, _ = loso_skill_batch(Yc, X[:, nocl].T)
        nul_c, _ = loso_skill_batch(Yc, F[:2000])
        nm = np.array([nul_c[rng.choice(2000, meff, replace=False)].max() for _ in range(1000)])
        hits += r_.max() > np.quantile(nm, .95); sk.append(r_.max())
    log(f'modern-ledger control ({lab}): best-variable skill {np.mean(sk):+.4f}, detected {hits}/20')

# Linear B (4 sites)
lbc, lbcats = lb_commodity()
LB = ['KN', 'PYL', 'THB', 'MYC']
Yb = np.array([[lbc[s][c] for c in lbcats] for s in LB], float)
Xb, kb = land_matrix(LB, land)
rb, gb = loso_skill_batch(Yb, Xb.T)
Fb = grf(LB, np.array([20.0, 40.0, 80.0]), 4000, rng)
nb, _ = loso_skill_batch(Yb, Fb)
ob = np.argsort(-rb)
log('Linear B 4 sites:', dict(zip(LB, Yb.sum(1).astype(int).tolist())))
for j in ob[:5]:
    log(f'  {kb[j]:16s} {rb[j]:+.4f}  P1 {(nb >= rb[j]).mean():.3f}')
log(f'  LB variables beating 95% field null: {(rb > np.quantile(nb, .95)).mean():.2f}')
json.dump({'sites': CRETE, 'cats': cats, 'keys': keys, 'real': real.tolist(), 'meff': meff,
           'size_skill': float(sk_size[0])}, open(os.path.join(CK, 'c1.json'), 'w'))
open(os.path.join(CK, 'c1.out'), 'w').write('\n'.join(out_lines) + '\n')
