#!/usr/bin/env python3
"""la29 cycle 2: the land as a decoder. Which signs' site-to-site frequency tracks a land variable
beyond spatially smooth random fields?

Per sign: empirical logit share at each site (sites with >= MIN_TOK sign tokens), inverse-variance
weighted correlation with each land variable. Null per sign: the same statistic against 10^4
Gaussian random fields at the site coordinates (measured correlation lengths); max |r| over M_eff
fields gives a per-sign P that covers the variable search; BH across signs.
Controls: planted signs (beta-binomial, overdispersion matched to real signs) driven by a land
variable; shuffled site labels (whole sign matrix); known logograms (VIN, OLE, GRA, CYP) read as
the positive anchor; the same with Hagia Triada dropped.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la29_common import *

rng = np.random.default_rng(292)
NNULL = int(os.environ.get('NNULL', 10000))
MIN_TOK = int(os.environ.get('MIN_TOK', 30))
MIN_SIGN = int(os.environ.get('MIN_SIGN', 15))
DROP = os.environ.get('DROP', '').split(',') if os.environ.get('DROP') else []
TAG = os.environ.get('TAG', 'c2')
out_lines = []


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out_lines.append(s)


land = land_table()
corr = json.load(open(os.path.join(CK, 'crete_corr.json')))
LS = np.array(sorted({max(3.0, v['efold_km']) for v in corr.values()} | {10.0, 20.0, 40.0}))
sg, nd = la_signs()
SITES_ = [s for s in sg if sum(sg[s].values()) >= MIN_TOK and s not in ('KEA', 'MI') and s not in DROP]
SITES_.sort(key=lambda s: -sum(sg[s].values()))
tot = np.array([sum(sg[s].values()) for s in SITES_], float)
allsig = sorted({g for s in SITES_ for g in sg[s]})
cnt = np.array([[sg[s][g] for s in SITES_] for g in allsig], float)
keep = cnt.sum(1) >= MIN_SIGN
signs = [g for g, k in zip(allsig, keep) if k]; K = cnt[keep]
log('sites', dict(zip(SITES_, tot.astype(int).tolist())), f'signs tested {len(signs)}')

X, keys = land_matrix(SITES_, land)
Rv = np.corrcoef(X.T); ev = np.linalg.eigvalsh(np.nan_to_num(Rv)); meff = int(round(ev.sum() ** 2 / (ev ** 2).sum()))
log(f'{len(keys)} land variables, M_eff {meff}')


def logit_w(K, tot):
    p = (K + 0.5) / (tot[None] + 1.0)
    l = np.log(p / (1 - p))
    w = (tot[None] + 1.0) * p * (1 - p)
    return l, w


def wcorr(l, w, Z):
    """l, w: [G, S]; Z: [F, S] -> r[G, F]"""
    W = w / w.sum(1, keepdims=True)
    lm = (W * l).sum(1, keepdims=True); lc = l - lm
    sl = np.sqrt((W * lc ** 2).sum(1))
    # weighted moments of Z under each sign's weights
    Zm = W @ Z.T                                   # [G, F]
    Z2 = W @ (Z ** 2).T
    sz = np.sqrt(np.clip(Z2 - Zm ** 2, 1e-12, None))
    cov = (W * lc) @ Z.T
    return cov / (sl[:, None] * sz)


NULLMODE = os.environ.get('NULLMODE', 'rank')


def null_bank(n):
    """rank: per variable an independent GRF draw whose site ranks carry the variable's own values
    (spatially constrained permutation; keeps skewed marginals); max over all variables.
    gauss: Gaussian fields, max over M_eff."""
    if NULLMODE == 'gauss':
        return grf(SITES_, LS, n, rng), 'gauss'
    V = Xs.shape[1]
    G = grf(SITES_, LS, n * V, rng).reshape(n, V, -1)
    rk = G.argsort(-1).argsort(-1)
    srt = np.sort(Xs, 0)                            # [S, V]
    Z = np.take_along_axis(np.broadcast_to(srt.T[None], G.shape), rk, -1)
    return Z.reshape(n * V, -1), 'rank'


def null_max(l, w, bank, n):
    Fz, mode = bank
    r = np.abs(wcorr(l, w, Fz))
    if mode == 'rank':
        return r.reshape(r.shape[0], n, -1).max(-1)            # [G, n]
    idx = np.array([rng.choice(r.shape[1], meff, replace=False) for _ in range(2000)])
    return np.stack([r[:, i].max(1) for i in idx], 1)


L, Wt = logit_w(K, tot)
Xs = (X - X.mean(0)) / X.std(0)
r_real = wcorr(L, Wt, Xs.T)                       # [G, V]
NB = int(os.environ.get('NBANK', 2000))
BANK = null_bank(NB)
BANKS = null_bank(400)
log(f'null mode {NULLMODE}, draws {NB}')


def sign_p(r_real, nm):
    best = np.abs(r_real).max(1)
    P = (1 + (nm >= best[:, None]).sum(1)) / (1 + nm.shape[1])
    return P, best


def bh(P, q=0.1):
    o = np.argsort(P); m = len(P)
    th = q * np.arange(1, m + 1) / m
    ok = P[o] <= th
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    sel = np.zeros(m, bool); sel[o[:k]] = True
    return sel


P, best = sign_p(r_real, null_max(L, Wt, BANK, NB))
sel = bh(P)
log(f'signs with P<0.05 (variable search covered): {(P < 0.05).sum()} of {len(signs)} (expected {0.05 * len(signs):.1f}); BH q0.1 survivors {sel.sum()}')
o = np.argsort(P)
for g in o[:15]:
    j = int(np.argmax(np.abs(r_real[g])))
    sh = (K[g] / tot * 1000).round(1)
    log(f'  {signs[g]:10s} n {int(K[g].sum()):4d}  best {keys[j]:14s} r {r_real[g, j]:+.2f}  P {P[g]:.4f}  {"BH" if sel[g] else ""}  per-mille ' +
        ' '.join(f'{s}:{v}' for s, v in zip(SITES_, sh)))
for lg in ('L:VIN', 'L:OLE', 'L:GRA', 'L:CYP', 'L:OLIV', 'L:VIR', 'L:LIV', 'L:S301', 'L:S304', 'L:S401'):
    if lg in signs:
        g = signs.index(lg); j = int(np.argmax(np.abs(r_real[g])))
        log(f'  anchor {lg:7s} best {keys[j]:14s} r {r_real[g, j]:+.2f} P {P[g]:.3f}')

# shuffled site labels
fp = []
for k in range(int(os.environ.get('NSHUF', 50))):
    p = rng.permutation(len(SITES_))
    Lp, Wp = L[:, p], Wt[:, p]
    rr = wcorr(Lp, Wp, Xs.T)
    Pp, _ = sign_p(rr, null_max(Lp, Wp, BANKS, 400))
    fp.append(((Pp < 0.05).sum(), bh(Pp).sum()))
fp = np.array(fp)
log(f'shuffled site labels: signs P<0.05 mean {fp[:, 0].mean():.1f} (95% {np.quantile(fp[:, 0], .95):.0f}); BH survivors mean {fp[:, 1].mean():.2f}, runs with >=1: {(fp[:, 1] > 0).mean():.2f}')

# planted signs, beta-binomial with matched dispersion
pr = K / tot[None]
pbar = K.sum(1) / tot.sum()
disp = np.median(((pr - pbar[:, None]) ** 2 / (pbar[:, None] * (1 - pbar[:, None]) / tot[None])).mean(1))
rho = max(1e-4, (disp - 1) / (np.mean(tot) - 1))     # beta-binomial intra-class correlation
log(f'real sign overdispersion factor (median) {disp:.2f} -> beta-binomial rho {rho:.4f}')
for v in [k for k in ('olive_5', 'upland_5', 'rain_10', 'coast_km', 'cl_pasture_5') if k in keys]:
    z = Xs[:, keys.index(v)]
    for base in (0.005, 0.02):
        for beta in (0.5, 1.0):
            hit = 0; right = 0; NP = 30
            Kp = np.zeros((NP, len(SITES_)))
            for i in range(NP):
                p = base * np.exp(beta * z); p = np.clip(p, 1e-5, 0.5)
                a = p * (1 / rho - 1); b = (1 - p) * (1 / rho - 1)
                Kp[i] = rng.binomial(tot.astype(int), rng.beta(a, b))
            Lp, Wp = logit_w(Kp, tot)
            rr = wcorr(Lp, Wp, Xs.T)
            Pp, _ = sign_p(rr, null_max(Lp, Wp, BANKS, 400))
            jj = np.abs(rr).argmax(1)
            right = sum(abs(np.corrcoef(Xs[:, j], z)[0, 1]) > 0.8 for j in jj)
            log(f'   planted {v:12s} base {base} beta {beta}: P<0.05 {int((Pp < 0.05).sum())}/{NP}, right variable {right}/{NP}')

json.dump({'sites': SITES_, 'signs': signs, 'P': P.tolist(), 'best': best.tolist(), 'keys': keys,
           'bestvar': [keys[int(np.argmax(np.abs(r_real[g])))] for g in range(len(signs))],
           'r': [float(r_real[g, int(np.argmax(np.abs(r_real[g])))]) for g in range(len(signs))]},
          open(os.path.join(CK, f'{TAG}.json'), 'w'))
open(os.path.join(CK, f'{TAG}.out'), 'w').write('\n'.join(out_lines) + '\n')
