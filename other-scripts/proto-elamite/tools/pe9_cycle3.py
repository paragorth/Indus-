"""pe9 cycle 3: which signs behave like depleted tokens vs inexhaustible ones, and the
pre-registered test on the 52 RESERVED tablets (never used in cycles 1-2).

(a) Per sign (fit set, signs on >= 6 tablets): X = number of within-tablet string
    pairs that both contain the sign.  Null = middles shuffled across tablets
    (1,000x; each sign's frequency and every string kept).  z <= -2: 'depleted'
    (avoids recurring on a tablet); z >= +2: 'inexhaustible'.  False-positive control:
    the same classification run with 20 shuffled corpora as the 'observed' data.
(b) Reserved tablets (pre-registered in loops/pe9_cycle2.txt):
    P1 use profile on reserved tablets (base bigram fitted on the fit set): f1 > 1 and
       f2 > 1; killed if f2 < 1 (depletion at c <= 2).
    P2 held-out bits: POLYA (theta and base fitted on the fit set) beats LM on reserved
       tablets by more than on reserved middles shuffled across reserved tablets (100x).
    P3 signs classed 'inexhaustible' on the fit set have a higher reserved within-tablet
       pair excess (obs - shuffle mean, per tablet-occurrence) than unclassed signs.
    P4 reserved p3_given_2 and repeat_rate lie inside the 5-95% posterior-predictive band of
       the ABC-accepted parameters (fit set) simulated on the reserved structure, and
       outside the band of the BAG_WOR c <= 2 accepted parameters.
"""
import json, os, sys
import numpy as np
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe9_common import *  # noqa
from pe9_cycle1 import pe_fit_tablets, shuffle_across  # noqa
from pe9_cycle1b import fit as fit_profile  # noqa
from pe9_cycle2 import load_table  # noqa
from pe9_cycle2_abc import abc  # noqa


def pair_counts(tabs):
    X = Counter()
    occ = Counter()
    for t in tabs:
        cnt = Counter(s for m in t for s in set(m))
        for s, n in cnt.items():
            X[s] += n * (n - 1) // 2
            occ[s] += 1
    return X, occ


def sign_z(tabs, nsh=1000, seed=0):
    X, occ = pair_counts(tabs)
    signs = [s for s in occ if occ[s] >= 6]
    sims = np.zeros((nsh, len(signs)))
    for r in range(nsh):
        Xs, _ = pair_counts(shuffle_across(tabs, seed * 100000 + r))
        sims[r] = [Xs[s] for s in signs]
    mu, sd = sims.mean(0), sims.std(0) + 1e-9
    return {s: {'obs': X[s], 'exp': float(mu[i]), 'z': float((X[s] - mu[i]) / sd[i]), 'tablets': occ[s]}
            for i, s in enumerate(signs)}


if __name__ == '__main__':
    C = load_corpora()
    fit, hold, _, _ = pe_fit_tablets(C)
    res = {}
    # (a) per-sign classes
    Z = sign_z(fit)
    dep = sorted([s for s in Z if Z[s]['z'] <= -2], key=lambda s: Z[s]['z'])
    inex = sorted([s for s in Z if Z[s]['z'] >= 2], key=lambda s: -Z[s]['z'])
    fp = []
    for k in range(20):
        Zs = sign_z(shuffle_across(fit, 777 + k), nsh=200, seed=k + 1)
        fp.append((sum(v['z'] <= -2 for v in Zs.values()), sum(v['z'] >= 2 for v in Zs.values())))
    res['signs'] = {'n_tested': len(Z), 'depleted': [(s, round(Z[s]['z'], 2), Z[s]['obs'], round(Z[s]['exp'], 1), Z[s]['tablets']) for s in dep],
                    'inexhaustible': [(s, round(Z[s]['z'], 2), Z[s]['obs'], round(Z[s]['exp'], 1), Z[s]['tablets']) for s in inex],
                    'shuffle_false_pos_mean': [float(np.mean([a for a, b in fp])), float(np.mean([b for a, b in fp]))],
                    'shuffle_false_pos_max': [int(max(a for a, b in fp)), int(max(b for a, b in fp))]}
    print('signs', json.dumps(res['signs'])[:3000], flush=True)
    # (b) reserved tablets
    allt = fit + hold
    E = Enc(allt)
    nf = len(fit)
    tr = np.arange(nf, dtype=np.int64)
    te = np.arange(nf, E.T, dtype=np.int64)
    # P1 profile on reserved, base from fit set
    w, B = fit_base(E.tok, E.ss, E.ts, tr, E.V, 0.5)
    from scipy.optimize import minimize
    sess = sess_flags(E.T, 1)
    r = minimize(lambda x: -score_profile(E.tok, E.ss, E.ts, te, sess, E.V, w, B, 1, *np.exp(x)).sum(),
                 np.zeros(4), method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 3000})
    f = np.exp(r.x).tolist()
    rng = np.random.RandomState(3)
    bo = []
    for b in range(200):
        idx = rng.randint(len(hold), size=len(hold))
        Eb = Enc(fit + [hold[i] for i in idx], vocab=E.vocab)
        teb = np.arange(nf, Eb.T, dtype=np.int64)
        rb = minimize(lambda x: -score_profile(Eb.tok, Eb.ss, Eb.ts, teb, sess_flags(Eb.T, 1), Eb.V, w, B, 1, *np.exp(x)).sum(),
                      r.x, method='Nelder-Mead', options={'xatol': 1e-2, 'fatol': 1e-2})
        bo.append(np.exp(rb.x))
    bo = np.array(bo)
    res['P1'] = {'f': f, 'q05': np.percentile(bo, 5, 0).tolist(), 'q95': np.percentile(bo, 95, 0).tolist(),
                 'P_f2_lt_1': float(np.mean(bo[:, 1] < 1)), 'P_f1_lt_1': float(np.mean(bo[:, 0] < 1)),
                 'pass': bool(f[0] > 1 and f[1] > 1)}
    print('P1', res['P1'], flush=True)
    # P2 POLYA vs LM on reserved, theta fitted on fit set
    best = None
    for th in [1, 3, 10, 30, 100, 300, 1000, 3000]:
        ll = score(E.tok, E.ss, E.ts, tr, sess, E.V, w, B, 1, 1, 0., 0., 0., float(th)).sum()
        if best is None or ll > best[0]:
            best = (ll, th)
    th = best[1]

    def gain(Ex):
        lm = score(Ex.tok, Ex.ss, Ex.ts, np.arange(nf, Ex.T, dtype=np.int64), sess_flags(Ex.T, 1), Ex.V, w, B, 1, 0, 0., 0., 0., 1.).sum()
        po = score(Ex.tok, Ex.ss, Ex.ts, np.arange(nf, Ex.T, dtype=np.int64), sess_flags(Ex.T, 1), Ex.V, w, B, 1, 1, 0., 0., 0., float(th)).sum()
        n = Ex.ss[-1] - Ex.ss[Ex.ts[nf]]
        return float((po - lm) / LN2 / n)
    g_real = gain(E)
    g_sh = [gain(Enc(fit + shuffle_across(hold, 500 + k), vocab=E.vocab)) for k in range(100)]
    res['P2'] = {'theta': th, 'gain_bits_per_sign': g_real, 'shuffled_mean': float(np.mean(g_sh)),
                 'shuffled_max': float(np.max(g_sh)), 'p': float(np.mean(np.array(g_sh) >= g_real)),
                 'pass': bool(g_real > np.max(g_sh))}
    print('P2', res['P2'], flush=True)
    # P3 sign classes on reserved
    Zh = sign_z(hold, nsh=1000, seed=9)
    Xh, occh = pair_counts(hold)
    exc = {}
    for s in Z:
        if occh[s] >= 2:
            # pair excess per reserved tablet-occurrence, using a shuffle expectation
            exc[s] = (Xh[s] - Zh[s]['exp']) / occh[s] if s in Zh else None
    inx = [exc[s] for s in inex if exc.get(s) is not None]
    oth = [exc[s] for s in Z if s not in inex and s not in dep and exc.get(s) is not None]
    dp = [exc[s] for s in dep if exc.get(s) is not None]
    # permutation p: label shuffle among signs with a value
    vals = np.array(inx + oth)
    lab = np.array([1] * len(inx) + [0] * len(oth))
    obs_d = vals[lab == 1].mean() - vals[lab == 0].mean() if len(inx) and len(oth) else float('nan')
    pr = np.random.RandomState(1)
    perm = [vals[l == 1].mean() - vals[l == 0].mean() for l in (pr.permutation(lab) for _ in range(5000))]
    res['P3'] = {'n_inex_scored': len(inx), 'n_other_scored': len(oth), 'n_dep_scored': len(dp),
                 'mean_excess_inex': float(np.mean(inx)) if inx else None,
                 'mean_excess_other': float(np.mean(oth)) if oth else None,
                 'mean_excess_dep': float(np.mean(dp)) if dp else None,
                 'diff': float(obs_d), 'p_perm': float(np.mean(np.array(perm) >= obs_d)),
                 'pass': bool(obs_d > 0 and np.mean(np.array(perm) >= obs_d) < 0.05)}
    print('P3', res['P3'], flush=True)
    # P4 posterior predictive on reserved structure
    P, S = load_table('PE')
    ok = np.isfinite(S).all(1)
    P, S = P[ok], S[ok]
    scale = np.median(np.abs(S - np.median(S, 0)), 0) * 1.4826 + 1e-9
    Ef = Enc(fit)
    obs_fit = stats(Ef.tok, Ef.ss, Ef.ts, Ef.V, 15, 0)
    _, idx = abc(P, S, obs_fit, scale)
    Eh = Enc(hold)
    obs_h = stats(Eh.tok, Eh.ss, Eh.ts, Eh.V, 15, 0)
    # base for the reserved simulations: fit-set frequencies restricted to signs, unigram+bigram
    wv = np.array([w[E.idx[s]] for s in Eh.vocab])
    # simulate on reserved structure with fit-set base mapped onto the full vocabulary
    hold_full = Enc(hold, vocab=E.vocab)

    def band(rows, nsim=400):
        rs = np.random.RandomState(11)
        out = []
        CB0 = np.cumsum(B, 1)
        for r_ in rs.choice(len(rows), min(nsim, len(rows) * 4), replace=True):
            m, base, g, K, c, eps, thh, Ss = rows[r_]
            ww = w ** g
            ww /= ww.sum()
            if base == 1:
                BB = B ** g
                BB /= BB.sum(1, keepdims=True)
                CB = np.cumsum(BB, 1)
            else:
                CB = CB0
            tk = simulate_fast(hold_full.ss, hold_full.ts, sess_flags(hold_full.T, int(Ss)), E.V, np.cumsum(ww), CB,
                               int(base), int(m), K, c, eps, thh, int(rs.randint(2**31 - 1)))
            out.append(stats(tk, hold_full.ss, hold_full.ts, E.V, 15, 0))
        out = np.array(out)
        return np.percentile(out, 5, 0), np.percentile(out, 95, 0)
    obs_h = stats(hold_full.tok, hold_full.ss, hold_full.ts, E.V, 15, 0)
    lo, hi = band(P[idx])
    bag = P[(P[:, 0] == 3) & (P[:, 4] <= 2) & (P[:, 5] < 0.5)]
    # bag band: the c <= 2 bag sims closest to the fit-set observation
    _, bidx = abc(bag, S[(P[:, 0] == 3) & (P[:, 4] <= 2) & (P[:, 5] < 0.5)], obs_fit, scale)
    blo, bhi = band(bag[bidx])
    keys = [STAT_NAMES.index('p3_given_2'), STAT_NAMES.index('repeat_rate')]
    res['P4'] = {STAT_NAMES[k]: {'reserved': float(obs_h[k]), 'abc_band': [float(lo[k]), float(hi[k])],
                                 'bag_c_le2_band': [float(blo[k]), float(bhi[k])]} for k in keys}
    res['P4']['pass'] = bool(all(lo[k] <= obs_h[k] <= hi[k] and not (blo[k] <= obs_h[k] <= bhi[k]) for k in keys))
    res['P4']['all_stats_reserved'] = dict(zip(STAT_NAMES, np.round(obs_h, 4).tolist()))
    res['P4']['abc_band_all'] = {STAT_NAMES[k]: [round(float(lo[k]), 4), round(float(hi[k]), 4)] for k in range(len(lo))}
    print('P4', json.dumps(res['P4']), flush=True)
    json.dump(res, open(os.path.join(DATA, 'pe9_cycle3.json'), 'w'), indent=1)
