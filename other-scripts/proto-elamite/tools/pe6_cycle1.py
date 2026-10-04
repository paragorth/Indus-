"""pe6 cycle 1: fit the fake-Susa economy to the real corpus by ABC.
 (1) prior-predictive coverage of each statistic;
 (2) rejection + local-linear ABC on the real corpus -> posterior of the economy;
 (3) CONTROL: parameter recovery on held-out bank simulations (which economy parameters are identifiable);
 (4) posterior predictive check on all statistics (new simulations at posterior draws);
 (5) CONTROL: same fit on a globally sign-shuffled real corpus (distance and parameter shifts).
Output: data/pe6_cycle1.json
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe6_common import load_real, stats, STATS, PNAMES, PRIOR, abc_fit, shuffle_global, from_unbounded, PEDATA
from pe6_bank import load_bank
from pe6_pipe import fit, summarize_post, post_sims


def main():
    bank = load_bank(); seeds, U, RAW, S = bank
    ok = np.isfinite(S).all(1)
    U, RAW, S = U[ok], RAW[ok], S[ok]
    bank = (seeds[ok], U, RAW, S)
    print('bank', len(S))
    R = load_real(); so = stats(R)
    out = {'n_bank': int(len(S))}
    # (1) prior predictive coverage
    cov = {}
    for k, name in enumerate(STATS):
        q = float((S[:, k] < so[k]).mean() + 0.5 * (S[:, k] == so[k]).mean())
        cov[name] = (float(so[k]), q, float(np.percentile(S[:, k], 1)), float(np.percentile(S[:, k], 99)))
    out['prior_cov'] = cov
    # (2) fit
    for frac in (0.005, 0.01):
        adj, idx, d = fit(so, bank, frac=frac)
        out[f'post_{frac}'] = summarize_post(adj)
        out[f'dist_{frac}'] = (float(d.min()), float(np.median(d)))
    adj, idx, d = fit(so, bank, frac=0.01)
    np.save(os.path.join(PEDATA, 'pe6_post_real.npy'), adj)
    # (3) recovery control: 200 held-out bank sims
    rng = np.random.default_rng(5)
    test = rng.choice(len(S), 200, replace=False)
    mask = np.ones(len(S), bool); mask[test] = False
    est = []; covr = []
    for i in test:
        a, _, _, _ = abc_fit(U[mask], S[mask], S[i], frac=0.01)
        med = np.median(a, 0); lo = np.percentile(a, 5, 0); hi = np.percentile(a, 95, 0)
        est.append(med); covr.append((U[i] >= lo) & (U[i] <= hi))
    est = np.array(est); covr = np.array(covr)
    rec = {}
    for j, p in enumerate(PNAMES):
        r = float(np.corrcoef(est[:, j], U[test, j])[0, 1])
        rec[p] = (r, float(covr[:, j].mean()))
    out['recovery'] = rec
    # (4) posterior predictive check
    res = post_sims(adj, 200, seed=11, want_stats=True)
    PS = np.array([r[2] for r in res])
    ppc = {}
    for k, name in enumerate(STATS):
        q = float((PS[:, k] < so[k]).mean() + 0.5 * (PS[:, k] == so[k]).mean())
        ppc[name] = (float(so[k]), float(np.median(PS[:, k])), float(np.percentile(PS[:, k], 5)),
                     float(np.percentile(PS[:, k], 95)), q)
    out['ppc'] = ppc
    # (5) shuffled corpus fit
    for nm, Cx in (('shuf_global', shuffle_global(R, 3)),):
        sx = stats(Cx)
        a2, i2, d2 = fit(sx, bank, frac=0.01)
        out[nm] = dict(dist=(float(d2.min()), float(np.median(d2))), post=summarize_post(a2),
                       stats={n: float(v) for n, v in zip(STATS, sx)})
    json.dump(out, open(os.path.join(PEDATA, 'pe6_cycle1.json'), 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('n_bank', 'dist_0.005', 'dist_0.01')}, indent=1))


if __name__ == '__main__':
    main()
