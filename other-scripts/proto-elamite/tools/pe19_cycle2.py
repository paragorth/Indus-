"""pe19 cycle 2: MASSIVE RANDOM TRAIT-SUBSET SERIATION + PARTIAL CA (blind).

Arm R: 3,000 random trait subsets (25-250 traits), CA axis 1 on each, oriented by
the fixed rule. Internal (blind) quality Q = trait concentration along the order
(mean over traits of 1 - 12 var(pos)) minus the same for a column-shuffled copy of
the subset. Q is computed on tablet half A; survivors (top 1%) re-tested on half B.
Frozen: RAND_TOP (consensus of the 30 best), RAND_BOT (30 worst), RAND_ALL.
Arm P: partial CA, numeral-code (number system ~ commodity) traits and the
format traits partialled out of the sign/variant table; axes 1-3 frozen.
Never reads the hidden labels.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from pe19_common import *


def concentration(pos, X):
    c = X.sum(0)
    ok = c >= 3
    Xk = X[:, ok]; ck = c[ok]
    m = (pos[:, None] * Xk).sum(0) / ck
    v = ((pos[:, None] ** 2) * Xk).sum(0) / ck - m ** 2
    return float(np.mean(1 - 12 * v)) if ok.any() else 0.0


def ca_safe(X):
    ok = X.sum(1) > 0
    s = np.full(len(X), np.nan)
    if ok.sum() > 10 and (X[ok].sum(0) > 0).sum() > 3:
        s[ok] = ca_scores(X[ok])
    return s


def main(nsub=3000):
    t0 = time.time()
    T = load_pe()
    ids, X, names, elen = build_matrix(T, 'BVNF')
    n, m = X.shape
    rng = np.random.default_rng(2024)
    half = rng.uniform(size=n) < 0.5
    recs, ranks = [], []
    for r in range(nsub):
        k = int(rng.integers(25, 251))
        cols = rng.choice(m, k, replace=False)
        Xs = X[:, cols]
        s = ca_safe(Xs)
        ok = ~np.isnan(s)
        if ok.sum() < 200:
            continue
        s_o = orient(s[ok], elen[ok])
        R = np.full(n, np.nan); R[ok] = rankdata(s_o) / ok.sum()
        Xsh = np.array([rng.permutation(c) for c in Xs.T]).T
        ss = ca_safe(Xsh); ok2 = ~np.isnan(ss)
        R2 = np.full(n, np.nan); R2[ok2] = rankdata(ss[ok2]) / ok2.sum()
        qA = concentration(R[ok & half], Xs[ok & half]) - concentration(R2[ok2 & half], Xsh[ok2 & half])
        qB = concentration(R[ok & ~half], Xs[ok & ~half]) - concentration(R2[ok2 & ~half], Xsh[ok2 & ~half])
        recs.append({'r': r, 'k': k, 'qA': qA, 'qB': qB, 'cov': int(ok.sum())})
        ranks.append(R.astype(np.float32))
        if r % 500 == 0:
            print(r, round(time.time() - t0), flush=True)
    ranks = np.array(ranks)
    qA = np.array([d['qA'] for d in recs]); qB = np.array([d['qB'] for d in recs])
    top = np.argsort(-qA)[:30]; bot = np.argsort(qA)[:30]
    out = {'n_sub': len(recs), 'rho_qA_qB': float(spearmanr(qA, qB)[0]),
           'top_qB_mean': float(qB[top].mean()), 'all_qB_mean': float(qB.mean()), 'bot_qB_mean': float(qB[bot].mean())}

    def cons(sel, name):
        Rm = np.nanmean(ranks[sel], axis=0)
        Rm = np.where(np.isnan(Rm), 0.5, Rm)
        return freeze(name, ids, orient(Rm, elen))
    out['hash'] = {'RAND_TOP': cons(top, 'C2_RAND_TOP'), 'RAND_BOT': cons(bot, 'C2_RAND_BOT'),
                   'RAND_ALL': cons(np.arange(len(recs)), 'C2_RAND_ALL')}
    # which traits are over-represented in the top subsets? (blind diagnostics saved)
    print(out, flush=True)

    # Arm P: partial CA (signs/variants | numeral codes + format traits)
    sel = [j for j, nm in enumerate(names) if nm.startswith(('B_', 'V_'))]
    zc = [j for j, nm in enumerate(names) if not nm.startswith(('B_', 'V_'))]
    Xa = X[:, sel]; Z = X[:, zc]
    ok = Xa.sum(1) > 0
    Xa, Z, el = Xa[ok], Z[ok], elen[ok]
    P = Xa / Xa.sum(); rw = P.sum(1); cw = P.sum(0)
    S = (P - np.outer(rw, cw)) / np.sqrt(np.outer(rw, cw))
    Zw = np.hstack([np.ones((len(Z), 1)), Z]) * np.sqrt(rw)[:, None]
    beta, *_ = np.linalg.lstsq(Zw, S, rcond=None)
    Sr = S - Zw @ beta
    U, sv, Vt = np.linalg.svd(Sr, full_matrices=False)
    pid = [ids[i] for i in np.where(ok)[0]]
    out['partial'] = {}
    for ax in range(3):
        sc = orient(U[:, ax] / np.sqrt(rw), el)
        out['partial']['ax%d' % (ax + 1)] = {'hash': freeze('C2_PARTIAL_ax%d' % (ax + 1), pid, sc),
                                              'sv': float(sv[ax])}
    json.dump({'summary': out, 'recs': recs}, open(os.path.join(CK, 'c2_fit.json'), 'w'))
    np.save(os.path.join(CK, 'c2_ranks.npy'), ranks)
    print(out, 'done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
