"""pe53 cycle 3e: same-tablet clock with a TYPE-MATCHED null.
Records are regrouped into pseudo-tablets only within their record type (stratum = coarse:
'has M362 count' x 'has any ~a/@g count'; fine: exact pattern of non-zero herd signs), so tablet-level clumping of record types cannot pass the test.
(Ur III Drehem cannot calibrate this: receipts are single records. Planted shared-time
archives are in cycle 3, job sametab_plant.)
"""
from pe53_common import *
from pe53_cycle1 import A8


def dll_fn(L, g):
    tot = 0.0
    for u in np.unique(g):
        Lu = L[g == u]
        if len(Lu) < 2:
            continue
        j = Lu.sum(0)
        tot += (np.log(np.mean(np.exp(j - j.max()))) + j.max()) - \
            sum(np.log(np.mean(np.exp(l - l.max()))) + l.max() for l in Lu)
    return tot


def strat_test(M, groups, strata, a, rng, nperm=1000):
    y, n = yn_from(M, a)
    r = scores(y, n, True)
    keep = r['keep']
    th = THETAS.index(tuple(r['theta']))
    pc = np.clip(C_TH[th], PGRID[0], PGRID[-1])
    L = bb_logpmf(y[keep], n[keep], pc, r['kappa'])
    g = np.asarray(groups)[keep]
    st = np.asarray(strata)[keep]
    obs = dll_fn(L, g)
    null = []
    for _ in range(nperm):
        g2 = g.copy()
        for s in np.unique(st):
            idx = np.where(st == s)[0]
            g2[idx] = rng.permutation(g[idx])
        null.append(dll_fn(L, g2))
    null = np.array(null)
    return dict(dLL=float(obs), null_mean=float(null.mean()), p_hi=float((np.sum(null >= obs) + 1) / (nperm + 1)))


def strata_of(M, fine=False):
    if fine:   # exact pattern of which herd signs carry a non-zero count
        return [tuple(int(v > 0) for v in m) for m in M]
    return [(int(m[0] > 0), int((m[4:] > 0).any())) for m in M]


if __name__ == '__main__':
    rng = np.random.default_rng(31)
    R, M = pe_matrix()
    groups = np.array([r[0] for r in R])
    st = [str(s) for s in strata_of(M)]
    S = np.load(os.path.join(CK, 'c1_pe_scores.npy'))[:, 0]
    top = np.argsort(-np.nan_to_num(S, nan=-1e9))[:5]
    out = {'pe': [dict(a=''.join('OYF'[c] for c in A8[i]), **strat_test(M, groups, st, A8[i], rng)) for i in top]}
    st2 = [str(s) for s in strata_of(M, fine=True)]
    out['pe_fine'] = [dict(a=''.join('OYF'[c] for c in A8[i]), **strat_test(M, groups, st2, A8[i], rng)) for i in top]
    # restricted to records whose young share is not fixed by the pattern (0 < y < n)
    a = A8[top[0]]
    y, n = yn_from(M, a)
    mid = (y > 0) & (y < n)
    out['n_mid'] = int(mid.sum())
    out['mid_tablets'] = sorted(set(groups[mid]))
    out['pe_mid'] = strat_test(M[mid], groups[mid], [str(s) for s in strata_of(M[mid], fine=True)], a, rng)
    out['pe_mid_coarse'] = strat_test(M[mid], groups[mid], ['all'] * int(mid.sum()), a, rng)
    print(out, flush=True)
    dump(out, os.path.join(DATA, 'pe53_cycle3e.json'))
