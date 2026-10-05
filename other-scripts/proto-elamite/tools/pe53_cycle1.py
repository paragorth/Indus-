"""pe53 cycle 1: calibration + first PE run.
 1a Drehem with KNOWN months: does the lamb/(lamb+ewe) ratio follow a birth pulse, and
    does its peak fall where biology puts lambing (Dec-Jan)?  Permutation null on months.
 1b Drehem, terms made opaque, PE size (45 multi-term tablets) and 300 tablets: rank of
    the true Y/F assignment among all 6,050; month order from the true assignment.
 1c Planted seasonal herd archives with PE's record structure (truth recovery), and
    non-seasonal plants (null distribution of the best score).
 1d PE: all 6,050 assignments of the 8 herd signs; best scores against non-seasonal
    plants and against records with each sign's counts shuffled across records.
"""
import sys, time
from multiprocessing import Pool
from pe53_common import *

DR8 = DR_TERMS[:8]
TR8 = DR_TRUTH[:8]
A8 = all_assignments(8)
TRUE_IDX = int(np.where((A8 == TR8).all(1))[0][0])


def rank_of(S, i):
    v = S[:, 0]
    return float(np.mean(v[np.isfinite(v)] > v[i]))


def job(args):
    kind, seed = args
    rng = np.random.default_rng(seed)
    if kind.startswith('dr'):
        X, mo, _ = drehem_matrix()
        X = X[:, :8]
        multi = np.where((X > 0).sum(1) >= 2)[0]
        size = 45 if kind == 'dr45' else 300
        pick = rng.choice(multi, size, replace=False)
        M = X[pick]
        S = score_all(M, A8)
        r1 = rank_of(S, TRUE_IDX)
        r2 = float(np.mean(S[:, 1][np.isfinite(S[:, 1])] > S[TRUE_IDX, 1]))
        top = np.nanargmax(S[:, 0])
        return dict(kind=kind, seed=seed, rank_S1=r1, rank_S2=r2, top=A8[top].tolist(),
                    S1_true=S[TRUE_IDX, 0], S1_max=float(np.nanmax(S[:, 0])),
                    yhit=float(np.mean(TR8[A8[top] == 1] == 1)) if (A8[top] == 1).any() else 0.0)
    R, Mpe = pe_matrix()
    if kind in ('plant', 'flat'):
        # truth: two Y signs, two F signs, rest O
        perm = rng.permutation(8)
        truth = np.zeros(8, np.int8)
        truth[perm[:2]] = 1
        truth[perm[2:4]] = 2
        th = THETAS[rng.integers(len(THETAS))]
        pc = curve(*th)
        k = KAPPAS[rng.integers(1, 4)]
        M = np.zeros_like(Mpe)
        p0 = np.clip(rng.choice(pc), 0.01, 0.99)      # flat herds: one timeless share
        for r in range(len(Mpe)):
            tot = Mpe[r].sum()
            nf = max(1, int(round(tot * rng.uniform(0.3, 0.7))))
            p = np.clip(pc[rng.integers(len(TGRID))], 0.01, 0.99) if kind == 'plant' else p0
            q = rng.beta(p * k, (1 - p) * k)
            # young per female ~ q/(1-q)
            ny = rng.binomial(int(nf / max(1 - q, 0.05)), q)
            no = int(rng.integers(0, max(2, tot // 2)))
            for cls, cnt in ((1, ny), (2, nf), (0, no)):
                idx = np.where(truth == cls)[0]
                if len(idx) and cnt:
                    M[r, idx] += rng.multinomial(cnt, rng.dirichlet(np.ones(len(idx))))
        S = score_all(M, A8)
        ti = int(np.where((A8 == truth).all(1))[0][0])
        top = int(np.nanargmax(S[:, 0]))
        return dict(kind=kind, seed=seed, rank_S1=rank_of(S, ti),
                    rank_S2=float(np.mean(S[:, 1] > S[ti, 1])),
                    S1_max=float(np.nanmax(S[:, 0])), S2_max=float(np.nanmax(S[:, 1])),
                    exact=bool((A8[top] == truth).all()),
                    yhit=float(np.mean(truth[A8[top] == 1] == 1)) if (A8[top] == 1).any() else 0.0)
    if kind == 'shuf':
        M = Mpe.copy()
        for j in range(8):
            M[:, j] = rng.permutation(M[:, j])
        S = score_all(M, A8)
        return dict(kind=kind, seed=seed, S1_max=float(np.nanmax(S[:, 0])), S2_max=float(np.nanmax(S[:, 1])))
    if kind == 'pe':
        S = score_all(Mpe, A8)
        np.save(os.path.join(CK, 'c1_pe_scores.npy'), S)
        return dict(kind='pe', S1_max=float(np.nanmax(S[:, 0])), S2_max=float(np.nanmax(S[:, 1])))


def drehem_known():
    X, mo, _ = drehem_matrix()
    y, n = yn_from(X[:, :8], TR8)
    k = n > 0
    y, n, mo = y[k], n[k], mo[k]

    def amp(m):
        r = np.array([y[m == i].sum() / max(n[m == i].sum(), 1) for i in range(1, 13)])
        return r.max() - r.min(), int(np.argmax(r)) + 1, r
    a, pk, r = amp(mo)
    rng = np.random.default_rng(5)
    null = [amp(rng.permutation(mo))[0] for _ in range(1000)]
    # tablet-level (mean of per-tablet ratios) as well
    rt = np.array([np.mean(y[mo == i] / n[mo == i]) for i in range(1, 13)])
    nullt = []
    for _ in range(1000):
        pm = rng.permutation(mo)
        q = np.array([np.mean(y[pm == i] / n[pm == i]) for i in range(1, 13)])
        nullt.append(q.max() - q.min())
    # month variance share of per-tablet ratio
    rr = y / n
    vs = np.var([rr[mo == i].mean() for i in range(1, 13)]) / np.var(rr)
    return dict(n=int(k.sum()), pooled_ratio=r.round(3).tolist(), peak_month=pk, amp=a,
                p_amp=float(np.mean(np.array(null) >= a)), tablet_ratio=rt.round(3).tolist(),
                p_amp_tab=float(np.mean(np.array(nullt) >= rt.max() - rt.min())),
                month_var_share=float(vs))


if __name__ == '__main__':
    t0 = time.time()
    out = {'known': drehem_known()}
    print(out['known'], flush=True)
    jobs = [('pe', 0)] + [('dr45', s) for s in range(12)] + [('dr300', 100 + s) for s in range(4)] + \
           [('plant', 200 + s) for s in range(12)] + [('flat', 300 + s) for s in range(12)] + \
           [('shuf', 400 + s) for s in range(12)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(job, jobs):
            res.append(r)
            print(round(time.time() - t0), r, flush=True)
            dump(dict(out, runs=res), os.path.join(CK, 'c1.json'))
    out['runs'] = res
    dump(out, os.path.join(DATA, 'pe53_cycle1.json'))
