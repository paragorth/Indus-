"""pe53 cycle 3: does the best PE assignment survive held-out tablets and resampling?
 3a Held-out halves: tablets split at random (20 splits); all 6,050 assignments scored on
    half A; the top-10 on A are re-ranked on half B (percentile among all assignments on B).
    Same for planted seasonal archives (should transfer), flat plants and Drehem PE-size
    samples with known truth.
 3b Written-only: a record counts only if at least one Y sign and one F sign are written
    (absent lines are missing, not zero).  Best assignment and its rank vs the main run.
 3c Bootstrap over tablets (40 resamples): how often is each sign the Y / F class at the top?
"""
import time
from multiprocessing import Pool
from pe53_common import *
from pe53_cycle1 import A8, TR8


def split_transfer(M, groups, rng, nsplit=10):
    ug = np.unique(groups)
    out = []
    for _ in range(nsplit):
        a = set(rng.choice(ug, len(ug) // 2, replace=False))
        ia = np.array([g in a for g in groups])
        SA = score_all(M[ia], A8)[:, 0]
        SB = score_all(M[~ia], A8)[:, 0]
        SA = np.nan_to_num(SA, nan=-1e9)
        SB = np.nan_to_num(SB, nan=-1e9)
        top = np.argsort(-SA)[:10]
        pct = [float(np.mean(SB > SB[i])) for i in top]   # 0 = best on B
        out.append(float(np.mean(pct)))
    return out


def gen_plant(rng, Mpe, flat=False):
    perm = rng.permutation(8)
    truth = np.zeros(8, np.int8)
    truth[perm[:2]] = 1
    truth[perm[2:4]] = 2
    th = THETAS[rng.integers(len(THETAS))]
    pc = curve(*th)
    k = KAPPAS[rng.integers(1, 4)]
    p0 = np.clip(rng.choice(pc), 0.01, 0.99)
    M = np.zeros_like(Mpe)
    for r in range(len(Mpe)):
        tot = Mpe[r].sum()
        nf = max(1, int(round(tot * rng.uniform(0.3, 0.7))))
        p = p0 if flat else np.clip(pc[rng.integers(len(TGRID))], 0.01, 0.99)
        q = rng.beta(p * k, (1 - p) * k)
        ny = rng.binomial(int(nf / max(1 - q, 0.05)), q)
        no = int(rng.integers(0, max(2, tot // 2)))
        for cls, cnt in ((1, ny), (2, nf), (0, no)):
            idx = np.where(truth == cls)[0]
            if len(idx) and cnt:
                M[r, idx] += rng.multinomial(cnt, rng.dirichlet(np.ones(len(idx))))
    return M, truth


def job(args):
    kind, seed = args
    rng = np.random.default_rng(seed)
    R, Mpe = pe_matrix()
    groups = np.array([r[0] for r in R])
    if kind == 'pe':
        return dict(kind=kind, seed=seed, transfer=split_transfer(Mpe, groups, rng, 10))
    if kind in ('plant', 'flat'):
        M, truth = gen_plant(rng, Mpe, flat=(kind == 'flat'))
        return dict(kind=kind, seed=seed, transfer=split_transfer(M, groups, rng, 3))
    if kind == 'dr':
        X, mo, ids = drehem_matrix()
        X = X[:, :8]
        multi = np.where((X > 0).sum(1) >= 2)[0]
        pick = rng.choice(multi, 45, replace=False)
        # 45 tablets in 23 pseudo-tablet groups (PE has 45 records on 23 tablets)
        g = rng.integers(0, 23, 45)
        return dict(kind=kind, seed=seed, transfer=split_transfer(X[pick], g, rng, 3))
    if kind == 'boot':
        ug = np.unique(groups)
        res = []
        for b in range(8):
            pick = rng.choice(ug, len(ug), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in pick])
            S = score_all(Mpe[idx], A8)[:, 0]
            res.append(''.join('OYF'[c] for c in A8[int(np.nanargmax(S))]))
        return dict(kind=kind, seed=seed, tops=res)
    if kind == 'written':
        _, Mw = pe_matrix(none_as_zero=False)
        S = np.full((len(A8), 2), np.nan)
        for i, a in enumerate(A8):
            okY = np.isfinite(Mw[:, a == 1]).any(1)
            okF = np.isfinite(Mw[:, a == 2]).any(1)
            ok = okY & okF
            if ok.sum() < 5:
                continue
            Z = np.nan_to_num(Mw[ok])
            y, n = yn_from(Z, a)
            r = scores(y, n)
            if r:
                S[i] = (r['S1'], r['S2'])
        np.save(os.path.join(CK, 'c3_written_scores.npy'), S)
        return dict(kind=kind, best=''.join('OYF'[c] for c in A8[int(np.nanargmax(S[:, 0]))]),
                    S1_max=float(np.nanmax(S[:, 0])), n_scored=int(np.isfinite(S[:, 0]).sum()))


if __name__ == '__main__':
    t0 = time.time()
    jobs = [('pe', 1), ('written', 2)] + [('boot', 10 + s) for s in range(3)] + \
           [('plant', 100 + s) for s in range(4)] + [('flat', 200 + s) for s in range(4)] + \
           [('dr', 300 + s) for s in range(4)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(job, jobs):
            res.append(r)
            print(round(time.time() - t0), r, flush=True)
            dump(res, os.path.join(CK, 'c3.json'))
    dump(res, os.path.join(DATA, 'pe53_cycle3.json'))
