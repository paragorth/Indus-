"""pe54 cycle 3: INVERT -- link by the strings, ask the flocks.
Pairs of sightings on different tablets that share an owner string are taken as candidate flock histories.
Statistic: mean symmetric pair score (ensemble of the top 20 of a fresh random-hypothesis search) over the
string-sharing pairs, against 20,000 permutations of strings across sightings (same-tablet pairs excluded);
same with a size-only score. Calibration on Ur III (owner = herdsman) and on planted archives at PE size.
For the PE string pairs: which direction and gap the top hypotheses prefer, and which classes grow.
usage: pe54_cycle3.py pe|ur3|plant zc|zm"""
import sys
import numpy as np
from pe54_lib import *
import pe54_lib as L

part, mode = sys.argv[1], sys.argv[2]
L.ZERO_MISSING = mode == 'zm'
rng = np.random.default_rng(5403)


def perm_test(S, strings, tabs, n_perm=20000):
    Ssym = np.maximum(S, S.T)
    n = len(strings)
    ok = tabs[:, None] != tabs[None, :]
    iu = np.triu_indices(n, 1)
    okp = ok[iu]
    lab = np.unique(strings, return_inverse=True)[1]

    def stat(lb):
        same = (lb[:, None] == lb[None, :])[iu] & okp
        return Ssym[iu][same].mean() if same.any() else np.nan, int(same.sum())
    obs, npair = stat(lab)
    null = np.array([stat(rng.permutation(lab))[0] for _ in range(n_perm)])
    null = null[~np.isnan(null)]
    return dict(obs=float(obs), n_pairs=npair, p=float((1 + np.sum(null >= obs)) / (1 + len(null))),
                null_mean=float(null.mean()))


def run(X, tabs, strings, n_hyp=4000):
    C = Corpus(X, tabs, strings)
    top, _ = search(C, n_hyp, rng, keep=200, climb=4)
    Sm = np.mean([pair_scores(C, r, th) for _, r, th in top[:20]], 0)
    out = dict(ens=perm_test(Sm, strings, np.asarray(tabs)), size=perm_test(size_scores(C), strings, np.asarray(tabs)))
    return out, top, C


res = {}
if part == 'pe':
    R, X, tabs, st = pe_corpus()
    out, top, C = run(X, tabs, st)
    res.update(out)
    lab = ['%s %s' % (r[0], r[1]) for r in R]
    pairs = [(i, j) for i in range(C.n) for j in range(i + 1, C.n) if st[i] == st[j] and tabs[i] != tabs[j]]
    det = []
    for i, j in pairs:
        fw, dts = [], []
        for _, r, th in top[:200]:
            sij = pair_scores(C, r, th)
            fw.append(sij[i, j] > sij[j, i])
            best_dt = int(np.argmax([pair_scores(C, r, th, dts=(d,))[i, j] for d in DTS]))
            dts.append(best_dt)
        det.append(dict(a=lab[i], b=lab[j], string=st[i], xa=X[i], xb=X[j], p_a_first=float(np.mean(fw)),
                        dt_hist=np.bincount(dts, minlength=4)))
    res['pairs'] = det
    # class roles among top 200: share of hypotheses giving each sign each role
    rolemat = np.zeros((C.K, NROLE))
    for _, r, th in top[:200]:
        rolemat[np.arange(C.K), r] += 1
    res['role_share'] = {PE_SIGNS[k]: (rolemat[k] / 200).round(2) for k in range(C.K)}
elif part == 'ur3':
    d, X, tabs, st, yr = ur3_corpus()
    out, top, C = run(X, tabs, st, n_hyp=800)
    res.update(out)
    S = pair_scores(C, np.array([1, 2, 3, 3, 4, 5, 6, 6]), dict(b=0.85, sy=0.65, sa=0.85, of=0.07, om=0.35, qf=0.5, k=6.0))
    res['truth'] = perm_test(S, st, np.asarray(tabs))
else:
    res['plants'] = []
    for rep in range(6):
        X, tabs, own, yrs, truth, th0 = plant_archive(rng, n_flocks=150, years=8, survive=0.04, het=1.0)
        out, top, C = run(X, tabs, list(own), n_hyp=2000)
        out['truth'] = perm_test(pair_scores(C, truth, th0), list(own), np.asarray(tabs))
        res['plants'].append(out)
        print(rep, out, flush=True)
dump(res, os.path.join(CK, 'c3_%s_%s.json' % (part, mode)))
print('done', part, mode, {k: v for k, v in res.items() if k in ('ens', 'size', 'truth')})
