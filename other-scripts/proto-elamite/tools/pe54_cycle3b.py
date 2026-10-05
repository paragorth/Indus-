"""pe54 cycle 3b: is the PE string-pair excess (zm) an artefact of how many cells two entries share?
Null = random cross-tablet pairs drawn with the SAME number of jointly written classes as each string pair
(20,000 draws). Also a role-free check: score = identity dynamics only (gap 0, every class F0).
usage: pe54_cycle3b.py pe|plant|ur3"""
import sys
import numpy as np
from pe54_lib import *
import pe54_lib as L
L.ZERO_MISSING = True
part = sys.argv[1]
rng = np.random.default_rng(5413)


def strat_test(S, X, strings, tabs, n=20000):
    Ssym = np.maximum(S, S.T)
    N = len(strings)
    ov = ((X[:, None, :] > 0) & (X[None, :, :] > 0)).sum(2)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if strings[i] == strings[j] and tabs[i] != tabs[j]]
    if not pairs:
        return None
    pool = {}
    for i in range(N):
        for j in range(i + 1, N):
            if tabs[i] != tabs[j] and strings[i] != strings[j]:
                pool.setdefault(ov[i, j], []).append(Ssym[i, j])
    obs = np.mean([Ssym[i, j] for i, j in pairs])
    draws = np.zeros(n)
    for i, j in pairs:
        p = np.array(pool.get(ov[i, j], [0.0]))
        draws += p[rng.integers(len(p), size=n)]
    draws /= len(pairs)
    return dict(obs=float(obs), p=float((1 + np.sum(draws >= obs)) / (n + 1)), null_mean=float(draws.mean()),
                overlaps=[int(ov[i, j]) for i, j in pairs], per_pair=[float(Ssym[i, j]) for i, j in pairs])


def run(X, tabs, st, n_hyp):
    C = Corpus(X, np.asarray(tabs), st)
    top, _ = search(C, n_hyp, rng, keep=50, climb=4)
    Sm = np.mean([pair_scores(C, r, th) for _, r, th in top[:20]], 0)
    old = L.DTS
    ident = np.ones(C.K, int)
    Si = pair_scores(C, ident, dict(top[0][2]), dts=(0,))
    return dict(ens=strat_test(Sm, X, st, np.asarray(tabs)), ident=strat_test(Si, X, st, np.asarray(tabs)),
                size=strat_test(size_scores(C), X, st, np.asarray(tabs)))


res = {}
if part == 'pe':
    R, X, tabs, st = pe_corpus()
    res = run(X, tabs, st, 4000)
elif part == 'ur3':
    d, X, tabs, st, yr = ur3_corpus()
    res = run(X, tabs, st, 800)
else:
    res['plants'] = []
    for rep in range(6):
        X, tabs, own, yrs, truth, th0 = plant_archive(rng, n_flocks=150, years=8, survive=0.04, het=1.0)
        o = run(X, tabs, list(own), 2000)
        res['plants'].append(o)
        print(rep, {k: (v['p'] if v else None) for k, v in o.items()}, flush=True)
dump(res, os.path.join(CK, 'c3b_%s.json' % part))
print('done', part, res if part != 'plant' else '')
