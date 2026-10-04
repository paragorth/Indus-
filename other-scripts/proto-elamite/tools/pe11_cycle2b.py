"""pe11 cycle 2b: is the small PE loop excess real, and what is it made of?
(i) ALL and HERD feature sets: real D vs 60 copula surrogates (sharper p), and the
    same statistic for the Drehem real control (n 638, 15 features, 60 surrogates) and
    for a COMPOSITIONAL TYPE-LOOP control: three static tablet types A, B, C (built
    from PE tablets with high shares of the 3 commonest final signs) whose tablets are
    pairwise mixtures A-B, B-C, C-A (a loop in commodity space with NO time).
(ii) Loop anatomy: mean feature profile along the inferred real PE phase (which
    features rise and fall around the loop)."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, pickle
import numpy as np
from multiprocessing import Pool
from pe11_common import *  # noqa
from pe11_cycle1 import task

NREP = 60


def typeloop(P, pool, seed):
    """Tablets as mixtures of two of three entry-pools, pairs cycling A-B, B-C, C-A."""
    rng = np.random.default_rng(seed)
    ids = sorted(P)
    pools = {s: [e for t in P.values() for e in t['entries'] if e[0] == s] for s in pool[:3]}
    rest = [e for t in P.values() for e in t['entries'] if e[0] not in pools]
    out = {}
    for tid in ids:
        k = int(rng.integers(0, 3))
        a, b = pool[k], pool[(k + 1) % 3]
        w = rng.random()
        E = []
        for _ in P[tid]['entries']:
            u = rng.random()
            src = pools[a] if u < 0.6 * w else (pools[b] if u < 0.6 else rest)
            E.append(src[rng.integers(len(src))])
        out[tid] = {'hdr': None, 'entries': E, 'signs': set(), 'design': ''}
    return out


def main():
    P, U = pickle.load(open(os.path.join(CKPT, 'frames.pkl'), 'rb'))
    pool = pe_pool(P)
    X, names, ids = pe_features(P, pool)
    counted = [names.index('sh_' + s) for s in ('M288', 'M218', 'M371', 'M263', 'M346', 'M054', 'M096', 'M066', 'M376', 'M387', 'M009', 'M057', 'M367', 'M001')]
    sets = {'ALL': X, 'HERD': X[:, counted + [names.index(k) for k in ('logS', 'mlogS', 'sh1', 'shFrac')]]}
    Xu, nu, idu = ur3_features(U)
    rng = np.random.default_rng(0)
    sets['UR3'] = Xu[rng.choice(len(idu), 638, replace=False)]
    sets['TYPELOOP'], _, _ = pe_features(typeloop(P, pool, 3), pool, ids)
    jobs = []
    for nm, M in sets.items():
        jobs.append(('real', 'b_%s_real' % nm, M, None, 5, {'set': nm}))
        for q in range(NREP):
            jobs.append(('copula', 'b_%s_copula%d' % (nm, q), M, None, 8000 + q, {'set': nm}))
    with Pool(2) as pl:
        res = list(pl.imap_unordered(task, jobs))
    out = {}
    for nm in sets:
        real = [r for r in res if r['key'] == 'b_%s_real' % nm][0]
        d = np.array([r['D'] for r in res if r['set'] == nm and r['kind'] == 'copula'])
        out[nm] = {'D': real['D'], 'null_mean': float(d.mean()), 'null_sd': float(d.std()), 'null_max': float(d.max()),
                   'z': float((real['D'] - d.mean()) / d.std()), 'p': float((1 + (d >= real['D']).sum()) / (1 + len(d)))}
        print(nm, out[nm], flush=True)
    # loop anatomy on the cycle-2 real ALL order
    z = np.array(json.load(open(os.path.join(CKPT, 'c2_ALL_real.json')))['z'])
    Xs = standardize(X)
    prof = {names[j]: [float(Xs[z == s, j].mean()) if (z == s).any() else None for s in range(S)] for j in range(len(names))}
    amp = {k: float(np.nanmax([v for v in p if v is not None]) - np.nanmin([v for v in p if v is not None])) for k, p in prof.items()}
    top = sorted(amp, key=lambda k: -amp[k])[:8]
    out['anatomy'] = {'counts': np.bincount(z, minlength=S).tolist(),
                      'top_features': {k: {'range_sd': amp[k], 'peak_phase': int(np.nanargmax([v if v is not None else -9 for v in prof[k]]))} for k in top}}
    print(json.dumps(out['anatomy'], default=lambda x: round(x, 3)))
    save('pe11_cycle2b.json', out)


if __name__ == '__main__':
    main()
