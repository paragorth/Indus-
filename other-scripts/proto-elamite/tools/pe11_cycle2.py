"""pe11 cycle 2: THE PROTO-ELAMITE FILM.
(a) Fixed feature sets on the 638 PE tablets with >= 3 valued entries:
    ALL (24 features), HERD (counted-goods shares + counted amounts), GRAIN (capacity
    share/amount + capacity-tied sign shares).  Real loop excess D and ring fit R2c vs
    20 copula, 20 shuffle and 20 random-walk surrogates each.
(b) Massive random guessing: 1,500 random feature subsets (3-6 of the 24) scored on a
    random half A of the tablets; the whole search repeated on 8 copula surrogates of
    half A (family-wise null for the maximum); the top 1% of real subsets re-tested on
    the held-out half B against 30 copula surrogates per subset (Bonferroni).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, pickle, sys, time
import numpy as np
from multiprocessing import Pool
from pe11_common import *  # noqa

NSUB = 1500


def score_task(args):
    key, X, kind, seed = args
    fn = os.path.join(CKPT, 'c2_%s.json' % key)
    if os.path.exists(fn):
        return json.load(open(fn))
    rng = np.random.default_rng(seed)
    if kind != 'real':
        X = NULLS[kind](X, rng)
    r = loop_score(X, rng, restarts=6, sweeps=25)
    res = {'key': key, 'kind': kind, 'D': r['D'], 'R2c': r['R2c'], 'R2l': r['R2l']}
    if kind == 'real':
        res['z'] = r['z'].tolist()
    json.dump(res, open(fn, 'w'))
    return res


def search_task(args):
    """Score all subsets on one (possibly surrogate) matrix; checkpoint whole replicate."""
    key, X, subsets, kind, seed = args
    fn = os.path.join(CKPT, 'c2s_%s.json' % key)
    if os.path.exists(fn):
        return json.load(open(fn))
    rng = np.random.default_rng(seed)
    if kind != 'real':
        X = null_copula(X, rng)
    out = []
    for s in subsets:
        r = loop_score(X[:, s], rng, restarts=3, sweeps=15)
        out.append(r['D'])
    res = {'key': key, 'D': out}
    json.dump(res, open(fn, 'w'))
    return res


def chunks(L, k):
    return [L[i:i + k] for i in range(0, len(L), k)]


def main():
    P, U = pickle.load(open(os.path.join(CKPT, 'frames.pkl'), 'rb'))
    pool = pe_pool(P)
    X, names, ids = pe_features(P, pool)
    print(len(ids), names, flush=True)
    counted = [names.index('sh_' + s) for s in ('M288', 'M218', 'M371', 'M263', 'M346', 'M054', 'M096', 'M066', 'M376', 'M387', 'M009', 'M057', 'M367', 'M001')]
    sets = {'ALL': list(range(len(names))),
            'HERD': counted + [names.index(k) for k in ('logS', 'mlogS', 'sh1', 'shFrac')],
            'GRAIN': [names.index(k) for k in ('shC', 'logC', 'sh_M297', 'sh_|M036+1(N30D)|')]}
    jobs = []
    for nm, cols in sets.items():
        jobs.append(('%s_real' % nm, X[:, cols], 'real', 5))
        for k in ('copula', 'shuffle', 'rw'):
            for q in range(20):
                jobs.append(('%s_%s%d' % (nm, k, q), X[:, cols], k, 300 + q))
    rng = np.random.default_rng(22)
    perm = rng.permutation(len(ids))
    A, B = perm[: len(ids) // 2], perm[len(ids) // 2:]
    subsets = [sorted(rng.choice(len(names), rng.integers(3, 7), replace=False).tolist()) for _ in range(NSUB)]
    sjobs = []
    for ci, ch in enumerate(chunks(subsets, 150)):
        sjobs.append(('real_c%d' % ci, X[A], ch, 'real', 900 + ci))
        for q in range(8):
            sjobs.append(('null%d_c%d' % (q, ci), X[A], ch, 'copula', 5000 + 100 * q))  # same surrogate across chunks of one replicate
    t = time.time()
    with Pool(2) as pl:
        res = []
        for i, r in enumerate(pl.imap_unordered(score_task, jobs)):
            res.append(r)
            if i % 30 == 0:
                print('fixed', i, len(jobs), round(time.time() - t), flush=True)
        sres = {}
        for i, r in enumerate(pl.imap_unordered(search_task, sjobs)):
            sres[r['key']] = r['D']
            print('search', i, len(sjobs), round(time.time() - t), flush=True)
    summ = {}
    for nm in sets:
        real = [r for r in res if r['key'] == nm + '_real'][0]
        row = {'D': real['D'], 'R2c': real['R2c'], 'R2l': real['R2l']}
        for k in ('copula', 'shuffle', 'rw'):
            N = [r for r in res if r['key'].startswith(nm + '_' + k)]
            d = np.array([r['D'] for r in N]); c = np.array([r['R2c'] for r in N])
            row[k] = {'D_mean': float(d.mean()), 'D_sd': float(d.std()), 'D_max': float(d.max()),
                      'p_D': float((1 + (d >= real['D']).sum()) / (1 + len(d))),
                      'R2c_mean': float(c.mean()), 'R2c_max': float(c.max()),
                      'p_R2c': float((1 + (c >= real['R2c']).sum()) / (1 + len(c)))}
        summ[nm] = row
    nch = len(chunks(subsets, 150))
    realD = np.concatenate([sres['real_c%d' % c] for c in range(nch)])
    nullmax = [float(np.max(np.concatenate([sres['null%d_c%d' % (q, c)] for c in range(nch)]))) for q in range(8)]
    nullall = np.concatenate([sres['null%d_c%d' % (q, c)] for q in range(8) for c in range(nch)])
    order = np.argsort(-realD)
    top = order[: NSUB // 100]
    # held-out half B
    hjobs = []
    for j in top:
        cols = subsets[j]
        hjobs.append(('hoB_%d_real' % j, X[B][:, cols], 'real', 7))
        for q in range(30):
            hjobs.append(('hoB_%d_copula%d' % (j, q), X[B][:, cols], 'copula', 700 + q))
    with Pool(2) as pl:
        hres = list(pl.imap_unordered(score_task, hjobs))
    ho = []
    for j in top:
        r = [x for x in hres if x['key'] == 'hoB_%d_real' % j][0]
        n = np.array([x['D'] for x in hres if x['key'].startswith('hoB_%d_copula' % j)])
        p = float((1 + (n >= r['D']).sum()) / (1 + len(n)))
        ho.append({'subset': [names[c] for c in subsets[j]], 'D_A': float(realD[j]),
                   'D_B': r['D'], 'nullB_mean': float(n.mean()), 'nullB_max': float(n.max()), 'p_B': p,
                   'p_B_bonf': min(1.0, p * len(top))})
    out = {'fixed': summ, 'search': {'n_subsets': NSUB, 'real_max': float(realD.max()),
                                      'real_q99': float(np.quantile(realD, 0.99)),
                                      'real_mean': float(realD.mean()),
                                      'null_mean': float(nullall.mean()), 'null_q99': float(np.quantile(nullall, 0.99)),
                                      'null_max_per_rep': nullmax,
                                      'fwer_p': float((1 + sum(m >= realD.max() for m in nullmax)) / 9),
                                      'n_real_above_null_q99': int((realD > np.quantile(nullall, 0.99)).sum())},
           'heldout': ho, 'names': names, 'ids': ids}
    save('pe11_cycle2.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('ids',)}, indent=1, default=float))


if __name__ == '__main__':
    main()
