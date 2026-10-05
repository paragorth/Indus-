#!/usr/bin/env python3
"""la51 engine: massive random word-to-context link search with held-out deposits, nulls and plants.

Statistic: support-stratified (Cochran-Mantel-Haenszel) z for 'documents carrying term t lie in deposits
satisfying context predicate P', strata = support family (tablet / sealing / vessel / other), so a word
is never linked to a context merely because of the object it is written on.
Held-out: deposits split at random 50/50 (S splits). A link is proposed on train deposits (z >= ZT) and
re-tested on test deposits (z >= ZV, needs >= 2 test docs with the term and P true for >= 1 and false
for >= 1 test deposit). A surviving link replicates in >= half of the splits where it could be tested
(>= MINTEST tests).
Nulls: N1 within-site room shuffle (deposit labels permuted among same-site, same-stratum documents);
N2 cross-site context shuffle (class vectors permuted among deposits).
"""
import numpy as np, collections
import la51_common as L

ZT, ZV, MINTEST = 3.0, 2.0, 5


def stratum(sup):
    s = (sup or '').lower()
    if any(k in s for k in ('nodule', 'roundel', 'sealing', 'label')): return 1
    if 'vessel' in s or 'jar' in s or 'pithos' in s or 'stirrup' in s: return 2
    if any(k in s for k in ('tablet', 'lame', 'bar', 'page', 'leaf')) or s == '': return 0
    return 3


def random_preds(rng, n):
    """Random context predicates over classes: 1-3 literal conjunctions, 2-3 literal disjunctions,
    'at least k of a random class set'. Returned as functions of the deposit x class matrix."""
    nc = len(L.CLASSES)
    out, names, seen = [], [], set()
    tries = 0
    for c in range(nc):                      # every single class is always in the hypothesis set
        key = (0, (c,), (0,), 0); seen.add(key); out.append(key); names.append(L.CLASSES[c])
    while len(out) < n and tries < n * 10:
        tries += 1
        kind = rng.integers(4)
        r = int(rng.integers(1, 4))
        cl = tuple(sorted(rng.choice(nc, r, replace=False).tolist()))
        neg = tuple(int(x) for x in rng.integers(0, 2, r)) if kind == 1 else (0,) * r
        k = int(rng.integers(1, r + 1)) if kind == 3 else 0
        key = (kind if r > 1 else 0, cl, neg, k)
        if kind == 2 and r == 1: key = (0, cl, neg, 0)
        if key in seen: continue
        seen.add(key)
        out.append(key)
        def nm(key=key):
            kd, cl, ng, k = key
            lit = [('NOT ' if n else '') + L.CLASSES[c] for c, n in zip(cl, ng)]
            if kd in (0, 1): return ' AND '.join(lit)
            if kd == 2: return ' OR '.join(lit)
            return f'>={k} of {{{",".join(lit)}}}'
        names.append(nm())
    return out, names


def pred_matrix(keys, C):
    """deposit x pred boolean."""
    P = np.zeros((C.shape[0], len(keys)), dtype=bool)
    for j, (kd, cl, ng, k) in enumerate(keys):
        A = np.stack([C[:, c] ^ bool(n) for c, n in zip(cl, ng)], 1)
        if kd in (0, 1): P[:, j] = A.all(1)
        elif kd == 2: P[:, j] = A.any(1)
        else: P[:, j] = A.sum(1) >= k
    return P


def cmh_z(X, D, strata, docmask):
    """X doc x term bool, D doc x pred bool. Returns z (term x pred), k (observed)."""
    num = np.zeros((X.shape[1], D.shape[1])); var = np.zeros_like(num); kk = np.zeros_like(num)
    for s in np.unique(strata):
        idx = np.where((strata == s) & docmask)[0]
        n = len(idx)
        if n < 3: continue
        Xs = X[idx].astype(np.float64); Ds = D[idx].astype(np.float64)
        k = Xs.T @ Ds
        nt = Xs.sum(0)[:, None]; K = Ds.sum(0)[None, :]
        E = nt * K / n
        V = nt * K * (n - K) * (n - nt) / (n * n * (n - 1))
        num += k - E; var += V; kk += k
    z = np.where(var > 0, num / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    return z, kk


def run(M, n_preds=2000, splits=40, seed=0, dep_of=None, C=None, X=None, track=None, probes=None):
    rng = np.random.default_rng(seed)
    dep_of = M['dep_of'] if dep_of is None else dep_of
    C = M['C'] if C is None else C
    X = M['X'] if X is None else X
    strata = np.array([stratum(s) for s in M['support']])
    keys, names = random_preds(rng, n_preds)
    P = pred_matrix(keys, C)
    keep = (P.sum(0) > 0) & (P.sum(0) < C.shape[0])
    keys = [k for k, f in zip(keys, keep) if f]; names = [n for n, f in zip(names, keep) if f]; P = P[:, keep]
    D = P[dep_of]                                  # doc x pred
    ndep = C.shape[0]
    tests = np.zeros((X.shape[1], P.shape[1]), dtype=np.int32)
    passes = np.zeros_like(tests)
    proposed = 0
    for sp in range(splits):
        tr = rng.random(ndep) < 0.5
        trd = tr[dep_of]; ted = ~trd
        ztr, _ = cmh_z(X, D, strata, trd)
        zte, _ = cmh_z(X, D, strata, ted)
        # testability on held-out deposits
        nte = (X[ted]).sum(0)[:, None] >= 2
        Pte = P[~tr]
        varied = (Pte.any(0) & (~Pte).any(0))[None, :]
        prop = ztr >= ZT
        proposed += int(prop.sum())
        t = prop & nte & varied
        tests += t
        passes += t & (zte >= ZV)
    rate = np.where(tests > 0, passes / np.maximum(tests, 1), 0)
    surv = (tests >= MINTEST) & (rate >= 0.5)
    res = dict(n_links=int(X.shape[1] * P.shape[1]), n_preds=len(keys), proposed=proposed,
               tested=int((tests > 0).sum()), survivors=int(surv.sum()),
               surv_terms=sorted({M['terms'][i] for i in np.where(surv.any(1))[0]}))
    rows = []
    for i, j in zip(*np.where(surv)):
        rows.append((M['terms'][i], names[j], int(tests[i, j]), float(rate[i, j])))
    # collapse: best predicate per term
    best = {}
    for r in rows:
        if r[0] not in best or (r[3], r[2]) > (best[r[0]][3], best[r[0]][2]): best[r[0]] = r
    res['best'] = sorted(best.values(), key=lambda r: (-r[3], -r[2]))
    if probes:
        # single-class probes: replication rate across splits + full-data z
        Cs = C[dep_of]
        zfull, kfull = cmh_z(X, Cs, strata, np.ones(len(dep_of), bool))
        single = {}
        for j, kk in enumerate(keys):
            if kk[0] == 0 and len(kk[1]) == 1: single[kk[1][0]] = j
        res['probes'] = {}
        for (t, c) in probes:
            if t not in M['terms']: continue
            ti = M['terms'].index(t); ci = L.CLASSES.index(c); j = single.get(ci)
            res['probes'][f'{t}~{c}'] = dict(z=float(zfull[ti, ci]), k=int(kfull[ti, ci]),
                                              n=int(X[:, ti].sum()),
                                              tests=int(tests[ti, j]) if j is not None else 0,
                                              rate=float(rate[ti, j]) if j is not None else None,
                                              surv=bool(surv[ti, j]) if j is not None else False)
    if track:
        ti = M['terms'].index(track[0]) if track[0] in M['terms'] else None
        if ti is not None:
            js = [j for j, n in enumerate(names) if n == track[1]]
            res['track'] = [(names[j], int(tests[ti, j]), float(rate[ti, j])) for j in js]
            res['track_any'] = bool(surv[ti].any())
    return res


def null_within_site(M, rng):
    """N1: permute deposit labels among documents of the same site and support stratum."""
    dep = M['dep_of'].copy()
    strata = np.array([stratum(s) for s in M['support']])
    for site in np.unique(M['sites']):
        for s in np.unique(strata):
            idx = np.where((M['sites'] == site) & (strata == s))[0]
            if len(idx) > 1: dep[idx] = dep[rng.permutation(idx)]
    return dep


def null_cross_site(M, rng):
    """N2: permute class vectors among deposits (contexts shuffled across sites)."""
    return M['C'][rng.permutation(M['C'].shape[0])]


def plant(M, rng, cls='METAL', p_in=0.3, p_out=0.01, name='w:PLANTED'):
    """Planted link: a new term written on a fraction p_in of documents in deposits with class cls."""
    ci = L.CLASSES.index(cls)
    indep = M['C'][M['dep_of'], ci]
    col = np.where(indep, rng.random(len(indep)) < p_in, rng.random(len(indep)) < p_out)
    M2 = dict(M)
    M2['X'] = np.concatenate([M['X'], col[:, None]], 1)
    M2['terms'] = M['terms'] + [name]
    return M2, int(col.sum()), int((col & indep).sum())
