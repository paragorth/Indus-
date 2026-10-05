#!/usr/bin/env python3
"""la51 cycle 3: invert the problem - read the ROOM from the TABLET.
Leave-one-deposit-out: for each deposit, learn term -> class links (support-stratified CMH z, single
classes) on all other deposits, then predict the held-out deposit's object classes from the terms on its
documents (mean z over its term tokens). Score: AUC over (deposit, class) cells vs the coded classes.
Nulls: N1 within-site room shuffle, N2 cross-site context shuffle (50 reps each).
Then: predictions for documents with no coded room (unpublished / unrecorded findspots).
usage: la51_c3.py la|lb|lbsmall [reps]
"""
import sys, json, os, numpy as np, collections
import la51_common as L, la51_engine as E

corpus = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 50


def auc(scores, truth):
    s = np.asarray(scores); t = np.asarray(truth, bool)
    if t.all() or (~t).all(): return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    # ties: average ranks
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = t.sum(); n0 = (~t).sum()
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def loo(M, dep_of, C, min_docs=2):
    strata = np.array([E.stratum(s) for s in M['support']])
    X = M['X']; nd = C.shape[0]
    Pc = C                                    # deposit x class (single-class predicates)
    S, T, W = [], [], []
    for d in range(nd):
        held = dep_of == d
        if held.sum() < min_docs or X[held].sum() == 0: continue
        z, _ = E.cmh_z(X, Pc[dep_of], strata, ~held)
        xs = X[held].sum(0).astype(float)                 # term token counts in held deposit
        if xs.sum() == 0: continue
        sc = (xs[:, None] * np.clip(z, -5, 5)).sum(0) / xs.sum()
        for c in range(C.shape[1]):
            if C[:, c].all() or (~C[:, c]).all(): continue
            S.append(sc[c]); T.append(C[d, c]); W.append((d, c))
    return auc(S, T), S, T, W


def get_M():
    if corpus == 'la': return L.build_matrices(L.load_la(), min_docs=2)
    docs = L.load_lb()
    if corpus == 'lbsmall':
        rng = np.random.default_rng(1000)
        docs = [docs[i] for i in rng.choice(len(docs), 1541, replace=False)]
    return L.build_matrices(docs, min_docs=2)


M = get_M()
a, S, T, W = loo(M, M['dep_of'], M['C'])
print(corpus, 'real AUC', round(a, 3), 'cells', len(T), 'deposits', len({w[0] for w in W}), flush=True)
per_class = {}
for c in range(len(L.CLASSES)):
    idx = [i for i, w in enumerate(W) if w[1] == c]
    if idx: per_class[L.CLASSES[c]] = (auc([S[i] for i in idx], [T[i] for i in idx]) if len(set(T[i] for i in idx)) > 1 else None, len(idx))
nul = {'n1': [], 'n2': []}
for r in range(reps):
    rng = np.random.default_rng(700 + r)
    nul['n1'].append(loo(M, E.null_within_site(M, rng), M['C'])[0])
    nul['n2'].append(loo(M, M['dep_of'], E.null_cross_site(M, rng))[0])
out = dict(corpus=corpus, auc=a, per_class=per_class, cells=len(T),
           n1=[float(x) for x in nul['n1']], n2=[float(x) for x in nul['n2']])
for k in ('n1', 'n2'):
    v = np.array(nul[k], float); v = v[~np.isnan(v)]
    out[k + '_mean'] = float(v.mean()); out[k + '_P'] = float(((v >= a).sum() + 1) / (len(v) + 1))
    print(k, 'mean', round(v.mean(), 3), 'P', round(out[k + '_P'], 3), flush=True)
print('per class', per_class)

# predictions for un-roomed documents (LA only): learn on all coded deposits
if corpus == 'la':
    strata = np.array([E.stratum(s) for s in M['support']])
    z, _ = E.cmh_z(M['X'], M['C'][M['dep_of']], strata, np.ones(len(M['docs']), bool))
    ti = {t: i for i, t in enumerate(M['terms'])}
    preds = []
    for d in L.load_la():
        if d['deposit']: continue
        ts = [ti[t] for t in d['terms'] if t in ti]
        if not ts: continue
        sc = np.clip(z[ts], -5, 5).mean(0)
        top = [(L.CLASSES[c], round(float(sc[c]), 2)) for c in np.argsort(-sc)[:3]]
        preds.append(dict(id=d['id'], site=d['site'], support=d['support'], n_terms=len(ts), top=top))
    out['unroomed'] = preds
    for p in preds: print(p)
json.dump(out, open(os.path.join(L.CK, f'c3_{corpus}.json'), 'w'), indent=1, default=str)
