#!/usr/bin/env python3
"""LA-39 cycle 4: modifier effect profiles with correct inference (the cycle-3 bootstrap had a floor
of 0.001, too coarse for Holm over 190 tests).
(a) Token-level features (log quantity, fraction, no number, position, after a word, relative
    quantity): permute the type label among tokens of the SAME BASE inside the SAME DOCUMENT
    (document-level features cancel). 20,000 permutations, Holm over all type x feature tests.
(b) Document-level features (site HT, support tablet, total word, number of commodities):
    documents holding base b are labelled by whether they hold type t; label permuted among those
    documents (20,000). Holm.
(c) The same on Linear B (sex markers, SI, adjuncts) as a calibration of what a known-meaning
    modifier looks like in these profiles.
"""
import sys, os, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la39_common as L

rng = np.random.default_rng(394)
TOK = ['logq', 'frac', 'noq', 'relpos', 'afterword', 'relq']
DOC = ['site', 'support', 'total', 'ncom']
NP = 20000


def holm(tests, alpha=0.05):
    tests = sorted(tests, key=lambda x: x[0]); M = len(tests); out = []
    for r, t in enumerate(tests):
        if t[0] * (M - r) > alpha: break
        out.append(t)
    return out, M


def within_doc(A, min_tok=4):
    F = [A['feats'].index(f) for f in TOK if f in A['feats']]
    names = [f for f in TOK if f in A['feats']]
    types = defaultdict(int)
    for b, m in zip(A['base'], A['mod']):
        if m: types[(b, m)] += 1
    tests = []; eff_all = {}
    for (b, m), n in types.items():
        if n < min_tok: continue
        kb = np.where(A['base'] == b)[0]
        # strata: documents with both t and other types of b
        strata = defaultdict(list)
        for i in kb: strata[A['doc'][i]].append(i)
        strata = [np.array(v) for v in strata.values()
                  if any(A['mod'][i] == m for i in v) and any(A['mod'][i] != m for i in v)]
        if not strata: continue
        idx = np.concatenate(strata)
        lab = np.array([A['mod'][i] == m for i in idx])
        X = A['X'][np.ix_(idx, F)]
        def st(l): return X[l].mean(0) - X[~l].mean(0)
        o = st(lab)
        # permutation within strata
        offs = np.cumsum([0] + [len(s) for s in strata])
        labs = [lab[offs[k]:offs[k + 1]] for k in range(len(strata))]
        cnt_ge = np.zeros(len(F)); cnt_le = np.zeros(len(F))
        for _ in range(NP):
            l = np.concatenate([rng.permutation(x) for x in labs])
            v = st(l); cnt_ge += v >= o - 1e-12; cnt_le += v <= o + 1e-12
        p = np.minimum(1, 2 * np.minimum(cnt_ge, cnt_le) / NP)
        p = np.maximum(p, 1 / NP)
        eff_all['%s+%s' % (b, m)] = dict(n_in_mixed_docs=int(lab.sum()), ndocs=len(strata),
                                         eff=dict(zip(names, np.round(o, 2).tolist())),
                                         p=dict(zip(names, np.round(p, 5).tolist())))
        for f, nm in enumerate(names):
            if np.isnan(o[f]) or (cnt_ge[f] == NP and cnt_le[f] == NP): continue   # no variation
            tests.append((p[f], '%s+%s' % (b, m), nm, round(float(o[f]), 2)))
    return eff_all, holm(tests)


def doc_level(A, min_tok=4):
    F = [A['feats'].index(f) for f in DOC if f in A['feats']]
    names = [f for f in DOC if f in A['feats']]
    types = defaultdict(int)
    for b, m in zip(A['base'], A['mod']):
        if m: types[(b, m)] += 1
    tests = []; out = {}
    for (b, m), n in types.items():
        if n < min_tok: continue
        kb = np.where(A['base'] == b)[0]
        docs = sorted(set(A['doc'][kb]))
        if len(docs) < 4: continue
        first = {}
        for i in kb: first.setdefault(A['doc'][i], i)
        Xd = A['X'][np.ix_([first[d] for d in docs], F)]
        has = np.array([any(A['mod'][i] == m for i in kb if A['doc'][i] == d) for d in docs])
        if has.all() or (~has).all(): continue
        o = Xd[has].mean(0) - Xd[~has].mean(0)
        P = np.array([rng.permutation(has) for _ in range(NP)])
        k = has.sum()
        means1 = (P.astype(float) @ Xd) / k; means0 = ((~P).astype(float) @ Xd) / (len(docs) - k)
        V = means1 - means0
        p = np.minimum(1, 2 * np.minimum((V >= o - 1e-12).mean(0), (V <= o + 1e-12).mean(0)))
        p = np.maximum(p, 1 / NP)
        out['%s+%s' % (b, m)] = dict(docs_with=int(k), docs_other=int(len(docs) - k),
                                     eff=dict(zip(names, np.round(o, 2).tolist())), p=dict(zip(names, np.round(p, 5).tolist())))
        for f, nm in enumerate(names):
            if Xd[:, f].std() == 0: continue
            tests.append((p[f], '%s+%s' % (b, m), nm, round(float(o[f]), 2)))
    return out, holm(tests)


if __name__ == '__main__':
    R = {}
    for nm, A, npm in (("LA", L.to_arrays(L.la_commodity_rows()), 20000), ("LB", L.to_arrays(L.lb_rows()), 4000)):
        NP = npm
        e1, (s1, M1) = within_doc(A)
        print(nm, 'within-document token tests:', M1, 'Holm-significant:', [(k, f, e, p) for p, k, f, e in s1], flush=True)
        e2, (s2, M2) = doc_level(A)
        print(nm, 'document-level tests:', M2, 'Holm-significant:', [(k, f, e, p) for p, k, f, e in s2], flush=True)
        R[nm] = dict(within=e1, within_sig=s1, M1=M1, doc=e2, doc_sig=s2, M2=M2)
    json.dump(R, open(os.path.join(L.CK, 'c4.json'), 'w'), indent=1, default=str)
