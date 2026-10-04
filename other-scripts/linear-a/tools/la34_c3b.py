#!/usr/bin/env python3
"""la34 cycle 3b: kill-tests for the cycle-3 positives (T1 spelling, T4 vocabulary, T0 hands).
Confound: sign-matched distances are means over the codes two units share, and units sharing words share more
codes and are larger (less noisy means).  Fix: residualise every same-site pair distance on log n drawings of
each unit (min, max) and log(1 + shared codes) by OLS, then rerun:
  T0 same published hand vs different; T4 Mantel with word Jaccard;
  T1a identical-word pairs vs one-sign-variant-only pairs; T1b variant-only pairs vs unrelated pairs."""
import numpy as np, json, os, collections
from scipy.stats import spearmanr
import la34_c3 as C
from la34_common import CK
from la34_img import occ_table

rng = np.random.default_rng(343)


def main():
    U, DD, DM, words, logos = C.build()
    meta = C.meta
    o_, names, Z, P, codes = occ_table(C.feat, scale_free=True, occ=C.occ)
    nocc = collections.Counter(o['unit'] for o in o_)
    cset = collections.defaultdict(set)
    for o, c in zip(o_, codes): cset[o['unit']].add(c)
    n = len(U); iu = np.triu_indices(n, 1)
    sites = np.array([meta[u]['site'] for u in U])
    same = sites[iu[0]] == sites[iu[1]]
    na = np.array([nocc[u] for u in U], float)
    sh = np.array([[len(cset[U[a]] & cset[U[b]]) for b in range(n)] for a in range(n)], float)
    hands = [meta[u]['scribe'] or None for u in U]
    J = np.array([[(1 - len(words[U[a]] & words[U[b]]) / len(words[U[a]] | words[U[b]])) if (words[U[a]] | words[U[b]]) else np.nan
                   for b in range(n)] for a in range(n)])
    out = {}
    for nm, Dm in (('D', DD), ('M', DM)):
        d = Dm[iu]; ok = same & np.isfinite(d)
        X = np.column_stack([np.ones(ok.sum()), np.log(np.minimum(na[iu[0]], na[iu[1]]))[ok],
                             np.log(np.maximum(na[iu[0]], na[iu[1]]))[ok], np.log1p(sh[iu])[ok]])
        beta = np.linalg.lstsq(X, d[ok], rcond=None)[0]
        R = np.full((n, n), np.nan)
        res = d.copy() * np.nan; res[ok] = d[ok] - X @ beta
        R[iu] = res; R[(iu[1], iu[0])] = res
        r = {'beta': beta.round(3).tolist()}
        r['T0_hands_resid'] = C.pair_test(R, U, list(sites), hands, lambda a, b: None if (a is None or b is None) else (1 if a == b else 0), 500)
        # T4 on residuals
        jj = J[iu]; f = ok & np.isfinite(jj)
        real = spearmanr(res[f], jj[f]).correlation; null = []
        for _ in range(300):
            p = C.perm_site(U, list(sites)); Jp = J[np.ix_(p, p)][iu]; g = ok & np.isfinite(Jp)
            null.append(spearmanr(res[g], Jp[g]).correlation)
        r['T4_vocab_resid'] = {'rho': float(real), 'null_mean': float(np.mean(null)), 'p': float((1 + np.sum(np.array(null) >= real)) / 301)}
        # T1 on residuals.  Jaccard cannot stratify (pairs without a shared word all have Jaccard 1), so:
        #  T1a identical-word pairs (A) vs one-sign-variant-only pairs (B)
        #  T1b variant-only pairs (B) vs unrelated pairs (C: no shared word, no one-sign variant).
        #      B closer than C = the same writer (or office) varies a spelling; B no closer = variants mark other writers.
        W = [words[u] for u in U]
        r['T1a_identical_vs_variant'] = C.pair_test(R, U, list(sites), W,
            lambda x, y: 1 if (x & y) else (0 if any(C.one_sign(p, q) for p in x for q in y) else None), 300)
        r['T1b_variant_vs_unrelated'] = C.pair_test(R, U, list(sites), W,
            lambda x, y: None if (x & y) else (1 if any(C.one_sign(p, q) for p in x for q in y) else 0), 300)
        out[nm] = r
        print(nm, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(CK, 'c3b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
