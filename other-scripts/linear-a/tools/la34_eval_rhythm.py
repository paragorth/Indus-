#!/usr/bin/env python3
"""Score arm R (box geometry) against held-out hands, within site; label-permutation null."""
import json, os, numpy as np, collections, sys
from la34_common import load, CK
from la34_score import standardize, dist, perm_test
occ, meta, cid = load()
R = json.load(open(os.path.join(CK, 'rhythm.json')))
nsyl = collections.Counter(o['unit'] for o in occ if o['role'] == 'syllabogram')
for minsyl in (3, 8):
    U = [u for u in R['units'] if meta[u]['scribe'] and nsyl[u] >= minsyl]
    c = collections.Counter(meta[u]['scribe'] for u in U)
    U = [u for u in U if c[meta[u]['scribe']] >= 2]
    X = np.array([R['units'][u] for u in U]); Z = standardize(X)
    h = [meta[u]['scribe'] for u in U]; s = [meta[u]['site'] for u in U]
    r = perm_test(dist(Z), h, s, 1000)
    print(f'minsyl {minsyl} units {len(U)} hands {len(set(h))}', {k: round(v, 3) for k, v in r.items()})
    # single-feature AUCs
    for j, nm in enumerate(R['names']):
        rj = perm_test(dist(Z[:, [j]]), h, s, 200, seed=j)
        print('  ', nm, round(rj['auc'], 3), 'p', round(rj['auc_p'], 3))
