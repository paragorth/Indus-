#!/usr/bin/env python3
"""LA-20 cycle 1 report: per-configuration z and P vs within-list shuffles; family-wise P from
max z over all configurations (types x rankers x {all pairs, never-met pairs})."""
import json, os, sys
import numpy as np
from la20_common import CK, row

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
cfg = {}
for T in 'WELF':
    p = os.path.join(CK, 'c1_%s.json' % T)
    if not os.path.exists(p): continue
    d = json.load(open(p))
    for m, r in d['real'].items():
        N = np.array(d['nulls'][m])
        if len(N) == 0: continue
        for k, (i, j) in {'all': (0, 1), 'unmet': (2, 3)}.items():
            ra = r[j] / r[i] if r[i] else np.nan
            na = np.where(N[:, i] > 0, N[:, j] / np.maximum(N[:, i], 1), np.nan)
            cfg[(T, m, k)] = (ra, na, r[i])
nrep = min(len(v[1]) for v in cfg.values())
Z = {}
for key, (ra, na, n) in cfg.items():
    na = na[:nrep]; mu, sd = np.nanmean(na), np.nanstd(na)
    Z[key] = ((ra - mu) / sd, (na - mu) / sd)
maxnull = np.nanmax(np.array([v[1] for v in Z.values()]), 0)
for key, (ra, na, n) in sorted(cfg.items()):
    z, nz = Z[key]
    p = (np.sum(na[:nrep] >= ra) + 1) / (nrep + 1)
    fw = (np.sum(maxnull >= z) + 1) / (nrep + 1)
    print('%s %-4s %-6s n=%5d acc %.3f null %.3f+-%.3f z %+.2f P %.4f FWER %.4f' % (key[0], key[1], key[2], n, ra, np.nanmean(na), np.nanstd(na), z, p, fw))
print('configurations', len(cfg), 'null reps', nrep)
