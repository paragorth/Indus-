#!/usr/bin/env python3
"""PE-61: does the world's vote add anything BEYOND layout?  Sign-level role scores are regressed on dumb layout
covariates of the same corpus (number adjacency t_adjN, log frequency, line-final share, document-first share,
position), and the residual is scored against the pe59 classes; also MEAS+CNT vs PERSON among number-adjacent
signs only (t_adjN >= 0.5).  Same for label-permuted training nulls and S2/S3 shuffles.
Usage: pe61_resid.py tag [minocc]"""
import os, sys, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe61_common as P
L = P.L
TAG = sys.argv[1]
MINOCC = int(sys.argv[2]) if len(sys.argv) > 2 else 6
OUT = os.path.join(P.CK, TAG)
V = pickle.load(open(os.path.join(OUT, 'votes.pkl'), 'rb'))
R = V['R']
CL = P.pe59_classes(); MC = CL['MEASURED'] | CL['COUNTED']
PE = pickle.load(open(os.path.join(P.CK, 'pe_feats.pkl'), 'rb'))
COV = ['t_adjN', 't_freq', 't_lfin', 't_first', 't_pos', 'adjN', 'lfin']
lines = []


def out(s):
    print(s, flush=True); lines.append(s)


cov_cache = {}


def cov(key):
    if key not in cov_cache:
        acc, cnt = collections.defaultdict(lambda: np.zeros(len(COV))), collections.Counter()
        for b in PE[key]:
            for t, x in zip(b['types'], b['X'][:, [L.FI[c] for c in COV]]):
                acc[t] += x; cnt[t] += 1
        cov_cache[key] = {t: acc[t] / cnt[t] for t in acc}
    return cov_cache[key]


def measures(key, rep, role):
    ms = [R[k][key] for k in R if k[0] == role and k[2] == rep and R[k] is not None and key in R[k]]
    if not ms:
        return None
    signs = [t for t, (v, c) in ms[0].items() if c >= MINOCC]
    s = np.array([np.mean([m[t][0] for m in ms]) for t in signs])
    C = cov(key)
    Z = np.column_stack([np.ones(len(signs))] + [np.array([C[t][j] for t in signs]) for j in range(len(COV))])
    beta = np.linalg.lstsq(Z, s, rcond=None)[0]
    res = s - Z @ beta
    y = np.array([t in MC for t in signs])
    adj = np.array([C[t][0] >= 0.5 for t in signs])
    yp = np.array([t in CL['PERSON'] for t in signs])
    keep = adj & (y | yp)
    return dict(resid_MC=L.auc(res, y), resid_PERSON=L.auc(res, yp),
                MC_vs_PERSON_adj=L.auc(s[keep], y[keep]) if keep.sum() > 3 else np.nan,
                n=(int(y.sum()), int(yp.sum()), int(keep.sum())))


roles = sorted({k[0] for k in R}, key=lambda r: L.ROLES.index(r))
nrep = max(k[2] for k in R)
out('PE-61 %s residual-beyond-layout check (covariates %s)' % (TAG, ','.join(COV)))
# baseline: same covariates, raw t_adjN for MC vs PERSON among adjacent signs
C = cov('REAL')
sg = [t for t in C]
for role in roles:
    m = measures('REAL', 0, role)
    nul = [measures('REAL', r, role) for r in range(1, nrep + 1)]
    sh = {tag: [measures('%s_%d' % (tag, j), 0, role) for j in range(P.NSHUF)] for tag in ('S2', 'S3')}
    for k in ('resid_MC', 'resid_PERSON', 'MC_vs_PERSON_adj'):
        nv = [x[k] for x in nul if x]
        out('%s %-17s real %.3f | label-perm null %s | S2 med %.3f q05-q95 %.3f-%.3f | S3 med %.3f q05-q95 %.3f-%.3f' % (
            role, k, m[k], ' '.join('%.2f' % v for v in nv),
            np.median([x[k] for x in sh['S2']]), np.quantile([x[k] for x in sh['S2']], .05), np.quantile([x[k] for x in sh['S2']], .95),
            np.median([x[k] for x in sh['S3']]), np.quantile([x[k] for x in sh['S3']], .05), np.quantile([x[k] for x in sh['S3']], .95)))
    out('%s n (MC, PERSON, adjacent MC|PERSON) %s' % (role, m['n']))
open(os.path.join(OUT, 'resid.txt'), 'w').write('\n'.join(lines) + '\n')
