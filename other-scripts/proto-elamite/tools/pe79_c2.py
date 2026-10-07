#!/usr/bin/env python3
"""PE-79 cycle 2: which PE sign makes each ledger's role read as Proto-Elamite?

Marginal effect of putting PE sign s in role r, on ledger k: M_k[r,s] = mean z_k over the ~830 random grafts
with H[r] = s.  Role-specific part d_k[r,s] = M_k(real labels) - mean M_k(3 label-permuted copies).
NULL for d: permuted copy 1 against the mean of copies 2-3 (same arithmetic, no real roles).
Averaged over architectures (all 48 by default: none was trusted in cycle 1).  A sign's predicted role = the role
with the largest mean d over the ledgers that have that role (>= 30 tokens), if above the null's 99th pct.
Output: data/pe79_roles_<half>.tsv (frozen, sha256 printed) and agreement with established PE results.
"""
import os, sys, json, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe79_common as C
from pe79_c1 import ARCHS, NH, NPERM


def marg(half, archs=None):
    archs = archs if archs is not None else range(len(ARCHS))
    H = C.random_H(NH, 120, np.random.default_rng(C.seed('pe79-H')))
    cnt = np.stack([np.bincount(H[:, r], minlength=120) for r in range(7)])
    acc, signs, names = None, None, None
    for ai in archs:
        d = np.load(os.path.join(C.CK, 'c1_%s_%02d.npz' % (half, ai)))
        Z = d['Z'].astype(np.float64)[:, :NH]
        signs, names = list(d['signs']), list(d['names'])
        M = np.stack([np.stack([np.bincount(H[:, r], weights=Z[i], minlength=120) / cnt[r] for r in range(7)])
                      for i in range(len(names))])                       # [corpus, role, sign]
        acc = M if acc is None else acc + M
    return acc / len(list(archs)), signs, names


def role_d(M, names):
    ix = {n: i for i, n in enumerate(names)}
    K = C.known_docs()
    elig = {r: [k for k in C.KNOWN if sum(1 for d in K[k] for x in d['toks'] if x[0] == 'T' and x[1] == r) >= 30]
            for r in C.ROLES}
    D = np.zeros((7, 120)); N = np.zeros((7, 120)); pos = np.zeros((7, 120))
    for r, ri in C.RI.items():
        ds, ns = [], []
        for k in elig[r]:
            real = M[ix[k + ':0'], ri]
            perm = np.mean([M[ix['%s:%d' % (k, j)], ri] for j in range(1, NPERM + 1)], 0)
            ds.append(real - perm)
            ns.append(M[ix[k + ':1'], ri] - np.mean([M[ix['%s:%d' % (k, j)], ri] for j in range(2, NPERM + 1)], 0))
        D[ri] = np.mean(ds, 0); N[ri] = np.mean(ns, 0); pos[ri] = np.mean(np.array(ds) > 0, 0)
    return D, N, pos, elig


def main(half='A', tag=None, archs=None):
    M, signs, names = marg(half, archs)
    D, N, pos, elig = role_d(M, names)
    thr = {r: np.quantile(N[C.RI[r]], 0.99) for r in C.ROLES}
    rows = []
    for j, s in enumerate(signs):
        best = int(np.argmax(D[:, j] - np.array([thr[r] for r in C.ROLES])))
        r = C.ROLES[best]
        call = r if D[best, j] > thr[r] else '-'
        rows.append((s, call) + tuple('%.4f' % D[i, j] for i in range(7)))
    fn = os.path.join(C.PED, 'pe79_roles_%s.tsv' % (tag or half))
    with open(fn, 'w') as f:
        f.write('sign\tcall\t' + '\t'.join('d_' + r for r in C.ROLES) + '\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    h = hashlib.sha256(open(fn, 'rb').read()).hexdigest()
    print('frozen', fn, 'sha256', h)
    print('eligible ledgers per role:', {r: elig[r] for r in C.ROLES})
    for r in C.ROLES:
        ri = C.RI[r]
        o = np.argsort(-D[ri])[:8]
        print('%s thr %.3f top: %s' % (r, thr[r], ', '.join('%s %.3f(%d/%d)' % (signs[j], D[ri, j], round(pos[ri, j] * len(elig[r])), len(elig[r])) for j in o)))
    return D, N, signs, h


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'A')
