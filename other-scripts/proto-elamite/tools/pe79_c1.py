#!/usr/bin/env python3
"""PE-79 cycle 1: every grammar architecture x 100,000 random grafts, leave-one-civilisation-out.

Usage: pe79_c1.py [half]   half = A (judge trained on PE half A; default), B (cycle 3 replication), S (half A, signs shuffled).
Per architecture: the PE judge is trained on one half of the PE tablets.  20,000 random role->sign grafts H
(same H for every architecture) are scored on every known administration (real role labels and 3 nulls with
labels permuted inside the corpus) and on PLANT (the other PE half, roles planted from sign identity).
Saved: z-scores per H x corpus (float16) -> pe79_ckpt/c1_<half>_<arch>.npz
"""
import os, sys, json, itertools, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe79_common as C

NH = 100000
NPERM = 3
PLANT_MAP = {'COM': 'M288', 'HDR': 'M157', 'UNI': 'M297', 'PER': 'M388', 'PLA': 'M387', 'TRA': 'M346',
             'TOT': 'M003'}
ARCHS = [dict(ncls=a, pos=b, lpos=c, prev=d, lam=e)
         for a, b, c, d, e in itertools.product([0, 1], [0, 1], [0, 1], [0, 1], [0.5, 0.8, 0.95])]


def halves():
    P = C.pe_docs()
    rng = random.Random(C.seed('pe79-halves'))
    idx = list(range(len(P))); rng.shuffle(idx)
    A = [P[i] for i in sorted(idx[:len(P) // 2])]
    B = [P[i] for i in sorted(idx[len(P) // 2:])]
    return A, B


def planted(docs):
    inv = {s: r for r, s in PLANT_MAP.items()}
    return [dict(d, toks=[('T', inv.get(x[1], 'O')) if x[0] == 'T' else x for x in d['toks']]) for d in docs]


def corpora():
    K = C.known_docs()
    out = {}
    for k in C.KNOWN:
        out[(k, 0)] = K[k]
        for j in range(1, NPERM + 1):
            out[(k, j)] = C.permute_roles(K[k], random.Random(C.seed('pe79-perm-%s-%d' % (k, j))), k == 'KH')
    return out


def run(args):
    half, ai = args
    fn = os.path.join(C.CK, 'c1_%s_%02d.npz' % (half, ai))
    if os.path.exists(fn):
        return fn
    A, B = halves()
    train, other = (A, B) if half in ('A', 'S') else (B, A)
    if half == 'S':      # kill control: PE judge trained on half A with signs shuffled over all slots
        train = C.L.shuffle_types(train, random.Random(C.seed('pe79-S1')))
    arch = ARCHS[ai]
    J = C.Judge(train, arch)
    Kn = len(J.signs)
    H = C.random_H(NH, Kn, np.random.default_rng(C.seed('pe79-H')))
    # the planted map is appended as the last row
    hp = np.array([[J.si[PLANT_MAP[r]] for r in C.ROLES]])
    H = np.vstack([H, hp])
    cor = corpora()
    cor[('PLANT', 0)] = planted(other)
    names, Z, raw = [], [], []
    for key, docs in cor.items():
        sk = C.skeleton(docs, J)
        s = C.score_H(H, sk, J)
        mu, sd = s[:NH].mean(), s[:NH].std() + 1e-12
        names.append('%s:%d' % key); Z.append(((s - mu) / sd).astype(np.float16)); raw.append(s.astype(np.float32))
    np.savez_compressed(fn, Z=np.array(Z), raw=np.array(raw), names=np.array(names), signs=np.array(J.signs),
                        arch=json.dumps(arch))
    return fn


if __name__ == '__main__':
    half = sys.argv[1] if len(sys.argv) > 1 else 'A'
    C.known_docs()
    with Pool(2) as p:
        for fn in p.imap_unordered(run, [(half, i) for i in range(len(ARCHS))]):
            print(fn, flush=True)
