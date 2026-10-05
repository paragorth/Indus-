#!/usr/bin/env python3
"""LA-57 cycle 1: massive random guessing across bureaucracies.
For each role, M random sparse linear scorers (1-5 abstract features, Gaussian weights; no training) are scored
(occurrence AUC) on every known system that carries the role.  Leave-one-system-out: survivors = scorers with
AUC >= THR on every OTHER system (>= 2); the survivors' rank-averaged ensemble is scored on the held-out system.
Null: role labels permuted within every training corpus (type level; khipu: occurrence level), NNULL replicates,
the whole selection redone, the ensemble scored on the TRUE held-out labels.  Universal scorers (AUC >= THR on all
systems) are then run on PLANT (never used), on Linear A, and on Linear A shuffled (S1 types, S2 order).
Usage: la57_c1.py [M] [NNULL] [THR] [tag]"""
import os, sys, pickle, json, time, collections
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la57_common as C

M = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
NNULL = int(sys.argv[2]) if len(sys.argv) > 2 else 20
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
TAG = sys.argv[4] if len(sys.argv) > 4 else 'c1'
CHUNK = 1000
TOPK = 200
OUT = os.path.join(C.CK, TAG)
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, 'log.txt')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    C.wlog(LOG, s)


def load():
    F = pickle.load(open(os.path.join(C.CK, 'feats.pkl'), 'rb'))
    sysd = {}
    for k in C.KNOWN:
        Xs, labs, types = [], [], []
        for j in range(4):
            f = F[(k, j)]
            Xs.append(f['X']); labs += f['labs']; types += [(j, t) for t in f['types']]
        sysd[k] = dict(X=np.vstack(Xs), labs=labs, types=types)
    return F, sysd


def hyps(M, rng):
    H = []
    for _ in range(M):
        k = rng.integers(1, 6)
        f = rng.choice(C.NF, k, replace=False)
        w = rng.normal(size=k)
        H.append((f, w))
    return H


def score(X, H):
    S = np.zeros((X.shape[0], len(H)), np.float32)
    for i, (f, w) in enumerate(H):
        S[:, i] = X[:, f] @ w
    return S


def label_sets(s, role, nnull, rng):
    """rows used for role (labelled rows), real y and nnull permuted y (type-level within corpus)."""
    labs = s['labs']
    rows = np.array([i for i, l in enumerate(labs) if l is not None])
    y = np.array([labs[i] == role for i in rows])
    Ys = [y]
    if s.get('kh'):
        for _ in range(nnull):
            Ys.append(rng.permutation(y))
    else:
        tl = sorted({s['types'][i][1] for i in rows})
        tr = {}
        for i in rows:
            tr[s['types'][i][1]] = labs[i]
        for _ in range(nnull):
            perm = rng.permutation(len(tl))
            m = {tl[a]: tr[tl[b]] for a, b in zip(range(len(tl)), perm)}
            Ys.append(np.array([m[s['types'][i][1]] == role for i in rows]))
    return rows, np.array(Ys)


def main():
    t0 = time.time()
    F, sysd = load()
    sysd['KH']['kh'] = True
    rng = np.random.default_rng(C.seed('la57-' + TAG))
    res = {}
    for role in C.ROLES:
        elig = []
        for k in C.KNOWN:
            labs = sysd[k]['labs']
            npos = sum(1 for l in labs if l == role)
            nneg = sum(1 for l in labs if l is not None and l != role)
            if npos >= 30 and nneg >= 30:
                elig.append(k)
        if len(elig) < 3:
            log('role', role, 'eligible', elig, '-> skipped (< 3 systems)')
            continue
        H = hyps(M, rng)
        L = {}
        A = {}           # system -> (nnull+1) x M AUCs
        for k in elig:
            rows, Ys = label_sets(sysd[k], role, NNULL, rng)
            L[k] = (rows, Ys)
            n1 = Ys.sum(1); n0 = Ys.shape[1] - n1
            out = np.zeros((Ys.shape[0], M))
            for c0 in range(0, M, CHUNK):
                S = score(sysd[k]['X'][rows], H[c0:c0 + CHUNK])
                R = rankdata(S, axis=0)
                out[:, c0:c0 + CHUNK] = (Ys.astype(np.float64) @ R - (n1 * (n1 + 1) / 2)[:, None]) / (n1 * n0)[:, None]
            A[k] = out
        # universal on real labels
        realA = np.array([A[k][0] for k in elig])
        univ = np.where((realA >= THR).all(0))[0]
        # LOSO (one system out) and LOCO (one civilisation out: cuneiform / Aegean / Andes)
        fold = {}
        units = [(h, [h]) for h in elig]
        civs = sorted({C.CIV2[k] for k in elig})
        if len(civs) >= 2:
            units += [('CIV:' + c, [k for k in elig if C.CIV2[k] == c]) for c in civs]
        for name, hold in units:
            tr = [k for k in elig if k not in hold]
            if not tr:
                continue
            for h in hold:
                rows_h, Ys_h = L[h]
                yh = Ys_h[0]
                Xh = sysd[h]['X'][rows_h]
                out = []
                for rep in range(NNULL + 1):
                    At = np.array([A[k][rep] for k in tr])
                    mn = At.min(0)
                    npass = int((mn >= THR).sum())
                    sel = np.argsort(-mn)[:TOPK]
                    S = score(Xh, [H[i] for i in sel])
                    ens = rankdata(S, axis=0).mean(1)
                    out.append((npass, C.auc(ens, yh), float(np.mean(A[h][0][sel] >= THR)), float(mn[sel].mean())))
                real = out[0]
                nulls = np.array([o[1] for o in out[1:]])
                p = (1 + int((nulls >= real[1]).sum())) / (1 + len(nulls))
                key = name if name == h else name + '>' + h
                fold[key] = dict(ntrain=len(tr), npass=real[0], auc=real[1], transfer=real[2], train_min=real[3],
                                 null_npass_med=float(np.median([o[0] for o in out[1:]])),
                                 null_auc_med=float(np.median(nulls)), null_auc_q95=float(np.quantile(nulls, 0.95)), p=p)
                log('%s held-out %-12s train %d sys | pass %5d (null med %5.0f) | top%d ens AUC %.3f (null med %.3f q95 %.3f) P %.3f | share>=thr %.2f' % (
                    role, key, len(tr), real[0], fold[key]['null_npass_med'], TOPK, real[1], fold[key]['null_auc_med'],
                    fold[key]['null_auc_q95'], p, real[2]))
        # universal survivors under the null (count only)
        nuniv = [int(((np.array([A[k][rep] for k in elig]) >= THR).all(0)).sum()) for rep in range(1, NNULL + 1)]
        log('%s universal scorers: %d of %d (null median %.0f, max %d) systems %s' % (role, len(univ), M, np.median(nuniv), max(nuniv), elig))
        res[role] = dict(elig=elig, fold=fold, nuniv=len(univ), null_univ=nuniv,
                         univ=[(H[i][0].tolist(), H[i][1].tolist()) for i in univ[:3000]],
                         univ_auc={k: realA[elig.index(k)][univ].tolist()[:3000] for k in elig})
    json.dump(res, open(os.path.join(OUT, 'res.json'), 'w'))
    log('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
