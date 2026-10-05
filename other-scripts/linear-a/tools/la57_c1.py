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
        # LOSO
        fold = {}
        for h in elig:
            tr = [k for k in elig if k != h]
            rows_h, Ys_h = L[h]
            yh = Ys_h[0]
            Xh = sysd[h]['X'][rows_h]
            out = []
            for rep in range(NNULL + 1):
                At = np.array([A[k][rep] for k in tr])
                sel = np.where((At >= THR).all(0))[0]
                if len(sel) > 2000:
                    sel = sel[np.argsort(-At[:, sel].min(0))[:2000]]
                if len(sel) == 0:
                    out.append((0, float('nan'), float('nan')))
                    continue
                S = score(Xh, [H[i] for i in sel])
                ens = rankdata(S, axis=0).mean(1)
                out.append((len(sel), C.auc(ens, yh), float(np.mean(A[h][0][sel] >= THR))))
            real = out[0]
            nulls = [o[1] for o in out[1:] if o[0] > 0]
            nn0 = sum(1 for o in out[1:] if o[0] == 0)
            p = (1 + sum(1 for v in nulls if not np.isnan(real[1]) and v >= real[1])) / (1 + len(nulls)) if nulls else float('nan')
            fold[h] = dict(nsel=real[0], auc=real[1], transfer=real[2], null_auc_med=float(np.median(nulls)) if nulls else None,
                           null_auc_max=float(max(nulls)) if nulls else None, null_nsel_med=float(np.median([o[0] for o in out[1:]])),
                           null_empty=nn0, p=p)
            log('%s held-out %-4s survivors %5d  ens AUC %.3f  share>=thr %.2f | null: n_sel med %.0f empty %d, AUC med %s max %s | P %.3f' % (
                role, h, real[0], real[1], real[2], fold[h]['null_nsel_med'], nn0,
                '%.3f' % fold[h]['null_auc_med'] if nulls else '-', '%.3f' % fold[h]['null_auc_max'] if nulls else '-', p))
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
