#!/usr/bin/env python3
"""LA-69 cycle 1 readout: per-feature calibration, length-corrected."""
import os, sys, pickle, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la69_common as C

R = pickle.load(open(os.path.join(C.CK, 'c1_feats.pkl'), 'rb'))
F = C.FEATS
CAL = ['LB_shape', 'LB_draft', 'UR3', 'OA']


def mat(rows):
    return np.array([[r[3][k] for k in F] for r in rows], float), np.array([[r[3]['_lt'], r[3]['_ll']] for r in rows])


def fit_len(rows):
    X, L = mat(rows)
    A = np.c_[np.ones(len(L)), L]
    B = []
    for j in range(len(F)):
        m = ~np.isnan(X[:, j])
        B.append(np.linalg.lstsq(A[m], X[m, j], rcond=None)[0])
    return np.array(B)


def resid(rows, B):
    X, L = mat(rows)
    A = np.c_[np.ones(len(L)), L]
    return X - A @ B.T


def auc(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    allv = np.r_[a, b]
    r = np.argsort(np.argsort(allv, kind='mergesort'), kind='mergesort') + 1.0
    # average ties
    from scipy.stats import rankdata
    r = rankdata(allv)
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def perm_p(a, b, obs, rng, n=500):
    allv = np.r_[a, b]; k = len(a); c = 0
    for _ in range(n):
        rng.shuffle(allv)
        if abs(auc(allv[:k], allv[k:]) - 0.5) >= abs(obs - 0.5) - 1e-12:
            c += 1
    return (c + 1) / (n + 1)


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    calrows = [r for r in R if r[0] in CAL and r[1] in ('NOTE', 'FINAL')]
    B = fit_len(calrows)
    print('sizes (median nT):')
    for c in CAL + ['LA', 'OB', 'PC', 'KH']:
        for cl in ('NOTE', 'FINAL', 'PLANT', 'LA', 'LA_PLANT', 'ARCH'):
            rr = [r for r in R if r[0] == c and r[1] == cl]
            if rr:
                print(' ', c, cl, len(rr), np.median([r[2]['nT'] for r in rr]))
    out = {}
    print('\nAUC(FINAL > NOTE) in sign-corrected, length-residualised features; P by label permutation')
    for mode in ('raw', 'res'):
        print('==', mode)
        print('%-9s' % 'feat' + ''.join('%17s' % c for c in CAL) + ''.join('%14s' % ('PL:' + c) for c in CAL))
        for j, k in enumerate(F):
            s = C.SIGN[k]
            line = '%-9s' % k
            pl = ''
            for c in CAL:
                rows = [r for r in R if r[0] == c]
                X = resid(rows, B) if mode == 'res' else mat(rows)[0]
                n = np.array([r[1] == 'NOTE' for r in rows]); f = np.array([r[1] == 'FINAL' for r in rows])
                p = np.array([r[1] == 'PLANT' for r in rows])
                a = auc(s * X[f, j], s * X[n, j])
                pp = perm_p(s * X[f, j][~np.isnan(X[f, j])], s * X[n, j][~np.isnan(X[n, j])], a, rng, 200) if not np.isnan(a) else np.nan
                ap = auc(s * X[f, j], s * X[p, j])
                out[(mode, k, c)] = (a, pp, ap)
                line += '   %5.2f (P%5.3f)' % (a, pp)
                pl += '        %5.2f' % ap
            print(line + pl)
    # placement of Linear A, feature by feature (length-residualised, sign-corrected)
    print('\nMeans of sign-corrected residual features (higher = reader-independent)')
    groups = [(c, cl) for c in CAL for cl in ('NOTE', 'FINAL', 'PLANT')] + [('LA', 'LA'), ('LA', 'LA_PLANT'), ('OB', 'ARCH'), ('PC', 'ARCH'), ('KH', 'ARCH')]
    print('%-18s' % 'set' + ''.join('%7s' % k for k in F))
    for g in groups:
        rows = [r for r in R if (r[0], r[1]) == g]
        X = resid(rows, B)
        print('%-18s' % ('%s/%s' % g) + ''.join('%7.2f' % (C.SIGN[k] * np.nanmean(X[:, j])) for j, k in enumerate(F)))
    pickle.dump({'B': B, 'auc': out}, open(os.path.join(C.CK, 'c1_calib.pkl'), 'wb'))
