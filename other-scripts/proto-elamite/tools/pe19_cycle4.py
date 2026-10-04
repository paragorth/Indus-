"""pe19 cycle 4: CHOOSE THE SERIATOR ON PLANTS, THEN LOOK. Within-content-cluster CA.

Cycles 1-3: no method recovered a planted drift; content dominates. Here the
content is clustered blindly (k-means on CA axes 1-6, K = 4, 8, 16, 32) and the
cluster dummies are partialled out of the CA residual matrix (time should cut
ACROSS content clusters). Candidate seriators (axis 1 or 2 of the partial CA,
plain CA, plant-only sanity) are ranked by recovery of planted drifts at three
strengths (3 seeds each; randomized SVD, 8 axes); only the winner is applied to the real data, frozen,
hashed and scored. The Uruk IV/III control is run with the winner.
Never reads the hidden labels.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from pe19_common import *
from pe19_cycle1 import planted
from sklearn.utils.extmath import randomized_svd as rsvd


def ca_resid(X):
    P = X / X.sum(); r = P.sum(1); c = P.sum(0)
    keep = c > 0
    P, c = P[:, keep], c[keep]
    return (P - np.outer(r, c)) / np.sqrt(np.outer(r, c)), r


def kmeans(Y, K, seed):
    from sklearn.cluster import KMeans
    return KMeans(K, n_init=4, random_state=seed).fit_predict(Y)


def partial_ca(X, K, axis=0, seed=0):
    S, r = ca_resid(X)
    U, s, Vt = rsvd(S, 8, random_state=seed)
    if K <= 1:
        return U[:, axis] / np.sqrt(r)
    rows = U[:, :6] * s[:6] / np.sqrt(r)[:, None]
    lab = kmeans(rows, K, seed)
    Z = np.eye(K)[lab] * np.sqrt(r)[:, None]
    beta, *_ = np.linalg.lstsq(Z, S, rcond=None)
    U2, s2, _ = rsvd(S - Z @ beta, 4, random_state=seed)
    return U2[:, axis] / np.sqrt(r)


def main():
    t0 = time.time()
    T = load_pe()
    ids, X, names, elen = build_matrix(T, 'BVNF')
    cands = [(K, ax) for K in (1, 4, 8, 16, 32) for ax in (0, 1)]
    cal = {}
    for (ntr, k) in ((40, 8.0), (100, 8.0), (200, 15.0)):
        for rep in range(3):
            Xp, t = planted(X, elen, seed=1000 + rep, n_tr=ntr, k=k)
            for K, ax in cands:
                sc = partial_ca(Xp, K, ax, seed=rep)
                cal.setdefault('K%d_ax%d' % (K, ax), {}).setdefault('n%d' % ntr, []).append(abs(float(spearmanr(sc, t)[0])))
            if rep == 0:  # sanity: planted traits alone
                cal.setdefault('PLANT_ONLY', {}).setdefault('n%d' % ntr, []).append(
                    abs(float(spearmanr(ca_scores(Xp[:, X.shape[1]:][Xp[:, X.shape[1]:].sum(1) > 0]),
                                        t[Xp[:, X.shape[1]:].sum(1) > 0])[0])))
        print('cal', ntr, {c: round(np.mean(v['n%d' % ntr]), 3) for c, v in cal.items()}, round(time.time() - t0), flush=True)
    summ = {c: float(np.mean([np.mean(x) for x in v.values()])) for c, v in cal.items() if c != 'PLANT_ONLY'}
    win = max(summ, key=summ.get)
    K, ax = int(win.split('_')[0][1:]), int(win[-1])
    print('winner', win, summ[win], flush=True)
    sc = orient(partial_ca(X, K, ax, seed=0), elen)
    h = freeze('C4_WIN_%s' % win, ids, sc)
    # stability of the winner across k-means seeds (blind)
    stab = [abs(float(spearmanr(partial_ca(X, K, ax, seed=s), sc)[0])) for s in (1, 2, 3)]
    # Uruk control with the winner
    PC = load_pc(); rng = np.random.default_rng(7)
    by = {'Uruk IV': [t for t in PC if t['period'] == 'Uruk IV'], 'Uruk III': [t for t in PC if t['period'] == 'Uruk III']}
    sub = []
    for kk, v in by.items():
        sub += [v[i] for i in rng.choice(len(v), min(700, len(v)), replace=False)]
    per = {t['id']: t['period'] for t in sub}
    pids, PX, pn, pel = build_matrix(sub, 'BVNF')
    uc = {}
    for c in ('K1_ax0', win):
        Kc, axc = int(c.split('_')[0][1:]), int(c[-1])
        v = orient(partial_ca(PX, Kc, axc), pel)
        pos = rankdata(v) / len(v)
        a = auc([pos[i] for i, p in enumerate(pids) if per[p] == 'Uruk IV'], [pos[i] for i, p in enumerate(pids) if per[p] == 'Uruk III'])
        uc[c] = {'auc_oriented': a, 'unoriented': max(a, 1 - a)}
    out = {'calibration': cal, 'cal_mean': summ, 'winner': win, 'hash': h, 'seed_stability': stab, 'uruk': uc}
    json.dump(out, open(os.path.join(CK, 'c4_fit.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'calibration'}), 'done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
