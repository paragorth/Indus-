"""pe19 cycle 1: blind seriation (fit + freeze) and the two calibration controls.

Does NOT read the hidden chronology. Writes data/pe19_ckpt/frozen_*.json and
c1_fit.json (controls + hashes). Scoring is pe19_score.py.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from pe19_common import *


def seriate_all(X, elen, tag, seed=0, unimodal_restarts=4, steps=600):
    res = {}
    ca = ca_scores(X)
    sp = spectral_scores(X)
    D = jaccard_dist(X)
    tsp = tsp_order(D, np.argsort(ca))
    best = None
    inits = [ca, sp] + [np.random.default_rng(seed + r).normal(size=len(X)) for r in range(unimodal_restarts - 2)]
    for r, ini in enumerate(inits):
        x, L = unimodal_fit(X, ini, steps=steps, seed=seed + r)
        if best is None or L < best[1]:
            best = (x, L, r)
    res['CA'] = orient(ca, elen)
    res['SPEC'] = orient(sp, elen)
    res['TSP'] = orient(tsp, elen)
    res['UNI'] = orient(best[0], elen)
    R = np.mean([rankdata(v) for v in res.values()], axis=0)
    res['CONS'] = R
    return res, {'uni_loss': best[1], 'uni_init': best[2]}


def planted(X, elen, seed, n_tr=40, base=(0.03, 0.2), k=8.0):
    rng = np.random.default_rng(seed)
    n = len(X)
    t = rng.uniform(0, 1, n)
    P = []
    for j in range(n_tr):
        c = rng.uniform(0.15, 0.85); s = rng.choice([-1, 1]); b = rng.uniform(*base)
        pr = b / (1 + np.exp(-s * k * (t - c)))
        P.append(rng.uniform(size=n) < pr)
    Xp = np.hstack([X, np.array(P, np.float32).T])
    return Xp, t


def main():
    t0 = time.time()
    out = {}
    T = load_pe()
    for sets in ('BVNF', 'VNF'):
        ids, X, names, elen = build_matrix(T, sets)
        res, info = seriate_all(X, elen, sets)
        hs = {m: freeze('PE_%s_%s' % (sets, m), ids, v) for m, v in res.items()}
        out['PE_' + sets] = {'n': len(ids), 'm': len(names), 'hash': hs, 'info': info,
                             'agree_rho': {m: float(spearmanr(res[m], res['CONS'])[0]) for m in res}}
        print(sets, len(ids), len(names), hs, round(time.time() - t0), flush=True)
        # shuffled-trait nulls (columns permuted independently) -> frozen too
        for r in range(10):
            rng = np.random.default_rng(100 + r)
            Xs = np.array([rng.permutation(col) for col in X.T]).T
            ok = Xs.sum(1) >= 1
            Xs2 = Xs[ok]
            sc = orient(ca_scores(Xs2), elen[ok])
            freeze('NULLSHUF_%s_%d' % (sets, r), [ids[i] for i in np.where(ok)[0]], sc)
    json.dump(out, open(os.path.join(CK, 'c1_fit.json'), 'w'), indent=1)

    # ---- control 1: planted drift on the real PE matrix ----
    ids, X, names, elen = build_matrix(T, 'BVNF')
    pl = {}
    for strength, ntr in (('strong', 40), ('weak', 12)):
        for rep in range(2):
            Xp, t = planted(X, elen, seed=rep, n_tr=ntr)
            res, _ = seriate_all(Xp, elen, 'pl', seed=rep, unimodal_restarts=3, steps=400)
            pl['%s_%d' % (strength, rep)] = {m: abs(float(spearmanr(v, t)[0])) for m, v in res.items()}
            print('planted', strength, rep, pl['%s_%d' % (strength, rep)], flush=True)
    out['planted'] = pl

    # ---- control 2: proto-cuneiform Uruk IV vs Uruk III, Uruk only ----
    PC = load_pc()
    rng = np.random.default_rng(7)
    by = {'Uruk IV': [t for t in PC if t['period'] == 'Uruk IV'], 'Uruk III': [t for t in PC if t['period'] == 'Uruk III']}
    sub = []
    for k, v in by.items():
        sub += [v[i] for i in rng.choice(len(v), min(700, len(v)), replace=False)]
    per = {t['id']: t['period'] for t in sub}
    pids, PX, pnames, pel = build_matrix(sub, 'BVNF')
    res, info = seriate_all(PX, pel, 'pc')
    pcout = {}
    for m, v in res.items():
        pos = rankdata(v) / len(v)
        e = [pos[i] for i, p in enumerate(pids) if per[p] == 'Uruk IV']
        l = [pos[i] for i, p in enumerate(pids) if per[p] == 'Uruk III']
        pcout[m] = auc(e, l)
    # size / entry-length-only baseline for the control
    pos = rankdata(pel) / len(pel)
    pcout['ELEN_ONLY'] = auc([pos[i] for i, p in enumerate(pids) if per[p] == 'Uruk IV'],
                             [pos[i] for i, p in enumerate(pids) if per[p] == 'Uruk III'])
    pcout['n'] = len(pids); pcout['m'] = len(pnames)
    out['uruk_control'] = pcout
    print('uruk', pcout, flush=True)
    json.dump(out, open(os.path.join(CK, 'c1_fit.json'), 'w'), indent=1)
    print('done', round(time.time() - t0))


if __name__ == '__main__':
    main()
