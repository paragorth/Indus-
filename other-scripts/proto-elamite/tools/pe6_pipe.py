"""pe6: ABC fit -> posterior simulations -> posterior role assignment of signs.  Shared pipeline."""
import os, sys, math, random, json
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe6_common import (simulate, stats, sign_features, from_unbounded, to_unbounded, abc_fit, PNAMES, PRIOR,
                        STATS, ROLES, FEATS, PEDATA)
from pe6_bank import load_bank


def _sim_feat(args):
    u, seed, n_tab, want_stats, min_n = args
    th = from_unbounded(u)
    C, role = simulate(th, seed, n_tab)
    F = sign_features(C, min_n)
    X = np.array([F[x] for x in F]) if F else np.zeros((0, len(FEATS)))
    y = [role[x] for x in F]
    return X, y, (stats(C) if want_stats else None)


def post_sims(adj, n, seed, n_tab=1585, want_stats=False, min_n=15, workers=2):
    r = np.random.default_rng(seed)
    rows = r.integers(len(adj), size=n)
    args = [(adj[i], int(seed * 100003 + k), n_tab, want_stats, min_n) for k, i in enumerate(rows)]
    with Pool(workers) as p:
        res = p.map(_sim_feat, args, chunksize=4)
    return res


def fit(sobs, bank=None, frac=0.01, use=None):
    seeds, U, RAW, S = bank if bank is not None else load_bank()
    ok = np.isfinite(S).all(1)
    adj, idx, d, mad = abc_fit(U[ok], S[ok], sobs, frac=frac, use=use)
    return adj, idx, d


def summarize_post(adj):
    out = {}
    TH = [from_unbounded(a) for a in adj]
    for p in PNAMES:
        v = np.array([t[p] for t in TH], float)
        out[p] = (float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95)))
    return out


def train_assign(res, Fobs, n_models=5, seed=0):
    """res: list of (X, y, _) from posterior sims.  Fobs: dict sign -> feature vector.
    Trains n_models random forests on disjoint groups of simulations; returns
    probs (sign -> mean prob over roles), sd, held-out calibration on the other groups."""
    from sklearn.ensemble import RandomForestClassifier
    groups = np.array_split(np.arange(len(res)), n_models)
    signs = sorted(Fobs); Xo = np.array([Fobs[s] for s in signs])
    P = []
    calib = []
    for gi, g in enumerate(groups):
        X = np.vstack([res[i][0] for i in g if len(res[i][1])]); y = sum([res[i][1] for i in g], [])
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed + gi)
        clf.fit(X, y)
        pr = np.zeros((len(signs), len(ROLES)))
        pp = clf.predict_proba(Xo)
        for j, c in enumerate(clf.classes_):
            pr[:, ROLES.index(c)] = pp[:, j]
        P.append(pr)
        # held-out calibration on the next group
        h = groups[(gi + 1) % n_models]
        Xh = np.vstack([res[i][0] for i in h if len(res[i][1])]); yh = sum([res[i][1] for i in h], [])
        ph = clf.predict_proba(Xh)
        top = ph.argmax(1); conf = ph.max(1)
        pred = clf.classes_[top]
        for c_, pr_, t_ in zip(conf, pred, yh):
            calib.append((float(c_), pr_, t_))
    P = np.array(P)
    mean = P.mean(0); sd = P.std(0)
    out = {s: (mean[i], sd[i]) for i, s in enumerate(signs)}
    cal = {}
    for lo in (0.5, 0.7, 0.9):
        sel = [(p, t) for c, p, t in calib if c >= lo]
        cal[lo] = (len(sel) / max(1, len(calib)), float(np.mean([p == t for p, t in sel])) if sel else float('nan'))
    per_role = {}
    for r in ROLES:
        sel = [(p, t) for c, p, t in calib if c >= 0.7 and p == r]
        per_role[r] = (len(sel), float(np.mean([p == t for p, t in sel])) if sel else float('nan'))
    return out, cal, per_role


def grade_table(assign, min_p=(0.9, 0.7, 0.5)):
    rows = []
    for s, (m, sd) in assign.items():
        k = int(np.argmax(m)); p = float(m[k])
        g = 'A' if p >= min_p[0] else 'B' if p >= min_p[1] else 'C' if p >= min_p[2] else '-'
        rows.append((s, ROLES[k], p, float(sd[k]), g))
    rows.sort(key=lambda r: -r[2])
    return rows


def pipeline(C, bank, nsim=200, seed=1, frac=0.01, use=None, n_tab=None, min_n=15):
    sobs = stats(C)
    adj, idx, d = fit(sobs, bank, frac=frac, use=use)
    res = post_sims(adj, nsim, seed, n_tab=n_tab or len(C), min_n=min_n)
    Fobs = sign_features(C, min_n)
    assign, cal, per_role = train_assign(res, Fobs, seed=seed)
    return dict(sobs=sobs, adj=adj, idx=idx, d=d, assign=assign, cal=cal, per_role=per_role,
                post=summarize_post(adj))
