#!/usr/bin/env python3
"""LA-58 ABC-RF: random forests trained on a simulation bank read off world type, site roles and parent links
for real, shuffled, planted and control corpora.  Calibration on held-out simulations."""
import sys, os, json, pickle, random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la58_common import *
from la58_run import corpus, nk_of, load_bank

ROLE = ['I', 'C', 'D']


def targets(T, K):
    typ = T[:, 0].astype(int)
    role = T[:, 2:2 + K].astype(int)
    par = T[:, 2 + K:2 + 2 * K].astype(int); par[par < 0] = K
    return typ, role, par


class Multi:
    """Three separate forests (type; roles; parents) to keep memory small."""
    def __init__(self, ntree, leaf, rs):
        mk = lambda: RandomForestClassifier(n_estimators=ntree, min_samples_leaf=leaf, max_features='sqrt',
                                            n_jobs=2, random_state=rs)
        self.f = [mk(), mk(), mk()]

    def fit(self, X, typ, role, par):
        self.f[0].fit(X, typ); self.f[1].fit(X, role); self.f[2].fit(X, par)
        self.classes_ = [self.f[0].classes_] + list(self.f[1].classes_) + list(self.f[2].classes_)
        return self

    def predict_proba(self, X):
        p1 = self.f[1].predict_proba(X); p2 = self.f[2].predict_proba(X)
        return [self.f[0].predict_proba(X)] + list(p1) + list(p2)


def train(tag, ntest=4000, ntree=100, leaf=10, rs=0):
    S, T, meta = load_bank(tag)
    K = meta['K']
    S = np.nan_to_num(S, nan=-2)
    typ, role, par = targets(T, K)
    n = len(S); idx = np.random.RandomState(rs).permutation(n)
    te, tr = idx[:ntest], idx[ntest:]
    rf = Multi(ntree, leaf, rs).fit(S[tr], typ[tr], role[tr], par[tr])
    P = rf.predict_proba(S[te])
    res = dict(K=K, abbr=meta['abbr'], ntrain=len(tr))
    pt = P[0]; res['type_acc'] = float((rf.classes_[0][pt.argmax(1)] == typ[te]).mean())
    res['type_calib'] = calib(pt, typ[te], rf.classes_[0])
    # confusion of world types (rows truth, cols predicted)
    cm = np.zeros((6, 6), int)
    for t, q in zip(typ[te], rf.classes_[0][pt.argmax(1)]):
        cm[int(t), int(q)] += 1
    res['type_confusion'] = cm.tolist()
    aucs, cal = [], []
    for k in range(K):
        pc = proba_of(P[1 + k], rf.classes_[1 + k], 1)
        aucs.append(auc(pc, role[te, k] == 1))
        cal.append((pc, role[te, k] == 1))
    res['centre_auc'] = [float(a) for a in aucs]
    pcs = np.concatenate([c[0] for c in cal]); ys = np.concatenate([c[1] for c in cal])
    res['centre_calib'] = bins(pcs, ys)
    acc = []
    for k in range(K):
        m = role[te, k] == 2
        if m.sum():
            pp = rf.classes_[1 + K + k][P[1 + K + k][m].argmax(1)]
            acc.append(float((pp == par[te, k][m]).mean()))
    res['parent_acc'] = acc
    return rf, res


def proba_of(p, classes, c):
    w = np.where(classes == c)[0]
    return p[:, w[0]] if len(w) else np.zeros(len(p))


def auc(s, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bins(p, y, edges=(0, .1, .3, .5, .7, .9, 1.01)):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            out.append([round(a, 2), round(b, 2), int(m.sum()), round(float(p[m].mean()), 3), round(float(y[m].mean()), 3)])
    return out


def calib(pt, typ, classes):
    p = pt.max(1); y = classes[pt.argmax(1)] == typ
    return bins(p, y)


def read(rf, s, K):
    s = np.nan_to_num(np.asarray(s, float)[None, :], nan=-2)
    P = rf.predict_proba(s)
    out = dict(type={TYPES[int(c)]: round(float(p), 3) for c, p in zip(rf.classes_[0], P[0][0])})
    out['centre'] = [round(float(proba_of(P[1 + k], rf.classes_[1 + k], 1)[0]), 3) for k in range(K)]
    out['dependent'] = [round(float(proba_of(P[1 + k], rf.classes_[1 + k], 2)[0]), 3) for k in range(K)]
    out['parent'] = []
    for k in range(K):
        cl = rf.classes_[1 + K + k]; pr = P[1 + K + k][0]
        out['parent'].append({int(c): round(float(p), 3) for c, p in zip(cl, pr) if p >= 0.05})
    return out
