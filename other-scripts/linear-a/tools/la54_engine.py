#!/usr/bin/env python3
"""LA-54 engine: thousands of random classifiers on random feature sets, scored on masked held-out docs."""
import warnings, math, random, collections
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import la54_common as C

FAMS = ['ridge', 'knn', 'nb', 'cent']


def random_configs(n, F, rng):
    cfgs = []
    for i in range(n):
        while True:
            gs = [g for g in C.GROUPS if rng.random() < 0.5]
            if gs: break
        keep = rng.uniform(0.4, 1.0)
        cols = [j for j, g in enumerate(F.group) if g in gs and rng.random() < keep]
        if not cols:
            cols = [j for j, g in enumerate(F.group) if g in gs]
        fam = rng.choice(FAMS)
        hp = {}
        if fam == 'ridge': hp['lam'] = 10 ** rng.uniform(-1, 3); hp['t'] = 10 ** rng.uniform(-1.3, 0.3)
        if fam == 'nb': hp['vs'] = 10 ** rng.uniform(-2, 0)
        if fam == 'knn': hp['k'] = rng.choice([3, 5, 7, 9, 13, 17]); hp['w'] = rng.choice(['uniform', 'distance'])
        if fam == 'rf': hp['d'] = rng.choice([3, 5, 8, None]); hp['leaf'] = rng.choice([1, 2, 4]); hp['mf'] = rng.choice(['sqrt', 0.3, 0.6])
        if fam == 'cent': hp['t'] = 10 ** rng.uniform(-1, 1)
        cfgs.append({'i': i, 'groups': gs, 'cols': cols, 'fam': fam, 'hp': hp, 'seed': rng.randrange(1 << 30)})
    return cfgs


class Centroid:
    """Softmax over negative standardized distance to class centroids (temperature t)."""
    def __init__(self, t=1.0): self.t = t
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X); Z = self.sc.transform(X)
        self.classes_ = np.unique(y)
        self.mu = np.array([Z[y == c].mean(0) for c in self.classes_]); return self
    def predict_proba(self, X):
        Z = self.sc.transform(X)
        d = ((Z[:, None, :] - self.mu[None]) ** 2).mean(2)
        a = -d / self.t; a -= a.max(1, keepdims=True); p = np.exp(a)
        return p / p.sum(1, keepdims=True)


def build(cfg):
    f, hp, s = cfg['fam'], cfg['hp'], cfg['seed']
    if f == 'lr': return make_pipeline(StandardScaler(), LogisticRegression(C=hp['C'], max_iter=400))
    if f == 'knn': return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=hp['k'], weights=hp['w']))
    if f == 'nb': return GaussianNB(var_smoothing=1e-3)
    if f == 'rf': return RandomForestClassifier(n_estimators=40, max_depth=hp['d'], min_samples_leaf=hp['leaf'], max_features=hp['mf'], random_state=s, n_jobs=1)
    if f == 'et': return ExtraTreesClassifier(n_estimators=40, max_depth=hp['d'], min_samples_leaf=hp['leaf'], max_features=hp['mf'], random_state=s, n_jobs=1)
    if f == 'cent': return Centroid(hp['t'])


def _std(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _softmax(a):
    a = a - a.max(1, keepdims=True); p = np.exp(a); return p / p.sum(1, keepdims=True)


def proba(cfg, Xtr, ytr, Xte, classes):
    cols = cfg['cols']; f = cfg['fam']; hp = cfg['hp']
    A, B = Xtr[:, cols], Xte[:, cols]
    K = len(classes)
    yi = np.array([classes.index(c) for c in ytr])
    Y = np.eye(K)[yi]
    present = Y.sum(0) > 0
    if f == 'ridge':
        Za, Zb = _std(A, B)
        Za = np.hstack([Za, np.ones((len(Za), 1))]); Zb = np.hstack([Zb, np.ones((len(Zb), 1))])
        W = np.linalg.solve(Za.T @ Za + hp['lam'] * np.eye(Za.shape[1]), Za.T @ (Y - Y.mean(0)))
        p = _softmax((Zb @ W) / hp['t'] + np.log(Y.mean(0) + 1e-3))
    elif f == 'knn':
        Za, Zb = _std(A, B)
        d = ((Zb[:, None, :] - Za[None]) ** 2).sum(2)
        k = min(hp['k'], len(Za))
        nn = np.argsort(d, 1)[:, :k]
        w = np.ones_like(nn, float) if hp['w'] == 'uniform' else 1.0 / (np.take_along_axis(d, nn, 1) + 1e-3)
        p = np.zeros((len(B), K))
        for j in range(k): p[np.arange(len(B)), yi[nn[:, j]]] += w[:, j]
        p += 0.5 * Y.mean(0)
    elif f == 'nb':
        mu = np.array([A[yi == c].mean(0) if (yi == c).any() else A.mean(0) for c in range(K)])
        var = np.array([A[yi == c].var(0) if (yi == c).sum() > 1 else A.var(0) for c in range(K)]) + hp['vs'] * (A.var(0).max() + 1e-6)
        ll = -0.5 * (((B[:, None, :] - mu[None]) ** 2) / var[None] + np.log(var[None])).sum(2)
        p = _softmax(ll / max(1.0, len(cols) / 10.0) + np.log(Y.mean(0) + 1e-3))
    elif f == 'cent':
        Za, Zb = _std(A, B)
        mu = np.array([Za[yi == c].mean(0) if (yi == c).any() else np.zeros(Za.shape[1]) for c in range(K)])
        d = ((Zb[:, None, :] - mu[None]) ** 2).mean(2)
        p = _softmax(-d / hp['t'])
    else:
        m = RandomForestClassifier(n_estimators=30, max_depth=hp['d'], min_samples_leaf=hp['leaf'], max_features=hp['mf'], random_state=cfg['seed'], n_jobs=1)
        m.fit(A, yi); q = m.predict_proba(B)
        p = np.zeros((len(B), K)); p[:, m.classes_] = q
    p = p * present + 0.0
    p = p + 1e-3
    return p / p.sum(1, keepdims=True)


def inner_scores(cfgs, X, y, classes, rng, nfold=4):
    """Inner CV accuracy and log-loss of every config on the training part."""
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=rng.randrange(1 << 30))
    folds = list(skf.split(X, y))
    yi = np.array([classes.index(c) for c in y])
    res = []
    for cfg in cfgs:
        P = np.zeros((len(y), len(classes)))
        for tr, te in folds:
            P[te] = proba(cfg, X[tr], y[tr], X[te], classes)
        acc = float((P.argmax(1) == yi).mean())
        ll = float(-np.log(P[np.arange(len(y)), yi]).mean())
        res.append((acc, ll))
    return res


def metrics(P, y, classes, prior):
    yi = np.array([classes.index(c) for c in y])
    pred = P.argmax(1)
    acc = float((pred == yi).mean())
    rec = [float((pred[yi == k] == k).mean()) for k in range(len(classes)) if (yi == k).any()]
    bacc = float(np.mean(rec))
    ll = float(-np.log2(P[np.arange(len(y)), yi]).mean())
    llp = float(-np.log2(np.array([prior[c] for c in yi])).mean())
    return {'acc': acc, 'bacc': bacc, 'bits_gain': llp - ll, 'n': len(y)}


def pipeline(docs, F, cfgs, rng, n_outer=4, topk=25, test_frac=0.3, X=None, return_detail=False):
    """docs: labelled docs. Repeated stratified holdouts; random search on the training part only."""
    if X is None: X = F.X(docs)
    y = np.array([d['label'] for d in docs])
    classes = sorted(set(y))
    out = []
    det = []
    for r in range(n_outer):
        idx = list(range(len(docs)))
        # stratified split
        bycls = collections.defaultdict(list)
        for i in idx: bycls[y[i]].append(i)
        te = []
        for c, ii in bycls.items():
            rng.shuffle(ii); te += ii[:max(1, int(round(len(ii) * test_frac)))]
        tes = set(te); tr = [i for i in idx if i not in tes]
        tr, te = np.array(tr), np.array(sorted(te))
        sc = inner_scores(cfgs, X[tr], y[tr], classes, rng)
        order = sorted(range(len(cfgs)), key=lambda k: (-sc[k][0], sc[k][1]))[:topk]
        P = np.mean([proba(cfgs[k], X[tr], y[tr], X[te], classes) for k in order], 0)
        cnt = collections.Counter(y[tr])
        prior = np.array([(cnt[c] + 0.5) / (len(tr) + 0.5 * len(classes)) for c in classes])
        m = metrics(P, y[te], classes, prior)
        m['maj'] = float(max((y[te] == c).mean() for c in classes))
        m['inner_best'] = sc[order[0]][0]
        out.append(m)
        if return_detail:
            det.append({'sc': sc, 'top': order, 'te': te.tolist(), 'P': P.tolist(), 'classes': classes})
    agg = {k: float(np.mean([m[k] for m in out])) for k in out[0]}
    if return_detail:
        return agg, out, det
    return agg, out


def group_importance(cfgs, sc):
    """Mean inner accuracy of configs with vs without each feature group."""
    res = {}
    for g in C.GROUPS:
        a = [sc[k][0] for k, c in enumerate(cfgs) if g in c['groups']]
        b = [sc[k][0] for k, c in enumerate(cfgs) if g not in c['groups']]
        res[g] = (float(np.mean(a)) - float(np.mean(b))) if a and b else 0.0
    return res
