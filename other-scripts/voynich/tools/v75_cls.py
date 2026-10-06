"""v75 random model types + leave-one-system-out (LOSO).

A model = (feature subset, layout features allowed or not, classifier type, its parameters). Classifiers (numpy):
  cent     nearest kind centroid (z-scored features)
  sysc     nearest system centroid -> its kind (kinds can be multimodal)
  knn      k nearest chunks, votes weighted 1/class size
  nb       diagonal Gaussian per kind
  lda      shrinkage LDA (pooled covariance, shrink lambda)
Open-set run (with_gen=True): classes = kinds + GEN (generators fitted to every reference system's surface).
Closed run (with_gen=False): kinds only (GEN rows dropped from training).
LOSO: hold out one reference system together with every generator fitted to it; its real chunks must come out as
its kind; its generator chunks must come out as GEN (open-set).
"""
import os, sys, pickle, random, json
import numpy as np
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_lib as X

KINDS = ['GLOSS', 'INDEX', 'CATAL', 'INGRED', 'APOTH', 'ASTRO', 'ALCH', 'NOMEN', 'SIGNS', 'MODERN', 'LANG']


def load(rep):
    D = pickle.load(open(os.path.join(X.CK, '%s_%s.pkl' % (os.environ.get('V75_FEATS', 'feats'), rep)), 'rb'))
    rows = D['rows']
    F = np.array([r['feats'] for r in rows], float)
    F[~np.isfinite(F)] = 0
    return D['feats'], rows, F


def random_model(rng, feats):
    lay = rng.random() < 0.5
    pool = [i for i, f in enumerate(feats) if lay or f not in X.LAYOUT]
    k = rng.randint(3, min(16, len(pool)))
    sub = sorted(rng.sample(pool, k))
    typ = rng.choice(['cent', 'sysc', 'knn', 'nb', 'lda'])
    par = {}
    if typ == 'knn': par['k'] = rng.choice([1, 3, 5, 9, 15])
    if typ == 'lda': par['lam'] = rng.choice([0.05, 0.2, 0.5, 0.9])
    return dict(sub=sub, typ=typ, layout=lay, **par)


class Fitted:
    def __init__(self, m, Xtr, ytr, gtr):
        self.m = m; s = m['sub']
        A = Xtr[:, s]; self.mu = A.mean(0); self.sd = A.std(0) + 1e-9
        Z = (A - self.mu) / self.sd
        self.classes = sorted(set(ytr)); self.y = np.array(ytr); self.Z = Z
        if m['typ'] in ('cent', 'nb', 'lda'):
            self.C = np.array([Z[self.y == c].mean(0) for c in self.classes])
            self.V = np.array([Z[self.y == c].var(0) + 0.05 for c in self.classes])
            if m['typ'] == 'lda':
                R = np.concatenate([Z[self.y == c] - Z[self.y == c].mean(0) for c in self.classes])
                S = np.cov(R.T) if R.shape[1] > 1 else np.array([[R.var()]])
                S = np.atleast_2d(S); S = (1 - m['lam']) * S + m['lam'] * np.eye(len(s))
                self.Si = np.linalg.pinv(S)
        if m['typ'] == 'sysc':
            g = np.array([a + '|' + b for a, b in zip(gtr, ytr)]); self.sysk = []; cs = []
            for b in sorted(set(g)):
                cs.append(Z[g == b].mean(0)); self.sysk.append(self.y[g == b][0])
            self.SC = np.array(cs)
        if m['typ'] == 'knn':
            cnt = Counter(ytr); self.w = np.array([1.0 / cnt[c] for c in ytr])

    def scores(self, Xte):
        """return (n, n_classes) scores, higher = closer."""
        Z = (Xte[:, self.m['sub']] - self.mu) / self.sd; t = self.m['typ']
        if t == 'cent':
            return -((Z[:, None, :] - self.C[None]) ** 2).sum(-1)
        if t == 'nb':
            return -0.5 * (((Z[:, None, :] - self.C[None]) ** 2) / self.V[None] + np.log(self.V[None])).sum(-1)
        if t == 'lda':
            D = Z[:, None, :] - self.C[None]
            return -np.einsum('nkd,de,nke->nk', D, self.Si, D)
        if t == 'sysc':
            d = ((Z[:, None, :] - self.SC[None]) ** 2).sum(-1)
            out = np.full((len(Z), len(self.classes)), -1e18)
            for j, c in enumerate(self.classes):
                idx = [i for i, k in enumerate(self.sysk) if k == c]
                out[:, j] = -d[:, idx].min(1)
            return out
        if t == 'knn':
            d = ((Z[:, None, :] - self.Z[None]) ** 2).sum(-1)
            k = self.m['k']; nn = np.argsort(d, 1)[:, :k]
            out = np.zeros((len(Z), len(self.classes)))
            ci = {c: j for j, c in enumerate(self.classes)}
            yi = np.array([ci[c] for c in self.y])
            for i in range(len(Z)):
                for j in nn[i]: out[i, yi[j]] += self.w[j]
            return out
        raise ValueError(t)

    def predict(self, Xte):
        s = self.scores(Xte)
        return [self.classes[j] for j in s.argmax(1)], s


def ref_mask(rows):
    return np.array([r['kind'] != '?' for r in rows])


def loso(m, rows, F, with_gen=True, bases=None):
    """returns per-base dict(acc_kind, gen_as_gen, gen_as_kind)"""
    R = [i for i, r in enumerate(rows) if r['kind'] != '?']
    base = np.array([rows[i]['base'] for i in R]); kind = [rows[i]['kind'] for i in R]
    role = [rows[i]['role'] for i in R]; FF = F[R]
    out = {}
    for b in (bases or sorted(set(base))):
        tr = [j for j in range(len(R)) if base[j] != b and (with_gen or kind[j] != 'GEN')]
        te = [j for j in range(len(R)) if base[j] == b]
        f = Fitted(m, FF[tr], [kind[j] for j in tr], [base[j] for j in tr])
        pred, _ = f.predict(FF[te])
        tks = [kind[j] for j in te if role[j] == 'real']
        if not tks or tks[0] not in {kind[j] for j in tr}: continue
        tk = tks[0]
        pr = [p for p, j in zip(pred, te) if role[j] == 'real']
        pg = [p for p, j in zip(pred, te) if role[j] != 'real']
        out[b] = dict(kind=tk, acc=np.mean([p == tk for p in pr]),
                      gen_as_gen=np.mean([p == 'GEN' for p in pg]) if pg else np.nan,
                      gen_as_kind=np.mean([p not in ('GEN', 'LANG') for p in pg]) if pg else np.nan,
                      gen_as_own=np.mean([p == tk for p in pg]) if pg else np.nan)
    return out


def summarize(L):
    byk = defaultdict(list)
    for b, d in L.items(): byk[d['kind']].append(d['acc'])
    kacc = {k: float(np.mean(v)) for k, v in byk.items()}
    return dict(kind_acc=float(np.mean(list(kacc.values()))), per_kind=kacc,
                gen_as_gen=float(np.nanmean([d['gen_as_gen'] for d in L.values()])),
                gen_as_own=float(np.nanmean([d['gen_as_own'] for d in L.values()])))


def readout(m, rows, F, with_gen=True):
    """train on all references (+ GEN), predict every Voynich-side chunk; returns {(sid, role): Counter}"""
    tr = [i for i, r in enumerate(rows) if r['kind'] != '?' and (with_gen or r['kind'] != 'GEN')]
    te = [i for i, r in enumerate(rows) if r['kind'] == '?']
    f = Fitted(m, F[tr], [rows[i]['kind'] for i in tr], [rows[i]['base'] for i in tr])
    pred, sc = f.predict(F[te])
    out = defaultdict(Counter); rank = defaultdict(list)
    for p, i, s in zip(pred, te, sc):
        r = rows[i]; key = (r['base'], r['role'])
        out[key][p] += 1
        order = [f.classes[j] for j in np.argsort(-s)]
        rank[key].append(order)
    return out, rank
