"""LA-17 ABC analysis: rejection ABC + ABC random forests, with planted, shuffled and
Linear B controls; inferred order scored afterwards against deposit dates and geography."""
import glob, os, sys, json
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from la17_common import *


def load_sims(ds, mode):
    fs = sorted(glob.glob(os.path.join(OUT, f'sims_{ds}_{mode}_w*_*.npz')))
    fs = [f for f in fs if not f.endswith('.tmp.npz')]
    T = np.concatenate([np.load(f)['theta'] for f in fs]); S = np.concatenate([np.load(f)['stats'] for f in fs])
    return T, S


def split_theta(T, K):
    return T[:, 0].astype(int), T[:, 1:1 + K], T[:, 1 + K:1 + 2 * K]


NJ = int(os.environ.get("NJ", "2"))

class ABC:
    def __init__(self, T, S, K, target='t', ntrain=None, seed=0):
        self.K = K
        self.src, self.t, self.tau = split_theta(T, K)
        self.S = S
        self.med = np.median(S, 0); self.mad = np.median(np.abs(S - self.med), 0) + 1e-6
        self.Z = (S - self.med) / self.mad
        rng = np.random.default_rng(seed)
        n = len(S); idx = rng.permutation(n)
        self.test = idx[:1000]; tr = idx[1000:] if ntrain is None else idx[1000:1000 + ntrain]
        self.tr = tr
        y = self.t if target == 't' else self.tau
        self.y = y
        self.clf = RandomForestClassifier(120, min_samples_leaf=5, max_features='sqrt', max_samples=0.5, n_jobs=NJ,
                                          random_state=seed).fit(S[tr], self.src[tr])
        self.reg = RandomForestRegressor(80, min_samples_leaf=10, max_features=0.33, max_samples=0.5, n_jobs=NJ,
                                         random_state=seed).fit(S[tr], y[tr])

    def reject(self, s, q=0.002):
        z = (s - self.med) / self.mad
        dist = np.sqrt(((self.Z[self.tr] - z) ** 2).sum(1))
        k = max(50, int(q * len(self.tr)))
        a = self.tr[np.argsort(dist)[:k]]
        ps = np.bincount(self.src[a], minlength=self.K) / k
        ranks = np.argsort(np.argsort(self.y[a], 1), 1) + 1  # rank of each site's time
        return ps, ranks

    def rf(self, s):
        return self.clf.predict_proba(s[None])[0], self.reg.predict(s[None])[0]

    def heldout(self):
        """planted outbreaks drawn from the prior, never seen in training"""
        P = self.clf.predict_proba(self.S[self.test]); yh = self.reg.predict(self.S[self.test])
        acc = np.mean(P.argmax(1) == self.src[self.test])
        cover = np.mean([np.sort(p)[::-1].cumsum().searchsorted(0.9) + 1 >= 0 and
                         self.src[i] in np.argsort(p)[::-1][:np.sort(p)[::-1].cumsum().searchsorted(0.9) + 1]
                         for p, i in zip(P, self.test)])
        rho = np.nanmedian([spearmanr(a, b)[0] for a, b in zip(yh, self.y[self.test])])
        return acc, cover, rho


def fmt_p(ps, codes, n=4):
    o = np.argsort(ps)[::-1][:n]
    return ', '.join(f'{codes[i]} {ps[i]:.2f}' for i in o)


def order_str(th, codes):
    return ' < '.join(codes[i] for i in np.argsort(th))


def date_match(th, codes, dates, nperm=20000, seed=0):
    x = np.array([th[i] for i, c in enumerate(codes) if c in dates])
    y = np.array([dates[c] for c in codes if c in dates])
    r = spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(nperm)])
    return r, np.mean(null >= r)
