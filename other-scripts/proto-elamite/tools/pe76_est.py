"""pe76 estimator: ABC (rejection + local-linear adjustment) and random-forest ABC on a bank."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, collections
from sklearn.ensemble import RandomForestRegressor
import pe76_common as P, common


class Est:
    def __init__(self, TH, ST, seed=0, n_rf=300):
        self.TH, self.ST = TH, ST
        self.med = np.median(ST, 0); self.mad = np.median(np.abs(ST - self.med), 0) + 1e-9
        self.Z = (ST - self.med) / self.mad
        self.rf = RandomForestRegressor(n_estimators=n_rf, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=seed, oob_score=True)
        self.rf.fit(ST, TH[:, 0])

    def rej(self, s, q=0.01, col=0):
        z = (s - self.med) / self.mad
        d = np.sqrt(((self.Z - z) ** 2).sum(1))
        k = max(50, int(q * len(d)))
        idx = np.argsort(d)[:k]
        y = self.TH[idx, col]; Xz = self.Z[idx] - z
        h = d[idx].max(); w = 1 - (d[idx] / h) ** 2
        A = np.c_[np.ones(k), Xz]
        beta = np.linalg.lstsq(A * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
        yadj = y - Xz @ beta[1:]
        return dict(raw_med=float(np.median(y)), adj_med=float(np.average(yadj, weights=w)),
                    adj_q10=float(np.quantile(yadj, .1)), adj_q90=float(np.quantile(yadj, .9)),
                    raw_q10=float(np.quantile(y, .1)), raw_q90=float(np.quantile(y, .9)), dist=float(d[idx].mean()),
                    d_all=d)

    def rf_pred(self, s):
        return float(self.rf.predict(s[None])[0])


def vshuffle(T, rng):
    """Within each base sign, permute the variant labels over all its variant tokens (keeps which base sign
    each tablet uses; destroys the habit structure)."""
    toks = collections.defaultdict(list)
    for i, t in enumerate(T):
        for l in t['lines']:
            for s in l['signs']:
                if common.is_sign(s) and '~' in s and not s.startswith('|'):
                    toks[common.base(s)].append((i, s))
    tab = collections.defaultdict(set)
    for b, lst in toks.items():
        lab = [s for _, s in lst]; rng.shuffle(lab)
        for (i, _), s in zip(lst, lab):
            tab[s].add(i)
    names = sorted(k for k, v in tab.items() if 2 <= len(v) <= 60)
    return [tab[k] for k in names]


def curveball(sets, n, rng, rounds=None):
    """Degree-preserving randomisation of the marker x tablet bipartite graph (curveball trades between markers)."""
    S = [set(s) for s in sets]
    m = len(S)
    rounds = rounds or 20 * m
    for _ in range(rounds):
        a, b = rng.choice(m, 2, replace=False)
        A, B = S[a], S[b]
        onlyA = list(A - B); onlyB = list(B - A)
        if not onlyA or not onlyB:
            continue
        pool = onlyA + onlyB; rng.shuffle(pool)
        k = len(onlyA)
        common_ = A & B
        S[a] = common_ | set(pool[:k]); S[b] = common_ | set(pool[k:])
    return S


def calib(E, THh, STh):
    """Blind recovery of planted (held-out simulated) worlds."""
    pr = E.rf.predict(STh)
    y = THh[:, 0]
    from scipy.stats import spearmanr
    r = spearmanr(pr, y).correlation
    r2 = 1 - ((pr - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    cov = []
    rj = []
    for i in range(min(200, len(y))):
        o = E.rej(STh[i])
        cov.append(o['adj_q10'] <= y[i] <= o['adj_q90']); rj.append(o['adj_med'])
    rr = spearmanr(rj, y[:len(rj)]).correlation
    # discrimination short (S<1) vs long (S>5)
    from sklearn.metrics import roc_auc_score
    sel = (y < 0) | (y > np.log(5))
    auc = roc_auc_score(y[sel] > 0, pr[sel]) if sel.sum() > 10 else np.nan
    return dict(rf_spearman=float(r), rf_r2=float(r2), rej_spearman=float(rr), cov80=float(np.mean(cov)), auc_short_long=float(auc))
