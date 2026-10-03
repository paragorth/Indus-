"""Loop 46 cycle 2: cluster texts into hands with a latent-class (Bernoulli mixture) model on variant profiles.

Data: per text, for every usable class present, label 1 = non-head form, 0 = head form (classes absent = missing).
Only texts with >= 2 labels carry information about K (a mixture of one Bernoulli is one Bernoulli).
Model: K hands, pi_k, theta_{k,h}; EM with Beta(1,1) smoothing, 20 restarts; K chosen by 5-fold held-out log-likelihood.
Pools: MD seals, Harappa seals, Harappa tablets, each city all types; POSITIVE CONTROL = all cities pooled
(the city dialect, S-DARK-14.2, should show up as K >= 2 if the method has power).
Null for Delta LL(K=2 vs 1): labels permuted among texts within site x type (200x), same CV folds.
If a pool shows K >= 2 beyond the null, hands are tested for locality (area-section, room-grid, period, material, type)
and for office alignment (first sign, last sign, number of distinct closers per hand) against hand labels
permuted within site x type (1,000x).
Usage: python3 tools/dark_loop46_c2.py [level]
"""
import sys, json, collections, itertools
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop46_common import *

LEVEL = sys.argv[1] if len(sys.argv) > 1 else 'all'
NSHUF = 200; NPERM = 1000; KMAX = 4; RESTARTS = 20; FOLDS = 5
rng = np.random.default_rng(462)
corpus = load_corpus(); N = len(corpus)
classes = load_classes(LEVEL); use = usable_classes(corpus, classes)
labels, _, _ = text_labels(corpus, classes, use)
heads = sorted(use); H = len(heads); hidx = {h: k for k, h in enumerate(heads)}
lines = [f'# loop 46 cycle 2: latent-class hands  level={LEVEL}  classes={heads}  restarts={RESTARTS} folds={FOLDS} shuffles={NSHUF}']


def matrix(texts):
    X = np.full((len(texts), H), -1, dtype=int)
    for r, i in enumerate(texts):
        for h, v in labels[i].items():
            X[r, hidx[h]] = v
    return X


def em(X, K, iters=200, seed=0):
    r = np.random.default_rng(seed)
    n = len(X); obs = X >= 0; x1 = (X == 1)
    pi = np.full(K, 1 / K); theta = r.uniform(0.2, 0.8, (K, H))
    for _ in range(iters):
        ll = np.log(pi)[None, :] + (x1[:, None, :] * np.log(theta)[None] + ((~x1) & obs)[:, None, :] * np.log(1 - theta)[None]).sum(2)
        m = ll.max(1, keepdims=True); resp = np.exp(ll - m); resp /= resp.sum(1, keepdims=True)
        pi = (resp.sum(0) + 1) / (n + K)
        num = resp.T @ x1 + 1; den = resp.T @ obs + 2
        theta = np.clip(num / den, 1e-3, 1 - 1e-3)
    return pi, theta


def loglik(X, pi, theta):
    obs = X >= 0; x1 = (X == 1)
    ll = np.log(pi)[None, :] + (x1[:, None, :] * np.log(theta)[None] + ((~x1) & obs)[:, None, :] * np.log(1 - theta)[None]).sum(2)
    m = ll.max(1); return float(np.sum(m + np.log(np.exp(ll - m[:, None]).sum(1))))


def fit_best(X, K):
    best = None
    for s in range(RESTARTS):
        pi, th = em(X, K, seed=s); l = loglik(X, pi, th)
        if best is None or l > best[0]:
            best = (l, pi, th)
    return best


def cv_ll(X, K, folds):
    tot = 0.0
    for f in range(FOLDS):
        tr = X[folds != f]; te = X[folds == f]
        _, pi, th = fit_best(tr, K)
        tot += loglik(te, pi, th)
    return tot


def pool_test(name, texts, do_shuffle=True):
    X = matrix(texts); n = len(X)
    if n < 25:
        lines.append(f'\n## {name}: {n} texts with >= 2 labels: too few'); return None
    folds = rng.integers(0, FOLDS, n)
    cv = {K: cv_ll(X, K, folds) for K in range(1, KMAX + 1)}
    bestK = max(cv, key=cv.get)
    d21 = cv[2] - cv[1]
    lines.append(f'\n## {name}: {n} texts with >= 2 labels; held-out LL by K: ' + ', '.join(f'K={K} {v:.1f}' for K, v in cv.items()) + f'; best K={bestK}; Delta(2-1)={d21:+.1f}')
    res = dict(n=n, cv=cv, bestK=bestK, d21=d21)
    if do_shuffle:
        st = np.array([hash((corpus[i]['site'], corpus[i]['type'])) for i in texts])
        null = []
        for s in range(NSHUF):
            Xs = X.copy()
            for k in range(H):
                col = Xs[:, k]; m = col >= 0
                if m.sum() > 1:
                    sub = col[m]; Xs[m, k] = permute_within(sub, st[m], rng)
            cvs = {K: cv_ll(Xs, K, folds) for K in (1, 2)}
            null.append(cvs[2] - cvs[1])
        null = np.array(null)
        P = (np.sum(null >= d21) + 1) / (NSHUF + 1)
        lines.append(f'   null Delta(2-1) under within-site x type shuffles: {null.mean():+.1f} +/- {null.std():.1f} (max {null.max():+.1f}); P={P:.3f}; shuffles preferring K>=2: {np.mean(null > 0):.2f}')
        res.update(null_mean=float(null.mean()), null_sd=float(null.std()), P=float(P))
    # hand description at K=2
    _, pi, th = fit_best(X, 2)
    lines.append('   K=2 hands: ' + ' | '.join(f'hand{k} pi={pi[k]:.2f} ' + ' '.join(f'W{h}:{th[k, j]:.2f}' for j, h in enumerate(heads)) for k in range(2)))
    res['pi'] = pi.tolist(); res['theta'] = th.tolist()
    return res, X, pi, th


def locality(name, texts, X, pi, th):
    obs = X >= 0; x1 = X == 1
    ll = np.log(pi)[None, :] + (x1[:, None, :] * np.log(th)[None] + ((~x1) & obs)[:, None, :] * np.log(1 - th)[None]).sum(2)
    hand = ll.argmax(1)
    sizes = collections.Counter(hand.tolist())
    st = np.array([hash((corpus[i]['site'], corpus[i]['type'])) for i in texts])
    feats = {'area-section': [corpus[i]['area-section'] for i in texts],
             'room-grid': [corpus[i]['room-grid'] for i in texts],
             'period': [corpus[i]['period'] for i in texts],
             'material': [corpus[i]['material'] for i in texts],
             'type': [corpus[i]['type'] for i in texts],
             'first sign': [str(corpus[i]['seq_all'][0]) if corpus[i]['seq_all'] else '-' for i in texts],
             'last sign': [str(corpus[i]['seq_all'][-1]) if corpus[i]['seq_all'] else '-' for i in texts]}
    def mi_cat(h, f):
        c = collections.Counter(zip(h.tolist(), f)); n = len(h)
        ch = collections.Counter(h.tolist()); cf = collections.Counter(f)
        return sum(v / n * np.log2(v * n / (ch[a] * cf[b])) for (a, b), v in c.items())
    out = [f'   hand sizes {dict(sizes)}']
    for fn, f in feats.items():
        m = np.array([v not in ('-', '--', '') for v in f])
        if m.sum() < 20:
            out.append(f'   {fn}: <20 texts with a value'); continue
        obsmi = mi_cat(hand[m], [f[i] for i in np.flatnonzero(m)])
        null = np.array([mi_cat(permute_within(hand, st, rng)[m], [f[i] for i in np.flatnonzero(m)]) for _ in range(NPERM)])
        P = (np.sum(null >= obsmi) + 1) / (NPERM + 1)
        out.append(f'   hand x {fn}: MI {obsmi:.3f} vs null {null.mean():.3f}+/-{null.std():.3f} P={P:.3f} (n={m.sum()})')
    # distinct closers per hand
    last = feats['last sign']
    nd = np.mean([len(set(last[i] for i in np.flatnonzero(hand == k))) / max(1, np.sum(hand == k)) for k in sizes])
    null = []
    for _ in range(NPERM):
        hp = permute_within(hand, st, rng)
        null.append(np.mean([len(set(last[i] for i in np.flatnonzero(hp == k))) / max(1, np.sum(hp == k)) for k in sizes]))
    null = np.array(null)
    out.append(f'   distinct closers per text within hand: {nd:.3f} vs null {null.mean():.3f}+/-{null.std():.3f} (P low = fewer closers per hand = office-bound) P={(np.sum(null <= nd) + 1) / (NPERM + 1):.3f}')
    lines.extend(out)


two = [i for i in range(N) if len(labels[i]) >= 2]
pools = {
    'POSITIVE CONTROL all sites, all types (city dialect should appear)': two,
    'Mohenjo-daro all types': [i for i in two if corpus[i]['site'] == 'Mohenjo-daro'],
    'Harappa all types': [i for i in two if corpus[i]['site'] == 'Harappa'],
    'Mohenjo-daro seals': [i for i in two if corpus[i]['site'] == 'Mohenjo-daro' and corpus[i]['type'].startswith('SEAL')],
    'Harappa seals': [i for i in two if corpus[i]['site'] == 'Harappa' and corpus[i]['type'].startswith('SEAL')],
    'Harappa tablets': [i for i in two if corpus[i]['site'] == 'Harappa' and corpus[i]['type'].startswith('TAB')],
    'other sites': [i for i in two if corpus[i]['site'] not in ('Harappa', 'Mohenjo-daro')],
}
summary = {}
for name, texts in pools.items():
    r = pool_test(name, texts)
    if r is None:
        continue
    res, X, pi, th = r
    summary[name] = {k: v for k, v in res.items()}
    locality(name, texts, X, pi, th)

json.dump(summary, open(OUT + f'loop46_cycle2_{LEVEL}.json', 'w'), indent=1, default=float)
open(OUT + f'loop46_cycle2_{LEVEL}_log.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
