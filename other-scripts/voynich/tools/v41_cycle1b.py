"""v41 cycle 1b: pure power of the clock on an exchangeable base.
Hand-1 A herbal tokens redealt across pages (page and line lengths kept: no native page
differences), then a planted drift with hidden times; and a shuffled-trait null pushed
through the same seriation (must give no held-out trajectory)."""
import sys, os, json, random
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from v41_lib import *

P = vpages()
T, _ = make_traits(P)
A = [p for p in P if p['lang'] == 'A']; B = [p for p in P if p['lang'] == 'B']
o = orient_from(A, B, T)
H1 = [p for p in P if p['hand'] == '1' and p['lang'] == 'A' and p['illus'] == 'H']
res = {}


def build(lines_list):
    return [dict(lines=L, all=[w for l in L for w in l],
                 h=[[w for i, l in enumerate(L) if i % 2 == k for w in l] for k in (0, 1)]) for L in lines_list]


def ev(pages, truth, seed):
    (X1, X2), names = rate_matrix(pages, T)
    coh = coherence_null(X1, X2, o, None, nperm=200, seed=seed)
    D1 = X1 - X1.mean(0); D2 = X2 - X2.mean(0)
    order, f, orders = seriate(D1, restarts=12, iters=15000, seed=seed)
    pos = position_of(order)
    z = lambda D: (D / (D.std(0) + 1e-12)) * o
    if spearmanr(pos, z(D1).mean(1))[0] < 0:
        pos = pos.max() - pos
    return dict(S=coh['S'], z=coh['z'], R=coh['R'], ratio=coh['ratio'],
                held=float(spearmanr(pos, z(D2).mean(1))[0]),
                truth=float(spearmanr(pos, truth)[0]) if truth is not None else None,
                clock_truth=float(spearmanr(z((X1 + X2) / 2 - ((X1 + X2) / 2).mean(0)).mean(1), truth)[0]) if truth is not None else None)


for seed in (0, 1):
    r = random.Random(500 + seed)
    toks = [w for p in H1 for w in p['all']]; r.shuffle(toks)
    it = iter(toks)
    base = [[[next(it) for _ in l] for l in p['lines']] for p in H1]
    for s in (0.0, 0.05, 0.1, 0.2, 0.4):
        t = np.array([r.random() for _ in H1])
        L = [[[plant(w, s * ti, r) for w in l] for l in pl] for pl, ti in zip(base, t)]
        e = ev(build(L), t, seed)
        res[f'redeal_{s}_{seed}'] = e
        print('redeal', s, seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()}, flush=True)
# shuffled traits on the real H1A herbal pages: each trait column permuted independently,
# then the same seriation; held-out trajectory must vanish
(X1, X2), names = rate_matrix(H1, T)
rng = np.random.default_rng(9)
hs = []
for k in range(20):
    Y1 = X1.copy(); Y2 = X2.copy()
    for j in range(X1.shape[1]):
        pi = rng.permutation(len(H1)); Y1[:, j] = X1[pi, j]; Y2[:, j] = X2[pi, j]
    D1 = Y1 - Y1.mean(0); D2 = Y2 - Y2.mean(0)
    order, f, _ = seriate(D1, restarts=4, iters=10000, seed=k)
    pos = position_of(order)
    z = lambda D: (D / (D.std(0) + 1e-12)) * o
    if spearmanr(pos, z(D1).mean(1))[0] < 0:
        pos = pos.max() - pos
    hs.append(spearmanr(pos, z(D2).mean(1))[0])
res['shuffled_traits_held'] = dict(mean=float(np.mean(hs)), sd=float(np.std(hs)), max=float(np.max(hs)))
print('shuffled traits held-out rho', res['shuffled_traits_held'], flush=True)
json.dump(res, open(os.path.join(CK, 'c1b.json'), 'w'), default=float)
