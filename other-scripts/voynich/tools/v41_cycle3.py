"""v41 cycle 3: the frozen clock against the physical book, and which hand moved first.
Physical evidence is used only as a check:
 (a) conjugate leaves (same quire, same bifolio, other leaf) and the two sides of one leaf:
     closer on the clock than other same-quire pages of the same group? Null: clock values
     permuted within quire x group (5000x). Control: a planted drift whose hidden times are
     assigned per BIFOLIO (all four pages share a time) must be caught; times per page must not.
 (b) clock against foliation (binding order) inside each group.
 (c) intermediate pages: whole-book oriented clock (not demeaned) between A and B medians.
 (d) Guttman test of 'which trait moved first': per hand x section group, the fraction of the
     A->B distance traversed for each trait; implicational scale (traits ordered the same way
     in every group?) vs groups with trait labels permuted.
"""
import sys, os, json, random, itertools
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from v41_lib import *
from v41_cycle2 import make_traits  # noqa  (cycle2 guarded below)

res, rows = {}, []
c2 = json.load(open(os.path.join(CK, 'c2.json')))
P = vpages('ZL3b')
byid = {p['id']: p for p in P}


def phys_test(gname, clock, pages, nperm=5000, seed=0):
    rng = np.random.default_rng(seed)
    clock = np.asarray(clock)
    q = np.array([p['quire'] or '?' for p in pages])
    pairs = {'conj': [], 'leaf': [], 'other': []}
    for i, j in itertools.combinations(range(len(pages)), 2):
        a, b = pages[i], pages[j]
        if a['quire'] != b['quire'] or a['quire'] is None:
            continue
        if a['leaf'] == b['leaf'] and a['bifolio'] == b['bifolio']:
            pairs['leaf'].append((i, j))
        elif a['bifolio'] == b['bifolio'] and a['bifolio'] is not None:
            pairs['conj'].append((i, j))
        else:
            pairs['other'].append((i, j))
    def stat(c):
        return {k: (np.mean([abs(c[i] - c[j]) for i, j in v]) if v else np.nan) for k, v in pairs.items()}
    obs = stat(clock)
    null = {k: [] for k in pairs}
    for _ in range(nperm):
        c = clock.copy()
        for s in set(q.tolist()):
            m = np.where(q == s)[0]
            c[m] = clock[m[rng.permutation(len(m))]]
        st = stat(c)
        for k in pairs:
            null[k].append(st[k])
    out = {}
    for k in pairs:
        nl = np.array(null[k])
        if not pairs[k] or np.isnan(obs[k]):
            continue
        out[k] = dict(n=len(pairs[k]), obs=float(obs[k]), null=float(np.nanmean(nl)),
                      z=float((obs[k] - np.nanmean(nl)) / (np.nanstd(nl) + 1e-12)),
                      p_low=float((1 + (nl <= obs[k]).sum()) / (1 + nperm)))
    return out


# ---- (a) control: planted per-bifolio vs per-page times in H1A herbal ----
from v41_cycle1_plant import plant  # small helper module
H1 = [p for p in P if p['hand'] == '1' and p['lang'] == 'A' and p['illus'] == 'H']
T, _ = make_traits(P)
A = [p for p in P if p['lang'] == 'A']; B = [p for p in P if p['lang'] == 'B']
o = orient_from(A, B, T)
ctrl = {}
for mode in ('bifolio', 'page'):
    for seed in (0, 1):
        r = random.Random(300 + seed)
        tb = {}
        pp = []
        for p in H1:
            key = (p['quire'], p['bifolio']) if mode == 'bifolio' else p['id']
            t = tb.setdefault(key, r.random())
            lines = [[plant(w, 0.3 * t, r) for w in l] for l in p['lines']]
            pp.append(dict(id=p['id'], quire=p['quire'], bifolio=p['bifolio'], leaf=p['leaf'], lines=lines,
                           all=[w for l in lines for w in l],
                           h=[[w for i, l in enumerate(lines) if i % 2 == k for w in l] for k in (0, 1)]))
        (X1, X2), _ = rate_matrix(pp, T)
        D = (X1 + X2) / 2; D = D - D.mean(0)
        clock = ((D / (D.std(0) + 1e-12)) * o).mean(1)
        ctrl[f'{mode}_{seed}'] = phys_test('ctrl', clock, pp, nperm=2000, seed=seed)
        print('ctrl', mode, seed, ctrl[f'{mode}_{seed}'], flush=True)
res['ctrl'] = ctrl

# ---- (a,b) real groups ----
real = {}
for name in ('ZL3b', 'IT2a'):
    for g in ('H1A_herbal', 'H1A_all', 'H2B_all', 'H3B_all', 'B_all', 'H2B_bio', 'H3B_stars'):
        d = c2[f'{name}_{g}']
        pages = [byid[i] for i in d['pages'] if i in byid]
        clock = [c for i, c in zip(d['pages'], d['clock']) if i in byid]
        pt = phys_test(g, clock, pages, seed=1)
        fol = spearmanr([p['order'] for p in pages], clock)[0]
        real[f'{name}_{g}'] = dict(phys=pt, foliation_rho=float(fol), n=len(pages))
        print(name, g, real[f'{name}_{g}'], flush=True)
res['real'] = real

# ---- (c) whole-book clock and intermediates; (d) Guttman ----
(X1, X2), names = rate_matrix(P, T)
X = (X1 + X2) / 2
Am = np.array([p['lang'] == 'A' for p in P]); Bm = np.array([p['lang'] == 'B' for p in P])
mu = X.mean(0); sd = X.std(0) + 1e-12
Z = (X - mu) / sd * o
clock = Z.mean(1)
mA, mB = np.median(clock[Am]), np.median(clock[Bm])
lo_, hi_ = mA + (mB - mA) / 3, mA + 2 * (mB - mA) / 3
inter = [(P[i]['id'], P[i]['hand'], P[i]['lang'], P[i]['illus'], P[i]['quire'], round(float(clock[i]), 2))
         for i in range(len(P)) if lo_ <= clock[i] <= hi_]
res['intermediate'] = inter
print('intermediates', inter, flush=True)
# per group trait progress: fraction of distance from A herbal hand 1 (0) to B bio hand 2 (1)? use pooled A and pooled B medians
grp = defaultdict(list)
for i, p in enumerate(P):
    if p['hand'] and p['lang'] and p['illus']:
        grp[(p['hand'], p['lang'], p['illus'])].append(i)
grp = {k: v for k, v in grp.items() if len(v) >= 5}
a0 = np.median(Z[Am], 0); b0 = np.median(Z[Bm], 0)
prog = {k: ((np.median(Z[v], 0) - a0) / (b0 - a0 + 1e-12)) for k, v in grp.items()}
res['progress'] = {'|'.join(k): dict(zip(names, np.round(v, 2).tolist())) for k, v in prog.items()}
for k, v in prog.items():
    print(k, len(grp[k]), dict(zip(names, np.round(v, 2))), flush=True)
# Guttman: if traits move in a fixed sequence, the trait rank order of progress is shared across groups.
# statistic = mean pairwise Spearman between groups' progress vectors (Kendall's W style), only the
# groups strictly between the two extremes matter, so use all groups; null = shuffle each group's vector.
K = np.array(list(prog.values()))
def concord(K):
    return np.mean([spearmanr(K[i], K[j])[0] for i in range(len(K)) for j in range(i + 1, len(K))])
obsW = concord(K)
rng = np.random.default_rng(5)
nullW = [concord(np.array([rng.permutation(r) for r in K])) for _ in range(2000)]
res['guttman'] = dict(obs=float(obsW), null=float(np.mean(nullW)), z=float((obsW - np.mean(nullW)) / np.std(nullW)),
                      p=float((1 + (np.array(nullW) >= obsW).sum()) / 2001))
print('guttman', res['guttman'], flush=True)
# which hand is furthest along at matched section (herbal): overall clock medians
hm = {'|'.join(k): float(np.median(clock[v])) for k, v in grp.items()}
res['group_clock'] = hm
print(hm)
json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), default=float)
