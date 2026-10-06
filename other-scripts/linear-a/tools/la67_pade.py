#!/usr/bin/env python3
"""LA-67 kill sweep, test K-PADE: la53 cycle 3's Linear A ranking re-run with FRESH seeds: 6,000 new random
attention specs, the 5 % best on BOTH controls (Linear B status AUC + Ur III status AUC, rank sum) applied to
Linear A, 1,000 within-stratum permutations (la53's transfer-permutation step is skipped: it does not enter
the Linear A ranking).  Re-tests the grade-C guess 'PA-DE is a specially marked entry' (p 0.010, n 3).
Output data/la67_ckpt/la53/c3.json (LA_rank)."""
import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from scipy.stats import rankdata
import la53_common as C
import la53_c3 as M

lb = C.lb_docs(); ur = C.ur3_docs()
CK = os.path.join(HERE, '..', 'data', 'la67_ckpt', 'la53'); os.makedirs(CK, exist_ok=True)
S = lambda n: C.seed('la67-' + n)
NSPEC, NPERM = 6000, 1000
rng = np.random.default_rng(S('la53-c3'))
specs = C.random_specs(NSPEC, rng)
for s in specs:
    s['docfe'] = bool(rng.random() < 0.5)
la = C.la_docs()
sets = {}
for tag, rows, tf in (('LB', C.occurrences(lb, 'w'), C.lb_status),
                      ('UR', C.occurrences(M.sample_docs(ur, 'qty', 20000, random.Random(S('urbig'))), 'qty'), C.ur_status),
                      ('LA', C.occurrences(la, 'w'), lambda t: False)):
    B = C.build(rows); Mx, ok = M.per_spec(B, specs)
    lab = np.array([tf(t) for t in B['types']])
    sets[tag] = {'B': B, 'M': Mx[ok], 'lab': lab[ok], 'cnt': np.bincount(B['ti'], minlength=len(B['types']))[ok],
                 'types': [t for t, o in zip(B['types'], ok) if o]}
a = {t: M.auc_cols(sets[t]['M'], sets[t]['lab']) for t in ('LB', 'UR')}
top = NSPEC // 20
both = np.argsort(-(rankdata(a['LB']) + rankdata(a['UR'])))[:top]
print('both LB %.3f UR %.3f' % (a['LB'][both].mean(), a['UR'][both].mean()), flush=True)
L = sets['LA']; sc = L['M'][:, both].mean(1)
B = L['B']; okm = np.bincount(B['ti'], minlength=len(B['types'])) >= 2
pr = np.random.default_rng(S('la-c3-null'))
sp = [specs[j] for j in both]
NUL = []
for k in range(NPERM):
    pi = C.perm_within(B['strata'], pr)
    Mp, _ = M.per_spec(dict(B, A=B['A'][pi]), sp)
    NUL.append(Mp[okm].mean(1))
    if k % 100 == 0:
        print('perm', k, flush=True)
NUL = np.array(NUL)
p = (np.sum(NUL >= sc[None, :], 0) + 1) / (NPERM + 1)
m = len(p); oo = np.argsort(p); q = np.empty(m); prev = 1.0
for j in range(m - 1, -1, -1):
    prev = min(prev, p[oo[j]] * m / (j + 1)); q[oo[j]] = prev
rank = sorted([{'type': t, 'n': int(n), 'score': float(s), 'p': float(pp), 'q': float(qq)}
               for t, n, s, pp, qq in zip(L['types'], L['cnt'], sc, p, q)], key=lambda r: -r['score'])
json.dump({'LA_rank': rank}, open(os.path.join(CK, 'c3.json'), 'w'))
print([r for r in rank if r['type'] == 'PA-DE'], flush=True)
