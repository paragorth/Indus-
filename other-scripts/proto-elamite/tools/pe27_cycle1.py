"""pe27 cycle 1: exact-ratio scan between number systems on the same tablet.
Controls: Ur III (counts of people/animals vs capacity amounts; truth = written '-ta' rates, never
scanned), planted conversions in PE, then the real PE system pairs.  Nulls: Y line-sets re-dealt
among tablets with the same pair of systems (200 permutations), search-corrected max score."""
import json, sys, os, time
import numpy as np
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe27_common import *  # noqa

NP = int(os.environ.get('NP', 200))
res = {}
t0 = time.time()

# ---- Ur III control
U = ur3_tablets()
xs, ys, ids = split_sys(U, 'CNT', 'CAP')
rates = {u['id']: set(Fr(r) for r in u['rates']) for u in U}
R = scan(xs, ys, nperm=NP, seed=1)
# how many of the top ratios are written rates (or rate x 30 days / x 360) on their hit tablets?
def truth_share(r, xs, ys, ids):
    hit, ok = 0, 0
    for i in range(len(xs)):
        if r in pairs_ratios(xs[i], ys[i]):
            hit += 1
            rr = rates[ids[i]]
            if rr and any(r == a * k for a in rr for k in (1, 30, 360)):
                ok += 1
    return hit, ok
for e in R['top'][:10]:
    e['hit_tabs'], e['written_rate_tabs'] = truth_share(Fr(e['r']), xs, ys, ids)
res['ur3_full'] = R
print('UR3 full', len(xs), R['max_real'], R['null_max_q95'], R['p_global'], flush=True)
for e in R['top'][:10]:
    print('  ', e, flush=True)
# subsamples of PE size
sub = []
rng = np.random.default_rng(5)
for k in range(5):
    idx = rng.choice(len(xs), 257, replace=False)
    Rk = scan([xs[i] for i in idx], [ys[i] for i in idx], nperm=NP // 2, seed=10 + k)
    sub.append({'max': Rk['max_real'], 'q95': Rk['null_max_q95'], 'p': Rk['p_global'],
                'top3': [(e['r'], e['H'], e['mu'], e['p_corr']) for e in Rk['top'][:3]]})
    print('UR3 sub', sub[-1], flush=True)
res['ur3_sub257'] = sub
print('t', time.time() - t0, flush=True)

# ---- PE
T = pe_tablets()
pairs = [('CNT', 'CAP'), ('CNT', 'CAP@'), ('CAP', 'CAP@'), ('CNT', 'CNT@'), ('CNT@', 'CAP'), ('CNT', 'B')]
# planted
pl = []
for r, frac in ((30, 0.15), (Fr(5, 2), 0.15), (7, 0.08), (30, 0.05)):
    rngp = np.random.default_rng(int(float(r) * 100 + frac * 1000))
    Tp = plant(T, 'CNT', 'CAP', r, frac, rngp)
    a, b, _ = split_sys(Tp, 'CNT', 'CAP')
    Rp = scan(a, b, nperm=NP // 2, seed=3)
    rank = [Fr(e['r']) for e in Rp['top']].index(Fr(r)) + 1 if Fr(r) in [Fr(e['r']) for e in Rp['top']] else None
    pl.append({'r': str(r), 'frac': frac, 'rank': rank, 'max': Rp['max_real'], 'q95': Rp['null_max_q95'],
               'p': Rp['p_global'], 'top1': Rp['top'][0] if Rp['top'] else None})
    print('PLANT', pl[-1], flush=True)
res['plant'] = pl
# real
for X, Y in pairs:
    a, b, ids = split_sys(T, X, Y)
    if len(a) < 5:
        continue
    Rr = scan(a, b, nperm=NP, seed=7)
    for e in Rr['top'][:8]:
        r = Fr(e['r'])
        ex = []
        for i in range(len(a)):
            for qa in a[i]:
                for qb in b[i]:
                    if qb['v'] / qa['v'] == r:
                        ex.append((ids[i], qa['fin'], str(qa['v']), qb['fin'], str(qb['v'])))
        e['examples'] = ex[:12]
    res[f'pe_{X}_{Y}'] = Rr
    print('PE', X, Y, len(a), Rr['max_real'], Rr['null_max_q95'], Rr['p_global'], flush=True)
    for e in Rr['top'][:6]:
        print('  ', {k: e[k] for k in ('r', 'H', 'mu', 'score', 'p_corr')}, e.get('examples', [])[:4], flush=True)
json.dump(res, open(os.path.join(CK, 'cycle1.json'), 'w'), indent=1, default=str)
print('done', time.time() - t0)
