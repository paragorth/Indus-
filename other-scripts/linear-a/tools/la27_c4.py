#!/usr/bin/env python3
"""LA-27 cycle 4: the Linear B ceiling search turned on the Linear A fraction signs.

In a mixed-radix weight notation a sub-unit sign is repeated at most r-1 times, where r is its
ratio to the next larger sign (LB: N never more than 3 under M, M:N = 4; M never more than 2
under LANA, LANA:M = 3; cycle 3). The same estimator (max repeat + 1) is applied to each LA
fraction letter ('DD' is read as D twice). Bootstrap by document (2,000) gives the stability of
the estimate. Planted control: amounts drawn from the real letter mix, written greedily in a
known value system (fifths with D, thirds with B and A = 1/6, binary J, E, F, K), as many as
the real amounts; the estimator should return 5 for D, 3 for B, 2 for E, F, K, A.
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la27_common import la_numbers, CK
rng = np.random.default_rng(427)

def mult(f):
    c = Counter()
    for x in f:
        if x == 'DD': c['D'] += 2
        else: c[x] += 1
    return c

def table(recs):
    mx, n, hist = defaultdict(int), Counter(), defaultdict(Counter)
    for d, f in recs:
        for k, v in mult(f).items():
            n[k] += 1; hist[k][v] += 1; mx[k] = max(mx[k], v)
    return mx, n, hist

def planted(nam, reps=200):
    fam = {'D': (5, ['D']), 'B': (3, ['B', 'A']), 'J': (16, ['J', 'E', 'F', 'K'])}
    val = {'D': [1/5], 'B': [1/3, 1/6], 'J': [1/2, 1/4, 1/8, 1/16]}
    out = defaultdict(Counter)
    for _ in range(reps):
        c = defaultdict(int)
        for fk, (den, signs) in fam.items():
            for _ in range(nam[fk]):
                k = rng.integers(1, den if fk != 'B' else 6)       # amount k/den (thirds: k/6)
                amt = k / (den if fk != 'B' else 6); used = Counter()
                for s, v in zip(signs, val[fk]):
                    while amt + 1e-9 >= v: amt -= v; used[s] += 1
                for s, m in used.items(): c[s] = max(c[s], m)
        for s in ['D', 'B', 'A', 'J', 'E', 'F', 'K']: out[s][c[s] + 1] += 1
    return {s: dict(v) for s, v in out.items()}

if __name__ == '__main__':
    R = [(d, f) for d, l, v, f in la_numbers() if f]
    mx, n, hist = table(R)
    docs = sorted({d for d, _ in R}); byd = defaultdict(list)
    for d, f in R: byd[d].append((d, f))
    boot = defaultdict(Counter)
    for _ in range(2000):
        s = [r for d in rng.choice(docs, len(docs)) for r in byd[d]]
        m2, _, _ = table(s)
        for k in mx: boot[k][m2.get(k, 0) + 1] += 1
    res = {k: dict(n=n[k], max=mx[k], ratio_ML=mx[k] + 1, hist=dict(hist[k]),
                   boot={r: round(c / 2000, 3) for r, c in sorted(boot[k].items())})
           for k in sorted(mx, key=lambda k: -n[k])}
    # repeats by site
    site = Counter((d[:2], k) for d, f in R for k, v in mult(f).items() if v > 1)
    nam = {'D': n['D'], 'B': n['B'] + n['A'], 'J': n['J'] + n['E'] + n['F'] + n['K']}
    out = dict(real=res, repeats_by_site={'%s %s' % k: v for k, v in site.items()},
               planted=planted(nam))
    json.dump(out, open(os.path.join(CK, 'c4.json'), 'w'), indent=1)
    for k, v in res.items(): print(k, v)
    print(out['repeats_by_site']); print(out['planted'])
