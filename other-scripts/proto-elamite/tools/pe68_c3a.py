"""pe68 cycle 3a: a lead seen while reading the joined team lists (P008043 line 9: 540 N39C after nine
1-person lines): the M288 line = 60 N39C x (sum of the count lines since the previous M288 / start).
Seen on joined tablets, so it is tested only on the NON-joined corpus, against (i) the frozen pe59 rule
(60 x the preceding line only) and (ii) nulls: M288 values re-dealt across tablets; run sums with run
boundaries shifted by one line; other multipliers (k x sum, k = 1..120)."""
import os, sys, json, random
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe68_lib import *  # noqa
import pe59_lib as P

cap, cnt = P.pe_maps()
cm = cnt['sex2']
runs = []
for t in P.build_pe():
    if t['id'] in JOINED_IDS:
        continue
    L = t['lines']
    acc, ok, prev_only = 0, True, None
    start = 0
    for i, l in enumerate(L):
        if l['role'] != 'E':
            continue
        if l['signs'] and l['signs'][-1] == 'M288':
            m = P.value(l['nums'], cap) if l['numclean'] else None
            if m and acc and ok:
                runs.append({'id': t['id'], 'i': i, 'sum': float(acc), 'prev': prev_only, 'm': float(m),
                             'k': i - start})
            acc, ok, start = 0, True, i + 1
            continue
        if not l['numclean'] or P.ncls(l['nums']) != 'AMB':
            ok = False
            prev_only = None
            continue
        v = P.value(l['nums'], cm)
        acc += v
        prev_only = float(v)
print('runs', len(runs), 'tablets', len({r['id'] for r in runs}))
hit_sum = sum(1 for r in runs if r['m'] == 60 * r['sum'])
hit_prev = sum(1 for r in runs if r['prev'] and r['m'] == 60 * r['prev'])
multi = [r for r in runs if r['k'] >= 2]
hit_sum_multi = sum(1 for r in multi if r['m'] == 60 * r['sum'])
hit_prev_multi = sum(1 for r in multi if r['prev'] and r['m'] == 60 * r['prev'])
print('60 x run-sum: %d/%d; 60 x previous line: %d/%d' % (hit_sum, len(runs), hit_prev, len(runs)))
print('runs of >= 2 count lines: 60 x sum %d/%d; 60 x previous %d/%d' % (hit_sum_multi, len(multi), hit_prev_multi, len(multi)))
rng = random.Random(seed('pe68-c3a'))
ms = [r['m'] for r in runs]
null = []
for _ in range(10000):
    null.append(sum(1 for r in multi if rng.choice(ms) == 60 * r['sum']))
null = np.array(null)
p = float(((null >= hit_sum_multi).sum() + 1) / (len(null) + 1))
print('re-dealt M288 values: mean %.2f, p %.4f' % (null.mean(), p))
ks = Counter()
for r in multi:
    for k in range(1, 241):
        if r['m'] == k * r['sum']:
            ks[k] += 1
print('best multipliers', ks.most_common(8))
json.dump({'runs': runs, 'hit_sum': hit_sum, 'hit_prev': hit_prev, 'n': len(runs), 'multi_n': len(multi),
           'hit_sum_multi': hit_sum_multi, 'hit_prev_multi': hit_prev_multi, 'null_mean': float(null.mean()), 'p': p,
           'multipliers': ks.most_common(12)}, open(os.path.join(CK, 'c3a.json'), 'w'), indent=1)
