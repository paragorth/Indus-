#!/usr/bin/env python3
"""la74 cycle 3a: freeze the vessel-ratio prediction BEFORE any capacity data is read."""
import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from la74_common import *
sets = []
for nm in ['c1_real', 'c1_real2', 'c1_real3', 'c2_site']:
    for r in json.load(open(os.path.join(CK, nm + '.json')))['top']:
        sz = r['H']['sizes']; sets.append(dict(src=nm, map=r['map'], sizes=sz, gte=r['gte'],
            ratios=[1 / sz[0]] + [sz[i] / sz[i + 1] for i in range(len(sz) - 1)]))
surv = [s for s in sets if s['gte'] > 0]
allr = np.array([x for s in surv for x in s['ratios']]); adj = np.array([x for s in surv for x in s['ratios'][1:]])
u1 = np.array([s['ratios'][0] for s in surv])
pred = dict(
    loop='la74', date='2026-10-07', n_survivors=len(surv), n_sets=len(sets),
    survivor_ratio_sets=[[round(x, 4) for x in s['ratios']] for s in surv],
    unit_over_largest_median=float(np.median(u1)), unit_over_largest_IQR=[float(np.percentile(u1, 25)), float(np.percentile(u1, 75))],
    adjacent_ratio_median=float(np.median(adj)), adjacent_ratio_IQR=[float(np.percentile(adj, 25)), float(np.percentile(adj, 75))],
    test=('Capacities: individual measured capacities of Minoan (MM III-LM I preferred) vessels, grouped by the source\'s own vessel '
          'classes. Observed ratio set O = all pairwise ratios (>1) between class medians. Score of a ratio set R = share of R within '
          '+-15 % (|ln| < 0.14) of some member of O. Statistic = mean score over the survivor sets. Null = the same over 10,000 vessel '
          'sets drawn from the search prior (sample_H, seed 74). One-sided P = share of 200 null batches (same count as survivors) '
          'scoring >= the survivors. Second statistic: |ln| distance of the median adjacent ratio to the nearest member of O vs null. '
          'Would support: P < 0.05 on both. Would kill: P > 0.2. Power note: with many classes O covers most ratios and the test is weak; '
          'the coverage of O over ln-ratio 0.2-2.3 is reported.'))
blob = json.dumps(pred, sort_keys=True).encode()
h = hashlib.sha256(blob).hexdigest(); pred['sha256_of_rest'] = h
json.dump(pred, open(os.path.join(DATA, 'la74_predictions.json'), 'w'), indent=1)
print(h, len(surv), pred['unit_over_largest_median'], pred['unit_over_largest_IQR'], pred['adjacent_ratio_median'], pred['adjacent_ratio_IQR'])
