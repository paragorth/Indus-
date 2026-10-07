#!/usr/bin/env python3
"""la74 cycle 3b: the frozen vessel-ratio prediction (data/la74_predictions.json) against measured
capacities of LM IA Petras vessels (data/la74_petras_capacities.json). Protocol as frozen."""
import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from la74_common import *
P = json.load(open(os.path.join(DATA, 'la74_predictions.json')))
h = P.pop('sha256_of_rest'); assert hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest() == h
cap = json.load(open(os.path.join(DATA, 'la74_petras_capacities.json')))['classes']
med = sorted(float(np.median(v)) for v in cap.values())
O = np.array(sorted({b / a for i, a in enumerate(med) for b in med[i + 1:] if b / a > 1.0001}))
lnO = np.log(O)
def score(R): return np.mean([np.min(np.abs(lnO - math.log(r))) < 0.14 for r in R])
def dist(r): return float(np.min(np.abs(lnO - math.log(r))))
surv = P['survivor_ratio_sets']; S = np.mean([score(R) for R in surv])
rng = np.random.default_rng(74)
nullsets = []
for _ in range(10000):
    H = sample_H(rng); sz = H['sizes']; nullsets.append([1 / sz[0]] + [sz[i] / sz[i + 1] for i in range(len(sz) - 1)])
nsc = np.array([score(R) for R in nullsets])
batches = [nsc[rng.choice(len(nsc), len(surv), replace=False)].mean() for _ in range(200)]
p1 = (np.sum(np.array(batches) >= S) + 1) / 201
d_obs = dist(P['adjacent_ratio_median'])
nadj = [x for R in nullsets for x in R[1:]]
dn = np.array([dist(np.median(rng.choice(nadj, 40))) for _ in range(2000)])
p2 = (np.sum(dn <= d_obs) + 1) / 2001
grid = np.linspace(0.2, 2.3, 400); cov = np.mean([np.min(np.abs(lnO - g)) < 0.14 for g in grid])
print(json.dumps(dict(sha=h, n_classes=len(med), class_medians=med, n_ratios=len(O), coverage_ln_0p2_2p3=float(cov),
      survivor_score=float(S), null_score_mean=float(nsc.mean()), p_score=float(p1),
      adj_median=P['adjacent_ratio_median'], nearest_O=float(O[np.argmin(np.abs(lnO - math.log(P['adjacent_ratio_median'])))]),
      dist=d_obs, p_dist=float(p2), unit_over_largest=P['unit_over_largest_median']), indent=0))
