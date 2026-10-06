#!/usr/bin/env python3
"""pe64 cycle 1: millions of random rate-detection rules; discovery on one half of the tablets,
held-out test of every surviving (rule, rate) on the other half.
usage: pe64_c1.py CORPUS VARIANT DIRECTION N_RANDOM
  CORPUS  PE | U3 | PC
  VARIANT real | small (U3: 120-tablet subsample) | plant (swap-null corpus + planted 3-rung ladder)
          | nullreal (a swap-null corpus run as if real: false-positive calibration)
  DIRECTION AB (discover on A, test on B) | BA
Output: data/pe64_ckpt/c1_<CORPUS>_<VARIANT>_<DIR>.json
"""
import sys, json, time, random, os
import numpy as np
import pe64_lib as L

C, V, D, NR = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rng = random.Random(L.seed('pe64c1' + C + V + D))
T = L.load(C)
truth = None
if V == 'small':
    T = [T[i] for i in sorted(rng.sample(range(len(T)), 120))]
elif V.startswith('plant'):
    T, truth = L.plant_ladder(L.null_swap(T, random.Random(7)), rng, rungs=(20, 40, 80), per=int(V[5:] or 10))
elif V.startswith('nullreal'):
    T = L.null_swap(T, random.Random(int(V[8:] or 11)))
A, B = L.split(T, 'pe64' + C)
if D == 'BA':
    A, B = B, A
TA = [T[i] for i in A]
TB = [T[i] for i in B]
t0 = time.time()
PA = L.PairTable(TA)
R = L.rule_space(PA, K=60, rng=rng, n_random=NR)
print(C, V, D, 'tabs A/B', len(TA), len(TB), 'pairs A', len(PA.rows), 'rules', len(R), flush=True)

cache = {}
real = np.zeros((len(R), 3), dtype=np.int64)
for k, r in enumerate(R):
    real[k] = L.score(PA, L.rule_mask(PA, r, cache))
cand = np.flatnonzero(real[:, 0] >= 3)
print('scored real', round(time.time() - t0), 's; rules with k>=3:', len(cand), flush=True)

# swap nulls on A, only for candidate rules, counted at the rule's own top rate
NN = 6
nullk = np.zeros((len(cand), NN))
for z in range(NN):
    Tn = L.null_swap(TA, random.Random(1000 + z))
    Pn = PA.revalue(Tn)
    cn = {}
    for q, k in enumerate(cand):
        nullk[q, z] = L.count_at(Pn, L.rule_mask(Pn, R[k], cn), int(real[k, 1]))
print('nulls A', round(time.time() - t0), 's', flush=True)
mu = nullk.mean(1)
mx = nullk.max(1)
kr = real[cand, 0]
surv = cand[(kr >= mx + 2) & (kr >= 2 * mu + 2)]
print('survivors', len(surv), flush=True)

# group survivors by rate. Selectors tested on B for each rate: 'union' (all A-survivor rules with that rate),
# 'grid' (best exhaustive-grid rule, at most one source sign), 'top' (best rule by excess on A)
byrate = {}
for k in surv:
    q = int(np.searchsorted(cand, k))
    ex = real[k, 0] - mu[q]
    byrate.setdefault(int(real[k, 1]), []).append((ex, int(k)))
tests = []
for rk, lst in byrate.items():
    lst.sort(reverse=True)
    tests.append((rk, 'union', [k for _, k in lst][:400], lst[0][0]))   # at most 400 best rules
    g = [(ex, k) for ex, k in lst if len(R[k][5]) <= 1]
    if g:
        tests.append((rk, 'grid', [g[0][1]], g[0][0]))
    tests.append((rk, 'top', [lst[0][1]], lst[0][0]))

PB = L.PairTable(TB)
NB = 200
NR1 = 20


def sel(Pp, ks, cc):
    m = [L.rule_mask(Pp, R[k], cc) for k in ks]
    return np.unique(np.concatenate(m)) if m else np.zeros(0, np.int64)


cb = {}
obs = [L.count_at(PB, sel(PB, ks, cb), rk) for rk, kind, ks, ex in tests]
nb = {'repair': np.zeros((len(tests), NR1)), 'swap': np.zeros((len(tests), NB))}
for z in range(NB):
    for kind, fn in (('repair', L.null_repair), ('swap', L.null_swap)):
        if kind == 'repair' and z >= NR1:
            continue
        Pn = PB.revalue(fn(TB, random.Random(5000 + z)))
        cn = {}
        for q, (rk, sk, ks, ex) in enumerate(tests):
            nb[kind][q, z] = L.count_at(Pn, sel(Pn, ks, cn), rk)
print('held-out', round(time.time() - t0), 's', flush=True)
res = []
for q, (rk, sk, ks, ex) in enumerate(tests):
    o = obs[q]
    p1 = (1 + (nb['repair'][q] >= o).sum()) / (NR1 + 1)
    p2 = (1 + (nb['swap'][q] >= o).sum()) / (NB + 1)
    res.append({'rate': rk / 1000, 'sel': sk, 'rules': [[R[k][0], R[k][1], R[k][2], R[k][3], R[k][4], list(R[k][5]), R[k][6]]
                                                      for k in ks[:400]], 'n_rules': len(ks),
                'kA': int(max(real[k, 0] for k in ks)), 'exA': float(ex), 'kB': int(o),
                'nullB_repair': float(nb['repair'][q].mean()), 'nullB_swap': float(nb['swap'][q].mean()),
                'p_repair': float(p1), 'p_swap': float(p2)})
out = {'corpus': C, 'variant': V, 'dir': D, 'n_rules': len(R), 'n_cand': int(len(cand)), 'n_surv': int(len(surv)),
       'n_rates': len(byrate), 'n_tests': len(tests), 'truth': truth, 'tests': res,
       'top_real': sorted(([int(real[k, 0]), int(real[k, 1]), R[k][0], R[k][1], R[k][2], list(R[k][5]), R[k][6]]
                           for k in cand), key=lambda x: -x[0])[:30],
       'secs': round(time.time() - t0)}
json.dump(out, open(os.path.join(L.CK, 'c1_%s_%s_%s.json' % (C, V, D)), 'w'), default=str)
ok = [x for x in res if x['p_swap'] <= 0.005 and x['kB'] >= 2]
print('validated tests', len(ok), 'rates', sorted({x['rate'] for x in ok}))
