"""pe81 outside comparison: frozen text axes vs environmental panel (exact permutation p over the 4 outposts)."""
import json, itertools, math, hashlib, sys
import numpy as np
from scipy.stats import spearmanr
F = json.load(open('data/pe81_frozen_axes.json')); E = json.load(open('data/pe81_environment.json'))
print('frozen sha256', hashlib.sha256(open('data/pe81_frozen_axes.json','rb').read()).hexdigest())
O = ['Malyan', 'Yahya', 'Sialk', 'Sofalin']
D = np.array([F['susa_likeness_D (lower = writes more like Susa)'][s] for s in O])
S = E['Susa']
rul = {'|dlogP| (primary)': [abs(math.log(E[s]['precip_mm_yr'] / S['precip_mm_yr'])) for s in O],
       'distance': [E[s]['dist_from_Susa_km'] for s in O],
       '|dT|': [abs(E[s]['T2M_C'] - S['T2M_C']) for s in O],
       '|delev|': [abs(E[s]['elev_m'] - S['elev_m']) for s in O]}
res = {}
for k, r in rul.items():
    rho = spearmanr(D, r).correlation
    perms = [spearmanr(D, [r[i] for i in p]).correlation for p in itertools.permutations(range(4))]
    p = np.mean([x >= rho - 1e-9 for x in perms])
    res[k] = (round(float(rho), 2), round(float(p), 3)); print(k, res[k])
pc = np.array([F['pc4'][s][0] for s in O])
for k, r in list(rul.items()) + [('precip', [E[s]['precip_mm_yr'] for s in O]), ('RH', [E[s]['RH2M_pct'] for s in O])]:
    print('PC1 undirected', k, round(abs(spearmanr(pc, r).correlation), 2))
