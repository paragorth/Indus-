"""pe57 cycle 2: test the frozen pe55 prediction on the vessel capacities.
Rule (frozen 5 Oct): survive if a capacity mode lies in 0.5-1.0 l or 4-9 l; support if a mode is near 2.8 l
(operationalised as 2.33-3.36 l, x/1.2); killed if no mode in either window.
Mode finders: KDE on log10 volume (Silverman bandwidth, peaks > 10% of max) and BIC Gaussian mixture; 2,000
bootstrap resamples each.  Nulls: (1) matched random draws from a broad vessel prior: n vessels from 1-3
types, type centres log-uniform 0.05-50 l, within-type CV 0.15-0.35 (the observed range), 2,000 sets; the
rate at which a random assemblage 'survives' is the chance level of the test. (2) vessel sets from an
unrelated period (Argaric bowls, Bronze Age Spain, same 3D pipeline)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pe57_common import *

OUT = os.path.join(CK, 'cycle2' + ('_argaric' if os.environ.get('PE57_ARGARIC') else '') + '.json')
res = {}
sets = {'AbuSalabikh_JN_all': abu_salabikh(), 'AbuSalabikh_round': abu_salabikh(' R'),
        'AbuSalabikh_flat': abu_salabikh(' F'), 'JebelAruda_all': mesh_set('aruda'),
        'JebelAruda_BRB': mesh_set('aruda', bowls_only=True)}
sets['Uruk_JN_pooled'] = np.r_[sets['AbuSalabikh_JN_all'], sets['JebelAruda_all']]
if os.environ.get('PE57_ARGARIC') and os.path.exists(os.path.join(CK, 'mesh_argaric.json')):
    sets['Argaric_null'] = mesh_set('argaric')
for name, x in sets.items():
    if len(x) < 5: continue
    km, bw = kde_modes(x)
    gm, gw = gmm_modes(x)
    bk, allk = bootstrap(x, 2000, use='kde')
    bg, allg = bootstrap(x, 100, use="gmm")
    res[name] = {'n': int(len(x)), 'median': float(np.median(x)), 'range': [float(x.min()), float(x.max())],
                 'kde_modes': km.round(3).tolist(), 'bw_log10': float(bw), 'gmm_means': gm.round(3).tolist(),
                 'gmm_w': gw.round(2).tolist(), 'verdict_kde': verdict(km), 'verdict_gmm': verdict(gm),
                 'boot_kde': bk, 'boot_gmm': bg,
                 'kde_mode_q': np.percentile(allk, [5, 25, 50, 75, 95]).round(3).tolist()}
    print(name, json.dumps(res[name]), flush=True)

# Null 1: matched random draws from a broad vessel prior
r = np.random.default_rng(7)
for n in (10, 33, 43):
    cnt = {'survive': 0, 'inA': 0, 'inB': 0, 'support28': 0}
    N = 2000
    for _ in range(N):
        k = r.integers(1, 4)
        cen = 10 ** r.uniform(np.log10(0.05), np.log10(50), k)
        cv = r.uniform(0.15, 0.35)
        w = r.dirichlet(np.ones(k))
        typ = r.choice(k, n, p=w)
        x = cen[typ] * np.exp(r.normal(0, cv, n))
        v = verdict(kde_modes(x)[0])
        for kk in cnt: cnt[kk] += v[kk]
    res[f'null_prior_n{n}'] = {kk: vv / N for kk, vv in cnt.items()}
    print('null prior n', n, res[f'null_prior_n{n}'], flush=True)
# Null 1b: prior restricted to 'small open bowls' (centres 0.2-5 l), i.e. given only that the vessels are hand-held bowls
for n in (10, 43):
    cnt = {'survive': 0, 'inA': 0, 'inB': 0, 'support28': 0}
    N = 2000
    for _ in range(N):
        k = r.integers(1, 4)
        cen = 10 ** r.uniform(np.log10(0.2), np.log10(5), k)
        cv = r.uniform(0.15, 0.35); w = r.dirichlet(np.ones(k)); typ = r.choice(k, n, p=w)
        x = cen[typ] * np.exp(r.normal(0, cv, n))
        v = verdict(kde_modes(x)[0])
        for kk in cnt: cnt[kk] += v[kk]
    res[f'null_bowlprior_n{n}'] = {kk: vv / N for kk, vv in cnt.items()}
    print('null bowl prior n', n, res[f'null_bowlprior_n{n}'], flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
