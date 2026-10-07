"""pe75 cycle 2b: the two cycle-2 per-target hits (M297 presence, sealed) against a stricter null:
physical rows permuted within publication volume x line-count bin (lot/genre confound), 200 perms;
leave-one-volume-out; univariate direction of each physical feature."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_cycle2 as c2
import common

R, B, Sx, P, Y, lb = c2.build()
vol = {t['id']: t.get('volume', '') for t in common.load()}
V = np.array([vol.get(r['id'], '') for r in R])
Xb = c2.std(np.c_[B, Sx]); Pz = c2.std(P)
names = ['logh', 'logw', 'logt', 'aspect', 'thick_resid', 'slack']
strata = np.array(['%s|%d' % (v, b) for v, b in zip(V, lb)])
out = {}
for tg in ('sign:M297', 'sealed', 'total', 'reverse', 'PLANT'):
    y = Y[tg]
    def g10(Pm):
        gs = []
        for s in range(10):
            sp = np.random.default_rng(500 + s).permutation(len(B)); tr, te = sp[: len(B) // 2], sp[len(B) // 2:]
            gs.append(c2.gain(Xb, Pm, y, tr, te))
        return np.mean(gs) * 1000
    real = g10(Pz)
    rng = np.random.default_rng(7520)
    nul = []
    for i in range(200):
        perm = np.arange(len(B))
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]; perm[idx] = rng.permutation(idx)
        nul.append(g10(Pz[perm]))
    # leave-one-volume-out (volumes with >= 30 tablets)
    lovo = {}
    for v, n in collections.Counter(V).items():
        if n >= 30:
            te = np.where(V == v)[0]; tr = np.where(V != v)[0]
            lovo[v] = round(1000 * c2.gain(Xb, Pz, y, tr, te), 1)
    b = c2.irls(np.c_[Xb, Pz], y * 1.0)[-6:]
    out[tg] = {'n_pos': int(y.sum()), 'gain_mbit': round(real, 1), 'null_vol_line_mean': round(float(np.mean(nul)), 1),
               'null_sd': round(float(np.std(nul)), 1), 'p': float(np.mean(np.array(nul) >= real)),
               'lovo': lovo, 'coef_physical': dict(zip(names, np.round(b, 2)))}
    print(tg, out[tg])
json.dump(out, open(os.path.join(c2.C.CK, 'cycle2b.json'), 'w'), indent=1, default=float)
