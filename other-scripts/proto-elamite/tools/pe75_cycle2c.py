"""pe75 cycle 2c: freeze the sealed and M297 physical models fitted on the 781 whole tablets
(data/pe75_frozen_physical.json, sha256 printed), then score them on the 175 '75%-preserved'
tablets never used (dims partly reduced by breakage: a noisy, biased test).  Statistic: held-out
log-loss gain over the text+header baseline (refitted on the 781) and AUC of the physical part."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, sys, hashlib, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_cycle2 as c2
import pe75_common as C
from pe18_common import seal_map

ALL = [r for r in C.pe_table() if r['h'] and r['w'] and r['t']]
S = seal_map()


def feats(R, ref=None):
    lg = lambda a: np.log(np.asarray(a, float) + 1)
    B = np.c_[lg([r['n_lines'] for r in R]), lg([r['glyphs'] for r in R]), [r['obv'] for r in R], [r['rev'] for r in R]]
    top = ref['top'] if ref else [s for s, _ in collections.Counter(r['header'][0] for r in R if r['header']).most_common(8)]
    Sx = np.c_[[[1.0 if r['header'] else 0] for r in R], [[1.0 if r['header'] and r['header'][0] == s else 0 for s in top] for r in R],
               [[1.0 if r['first_sys'] == s else 0 for s in ('SDB', 'C', 'NONE')] for r in R]]
    h, w, t = (np.log([r[k] for r in R]) for k in 'hwt')
    area = h + w
    if ref is None:
        ref = {'top': top, 'ta': list(np.polyfit(area, t, 1)), 'sa': list(np.polyfit(B[:, 1], area, 1))}
    P = np.c_[h, w, t, h - w, t - np.polyval(ref['ta'], area), area - np.polyval(ref['sa'], B[:, 1])]
    return np.c_[B, Sx], P, ref


tr = [r for r in ALL if r['complete_cat']]
te = [r for r in ALL if r['pres'] == '75%']
Xtr, Ptr, ref = feats(tr)
Xte, Pte, _ = feats(te, ref)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; pm, ps = Ptr.mean(0), Ptr.std(0) + 1e-9
Z = lambda X, m, s: (X - m) / s
targets = {'sealed': lambda r: S.get(r['id'], [False])[0], 'sign:M297': lambda r: 'M297' in r['signs']}
frozen = {'loop': 'pe75', 'date': '2026-10-07', 'test_set': "CDLI PE tablets with object_preservation '75%' (not used in fitting)", 'models': {}}
for k, f in targets.items():
    y = np.array([f(r) for r in tr]) * 1.0
    b0 = c2.irls(Z(Xtr, mu, sd), y); b1 = c2.irls(np.c_[Z(Xtr, mu, sd), Z(Ptr, pm, ps)], y)
    frozen['models'][k] = {'b0': list(np.round(b0, 6)), 'b1': list(np.round(b1, 6))}
frozen['norm'] = {'mu': list(mu), 'sd': list(sd), 'pm': list(pm), 'ps': list(ps), 'ref': ref}
s = json.dumps(frozen, sort_keys=True, default=float)
open(os.path.join(C.DATA, 'pe75_frozen_physical.json'), 'w').write(s)
print('sha256', hashlib.sha256(s.encode()).hexdigest())
out = {}
for k, f in targets.items():
    y = np.array([f(r) for r in te])
    b0 = np.array(frozen['models'][k]['b0']); b1 = np.array(frozen['models'][k]['b1'])
    X0 = Z(Xte, mu, sd); X1 = np.c_[X0, Z(Pte, pm, ps)]
    g = c2.llb(b1, X1, y) - c2.llb(b0, X0, y)
    phys = Z(Pte, pm, ps) @ b1[-6:]
    pos, neg = phys[y], phys[~y]
    auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else None
    rng = np.random.default_rng(1)
    nul = []
    for i in range(1000):
        yy = rng.permutation(y); nul.append(c2.llb(b1, X1, yy) - c2.llb(b0, X0, yy))
    out[k] = {'n': len(te), 'pos': int(y.sum()), 'gain_mbit': round(1000 * g, 1), 'auc_physical_part': auc,
              'p_label_perm': float(np.mean(np.array(nul) >= g))}
    print(k, out[k])
json.dump(out, open(os.path.join(C.CK, 'cycle2c.json'), 'w'), indent=1)
