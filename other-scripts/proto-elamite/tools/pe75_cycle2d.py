"""pe75 cycle 2d: is the M297 physical signal just 'a capacity (C-system) document'?  Same gain test
with the baseline extended by 'C system anywhere on the tablet' and 'number of C lines'; also the
other capacity signs (M002) and C-system presence itself as targets.  Train = 781 whole tablets
(10 random halves, null: physical rows permuted within volume x line bin, 200), held-out = 172
'75%' tablets (frozen-style fit on the 781)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_cycle2 as c2
import pe75_cycle2c as cc
import common

vol = {t['id']: t.get('volume', '') for t in common.load()}


def run(tr, te, tg, extra):
    Xtr, Ptr, ref = cc.feats(tr); Xte, Pte, _ = cc.feats(te, ref)
    E = lambda R: np.c_[[[('C' in r['systems']) * 1.0, sum(1 for s in r['systems'] if s.startswith('C'))] for r in R]] if extra else np.zeros((len(R), 0))
    Xtr = np.c_[Xtr, E(tr)]; Xte = np.c_[Xte, E(te)]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; pm, ps = Ptr.mean(0), Ptr.std(0) + 1e-9
    Xz, Pz = (Xtr - mu) / sd, (Ptr - pm) / ps
    y = np.array([tg(r) for r in tr])
    def g10(Pm):
        gs = []
        for s in range(10):
            sp = np.random.default_rng(500 + s).permutation(len(tr)); a, b = sp[: len(tr) // 2], sp[len(tr) // 2:]
            gs.append(c2.gain(Xz, Pm, y, a, b))
        return 1000 * np.mean(gs)
    real = g10(Pz)
    V = np.array(['%s|%d' % (vol.get(r['id'], ''), min(r['n_lines'] // 3, 5)) for r in tr])
    rng = np.random.default_rng(75); nul = []
    for i in range(200):
        perm = np.arange(len(tr))
        for s in np.unique(V):
            idx = np.where(V == s)[0]; perm[idx] = rng.permutation(idx)
        nul.append(g10(Pz[perm]))
    yt = np.array([tg(r) for r in te])
    b0 = c2.irls(Xz, y * 1.0); b1 = c2.irls(np.c_[Xz, Pz], y * 1.0)
    X0 = (Xte - mu) / sd; X1 = np.c_[X0, (Pte - pm) / ps]
    gh = 1000 * (c2.llb(b1, X1, yt) - c2.llb(b0, X0, yt))
    return {'train_gain': round(real, 1), 'null_mean': round(float(np.mean(nul)), 1), 'p': float(np.mean(np.array(nul) >= real)),
            'heldout_gain': round(gh, 1), 'pos_tr': int(y.sum()), 'pos_te': int(yt.sum()), 'coef': list(np.round(b1[-6:], 2))}


tr, te = cc.tr, cc.te
T = {'M297': lambda r: 'M297' in r['signs'], 'M002': lambda r: 'M002' in r['signs'],
     'Csys': lambda r: 'C' in r['systems'], 'M297|C': lambda r: 'M297' in r['signs']}
out = {}
for k, f in T.items():
    for extra in (False, True):
        if k == 'Csys' and extra:
            continue
        out['%s extraC=%s' % (k, extra)] = run(tr, te, f, extra)
        print(k, extra, out['%s extraC=%s' % (k, extra)])
json.dump(out, open(os.path.join(c2.C.CK, 'cycle2d.json'), 'w'), indent=1)
