"""pe33 cycle 1: do seal motifs predict tablet content? Massive motif x feature search with
tablet-level (volume x size) and seal-block nulls, plus a planted motif-sign link."""
import json, sys, collections
import numpy as np
from pe33_common import load, matrices, strata_of, perm_tablet, perm_seal, assoc, CK

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rows, T = load(True)
X, F, Y, mot = matrices(rows)
print(len(rows), 'tablets', len({r['seal'] for r in rows}), 'seals', X.shape, mot, flush=True)
st = strata_of(rows)
rng = np.random.default_rng(33)


def summ(Y_):
    lor, z = assoc(X, Y_)
    az = np.abs(z)
    return az.max(), (az > 2.5).sum(), np.sort(az.ravel())[::-1][:20].sum(), az


real_max, real_n, real_top, real_az = summ(Y)
res = {}
for name, fn in [('tablet', lambda: perm_tablet(Y, st, rng)), ('seal', lambda: perm_seal(Y, rows, rng)),
                 ('seal_free', lambda: perm_seal(Y, rows, rng, by_vol=False))]:
    mx, nn, tp = [], [], []
    cell_exceed = np.zeros_like(real_az)
    for i in range(NP):
        m_, n_, t_, az = summ(fn())
        mx.append(m_); nn.append(n_); tp.append(t_)
        cell_exceed += az >= real_az
    mx, nn, tp = map(np.array, (mx, nn, tp))
    res[name] = dict(p_max=float((mx >= real_max).mean()), p_n=float((nn >= real_n).mean()),
                     p_top=float((tp >= real_top).mean()), null_n=float(nn.mean()), null_max95=float(np.quantile(mx, .95)),
                     fwer_links=[(mot[i], F[j], round(float(real_az[i, j]), 2)) for i, j in zip(*np.where(real_az > np.quantile(mx, .95)))],
                     cell_p_min=float((cell_exceed / NP).min()))
    print(name, real_max, real_n, real_top, res[name], flush=True)

lor, z = assoc(X, Y)
top = sorted([(abs(z[i, j]), mot[i], F[j], round(float(lor[i, j]), 2), int((Y[:, i] & X[:, j]).sum()), int(Y[:, i].sum()), int(X[:, j].sum()))
              for i in range(len(mot)) for j in range(len(F))], reverse=True)[:25]

# planted control: add a random mid-frequency sign to 60% of BOVID tablets (and remove from none)
plant = []
bi = mot.index('BOVID')
for k in range(20):
    r2 = np.random.default_rng(1000 + k)
    Xp = X.copy()
    cand = [j for j, f in enumerate(F) if f.startswith('S:') and 4 <= X[:, j].sum() <= 12]
    j = r2.choice(cand)
    on = np.where(Y[:, bi] == 1)[0]
    add = r2.choice(on, int(0.6 * len(on)), replace=False)
    Xp[add, j] = 1
    _, zp = assoc(Xp, Y)
    # null max under seal-block shuffles (200)
    mx = []
    for i in range(200):
        _, zz = assoc(Xp, perm_seal(Y, rows, r2))
        mx.append(np.abs(zz).max())
    thr = np.quantile(mx, .95)
    rank = int((np.abs(zp) > abs(zp[bi, j])).sum()) + 1
    plant.append(dict(sign=F[j], z=round(float(zp[bi, j]), 2), thr=round(float(thr), 2), found=bool(abs(zp[bi, j]) > thr), rank=rank))
print('planted', sum(p['found'] for p in plant), '/', len(plant), plant[:5])
json.dump(dict(n=len(rows), seals=len({r['seal'] for r in rows}), shape=X.shape, motifs=mot,
               real=dict(max=float(real_max), n25=int(real_n), top20=float(real_top)), nulls=res,
               top=[list(map(lambda v: v if not isinstance(v, np.generic) else v.item(), t)) for t in top],
               planted=plant), open(f'{CK}/cycle1.json', 'w'), indent=1, default=float)
for t in top[:15]:
    print(t)
