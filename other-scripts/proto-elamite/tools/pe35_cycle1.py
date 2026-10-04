"""pe35 cycle 1: the pe33 single-link (cycle 1) and pooled cross-seal (cycle 2) tests re-run on the ENLARGED seal set
(pe33 Legrain rows + pe35 Louvre/Amiet rows), with the same nulls (volume x size tablet shuffle; seal-block within volume;
free seal-block) and planted controls. Usage: pe35_cycle1.py [NPERM]"""
import json, sys, collections
import numpy as np
from pe35_common import load, CK
from pe33_common import matrices, strata_of, perm_tablet, perm_seal, assoc, MOTIFS
from pe33_cycle2 import vecs, run

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
which = sys.argv[2] if len(sys.argv) > 2 else 'all'
rows, T = load(which)
X, F, Y, mot = matrices(rows)
nseal = len({r['seal'] for r in rows})
print(len(rows), 'tablets', nseal, 'seals', X.shape, mot, flush=True)
st = strata_of(rows)
rng = np.random.default_rng(35)
out = dict(n=len(rows), seals=nseal, motifs=mot, which=which,
           motif_counts={m: int(sum(m in r['motif'] for r in rows)) for m in MOTIFS})

# --- single links (pe33.1)
_, z = assoc(X, Y); az = np.abs(z)
for name, fn in [('tablet', lambda: perm_tablet(Y, st, rng)), ('seal', lambda: perm_seal(Y, rows, rng))]:
    mx = np.array([np.abs(assoc(X, fn())[1]).max() for _ in range(NP)])
    thr = float(np.quantile(mx, .95))
    out['single_' + name] = dict(max=float(az.max()), p_max=float((mx >= az.max()).mean()), fwer95=thr,
                                 links=[(mot[i], F[j], round(float(az[i, j]), 2)) for i, j in zip(*np.where(az > thr))])
    print('single', name, out['single_' + name], flush=True)
top = sorted([(round(float(az[i, j]), 2), mot[i], F[j], int((Y[:, i] & X[:, j]).sum()), int(Y[:, i].sum()), int(X[:, j].sum()))
              for i in range(len(mot)) for j in range(len(F))], reverse=True)[:12]
out['top'] = top
print(top[:8], flush=True)

# --- pooled cross-seal similarity (pe33.2)
S = vecs([r['content'] for r in rows])
real, p, pp, _ = run(rows, S, mot, NP, rng, lambda Yx: perm_seal(Yx, rows, rng))
real2, p2, pp2, _ = run(rows, S, mot, NP, rng, lambda Yx: perm_seal(Yx, rows, rng, by_vol=False))
out['pooled'] = dict(p_vol=pp, p_free=pp2, per_motif={m: [round(float(a), 4), float(b), float(c)] for m, a, b, c in zip(mot, real, p, p2)},
                     nseal={m: len({r['seal'] for r in rows if m in r['motif']}) for m in mot})
print('pooled', out['pooled'], flush=True)

# --- planted controls at the new size
bi = mot.index('BOVID')
found1, found2 = [], []
for k in range(20):
    r2 = np.random.default_rng(3500 + k)
    cand = [j for j, f in enumerate(F) if f.startswith('S:') and 4 <= X[:, j].sum() <= 12]
    j = r2.choice(cand)
    on = np.where(Y[:, bi] == 1)[0]
    Xp = X.copy(); Xp[r2.choice(on, int(0.6 * len(on)), replace=False), j] = 1
    _, zp = assoc(Xp, Y)
    mx = [np.abs(assoc(Xp, perm_seal(Y, rows, r2))[1]).max() for _ in range(200)]
    found1.append(bool(abs(zp[bi, j]) > np.quantile(mx, .95)))
    signs = collections.Counter(f for r in rows for f in r['content'] if f.startswith('S:'))
    pl = set(r2.choice([s for s, c in signs.items() if 3 <= c <= 10], 3, replace=False))
    rows2 = [dict(r, content=set(r['content']) | (pl if ('BOVID' in r['motif'] and r2.random() < 0.4) else set())) for r in rows]
    rr, pq, ppq, _ = run(rows2, vecs([r['content'] for r in rows2]), mot, 200, r2, lambda Yx: perm_seal(Yx, rows2, r2))
    found2.append(float(pq[bi]))
out['planted_single_power'] = float(np.mean(found1))
out['planted_pooled_power'] = float(np.mean(np.array(found2) < 0.05))
print('planted', out['planted_single_power'], out['planted_pooled_power'], flush=True)
json.dump(out, open(f'{CK}/cycle1_{which}.json', 'w'), indent=1, default=str)
