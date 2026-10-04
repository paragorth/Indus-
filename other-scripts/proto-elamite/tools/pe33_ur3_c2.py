"""pe33 cycle 2 control: the pooled cross-seal similarity statistic on Ur III legends (title words as
'pictures'), at PE size (39 seals, ~50 tablets) and at 10x PE size."""
import sys, json, collections
import numpy as np
import pe33_ur3 as U   # builds rows only when NSUB=0 (runs its own full test once)
from pe33_cycle2 import vecs, run
from pe33_common import CK

rows = U.rows
byseal = collections.defaultdict(list)
for r in rows:
    byseal[r['seal']].append(r)
seals = [s for s, v in byseal.items() if any(r['motif'] for r in v)]
other = [s for s, v in byseal.items() if not any(r['motif'] for r in v)]


def seal_perm(R, rng):
    return lambda Y: U.seal_perm(Y, R, rng)


res = {}
for label, nS, nO, cap in (('PE', 30, 9, 63), ('10xPE', 300, 90, 630)):
    ps, pooled = collections.defaultdict(list), []
    for k in range(25 if label == 'PE' else 8):
        rr = np.random.default_rng(900 + k)
        S = list(rr.choice(seals, nS, replace=False)) + list(rr.choice(other, nO, replace=False))
        R = []
        for s in S:
            v = byseal[s]
            R += [v[i] for i in rr.choice(len(v), min(len(v), int(rr.integers(1, 4))), replace=False)]
        R = R[:cap]
        mot = [m for m in U.MOT if sum(m in r['motif'] for r in R) >= 4 and len({r['seal'] for r in R if m in r['motif']}) >= 2]
        if not mot:
            continue
        Sm = vecs([r['content'] for r in R])
        real, p, pp, _ = run(R, Sm, mot, 300 if label == 'PE' else 200, rr, seal_perm(R, rr))
        pooled.append(pp)
        for m, v in zip(mot, p):
            ps[m].append(float(v))
        print(label, k, len(R), 'pooled p', pp, {m: round(float(v), 3) for m, v in zip(mot, p)}, flush=True)
    res[label] = dict(pooled_power=float(np.mean(np.array(pooled) < 0.05)), pooled_p=pooled,
                      motif_power={m: float(np.mean(np.array(v) < 0.05)) for m, v in ps.items()})
    print(label, res[label]['pooled_power'], res[label]['motif_power'], flush=True)
json.dump(res, open(f'{CK}/ur3_c2.json', 'w'), indent=1)
