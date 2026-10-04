"""pe35 cycle 3: power at the enlarged size. The Ur III calibration of pe33 (seal-legend title words as 'pictures'),
pooled cross-seal statistic and single-link FWER test, subsampled to the enlarged PE size (NT tablets, NS seals)
and to the maximum reachable size (all CDLI-flagged sealed PE tablets).
Usage: pe35_cycle3.py SCRATCH NT NS [DRAWS]"""
import sys, json, collections
import numpy as np
SCR, NT, NS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
DR = int(sys.argv[4]) if len(sys.argv) > 4 else 20
sys.argv = [sys.argv[0], SCR, '0']
import pe33_ur3 as U
from pe33_cycle2 import vecs, run
from pe35_common import CK

byseal = collections.defaultdict(list)
for r in U.rows:
    byseal[r['seal']].append(r)
seals = [s for s, v in byseal.items() if any(r['motif'] for r in v)]
other = [s for s, v in byseal.items() if not any(r['motif'] for r in v)]
res = {}
SETS = [('pe35', NT, NS)] if NT else [('max183', 183, 120), ('480', 480, 300)]
for label, nt, ns in SETS:
    pooled, single = [], []
    for k in range(DR):
        rr = np.random.default_rng(3500 + k)
        nm = int(round(ns * 0.77))
        S = list(rr.choice(seals, nm, replace=False)) + list(rr.choice(other, ns - nm, replace=False))
        R = []
        for s in S:
            v = byseal[s]
            R += [v[i] for i in rr.choice(len(v), min(len(v), int(rr.integers(1, 4))), replace=False)]
        R = R[:nt]
        mot = [m for m in U.MOT if sum(m in r['motif'] for r in R) >= 4 and len({r['seal'] for r in R if m in r['motif']}) >= 2]
        if not mot:
            continue
        _, p, pp, _ = run(R, vecs([r['content'] for r in R]), mot, 200, rr, lambda Y: U.seal_perm(Y, R, rr))
        pooled.append(pp)
        o = U.test(R, 200, rr)
        if o:
            single.append(o['p_max'])
        print(label, k, len(R), 'pooled', pp, 'single', o and o['p_max'], flush=True)
    res[label] = dict(n=nt, seals=ns, draws=len(pooled), pooled_power=float(np.mean(np.array(pooled) < .05)),
                      single_power=float(np.mean(np.array(single) < .05)))
    print(label, res[label], flush=True)
json.dump(res, open(f'{CK}/cycle3_%s.json' % ('pe35' if NT else 'ref'), 'w'), indent=1)
