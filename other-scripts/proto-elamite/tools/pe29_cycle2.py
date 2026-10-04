"""pe29 cycle 2: cluster into sessions with all-but-one evidence (random weights, kNN graph, Louvain, many restarts),
then ask whether the held-out evidence is enriched inside the sessions.
Statistic: lift = mean standardised held-out similarity over pairs co-clustered in >= 50% of restarts.
Null A (shuffled evidence): every clustering evidence relabelled independently inside strata, same pipeline.
Datasets: PE (1,300 Susa), UR3 (Amar-Suen 5 Drehem, plus pair AUC for the true day), PLANT (cycle-1 planted PE).
usage: pe29_cycle2.py DATASET NREST NNULL
"""
import sys, json, time
import numpy as np
from pe29_common import *
from pe29_cycle1 import build, strata, plant, nlbin
import pe29_cycle1

D = sys.argv[1]; NR = int(sys.argv[2]); NN = int(sys.argv[3])
rng = np.random.default_rng(292)
pe29_cycle1.rng = np.random.default_rng(29)

if D in ('PE', 'PLANT'):
    T = pe_tablets(); V, B = pe_clay(T)
    if D == 'PLANT':
        T, V, lab = plant(T, V, B)
    S = build(T, (V, B))
    stB = [b if b is not None else f'nob{i}' for i, b in enumerate(B)]
else:
    T = ur3_tablets()
    for t in T:
        t['sys'] = ''
    S = build(T); stB = None
    lab = np.array([hash(t['date']) % (10 ** 9) for t in T])
st, _ = strata(T)
Z = {k: zmat(v) for k, v in S.items()}
evs = list(S)
out = {'D': D, 'n': len(T), 'rows': []}
t0 = time.time()
if D != 'PE':
    C, parts = cocluster(Z, evs, NR, rng)
    pl = lab if D == 'UR3' else np.where(lab >= 0, lab, -np.arange(1, len(lab) + 1))
    a = pair_auc(C, pl)
    sizes = np.bincount(parts[0])
    out['all_auc'] = a
    print(D, 'all-evidence co-clustering AUC for true sessions %.3f' % a, 'median cluster size', np.median(sizes), 'time', time.time() - t0, flush=True)
    # null: shuffled evidence
    nul = []
    for r in range(max(2, NN // 2)):
        Zs = {k: zmat(permute(S[k], stB if k == 'clay' else st, rng)) for k in evs}
        Cs, _ = cocluster(Zs, evs, NR, rng)
        nul.append(pair_auc(Cs, pl))
    out['all_auc_null'] = nul
    print(D, 'shuffled-evidence AUC', np.round(nul, 3), flush=True)
for e in evs:
    rest = [k for k in evs if k != e]
    C, parts = cocluster(Z, rest, NR, rng)
    L = lift(C, S[e])
    nl = []
    for r in range(NN):
        Zs = {k: zmat(permute(S[k], stB if k == 'clay' else st, rng)) for k in rest}
        Cs, _ = cocluster(Zs, rest, NR, rng)
        nl.append(lift(Cs, S[e])[0])
    nl = np.array(nl)
    p = (1 + (nl >= L[0]).sum()) / (NN + 1)
    row = dict(heldout=e, lift=L[0], npairs=L[1], corr=L[2], null_mu=float(np.nanmean(nl)), null_max=float(np.nanmax(nl)), p=float(p))
    out['rows'].append(row)
    print(D, 'held out', e, 'lift %.3f (pairs %d, corr %.4f) null %.3f max %.3f p %.3f' % (L[0], L[1], L[2], nl.mean(), nl.max(), p), 'time %.0f' % (time.time() - t0), flush=True)
    if D == 'PE' and e == evs[0]:
        pass
# keep the full-evidence consensus for cycle 3
C, parts = cocluster(Z, evs, NR, rng)
np.save(os.path.join(CK, f'C_{D}.npy'), C.astype(np.float16))
json.dump(out, open(os.path.join(CK, f'cycle2_{D}.json'), 'w'), indent=1)
