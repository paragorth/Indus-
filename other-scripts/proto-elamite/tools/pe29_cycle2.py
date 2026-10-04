"""pe29 cycle 2: link tablets into sessions with all-but-one evidence (random weights and evidence subsets, mutual 2-NN links, many restarts),
then ask whether the held-out evidence is enriched inside the sessions.
Statistic: lift = mean standardised held-out similarity over pairs linked in >= 30% of restarts; truth precision for UR3/PLANT.
Null A (shuffled evidence): every clustering evidence relabelled independently inside strata, same pipeline.
Datasets: PE (1,300 Susa), UR3 (Amar-Suen 5 Drehem, plus pair AUC for the true day), PLANT (cycle-1 planted PE).
usage: pe29_cycle2.py DATASET NREST NNULL
"""
import sys, json, time
import numpy as np
from pe29_common import *
from pe29_cycle1 import build, strata, plant, nlbin
import pe29_cycle1

THR = 0.3
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
EXCL = os.environ.get('EXCL', '')
BLK = int(os.environ.get('NULLBLK', '0'))
for e in [x for x in EXCL.split(',') if x]:
    S.pop(e, None)
if BLK:  # finer null: relabel only inside runs of BLK consecutive museum numbers (keeps excavation-lot structure)
    st = [f"{t['pre']}|{(t['no'] // BLK) if t['no'] is not None else 'na' + str(i)}" for i, t in enumerate(T)]
    stB = [f'{a}|{b}' for a, b in zip(stB, st)] if stB is not None else None
TAG = D + (('_ex' + EXCL.replace(',', '')) if EXCL else '') + (f'_blk{BLK}' if BLK else '') + ('_hf' if HANDFIX else '')
Z = {k: zmat(v) for k, v in S.items()}
evs = list(S)
out = {'D': D, 'n': len(T), 'rows': []}
t0 = time.time()
if D != 'PE':
    pl = lab if D == 'UR3' else np.where(lab >= 0, lab, -np.arange(1, len(lab) + 1))
    same = pl[:, None] == pl[None, :]
    iu = np.triu_indices(len(pl), 1); base_rate = same[iu].mean()
    def prec(C, thr=THR):
        L = np.triu(C >= thr, 1); return float(same[L].mean()) if L.sum() else float('nan'), int(L.sum())
    for sub in [evs] + [[e] for e in evs]:
        C = colink(Z, sub, NR, rng, drop=0.3 if len(sub) > 1 else 0)
        pr, nlk = prec(C)
        out.setdefault('truth', []).append(dict(evs=sub, precision=pr, links=nlk, base=float(base_rate)))
        print(D, 'evidence', '+'.join(sub), 'links %d precision %.3f base %.4f enrichment %.1f' % (nlk, pr, base_rate, pr / base_rate), flush=True)
    nul = []
    for r in range(max(2, NN // 2)):
        Zs = {k: zmat(permute(S[k], stB if k == 'clay' else st, rng)) for k in evs}
        nul.append(prec(colink(Zs, evs, NR, rng, drop=0.3))[0])
    out['truth_null'] = nul
    print(D, 'shuffled-evidence precision', np.round(nul, 4), flush=True)
for e in evs:
    rest = [k for k in evs if k != e]
    C = colink(Z, rest, NR, rng, drop=0.3)
    L = lift(C, S[e], THR)
    nl = []
    for r in range(NN):
        Zs = {k: zmat(permute(S[k], stB if k == 'clay' else st, rng)) for k in rest}
        Cs = colink(Zs, rest, NR, rng, drop=0.3)
        nl.append(lift(Cs, S[e], THR)[0])
    nl = np.array(nl)
    p = (1 + (nl >= L[0]).sum()) / (NN + 1)
    row = dict(heldout=e, lift=L[0], npairs=L[1], corr=L[2], null_mu=float(np.nanmean(nl)), null_max=float(np.nanmax(nl)), p=float(p))
    out['rows'].append(row)
    print(D, 'held out', e, 'lift %.3f (pairs %d, corr %.4f) null %.3f max %.3f p %.3f' % (L[0], L[1], L[2], nl.mean(), nl.max(), p), 'time %.0f' % (time.time() - t0), flush=True)
    if D == 'PE' and e == evs[0]:
        pass
# keep the full-evidence consensus for cycle 3
C = colink(Z, evs, NR, rng, drop=0.3)
np.save(os.path.join(CK, f'C_{TAG}.npy'), C.astype(np.float16))
json.dump(out, open(os.path.join(CK, f'cycle2_{TAG}.json'), 'w'), indent=1)
