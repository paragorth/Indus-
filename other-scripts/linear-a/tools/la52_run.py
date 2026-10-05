#!/usr/bin/env python3
"""LA-52 chain runner. usage: la52_run.py TAG COND NCHAINS WORKER [G] [kind]
COND: LA, LAS (shuffled start), LAP (planted), LAO (arbitrary-preservation objective),
      LB, LBS, UR, URS, LAA / LAB (disjoint halves of LA, cycle 3), LAAS / LABS.
Writes data/la52_ckpt/TAG_COND_wWORKER.npy (chains x gens x 2 x F: count, shuffle baseline)
and TAG_COND_feats.json (tracked features, gen-0 real counts and 20-shuffle baselines)."""
import sys, os, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la52_common as C

tag, cond, n, w = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
G = int(sys.argv[5]) if len(sys.argv) > 5 else 6
kind = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != 'mix' else None
MEAS = [0, 1, 2, 3, 4, 6] if G == 6 else sorted(set([0, 1, 2, G // 2, G]))


def start_corpus(base):
    la = C.la_docs()
    T = C.ntok(la)
    if base == 'LA':
        return la
    if base == 'LAP':
        return C.plant(la, random.Random(C.seed('la52-plant')))
    if base == 'LB':
        return C.lb_docs(T, random.Random(C.seed('la52-lb')))
    if base == 'UR':
        return C.ur_docs(T, random.Random(C.seed('la52-ur')))
    if base in ('LAA', 'LAB'):
        r = random.Random(C.seed('la52-halves'))
        idx = list(range(len(la))); r.shuffle(idx)
        h = idx[:len(la) // 2] if base == 'LAA' else idx[len(la) // 2:]
        return [la[i] for i in sorted(h)]
    if base in ('LBA', 'LBB'):
        lb = C.lb_docs(T, random.Random(C.seed('la52-lb')))
        r = random.Random(C.seed('la52-halves-lb'))
        idx = list(range(len(lb))); r.shuffle(idx)
        h = idx[:len(lb) // 2] if base == 'LBA' else idx[len(lb) // 2:]
        return [lb[i] for i in sorted(h)]
    if base in ('URA', 'URB'):
        ur = C.ur_docs(T, random.Random(C.seed('la52-ur')))
        r = random.Random(C.seed('la52-halves-ur'))
        idx = list(range(len(ur))); r.shuffle(idx)
        h = idx[:len(ur) // 2] if base == 'URA' else idx[len(ur) // 2:]
        return [ur[i] for i in sorted(h)]
    raise ValueError(base)


base = {'LAS': 'LA', 'LBS': 'LB', 'URS': 'UR', 'LAO': 'LA', 'LAAS': 'LAA', 'LABS': 'LAB', 'LBAS': 'LBA', 'LBBS': 'LBB', 'URAS': 'URA', 'URBS': 'URB'}.get(cond, cond)
docs0 = start_corpus(base)
fj = os.path.join(C.CK, f'{tag}_{base}_feats.json')
if not os.path.exists(fj):
    r = random.Random(C.seed('la52-f0' + base))
    c, b = C.excess(docs0, r, 20)
    feats = sorted([f for f, v in c.items() if v >= 2])
    json.dump({'feats': [C.fkey(f) for f in feats], 'c0': [c[f] for f in feats],
               'b0': [b.get(f, 0.0) for f in feats], 'ndocs': len(docs0), 'ntok': C.ntok(docs0)}, open(fj + '.tmp', 'w'))
    os.replace(fj + '.tmp', fj)
meta = json.load(open(fj))
feats = [C.unkey(s) for s in meta['feats']]
objective = None
if cond == 'LAO':
    r = random.Random(C.seed('la52-objective'))
    e0 = np.array(meta['c0']) - np.array(meta['b0'])
    cand = [i for i in range(len(feats)) if e0[i] >= 3]
    pick = r.sample(cand, 40)
    objective = {feats[i]: meta['c0'][i] for i in pick}
    json.dump([meta['feats'][i] for i in pick], open(os.path.join(C.CK, f'{tag}_LAO_targets.json'), 'w'))

rng = random.Random(C.seed(f'la52-{tag}-{cond}-{w}-{kind}'))
out = os.path.join(C.CK, f'{tag}_{cond}{"_" + kind if kind else ""}_w{w}.npy')
logf = open(out.replace('.npy', '.log'), 'a')
arr = np.zeros((n, len(MEAS), 2, len(feats)), np.float32)
names = []
done = 0
t0 = time.time()
for k in range(n):
    start = C.kind_shuffle(docs0, rng) if cond.endswith('S') else docs0
    res, nm = C.chain(start, rng, G, feats, set(MEAS), kind=kind, objective=objective)
    for gi, g in enumerate(MEAS):
        arr[k, gi, 0] = res[g][0]; arr[k, gi, 1] = res[g][1]
    names.append(nm)
    done = k + 1
    if done % 10 == 0 or done == n:
        np.save(out + '.tmp.npy', arr[:done]); os.replace(out + '.tmp.npy', out)
        json.dump({'meas': MEAS, 'names': names, 'G': G}, open(out.replace('.npy', '_meta.json'), 'w'))
        print(f'{cond} {done}/{n} {time.time() - t0:.0f}s', file=logf, flush=True)
