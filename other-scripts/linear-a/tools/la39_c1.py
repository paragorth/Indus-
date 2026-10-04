#!/usr/bin/env python3
"""LA-39 cycle 1: does a ligature modifier's effect learned on one base predict its effect on another
base, on held-out documents?  Shuffled-modifier null; Linear B known-meaning controls (sex markers
:m/:f/:x on animals, SI on SUS/BOS) vs same-sign different-word adjuncts (TE, PA, KU, A, O, QE);
planted modifier system on Linear A; base-sharing test.
"""
import sys, json, os
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la39_common as L

rng = np.random.default_rng(39)
NS, NPERM = 20, 400
GROUPS = {'LB': {'sex': {'m', 'f', 'x'}, 'SI': {'SI'}, 'acro': {'TE', 'PA', 'KU', 'A', 'O', 'QE'}}}


def wstat(pm, allowed=None):
    num = den = 0.0
    for m, v in pm.items():
        if allowed is not None and m not in allowed: continue
        for b, c, n in v:
            w = min(n, 20); num += w * c; den += w
    return num / den if den else np.nan


def run(A, groups, label):
    sp = L.make_splits(A, NS, rng)
    pm = defaultdict(list)
    c, g, npair = L.transfer_score(A, sp, per_mod=pm)
    obs = {'all_w': wstat(pm), 'all_cos': c, 'gain': g}
    for gn, gs in groups.items(): obs[gn] = wstat(pm, gs)
    null = defaultdict(list)
    for _ in range(NPERM):
        lab = L.shuffle_mods(A, rng)
        pm2 = defaultdict(list)
        c2, g2, _ = L.transfer_score(A, sp, modlabel=lab, per_mod=pm2)
        null['all_w'].append(wstat(pm2)); null['all_cos'].append(c2); null['gain'].append(g2)
        for gn, gs in groups.items(): null[gn].append(wstat(pm2, gs))
    out = {}
    for k, v in obs.items():
        nv = np.array([x for x in null[k] if not np.isnan(x)])
        p = (1 + (nv >= v).sum()) / (1 + len(nv)) if not np.isnan(v) else np.nan
        out[k] = (round(v, 3), round(float(nv.mean()), 3), round(float(p), 4))
    permod = {m: (round(float(np.average([c for _, c, _ in v], weights=[min(n, 20) for *_, n in v])), 3),
                  sorted(set(b for b, _, _ in v))) for m, v in pm.items()}
    print(label, 'pairs/split', npair, json.dumps(out), flush=True)
    return out, permod


def base_share(A, label, nperm=1000):
    """Held-out: type means on half A vs half B; within-base type pairs closer than cross-base?"""
    res = []
    for s in range(NS):
        tr = L.make_splits(A, 1, rng)[0]
        types = sorted(set(zip(A['base'].tolist(), A['mod'].tolist())))
        MA, MB, keep = [], [], []
        for t in types:
            k = (A['base'] == t[0]) & (A['mod'] == t[1])
            if (k & tr).sum() and (k & ~tr).sum():
                MA.append(A['X'][k & tr].mean(0)); MB.append(A['X'][k & ~tr].mean(0)); keep.append(t)
        MA, MB = np.array(MA), np.array(MB)
        Dm = ((MA[:, None] - MB[None]) ** 2).sum(-1)
        bases = np.array([b for b, _ in keep])
        iu = ~np.eye(len(keep), dtype=bool)

        def stat(bb):
            same = (bb[:, None] == bb[None]) & iu
            if same.sum() == 0: return np.nan
            return Dm[iu & ~same].mean() - Dm[same].mean()
        o = stat(bases)
        nv = np.array([stat(rng.permutation(bases)) for _ in range(nperm // NS)])
        res.append((o, nv))
    o = np.nanmean([r[0] for r in res]); nv = np.nanmean(np.array([r[1] for r in res]), 0)
    p = (1 + (nv >= o).sum()) / (1 + len(nv))
    print(label, 'base-share', round(o, 3), 'null', round(float(nv.mean()), 3), 'P', round(p, 4), flush=True)
    return round(float(o), 3), round(float(nv.mean()), 3), round(float(p), 4)


def plant(A, s, reps=10, nperm=100):
    """Plant: each shared modifier gets a random unit direction * s (SD units) added to its tokens."""
    sh = L.shared_mods(A)
    hits = []
    for r in range(reps):
        X = A['X'].copy()
        for m in sh:
            u = rng.normal(size=X.shape[1]); u /= np.linalg.norm(u)
            X[A['mod'] == m] += s * u
        sp = L.make_splits(A, 10, rng)
        o = L.transfer_score(A, sp, Xover=X)[0]
        nv = [L.transfer_score(A, sp, modlabel=L.shuffle_mods(A, rng), Xover=X)[0] for _ in range(nperm)]
        hits.append((1 + sum(v >= o for v in nv)) / (1 + nperm) <= 0.05)
    return sum(hits), reps


def shuffled_arm(A, reps=10, nperm=100):
    fp = 0
    for r in range(reps):
        lab = L.shuffle_mods(A, rng)
        B = dict(A); B['mod'] = lab
        sp = L.make_splits(B, 10, rng)
        o = L.transfer_score(B, sp)[0]
        nv = [L.transfer_score(B, sp, modlabel=L.shuffle_mods(B, rng))[0] for _ in range(nperm)]
        fp += (1 + sum(v >= o for v in nv)) / (1 + nperm) <= 0.05
    return fp, reps


if __name__ == '__main__':
    R = {}
    LAc = L.to_arrays(L.la_commodity_rows()); LAa = L.to_arrays(L.la_rows()); LB = L.to_arrays(L.lb_rows())
    R['LAc'] = run(LAc, {}, 'LA commodity')
    R['LAall'] = run(LAa, {}, 'LA all ligatures')
    R['LB'] = run(LB, GROUPS['LB'], 'LB')
    R['base_LAc'] = base_share(LAc, 'LA commodity')
    R['base_LB'] = base_share(LB, 'LB')
    R['plant'] = {s: plant(LAc, s) for s in (0.5, 1.0, 2.0)}
    print('plant', R['plant'], flush=True)
    R['shuffled'] = shuffled_arm(LAc)
    print('shuffled-arm false positives', R['shuffled'], flush=True)
    json.dump(R, open(os.path.join(L.CK, 'c1.json'), 'w'), indent=1, default=str)
