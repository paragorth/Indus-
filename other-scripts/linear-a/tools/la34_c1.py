#!/usr/bin/env python3
"""la34 cycle 1: do blind shape features recover published Linear A hands?
Arms D (sign-agnostic ductus) and M (sign-matched form), full and scale-free; within-site label permutation.
Positive control 1 (real): the two sides of one tablet (same writer by construction) vs other documents of the site.
Positive control 2 (planted): fake hands (real hand partition permuted within site), each fake hand's drawings
sheared / stretched by a random amount of strength s before feature extraction."""
import numpy as np, json, os, sys, collections
from multiprocessing import Pool
from la34_common import load, CK
from la34_img import occ_table, unit_D, unit_M, labelled
from la34_score import standardize, dist, perm_test
import la34_feat

occ, meta, cid = load()
feat = json.load(open(os.path.join(CK, 'feat.json')))
res = {}


def score_units(o_, Z, P, codes, key, U, hands, sites, nperm=500):
    _, V, _ = unit_D(o_, Z, key)
    allU = sorted({o[key] for o in o_})
    idx = {u: i for i, u in enumerate(allU)}
    sel = [idx[u] for u in U]
    VD = V[sel]
    rD = perm_test(dist(VD), hands, sites, nperm)
    DM = unit_M(o_, Z, P, codes, U, key)
    rM = perm_test(DM, hands, sites, nperm)
    return rD, rM


def main():
    out = []
    for sf in (False, True):
        o_, names, Z, P, codes = occ_table(feat, scale_free=sf, occ=occ)
        nocc = collections.Counter(o['unit'] for o in o_)
        for mo in (3, 8):
            U = labelled(meta, sorted(nocc), nocc, mo)
            h = [meta[u]['scribe'] for u in U]; s = [meta[u]['site'] for u in U]
            rD, rM = score_units(o_, Z, P, codes, 'unit', U, h, s)
            row = {'scale_free': sf, 'min_occ': mo, 'units': len(U), 'hands': len(set(h)), 'D': rD, 'M': rM}
            out.append(row); print(json.dumps(row), flush=True)
        # side-pair positive control (documents as units, label = tablet)
        docs_by_tab = collections.defaultdict(set)
        for o in o_: docs_by_tab[o['unit']].add(o['doc'])
        ndoc = collections.Counter(o['doc'] for o in o_)
        Ud = sorted(d for d in ndoc if ndoc[d] >= 3)
        tab = {o['doc']: o['unit'] for o in o_}
        site = {o['doc']: o['site'] for o in o_}
        tabs2 = collections.Counter(tab[d] for d in Ud)
        Ud = [d for d in Ud if site[d] in {site[x] for x in Ud if tabs2[tab[x]] == 2}]
        h = [tab[d] for d in Ud]; s = [site[d] for d in Ud]
        rD, rM = score_units(o_, Z, P, codes, 'doc', Ud, h, s, 200)
        row = {'scale_free': sf, 'control': 'sides', 'docs': len(Ud), 'pairs': sum(v == 2 for v in tabs2.values()), 'D': rD, 'M': rM}
        out.append(row); print(json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)


def planted(strengths=(0.05, 0.1, 0.2), reps=3):
    o_all, names, Z, P, codes = occ_table(feat, scale_free=True, occ=occ)
    nocc = collections.Counter(o['unit'] for o in o_all)
    U = labelled(meta, sorted(nocc), nocc, 3)
    Uset = set(U)
    sub = [o for o in o_all if o['unit'] in Uset]
    out = []
    rng = np.random.default_rng(7)
    for s_ in strengths:
        for r in range(reps):
            h = np.array([meta[u]['scribe'] for u in U]); st = np.array([meta[u]['site'] for u in U])
            for site in np.unique(st):
                ix = np.where(st == site)[0]; h[ix] = rng.permutation(h[ix])
            fh = dict(zip(U, h))
            par = {x: (rng.uniform(-s_, s_), float(np.exp(rng.uniform(-s_, s_))), 0.0) for x in set(h)}
            files = [o['file'] for o in sub]; pa = [par[fh[o['unit']]] for o in sub]
            B = (len(files) + 1) // 2
            with Pool(2) as pool:
                rs = pool.starmap(la34_feat.run, [(files[:B], pa[:B]), (files[B:], pa[B:])])
            f2 = dict(zip(files, rs[0] + rs[1]))
            o2, _, Z2, P2, c2 = occ_table(f2, scale_free=True, occ=sub)
            rD, rM = score_units(o2, Z2, P2, c2, 'unit', U, list(h), list(st), 200)
            row = {'planted_strength': s_, 'rep': r, 'units': len(U), 'D': rD, 'M': rM}
            out.append(row); print(json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(CK, 'c1_planted.json'), 'w'), indent=1)


if __name__ == '__main__':
    if 'planted' in sys.argv: planted()
    else: main()
