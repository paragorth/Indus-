#!/usr/bin/env python3
"""LA-56 cycle 2: minimal explanations of every total that does not close, and who gets the blame.

For every section that does not close under the default reading, enumerate EVERY reading with
1, 2 or 3 changes: a main item dropped (0), subtracted (-1) or doubled (2); a sub-total,
after-total item or other-side total (osub) added. Unknown fraction values: D random draws;
a reading 'explains' the section with weight = share of draws under which it closes.
Read-off per item: posterior share of minimal explanations that change it (and how).
Feature enrichment: sum over sections of that posterior for items with feature f, against
a within-section label permutation (feature sets permuted among the items of each section,
2,000 runs), family-wise max over features.
Also the cost profile: share of non-closing sections explained with <= 1 / <= 2 / <= 3 changes,
real vs N1 / N2 / N3 / planted corpora (built with la56_c1.make).

Usage: python3 la56_c2.py CORPUS JOB nseeds
"""
import sys, os, json, random, itertools
import numpy as np
import la56_common as C
import la56_c1 as C1

MAXC = 3
MAX_ITEMS = 22


def explain(eng, k):
    p = eng.pre[k]; s = eng.secs[k]
    items = [j for j, it in enumerate(s['items']) if it['role'] in ('main', 'sub', 'after', 'osub')]
    items = items[:MAX_ITEMS]
    alts = {}
    for j in items:
        alts[j] = (0, -1, 2) if s['items'][j]['role'] == 'main' else (1,)
    ev = p['ints'][None, :] + (eng.V @ p['Lm'].T)          # D x n
    tot = p['tint'] + eng.V @ p['tl']
    base = ev @ p['base_w']
    need = tot - base                                       # D
    sols = []
    for c in range(1, MAXC + 1):
        for comb in itertools.combinations(items, c):
            for ws in itertools.product(*[alts[j] for j in comb]):
                delta = np.zeros_like(need)
                for j, w in zip(comb, ws):
                    delta = delta + (w - p['base_w'][j]) * ev[:, j]
                pr = float(np.mean(delta == need))
                if pr > 0: sols.append((c, comb, ws, pr))
        if sols: break
    return sols


def run(name, job, seed):
    secs = C1.load(name)
    S, info = C1.make(secs, job, random.Random(seed * 7919 + sum(map(ord, job)))) if job != 'real' else (secs, {})
    eng = C.Engine(S, D=400, seed=seed)
    pc = eng.base.mean(1)
    res = []
    for k in range(len(S)):
        if pc[k] >= 0.5: continue
        sols = explain(eng, k)
        cost = sols[0][0] if sols else None
        Z = sum(x[3] for x in sols) or 1.0
        post = {}
        for c, comb, ws, pr in sols:
            for j, w in zip(comb, ws):
                post.setdefault(j, {}); post[j][w] = post[j].get(w, 0) + pr / Z
        res.append({'k': k, 'id': S[k]['id'], 'cost': cost, 'nsol': len(sols),
                    'post': {str(j): v for j, v in post.items()},
                    'sols': [(c, list(cm), list(w), pr) for c, cm, w, pr in sols[:12]]})
    out = {'corpus': name, 'job': job, 'seed': seed, 'info': info, 'n': len(S), 'n_bad': len(res), 'res': res}
    # feature enrichment
    out['enrich'] = enrich(S, res)
    json.dump(out, open(os.path.join(C.CK, 'c2_%s_%s_%d.json' % (name, job, seed)), 'w'))
    costs = [r['cost'] for r in res]
    print(name, job, seed, 'bad', len(res), 'cost<=1 %d <=2 %d <=3 %d none %d' % (
        sum(c == 1 for c in costs), sum(c is not None and c <= 2 for c in costs),
        sum(c is not None for c in costs), sum(c is None for c in costs)),
        'top', out['enrich']['top'][:3], flush=True)


def enrich(S, res, nperm=2000, seed=1):
    """Observed blame per feature (drop / negate / double / add) vs within-section label permutation."""
    rng = np.random.default_rng(seed)
    feats = sorted({f for r in res for it in S[r['k']]['items'] for f in it['f']
                    if not f.startswith('pos=') or True})
    fi = {f: i for i, f in enumerate(feats)}
    blocks = []   # per section: (F matrix items x feats, blame vector items x 4)
    ops = {0: 0, -1: 1, 2: 2, 1: 3}
    for r in res:
        if not r['post']: continue
        its = [it for it in S[r['k']]['items'] if it['role'] in ('main', 'sub', 'after', 'osub')]
        keep = {j for j, it in enumerate(S[r['k']]['items']) if it['role'] in ('main', 'sub', 'after', 'osub')}
        remap = {j: i for i, j in enumerate(sorted(keep))}
        n = len(its)
        F = np.zeros((n, len(feats)), dtype=np.float32)
        for j, it in enumerate(its):
            for f in it['f']: F[j, fi[f]] = 1
        Bm = np.zeros((n, 4), dtype=np.float32)
        for j, d in r['post'].items():
            for w, v in d.items(): Bm[remap[int(j)], ops[int(w)]] += v
        # permutation only among items of the same role (main with main, extra with extra)
        roles = np.array([it['role'] == 'main' for it in its])
        blocks.append((F, Bm, roles))
    if not blocks: return {'top': []}
    obs = sum(F.T @ Bm for F, Bm, _ in blocks)          # feats x 4
    perm = np.zeros((nperm,) + obs.shape, dtype=np.float32)
    for t in range(nperm):
        acc = np.zeros_like(obs)
        for F, Bm, roles in blocks:
            idx = np.arange(len(roles))
            for g in (True, False):
                sel = np.where(roles == g)[0]
                idx[sel] = rng.permutation(sel)
            acc += F[idx].T @ Bm
        perm[t] = acc
    mu = perm.mean(0); sd = perm.std(0) + 1e-6
    z = (obs - mu) / sd
    zp = (perm - mu) / sd
    # family-wise: max z over features with support (obs or mean >= 0.5) per op
    names = ['drop', 'neg', 'dbl', 'add']
    top = []
    for o in range(4):
        sup = (obs[:, o] >= 0.75)
        if not sup.any(): continue
        mx = zp[:, sup, o].max(1)
        for i in np.where(sup)[0]:
            pfw = float(np.mean(mx >= z[i, o]))
            praw = float(np.mean(perm[:, i, o] >= obs[i, o]))
            top.append((feats[i], names[o], round(float(obs[i, o]), 2), round(float(mu[i, o]), 2),
                        round(float(z[i, o]), 2), praw, pfw))
    top.sort(key=lambda x: (x[6], x[5], -x[4]))
    return {'top': top[:40]}


if __name__ == '__main__':
    name, job = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for sd in range(n):
        if os.path.exists(os.path.join(C.CK, 'c2_%s_%s_%d.json' % (name, job, sd))): continue
        run(name, job, sd)
