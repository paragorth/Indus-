#!/usr/bin/env python3
"""LA-52 cycle 2 analysis: single-learner-type chains (gentle regime). A feature counts as
world-anchored only if EVERY learner type loses it: anchoring = 1 - max over types of AUC.
usage: la52_c2.py TAG"""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la52_common as C
import la52_report as R

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c2'
R.TAG = TAG
KINDS = C.TYPES


def per_kind(base, real, shuf):
    out = {}
    for k in KINDS:
        R.SUF = '_' + k
        try:
            r = R.retention(base, real, shuf, emin=3.0, boot=50)
        except FileNotFoundError:
            r = None
        if r is not None:
            out[k] = r
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    m = ~(np.isnan(a) | np.isnan(b))
    return spearmanr(a[m], b[m]).correlation if m.sum() > 5 else float('nan')


for base, real, shuf, extra in [('LAP', 'LAP', 'LAS', list(C.PLANT_FEATS)), ('LA', 'LA', 'LAS', None), ('LB', 'LB', 'LBS', None)]:
    rk = per_kind(base, real, shuf)
    if not rk:
        continue
    ks = list(rk)
    r0 = rk[ks[0]]
    F = r0['F']; ok = r0['ok']
    M = np.array([rk[k]['d']['auc'] for k in ks])
    print(f'== {base}: learner types {[(k, rk[k]["n"], rk[k]["nS"]) for k in ks]}, features {int(ok.sum())}')
    fam = collections.defaultdict(list)
    for j, f in enumerate(F):
        if ok[j]:
            fam[f[0]].append(j)
    print('   family   n  ' + '  '.join(f'{k[:6]:>6s}' for k in ks) + '   best')
    best = np.nanmax(M, 0)
    for f in sorted(fam, key=lambda f: np.nanmedian(best[fam[f]])):
        js = fam[f]
        print(f'   {f:5s} {len(js):4d}  ' + '  '.join(f'{np.nanmedian(M[i, js]):6.2f}' for i in range(len(ks))) + f'  {np.nanmedian(best[js]):6.2f}')
    print('   Spearman of feature AUC between learner types:')
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            print(f'     {ks[i]} vs {ks[j]}: {spearman(M[i], M[j]):.2f}')
    if extra:
        for f in extra:
            if f in F:
                j = F.index(f)
                print(f'   {"/".join(f):28s} e0 {r0["e0"][j]:.1f} ' + ' '.join(f'{k}:{M[i, j]:.2f}' for i, k in enumerate(ks)) + f' best {best[j]:.2f}  pct(best) {np.mean(best[ok] < best[j]):.2f}')
    # world-anchored candidates: lowest best-learner retention, with every type's SE small enough
    js = [j for j in np.argsort(best) if ok[j] and not np.isnan(best[j])]
    print('   lowest best-learner retention (candidates for world anchoring):')
    for j in js[:20]:
        print(f'     {"/".join(F[j]):40s} c0 {r0["c0"][j]:.0f} e0 {r0["e0"][j]:.1f} best {best[j]:.2f} by {ks[int(np.nanargmax(M[:, j]))]}')
    print('   highest best-learner retention:')
    for j in js[-8:]:
        print(f'     {"/".join(F[j]):40s} c0 {r0["c0"][j]:.0f} e0 {r0["e0"][j]:.1f} best {best[j]:.2f} by {ks[int(np.nanargmax(M[:, j]))]}')
    json.dump({'F': [C.fkey(f) for f in F], 'kinds': ks, 'auc': [[None if np.isnan(x) else float(x) for x in row] for row in M]},
              open(os.path.join(C.CK, f'{TAG}_{base}_perkind.json'), 'w'))
    # word level, pooled long-range retention per type, best over types
    agg = {}
    for k in ks:
        rk[k]['sites'] = set(f[1] for f in F if f[0] == 'SW')
        agg[k] = R.word_agg(rk[k], boot=30)
    words = set.intersection(*[set(x for x, v in agg[k].items() if 'long' in v) for k in ks])
    rows = []
    for x in words:
        b = max(agg[k][x]['long'][0] for k in ks)
        rows.append((b, x, agg[ks[0]][x]['long'][2], {k: round(agg[k][x]['long'][0], 3) for k in ks}))
    rows.sort()
    print(f'   words/logograms by best-learner pooled long-range retention (n {len(rows)}):')
    for row in rows[:15] + rows[-6:]:
        print(f'     {row[1]:14s} best {row[0]:.3f} excess {row[2]:.1f} {row[3]}')
    json.dump(rows, open(os.path.join(C.CK, f'{TAG}_{base}_wordbest.json'), 'w'))
