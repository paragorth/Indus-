#!/usr/bin/env python3
"""LA-36 cycle 1: transformation-family search on aligned list pairs.

Per unit (aligned pair of lists): observed pair score s (best of RATIO/DIFF/COMP/AFF, see common)
and its P against N1 (alignment shuffled: values drawn within each list) and N2 (list B replaced
by a random list of the same site and class type). The max over families and parameters is inside
both the observed score and the null, so the per-unit P is search-corrected.
Per class: S = sum of s, H = number of units with P_N1 <= 0.05, joint nulls (same replicate across
units). Controls: PY Ma target columns (known 7:7:2:3:1.5:150), Linear B series pairs, planted
transformations on the real Linear A units (clean, and with one aligned entry left unchanged).
"""
import json, os, sys, time
import numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la36_common import *

P = int(os.environ.get('NP', 1000))


def run_units(U, V, rng, tag, n2=True):
    by_site = defaultdict(list)
    for u in U:
        by_site[(u['site'], u['cls'] in ('COLUMNS', 'ROWS'))].append(u)
    res = []
    Sn1 = defaultdict(lambda: np.zeros(P)); Hn1 = defaultdict(lambda: np.zeros(P))
    Sn2 = defaultdict(lambda: np.zeros(P))
    for u in U:
        f, s = unit_obs(u, V)
        X, Y = unit_null(u, V, P, rng)
        sn = pair_score(fam_scores(X, Y))
        p1 = (np.sum(sn >= s) + 1) / (P + 1)
        # per-replicate P of the null itself (for the joint H statistic): rank within the null
        srt = np.sort(sn)
        pn = 1 - np.searchsorted(srt, sn, 'left') / P
        Sn1[u['cls']] += sn; Hn1[u['cls']] += (pn <= 0.05)
        p2 = None
        if n2:
            pools = [w['poolB'] for w in by_site[(u['site'], u['cls'] in ('COLUMNS', 'ROWS'))] if w['id'] != u['id']]
            pools = [p for p in pools if len(p) >= len(u['x'])]
            if pools:
                X2, Y2 = unit_null(u, V, P, rng, pools)
                sn2 = pair_score(fam_scores(X2, Y2))
                p2 = (np.sum(sn2 >= s) + 1) / (P + 1)
                Sn2[u['cls']] += sn2
            else:
                Sn2[u['cls']] += sn
        d, _ = detail(u, V)
        res.append({'id': u['id'], 'cls': u['cls'], 'n': len(u['x']), 's': s, 'P1': float(p1),
                    'P2': None if p2 is None else float(p2), 'detail': {k: (float(v) if v is not None and not isinstance(v, str) else v) for k, v in d.items()},
                    'x': [list(map(str, a)) for a in u['x']], 'y': [list(map(str, a)) for a in u['y']], 'keys': u['keys']})
    agg = {}
    for c in sorted({u['cls'] for u in U}):
        r = [x for x in res if x['cls'] == c]
        S = sum(x['s'] for x in r); H = sum(x['P1'] <= 0.05 for x in r)
        agg[c] = {'units': len(r), 'S': S, 'S_null1': float(Sn1[c].mean()),
                  'P_S1': float((np.sum(Sn1[c] >= S) + 1) / (P + 1)),
                  'S_null2': float(Sn2[c].mean()), 'P_S2': float((np.sum(Sn2[c] >= S) + 1) / (P + 1)),
                  'H': H, 'H_null': float(Hn1[c].mean()), 'P_H': float((np.sum(Hn1[c] >= H) + 1) / (P + 1))}
        print(tag, c, json.dumps(agg[c]), flush=True)
    return res, agg


def plant(U, rng, partial=False):
    out = []; truth = {}
    fams = ['RATIO', 'DIFF', 'COMP']
    for u in U:
        v = dict(u); x = vals(u['x'], CONV)
        f = fams[rng.integers(3)]
        if f == 'RATIO':
            r = float(RAT[rng.integers(len(RAT))]); m = RMODES[rng.integers(len(RMODES))]
            y = np.maximum(_apply(x * r, m), 0)
        elif f == 'DIFF':
            d = int(rng.integers(1, 6)) * (1 if rng.random() < .5 else -1); y = x + d
        else:
            T = float(np.ceil(x.max()) + rng.integers(1, 10)); y = T - x
        yy = [(float(a), ()) for a in y]
        if partial and len(yy) >= 3:
            j = rng.integers(len(yy)); yy[j] = u['y'][j]
        v['y'] = yy
        v['poolB'] = yy + [p for p in u['poolB']][len(yy):]
        out.append(v); truth[u['id']] = f
    return out, truth


def main():
    rng = np.random.default_rng(36)
    out = {}
    LA = la_units(); MA = ma_units(); LB = lb_units()
    t = time.time()
    out['LA'] = run_units(LA, CONV, rng, 'LA')
    out['MA'] = run_units(MA, CONV, rng, 'MA', n2=False)
    out['LB'] = run_units(LB, CONV, rng, 'LB', n2=False)
    for part in (False, True):
        Pl, truth = plant(LA, rng, part)
        res, agg = run_units(Pl, CONV, rng, 'PLANT' + ('_partial' if part else ''), n2=False)
        det = [r['P1'] <= 0.05 for r in res]
        print('planted', part, 'detected', sum(det), 'of', len(det), flush=True)
        out['PLANT' + ('_partial' if part else '')] = (res, agg, truth)
    print('time', time.time() - t)
    json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
