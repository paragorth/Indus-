"""pe75 cycle 3: BLANKS FROM ONE BATCH.  If clerks took pre-formed blanks from a stock made in one
session, tablets cut from one batch share all three dimensions (the text never sees them).  Do
dimension twins share content beyond what size alone explains?

Pairs of PE tablets (781 with h, w, t).  Batch rule = tolerances (dh, dw, dt) in mm.  Content
similarity of a pair = Jaccard of base-sign sets + same header sign + same first numeral system.
Score of a rule = mean similarity of rule pairs minus mean similarity of SIZE-MATCHED decoy pairs
(pairs whose dims differ by 2-5 mm in one random dimension, same other two within tolerance + 2).
Massive random guessing: 4,000 random rules (tolerances 0-3 mm each, plus optional same-volume
exclusion) scored on train tablets, top 1% re-tested on held-out tablets (pairs inside the half).
Nulls: thickness shuffled among tablets of similar area (kills batch identity, keeps size), 20 runs.
Planted: 40 'batches' of 3 tablets sharing a header sign get one common (h,w,t) +/- 0.5 mm.
Lot control: pairs from the same publication volume vs different volumes reported separately.
"""
import json, os, sys, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_common as C
import common

NH = 4000


def build():
    R = [r for r in C.pe_table() if r['complete_cat'] and r['h'] and r['w'] and r['t']]
    vol = {t['id']: t.get('volume', '') for t in common.load()}
    for r in R:
        r['vol'] = vol.get(r['id'], '')
    return R


def sims(R):
    n = len(R)
    sets = [set(r['signs']) for r in R]
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = len(sets[i] | sets[j])
            J[i, j] = J[j, i] = len(sets[i] & sets[j]) / u if u else 0
    H = np.array([r['header'][0] if r['header'] else '#%d' % i for i, r in enumerate(R)])
    F = np.array([r['first_sys'] for r in R])
    S = np.stack([J, (H[:, None] == H[None, :]) * 1.0, (F[:, None] == F[None, :]) * 1.0])
    V = np.array([r['vol'] for r in R])
    return S, V[:, None] == V[None, :]


def rule_score(D, S, SV, idx, rule):
    dh, dw, dt, xvol, wts = rule
    sub = np.ix_(idx, idx)
    A = [np.abs(D[k][idx][:, None] - D[k][idx][None, :]) for k in range(3)]
    iu = np.triu(np.ones((len(idx), len(idx)), bool), 1)
    tw = iu & (A[0] <= dh) & (A[1] <= dw) & (A[2] <= dt)
    tol = (dh, dw, dt)
    dec = np.zeros_like(tw)
    for k in range(3):
        o = [m for m in range(3) if m != k]
        dec |= iu & (A[k] >= tol[k] + 2) & (A[k] <= tol[k] + 5) & (A[o[0]] <= tol[o[0]] + 2) & (A[o[1]] <= tol[o[1]] + 2)
    if xvol:
        tw &= ~SV[sub]; dec &= ~SV[sub]
    if tw.sum() < 10 or dec.sum() < 10:
        return 0.0, int(tw.sum())
    s = np.tensordot(np.array(wts), S[:, idx][:, :, idx], 1)
    return float(s[tw].mean() - s[dec].mean()), int(tw.sum())


def pipeline(a):
    tag, seed, D, S, SV, rules = a
    rng = np.random.default_rng(seed)
    n = len(D[0]); m = rng.random(n) < 0.5
    tr, te = np.where(m)[0], np.where(~m)[0]
    sc = np.array([rule_score(D, S, SV, tr, r)[0] for r in rules])
    top = np.argsort(-sc)[: len(rules) // 100]
    tt = [rule_score(D, S, SV, te, rules[i]) for i in top]
    return tag, float(np.mean([x[0] for x in tt])), [(rules[i], round(sc[i], 4), round(tt[j][0], 4), tt[j][1]) for j, i in enumerate(top[:8])]


def main():
    R = build()
    S, SV = sims(R)
    D = [np.array([r[k] for r in R], float) for k in 'hwt']
    rng = np.random.default_rng(75030)
    rules = [(int(rng.integers(0, 4)), int(rng.integers(0, 4)), int(rng.integers(0, 4)), bool(rng.random() < 0.5),
              tuple(np.round(rng.dirichlet([1, 1, 1]), 3))) for _ in range(NH)]
    area = D[0] * D[1]
    ab = np.digitize(area, np.percentile(area, np.arange(5, 100, 5)))
    jobs = [('real', 75300 + i, D, S, SV, rules) for i in range(10)]
    for i in range(20):
        r2 = np.random.default_rng(9300 + i); t = D[2].copy()
        for b in np.unique(ab):
            idx = np.where(ab == b)[0]; t[idx] = t[r2.permutation(idx)]
        jobs.append(('null', 75300 + i % 10, [D[0], D[1], t], S, SV, rules))
    # planted batches
    Dp = [d.copy() for d in D]
    hc = collections.defaultdict(list)
    for i, r in enumerate(R):
        if r['header']:
            hc[r['header'][0]].append(i)
    pool = [g for g in hc.values() if len(g) >= 3]
    used = set()
    for b in range(40):
        g = pool[rng.integers(len(pool))]
        pick = [i for i in rng.permutation(g) if i not in used][:3]
        if len(pick) < 2:
            continue
        used.update(pick)
        base = [Dp[k][pick[0]] for k in range(3)]
        for i in pick:
            for k in range(3):
                Dp[k][i] = base[k] + rng.choice([-1, 0, 0, 1]) * 0.0 + np.round(rng.uniform(-0.5, 0.5))
    jobs += [('plant', 75300 + i, Dp, S, SV, rules) for i in range(5)]
    with Pool(2) as P:
        res = P.map(pipeline, jobs)
    agg = collections.defaultdict(list)
    for tag, v, top in res:
        agg[tag].append(v)
    out = {'n': len(R), 'n_rules': len(rules), 'heldout': {k: v for k, v in agg.items()},
           'p_real_vs_null': float(np.mean([x >= np.mean(agg['real']) for x in agg['null']])),
           'top_real': res[0][2]}
    # descriptive: exact twins (0,0,0) and (1,1,1), same vs different volume
    allidx = np.arange(len(R))
    W = (0.57, 0.29, 0.14)
    for rule in [(0, 0, 0, False, W), (0, 0, 0, True, W), (1, 1, 1, False, W), (1, 1, 1, True, W), (2, 2, 2, True, W)]:
        out['rule_%s' % (rule[:4],)] = rule_score(D, S, SV, allidx, rule)
        nn = [rule_score([D[0], D[1], j[2][2]], S, SV, allidx, rule)[0] for j in jobs[10:30]]
        out['rule_%s_null' % (rule[:4],)] = (float(np.mean(nn)), float(np.std(nn)))
    json.dump(out, open(os.path.join(C.CK, 'cycle3.json'), 'w'), indent=1, default=str)
    print(json.dumps(out, indent=0, default=str)[:4000])


if __name__ == '__main__':
    main()
