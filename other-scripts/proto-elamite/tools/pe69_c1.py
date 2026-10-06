"""pe69 cycle 1: try to kill the pe68 team rule (M288 = 60 N39C x sum of the count lines since the last M288).

1a decomposition: which team-rule hits are also per-line hits (60 or 120 x the last line)?
1b multiple comparison: 13 run definitions x multipliers 1..240 (3,120 rules) and the prior-60 family (13 rules);
   Westfall-Young max-z under (i) re-dealt M288 values, (ii) M288 values permuted within tablets,
   (iii) all numbers re-sampled from the same written shapes.
1c blind re-derivation: 1,000 random half splits of the tablets; best rule chosen on half A, scored on half B.
1d targeted null: only the count lines NOT directly before the M288 re-sampled from same shapes (keeps per-line
   pairs, scrambles the rest of the team). A real team rule loses its multi-line hits; a per-line artefact does not.
"""
import os, sys, json, math, random, time
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe69_lib import *  # noqa

NREP = int(os.environ.get('NREP', 1000))
K = np.arange(1, 241, dtype=float)
AGGS = list(AGG)


def family_z(m, A, mask=None, pool=None):
    """z for every (agg, k) with analytic re-deal expectation. returns dict agg -> (hits[K], e[K])"""
    if pool is None:
        pool = m[np.isfinite(m)]
    u, c = np.unique(np.round(pool, 6), return_counts=True)
    fr = c / len(pool)
    out = {}
    for a in AGGS:
        x = A[a]
        ok = np.isfinite(x) & np.isfinite(m) & (x > 0)
        if mask is not None:
            ok &= mask
        xs, ms = x[ok], m[ok]
        if not len(xs):
            out[a] = (np.zeros(len(K)), np.zeros(len(K)), 0)
            continue
        V = np.round(np.outer(K, xs), 6)              # K x n
        pos = np.searchsorted(u, V)
        pos = np.clip(pos, 0, len(u) - 1)
        e = np.where(u[pos] == V, fr[pos], 0.0).sum(1)
        h = (np.abs(V - np.round(ms, 6)[None, :]) < 1e-6).sum(1)
        out[a] = (h.astype(float), e, int(ok.sum()))
    return out


def zmat(F, minhits=0):
    Z = np.array([(F[a][0] - F[a][1]) / np.sqrt(F[a][1] + 0.25) for a in AGGS])  # agg x K
    if minhits:
        H = np.array([F[a][0] for a in AGGS])
        Z = np.where(H >= minhits, Z, -np.inf)
    return Z


def team_z(F, k=60):
    h, e, n = F['SUM_BEFORE']
    return float((h[k - 1] - e[k - 1]) / math.sqrt(e[k - 1] + 0.25)), int(h[k - 1]), float(e[k - 1]), n


def main():
    t0 = time.time()
    T = tablets()
    S = slots(T)
    m, A = feats(S)
    nc = np.array([nclean(s) for s in S])
    tab = np.array([s['id'] for s in S])
    multi = nc >= 2
    res = {'n_slots': len(S), 'n_clean_m288': int(np.isfinite(m).sum())}

    # ---------------- 1a decomposition
    team = A['SUM_BEFORE'] * 60
    last = A['LAST']
    th = np.isfinite(team) & (np.abs(m - team) < 1e-9)
    dec = {'team_hits': int(th.sum()), 'single_line_runs': int((th & (nc == 1)).sum()),
           'multi_line_runs': int((th & multi).sum()),
           'multi_also_120xlast': int((th & multi & (np.abs(m - 120 * last) < 1e-9)).sum()),
           'multi_also_60xlast': int((th & multi & (np.abs(m - 60 * last) < 1e-9)).sum())}
    uniq = th & multi & ~(np.abs(m - 120 * last) < 1e-9) & ~(np.abs(m - 60 * last) < 1e-9)
    dec['team_only'] = [{'id': S[i]['id'], 'line': S[i]['line'], 'sum': float(A['SUM_BEFORE'][i]), 'last': float(last[i]),
                         'm': float(m[i])} for i in np.where(uniq)[0]]
    dec['team_hit_list'] = [{'id': S[i]['id'], 'line': S[i]['line'], 'n_lines': int(nc[i]), 'sum': float(A['SUM_BEFORE'][i]),
                             'last': float(last[i]), 'm': float(m[i])} for i in np.where(th)[0]]
    # fully discriminating runs: >= 2 count lines, sum != last and sum != 2 x last (so 60 x sum differs from 60 and 120 x last)
    S_ = A['SUM_BEFORE']
    disc = multi & np.isfinite(m) & (S_ != last) & (S_ != 2 * last)
    dec['discriminating_runs'] = int(disc.sum())
    pool = m[np.isfinite(m)]
    vc = Counter(np.round(pool, 6).tolist())
    for nm, x in (('team60', 60 * S_), ('last60', 60 * last), ('last120', 120 * last)):
        h = int((disc & (np.abs(m - x) < 1e-9)).sum())
        e = sum(vc.get(round(v, 6), 0) / len(pool) for v in x[disc])
        dec['disc_' + nm] = {'h': h, 'e': round(e, 2)}
    res['1a'] = dec
    print('1a', json.dumps({k: v for k, v in dec.items() if k != 'team_hit_list'}), flush=True)

    # ---------------- 1b multiple comparison
    obs = {}
    for scope, mask in (('all', None), ('multi', multi)):
        F = family_z(m, A, mask)
        Z = zmat(F, minhits=3)
        tz = team_z(F)
        rank = int((Z > tz[0]).sum()) + 1
        best = np.unravel_index(np.argmax(Z), Z.shape)
        obs[scope] = {'team60_z': tz[0], 'team60_h': tz[1], 'team60_e': tz[2], 'n': tz[3],
                      'team60_rank_of_3120': rank, 'best_rule': [AGGS[best[0]], int(K[best[1]])], 'best_z': float(Z[best]),
                      'prior60_z': {a: float(Z[i, 59]) for i, a in enumerate(AGGS)},
                      'top10': sorted([[AGGS[i], int(K[j]), int(F[AGGS[i]][0][j]), round(float(F[AGGS[i]][1][j]), 2),
                                        round(float(Z[i, j]), 2)] for i in range(len(AGGS)) for j in range(len(K))
                                       if np.isfinite(Z[i, j])], key=lambda r: -r[4])[:10]}
        print('1b obs', scope, json.dumps({k: v for k, v in obs[scope].items() if k != 'prior60_z'}), flush=True)
    res['1b_obs'] = obs
    rng = np.random.default_rng(seed('pe69-c1'))
    prng = random.Random(seed('pe69-c1-shape'))
    pools = shape_pools(T)
    nulls = {}
    for nul in ('redeal', 'within_tablet', 'shape', 'shape_nonlast'):
        rec = {sc: {'max_all': [], 'max_60': [], 'team': [], 'team_h': []} for sc in ('all', 'multi')}
        for r in range(NREP):
            if nul == 'redeal':
                m2, A2, mask_m = redeal_m(m, rng), A, multi
            elif nul == 'within_tablet':
                m2, A2, mask_m = redeal_m(m, rng, tab), A, multi
            else:
                T2 = resample_shapes(T, pools, prng, only_nonlast=(nul == 'shape_nonlast'),
                                     which=('CNT',) if nul == 'shape_nonlast' else ('CNT', 'M288'))
                S2 = slots(T2)
                m2, A2 = feats(S2)
                mask_m = np.array([nclean(s) for s in S2]) >= 2
            for sc, mask in (('all', None), ('multi', mask_m)):
                F = family_z(m2, A2, mask, pool=m2[np.isfinite(m2)])
                Z = zmat(F, minhits=3)
                rec[sc]['max_all'].append(float(Z.max()))
                rec[sc]['max_60'].append(float(Z[:, 59].max()))
                tz = team_z(F)
                rec[sc]['team'].append(tz[0])
                rec[sc]['team_h'].append(tz[1])
        out = {}
        for sc in rec:
            o = obs[sc]['team60_z']
            ho = obs[sc]['team60_h']
            R = {k: np.array(v) for k, v in rec[sc].items()}
            out[sc] = {'p_raw': float(((R['team'] >= o).sum() + 1) / (NREP + 1)),
                       'p_hits_raw': float(((R['team_h'] >= ho).sum() + 1) / (NREP + 1)),
                       'null_team_hits_mean': float(R['team_h'].mean()),
                       'p_corr_prior60_family': float(((R['max_60'] >= o).sum() + 1) / (NREP + 1)),
                       'p_corr_full_family': float(((R['max_all'] >= o).sum() + 1) / (NREP + 1)),
                       'null_max_all_median': float(np.median(R['max_all'])),
                       'null_max_all_95': float(np.percentile(R['max_all'], 95))}
        nulls[nul] = out
        print('1b null', nul, json.dumps(out), '%.0fs' % (time.time() - t0), flush=True)
        json.dump(res | {'1b_null': nulls}, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)
    res['1b_null'] = nulls

    # ---------------- 1c blind split halves
    ids = sorted(set(tab.tolist()))
    sel = Counter()
    sel_m = Counter()
    hz, hz_team, hz_m, hz_team_m = [], [], [], []
    for r in range(NREP):
        rng2 = np.random.default_rng(seed('pe69-split-%d' % r))
        half = set(rng2.choice(ids, len(ids) // 2, replace=False).tolist())
        a = np.array([x in half for x in tab])
        for sc, base, SEL, HZ, HZT in (('all', np.ones(len(S), bool), sel, hz, hz_team),
                                       ('multi', multi, sel_m, hz_m, hz_team_m)):
            FA = family_z(m, A, base & a, pool=m[a & np.isfinite(m)])
            ZA = zmat(FA, minhits=3)
            bi = np.unravel_index(np.argmax(ZA), ZA.shape)
            rule = (AGGS[bi[0]], int(K[bi[1]]))
            SEL[rule] += 1
            FB = family_z(m, A, base & ~a, pool=m[~a & np.isfinite(m)])
            h, e, _ = FB[rule[0]]
            HZ.append(float((h[rule[1] - 1] - e[rule[1] - 1]) / math.sqrt(e[rule[1] - 1] + 0.25)))
            HZT.append(team_z(FB)[0])
    res['1c'] = {'all': {'selected_top': [[list(k), v] for k, v in sel.most_common(8)],
                         'heldout_z_selected_median': float(np.median(hz)), 'heldout_z_selected_gt2': float(np.mean(np.array(hz) > 2)),
                         'heldout_z_team60_median': float(np.median(hz_team)), 'heldout_team60_gt2': float(np.mean(np.array(hz_team) > 2))},
                 'multi': {'selected_top': [[list(k), v] for k, v in sel_m.most_common(8)],
                           'heldout_z_selected_median': float(np.median(hz_m)), 'heldout_z_selected_gt2': float(np.mean(np.array(hz_m) > 2)),
                           'heldout_z_team60_median': float(np.median(hz_team_m)), 'heldout_team60_gt2': float(np.mean(np.array(hz_team_m) > 2))}}
    print('1c', json.dumps(res['1c']), '%.0fs' % (time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
