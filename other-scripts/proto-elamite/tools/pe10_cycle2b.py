"""pe10 cycle 2b: pair-SPECIFIC rescoring of cycle 2.  Score(S,L) = min(z_S, -z_L): the short
form must be enriched in Q AND the long form depleted in Q; S needs >= 3 whole-entry
occurrences.  Aggregate statistic = number of gated pairs with score > 1.5 (and > 2), compared
with the within-tablet Q-shuffle null (200 reps); planted pairs (cycle-2 planting, m = 3/5/8)
must rank in the top 1% and push the aggregate above the null 95th percentile.
"""
import json, sys
import numpy as np
from pe10_common import *
import pe10_cycle2 as c2

rng = c2.rng


def scores(E, P, q):
    S_types, L_types, occS, occL, gate, sub = P
    p = q.mean()
    zS = c2.zmat([np.array(occS[s]) for s in S_types], q, p)
    zL = c2.zmat([np.array(sorted(occL[g])) for g in L_types], q, p)
    nS = np.array([len(occS[s]) for s in S_types])
    sc = np.minimum(zS[:, None], -zL[None, :])
    return np.where(gate & (nS[:, None] >= 3), sc, -np.inf)


def run(E, nrep=200, planted=None):
    P = c2.prepare(E)
    S_types, L_types = P[0], P[1]
    q = np.array([e['q'] for e in E], float)
    sc = scores(E, P, q)
    obs = {t: int((sc > t).sum()) for t in (1.5, 2.0)}
    nul = {t: [] for t in (1.5, 2.0)}; nmax = []
    for _ in range(nrep):
        s2 = scores(E, P, c2.shuffle_q(E, q))
        for t in nul:
            nul[t].append(int((s2 > t).sum()))
        nmax.append(float(s2.max()))
    thr = float(np.quantile(nmax, 0.95))
    out = {'n_pairs_scored': int(np.isfinite(sc).sum()), 'obs_max': float(sc.max()), 'fwer_thr': thr}
    for t in nul:
        a = np.array(nul[t])
        out['count>%.1f' % t] = obs[t]; out['null_mean>%.1f' % t] = float(a.mean())
        out['p>%.1f' % t] = float((np.sum(a >= obs[t]) + 1) / (nrep + 1))
    flat = np.argsort(sc, axis=None)[::-1][:20]
    out['top'] = []
    for f in flat:
        i, j = np.unravel_index(f, sc.shape)
        out['top'].append((' '.join(S_types[i]), ' '.join(L_types[j]), round(float(sc[i, j]), 2), bool(P[5][i, j])))
    out['survivors'] = [t for t in out['top'] if t[2] > thr]
    if planted:
        fin = sc[np.isfinite(sc)]
        ranks = []
        for s, l in planted:
            try:
                i = S_types.index(tuple(s.split())); j = L_types.index(tuple(l.split()))
                v = sc[i, j]
                ranks.append(None if not np.isfinite(v) else round(float((fin > v).mean()), 4))
            except ValueError:
                ranks.append('absent')
        out['planted_rank_frac'] = ranks
    return out


if __name__ == '__main__':
    T = load(); E, q2 = c2.collect(T)
    ck = os.path.join(CKPT, 'c2b.json')
    out = json.load(open(ck)) if os.path.exists(ck) else {}
    if 'real' not in out:
        out['real'] = run(E); json.dump(out, open(ck, 'w'), indent=1); print('real', out['real'], flush=True)
    for m in (3, 5, 8):
        k = 'plant_m%d' % m
        if k in out:
            continue
        E2, planted = c2.plant(E, m)
        out[k] = run(E2, nrep=100, planted=planted); out[k]['planted'] = planted
        json.dump(out, open(ck, 'w'), indent=1); print(k, out[k], flush=True)
