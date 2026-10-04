"""pe10 cycle 2c: the Ur III-calibrated pair miner (pe10_ur3_pairs.py) applied to PE unchanged.
Units = clean entry strings (base forms).  Types with n >= 3.  Pairs gated on S being a
proper subsequence of L.  Score = min(z_S, -z_L).
Q definitions:
  small  : tablet height in the bottom quartile of its opener stratum (header sign; '-' = none)
           -- the exact analogue of 'small tablet within year'; null shuffles Q among tablets
           of the same stratum (tablet-level labels, so all entries of a tablet move together).
  vert   : cycle-2 end-of-face positions on crowded faces; null shuffles within tablet.
  small_v: variant-level strings (with ~ letters) under 'small'.
Planted (mirrors the Ur III plant): 5 random multi-sign types (n >= 4) with a shorter
subsequence type (n >= 3); 60% of their occurrences on small tablets are replaced by that form.
"""
import json
import numpy as np
from pe10_common import *
import pe10_cycle2 as c2
from pe10_cycle3 import collect_raw

rng = np.random.default_rng(23)


def is_subseq(a, b):
    it = iter(b)
    return all(x in it for x in a)


def units(T, E, key='s'):
    hs = defaultdict(list)
    hdr_of = {}
    for e in E:
        hdr_of[e['tab']] = e['hdr']
    for ti, h in hdr_of.items():
        if T[ti]['h']:
            hs[h].append((T[ti]['h'], ti))
    small = {}
    for h, lst in hs.items():
        if len(lst) < 8:
            for _, ti in lst:
                small[ti] = None
            continue
        q1 = np.quantile([x for x, _ in lst], 0.25)
        for x, ti in lst:
            small[ti] = x <= q1
    U = []
    for e in E:
        sm = small.get(e['tab'])
        U.append({'tab': e['tab'], 'hdr': e['hdr'], 's': e[key], 'small': sm, 'vert': e['q']})
    return U


def run(U, qkey, nrep=300, planted=None):
    U = [u for u in U if u[qkey] is not None]
    q = np.array([bool(u[qkey]) for u in U], float)
    occ = defaultdict(list)
    for i, u in enumerate(U):
        occ[u['s']].append(i)
    types = [s for s, o in occ.items() if len(o) >= 3]
    O = [np.array(occ[s]) for s in types]
    gate = np.array([[len(a) < len(b) and is_subseq(a, b) for b in types] for a in types])
    if qkey == 'vert':
        gk = [u['tab'] for u in U]
    else:
        gk = [u['hdr'] for u in U]
    # tablet-level shuffle for 'small': permute labels among tablets inside stratum
    tabs_by = defaultdict(list)
    for i, u in enumerate(U):
        tabs_by[u['tab']].append(i)

    def shuffle():
        q2 = q.copy()
        if qkey == 'vert':
            for idx in tabs_by.values():
                idx = np.array(idx); q2[idx] = q[rng.permutation(idx)]
        else:
            strat = defaultdict(list)
            for t, idx in tabs_by.items():
                strat[U[idx[0]]['hdr']].append(t)
            for ts in strat.values():
                lab = [q[tabs_by[t][0]] for t in ts]
                lab = rng.permutation(lab)
                for t, l in zip(ts, lab):
                    q2[tabs_by[t]] = l
        return q2

    def score(qv):
        p = qv.mean()
        z = np.array([(qv[o].sum() - len(o) * p) / np.sqrt(len(o) * p * (1 - p)) for o in O])
        return np.where(gate, np.minimum(z[:, None], -z[None, :]), -np.inf), z
    sc, z = score(q)
    nm, nc = [], []
    for _ in range(nrep):
        s2, _ = score(shuffle()); nm.append(float(s2.max())); nc.append(int((s2 > 1.5).sum()))
    thr = float(np.quantile(nm, 0.95))
    top = np.argsort(sc, axis=None)[::-1][:10]
    out = {'Q': qkey, 'n_units': len(U), 'Q_rate': round(float(q.mean()), 3), 'n_types': len(types),
           'n_pairs_gated': int(gate.sum()), 'obs_max': float(sc.max()), 'fwer_thr': thr,
           'p_max': float((np.sum(np.array(nm) >= sc.max()) + 1) / (nrep + 1)),
           'count>1.5': int((sc > 1.5).sum()), 'null_count>1.5_mean': float(np.mean(nc)),
           'p_count': float((np.sum(np.array(nc) >= (sc > 1.5).sum()) + 1) / (nrep + 1)),
           'survivors': [(' '.join(types[i]), ' '.join(types[j]), round(float(sc[i, j]), 2)) for i, j in np.argwhere(sc > thr)],
           'top': [(' '.join(types[i]), ' '.join(types[j]), round(float(sc[i, j]), 2))
                   for i, j in (np.unravel_index(f, sc.shape) for f in top) if np.isfinite(sc[i, j])]}
    if planted:
        surv = {(a, b) for a, b, _ in out['survivors']}
        out['planted'] = planted
        out['recovered'] = sum(1 for p in planted if tuple(p) in surv)
    return out


def plant(U):
    U2 = [dict(u) for u in U]
    occ = defaultdict(list)
    for i, u in enumerate(U2):
        if u['small'] is not None:
            occ[u['s']].append(i)
    types = [s for s, o in occ.items() if len(o) >= 3]
    longs = [s for s in types if len(s) >= 2 and len(occ[s]) >= 4 and
             any(len(a) < len(s) and is_subseq(a, s) for a in types)]
    pick = [longs[i] for i in rng.choice(len(longs), min(5, len(longs)), replace=False)]
    planted = []
    for L in pick:
        shorts = [a for a in types if len(a) < len(L) and is_subseq(a, L)]
        S = shorts[rng.integers(len(shorts))]
        for i in occ[L]:
            if U2[i]['small'] and rng.random() < 0.6:
                U2[i]['s'] = S
        planted.append((' '.join(S), ' '.join(L)))
    return U2, planted


if __name__ == '__main__':
    T = load()
    E = collect_raw(T)
    out = {}
    U = units(T, E)
    for qk in ('small', 'vert'):
        out[qk] = run(U, qk); print(json.dumps(out[qk])[:1500], flush=True)
    Uv = units(T, E, key='raw')
    out['small_variant'] = run(Uv, 'small'); print(json.dumps(out['small_variant'])[:800], flush=True)
    rec = []
    for r in range(10):
        U2, planted = plant(U)
        o = run(U2, 'small', nrep=100, planted=planted)
        rec.append(o['recovered']); print('plant', r, o['recovered'], planted, flush=True)
    out['planted_recovered_per_run'] = rec
    # negative: tablets' small labels shuffled once within stratum, then searched
    json.dump(out, open(os.path.join(CKPT, 'c2c.json'), 'w'), indent=1)
