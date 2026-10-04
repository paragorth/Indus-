"""pe10 positive control for pair mining: Ur III year formulas.
Units = year formulas (grapheme tuples) on 9,997 dated intact tablets.  Q (squeezed) =
tablet in the smallest height quartile of its year.  Short S / long L = formula strings
(n >= 3), S shorter than L.  Score = min(z_S, -z_L) (S enriched on small tablets, L depleted),
gated on S being a grapheme subsequence of L.  Null = shuffle Q among tablets of the same
year (200 reps).  Truth: a recovered pair is 'correct' if S and L are attested for the
same catalogue year.  Also a planted version: for 5 random years, convert 60% of the
long formulas on small tablets into the year's shortest form.
"""
import json
import numpy as np
from collections import defaultdict, Counter
from pe10_common import CKPT
from pe10_ur3 import graphemes

rng = np.random.default_rng(5)


def is_subseq(a, b):
    it = iter(b)
    return all(x in it for x in a)


def setup(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['year']].append(r)
    U = []
    for y, g in by.items():
        if len(g) < 20:
            continue
        H = np.array([r['h'] for r in g]); q1 = np.quantile(H, 0.25)
        for r in g:
            U.append({'year': y, 's': tuple(graphemes(r['txt'])), 'q': r['h'] <= q1})
    return U


def run(U, nrep=200):
    yrs = np.array([u['year'] for u in U])
    q = np.array([u['q'] for u in U], float)
    occ = defaultdict(list)
    for i, u in enumerate(U):
        occ[u['s']].append(i)
    types = [s for s, o in occ.items() if len(o) >= 3]
    O = [np.array(occ[s]) for s in types]
    gate = np.array([[len(a) < len(b) and is_subseq(a, b) for b in types] for a in types])
    same = np.array([[len({U[i]['year'] for i in occ[a]} & {U[i]['year'] for i in occ[b]}) > 0 for b in types] for a in types])
    groups = defaultdict(list)
    for i, y in enumerate(yrs):
        groups[y].append(i)
    groups = [np.array(v) for v in groups.values()]

    def score(qv):
        p = qv.mean()
        z = np.array([(qv[o].sum() - len(o) * p) / np.sqrt(len(o) * p * (1 - p)) for o in O])
        return np.where(gate, np.minimum(z[:, None], -z[None, :]), -np.inf)
    sc = score(q)
    nm = []; nc = []
    for _ in range(nrep):
        q2 = q.copy()
        for g in groups:
            q2[g] = q[rng.permutation(g)]
        s2 = score(q2); nm.append(float(s2.max())); nc.append(int((s2 > 2).sum()))
    thr = float(np.quantile(nm, 0.95))
    surv = np.argwhere(sc > thr)
    top = np.argsort(sc, axis=None)[::-1][:10]
    return {'n_units': len(U), 'n_types': len(types), 'n_pairs_gated': int(gate.sum()),
            'share_gated_pairs_same_year': float(same[gate].mean()) if gate.any() else None,
            'obs_max': float(sc.max()), 'fwer_thr': thr, 'n_survivors': int(len(surv)),
            'survivors_same_year': int(sum(same[i, j] for i, j in surv)),
            'count>2': int((sc > 2).sum()), 'null_count>2_mean': float(np.mean(nc)),
            'p_count': float((np.sum(np.array(nc) >= (sc > 2).sum()) + 1) / (nrep + 1)),
            'top': [(' '.join(types[i]), ' '.join(types[j]), round(float(sc[i, j]), 2), bool(same[i, j]))
                    for i, j in (np.unravel_index(f, sc.shape) for f in top)]}


def plant(U):
    U2 = [dict(u) for u in U]
    by = defaultdict(list)
    for i, u in enumerate(U2):
        by[u['year']].append(i)
    yrs = rng.choice(sorted(by), 5, replace=False)
    for y in yrs:
        forms = Counter(U2[i]['s'] for i in by[y])
        common = [f for f, c in forms.items() if c >= 3]
        short = min(common, key=len)
        for i in by[y]:
            if U2[i]['q'] and len(U2[i]['s']) > len(short) and rng.random() < 0.6:
                U2[i]['s'] = short
    return U2, [str(y) for y in yrs]


if __name__ == '__main__':
    d = json.load(open(f'{CKPT}/ur3_years.json'))
    U = setup(d['rows'])
    out = {'real': run(U)}
    print(json.dumps(out['real'], indent=1), flush=True)
    U2, yrs = plant(U)
    out['planted'] = run(U2, nrep=100); out['planted']['years'] = yrs
    print(json.dumps(out['planted'], indent=1), flush=True)
    json.dump(out, open(f'{CKPT}/ur3_pairs.json', 'w'), indent=1)
