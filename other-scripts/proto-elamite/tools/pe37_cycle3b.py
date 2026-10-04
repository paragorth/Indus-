"""pe37 cycle 3b: nulls for the triple ruler (cycle 3 T).
Random rulers: log-ratio means drawn uniformly (young/F 0.2-1.5, males/F 0.05-2.0), the
real ABC spreads kept. Question: is the real-bone top score, or the rank of a given
triple, unusual against random rulers? Also posterior mass (softmax) of the top triples."""
from pe37_common import *
import pe20_common as P20
import itertools
C3 = json.load(open(os.path.join(DATA, 'pe37_cycle3.json')))
rng = np.random.default_rng(3738)


def trip_table(recs, signs):
    V = P20.to_matrix(recs, signs)
    rows = []
    for f, y, m in itertools.permutations(range(len(signs)), 3):
        w = ~np.isnan(V[:, f]) & ~np.isnan(V[:, y]) & ~np.isnan(V[:, m])
        if w.sum() < 3:
            continue
        F, Y, M = V[w, f].sum(), V[w, y].sum(), V[w, m].sum()
        if min(F, Y, M) <= 0:
            continue
        rows.append(((signs[f], signs[y], signs[m]), math.log(Y / F), math.log(M / F)))
    return rows


def scores(rows, d):
    ly = np.array([r[1] for r in rows]); lm = np.array([r[2] for r in rows])
    return stats.norm.logpdf(ly, *d['ly']) + stats.norm.logpdf(lm, *d['lm'])


out = {}
for lab, recs, signs, dk, key in (('ur3', P20.ur_herd_records(), [s for s in P20.UR_SIGNS if s != 'asz2-gar3'], 'kaftari',
                                   [('ud5', 'masz2', 'masz2-nita2'), ('u8', 'sila4', 'udu-nita2')]),
                                  ('pe', P20.pe_records(), P20.PE_SIGNS, 'banesh', [('M362', 'M367', 'M006'), ('M346', 'M346~a', 'M006')])):
    rows = trip_table(recs, signs)
    d = C3['ratios'][dk]
    s = scores(rows, d)
    p = np.exp(s - s.max()); p /= p.sum()
    order = np.argsort(-s)
    keyidx = {k: [r[0] for r in rows].index(k) for k in key}
    realtop = float(s.max())
    nul_top, nul_rank = [], {k: [] for k in key}
    for i in range(2000):
        dn = {'ly': (rng.uniform(math.log(0.2), math.log(1.5)), d['ly'][1]),
              'lm': (rng.uniform(math.log(0.05), math.log(2.0)), d['lm'][1])}
        sn = scores(rows, dn)
        nul_top.append(float(sn.max()))
        for k, j in keyidx.items():
            nul_rank[k].append(int(np.sum(sn > sn[j])))
    res = {'n_triples': len(rows), 'real_top': realtop, 'p_top': float(np.mean(np.array(nul_top) >= realtop)),
           'top5_post': [(rows[j][0], float(p[j])) for j in order[:5]]}
    for k, j in keyidx.items():
        rk = int(np.sum(s > s[j]))
        res['|'.join(k)] = {'rank': rk, 'post': float(p[j]),
                            'p_rank_le_real_under_random_rulers': float(np.mean(np.array(nul_rank[k]) <= rk))}
    out[lab] = res
    print(lab, json.dumps(res, default=str), flush=True)
dump(out, os.path.join(DATA, 'pe37_cycle3b.json'))
