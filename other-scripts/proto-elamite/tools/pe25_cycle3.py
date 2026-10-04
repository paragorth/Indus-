"""pe25 cycle 3: a recurring checker?

If one official writes the reverses of many tablets, reverses of two different
tablets agree with each other (RR) more than one tablet's obverse agrees with the
other's reverse (OR/RO) or obverse (OO).  Content is matched: for a tablet pair
(i, j) of the same stratum only bases present on all four faces (obv i, rev i,
obv j, rev j) are compared.  Statistic D = RR - (OR + RO)/2 and D2 = RR - OO.
Controls: planted corpora (variants re-drawn): S1 one hand per tablet with
tablets of a stratum shared among 20 writers; T2 same writers but each reverse
by one of 3 checkers per stratum; must give D ~ 0 and D > 0 respectively.
Linking: if D > 0, cluster tablets by reverse agreement (not done if D ~ 0).
"""
import sys, random, json, itertools
import numpy as np
from pe25_common import *
import pe25_cycle1 as c1

OUT = os.path.join(HERE, '..', 'loops', 'pe25_cycle3.txt')
recs = c1.recs
d = c1.d

elig = [i for i, r in enumerate(recs) if r['REV'] and r['OBV']]
# pairs within the same series (publication) and same header class
pairs = []
for g, idx in c1.groups.items():
    idx = [i for i in idx if i in set(elig)]
    for i, j in itertools.combinations(idx, 2):
        pairs.append((i, j))
print('pairs', len(pairs), flush=True)


def agree(X, Y, bset):
    a = n = 0
    out = {}
    for b in bset:
        xs = [v for bb, v, k in X if bb == b]
        ys = [v for bb, v, k in Y if bb == b]
        s = sum(v1 == v2 for v1 in xs for v2 in ys)
        out[b] = s / (len(xs) * len(ys))
    return out


def Dstat(R, pairs):
    rr, orr, oo = [], [], []
    for i, j in pairs:
        bs = set(b for b, v, k in R[i]['OBV']) & set(b for b, v, k in R[i]['REV']) & set(b for b, v, k in R[j]['OBV']) & set(b for b, v, k in R[j]['REV'])
        if not bs:
            continue
        a = agree(R[i]['REV'], R[j]['REV'], bs)
        b1 = agree(R[i]['OBV'], R[j]['REV'], bs)
        b2 = agree(R[i]['REV'], R[j]['OBV'], bs)
        c = agree(R[i]['OBV'], R[j]['OBV'], bs)
        for b in bs:
            rr.append(a[b]); orr.append((b1[b] + b2[b]) / 2); oo.append(c[b])
    rr, orr, oo = map(np.array, (rr, orr, oo))
    return dict(n=len(rr), RR=float(rr.mean()), OR=float(orr.mean()), OO=float(oo.mean()), D=float((rr - orr).mean()), D2=float((rr - oo).mean()), dd=(rr - orr))


def plant(kind, alpha, seed, nwriters=20, ncheck=3):
    r = random.Random(seed)
    out = list(recs)
    for g, idx in c1.groups.items():
        writers = [c1.profile(alpha, r) for _ in range(nwriters)]
        checkers = [c1.profile(alpha, r) for _ in range(ncheck)]
        for i in idx:
            rec = recs[i]
            h = r.choice(writers)
            hr = r.choice(checkers) if kind == 'T2' else h
            n = dict(rec)
            n['OBV'] = [(b, c1.draw(h, b, r), k) for b, v, k in rec['OBV']]
            n['REV'] = [(b, c1.draw(hr, b, r), k) for b, v, k in rec['REV']]
            out[i] = n
    return out


def boot_ci(dd, r, B=2000):
    m = [np.mean(dd[np.array([r.randrange(len(dd)) for _ in dd])]) for _ in range(B)]
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


if __name__ == '__main__':
    r = random.Random(33)
    real = Dstat(recs, pairs)
    ci = boot_ci(real['dd'], r, 1000)
    print('REAL', {k: v for k, v in real.items() if k != 'dd'}, ci, flush=True)
    pl = {}
    for kind in ('S1', 'T2'):
        pl[kind] = []
        for s in range(3):
            x = Dstat(plant(kind, 1.0, 300 + s), pairs)
            pl[kind].append(round(x['D'], 3))
        print(kind, pl[kind], flush=True)
    # tablet-pair bootstrap ignores pair dependence: tablet-level jackknife-ish via resampling strata
    gkeys = [g for g in c1.groups]
    pg = {}
    for i, j in pairs:
        pg.setdefault(c1.keyof[i], []).append((i, j))
    gl = list(pg)
    Ds = []
    for _ in range(200):
        samp = [p for g in (r.choice(gl) for _ in gl) for p in pg[g]]
        Ds.append(Dstat(recs, samp)['D'])
    sci = (float(np.percentile(Ds, 5)), float(np.percentile(Ds, 95)))
    print('stratum bootstrap', sci, flush=True)
    dump('cycle3.json', dict(real={k: v for k, v in real.items() if k != 'dd'}, ci=ci, stratum_ci=sci, plants=pl, npairs=len(pairs)))
    verdict = 'recurring checker supported' if sci[0] > max(pl['S1']) else 'no recurring checker (B)' if sci[1] < min(pl['T2']) else 'open (C)'
    row(OUT, 'PE-25.3a', 'RECURRING CHECKER: same-stratum tablet pairs (%d), content-matched bases present on all four faces; D = RR - (OR+RO)/2 (reverse-reverse minus obverse-reverse agreement); D2 = RR - OO; stratum bootstrap; plants S1 (20 writers per stratum, one hand per tablet) vs T2 (same, reverse by 1 of 3 checkers per stratum), 3 seeds' % len(pairs),
        'n %d base cells; RR %.3f, OR %.3f, OO %.3f; D %.3f (pair CI %.3f to %.3f; stratum CI %.3f to %.3f); D2 %.3f; plants S1 D %s, T2 D %s' % (real['n'], real['RR'], real['OR'], real['OO'], real['D'], ci[0], ci[1], sci[0], sci[1], real['D2'], pl['S1'], pl['T2']), verdict)
