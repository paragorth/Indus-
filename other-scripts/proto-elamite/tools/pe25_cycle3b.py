"""pe25 cycle 3b: recurring checker, looser content match (more power).
For ordered same-stratum pairs (i, j) and bases b on rev i, rev j and obv i:
D' = agree(rev i, rev j) - agree(obv i, rev j).  One hand per tablet -> D' ~ 0;
a pool of checkers writing reverses -> D' > 0.  Plants as in cycle 3 (3 and 1
checkers per stratum); stratum bootstrap."""
import random, json
import numpy as np
from pe25_common import *
import pe25_cycle1 as c1
import pe25_cycle3 as c3

OUT = c3.OUT
pairs = c3.pairs + [(j, i) for i, j in c3.pairs]


def Dp(R, pairs):
    out = []
    for i, j in pairs:
        bs = set(b for b, v, k in R[i]['OBV']) & set(b for b, v, k in R[i]['REV']) & set(b for b, v, k in R[j]['REV'])
        if not bs:
            continue
        a = c3.agree(R[i]['REV'], R[j]['REV'], bs)
        o = c3.agree(R[i]['OBV'], R[j]['REV'], bs)
        out += [(c1.keyof[i], a[b] - o[b]) for b in bs]
    return out


def mean(x):
    return float(np.mean([v for g, v in x]))


if __name__ == '__main__':
    r = random.Random(35)
    real = Dp(c1.recs, pairs)
    by = {}
    for g, v in real:
        by.setdefault(g, []).append(v)
    gl = list(by)
    bs = [np.mean([v for g in (r.choice(gl) for _ in gl) for v in by[g]]) for _ in range(2000)]
    ci = (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))
    pl = {}
    for kind, nc in (('S1', 3), ('T2', 3), ('T2one', 1)):
        pl[kind] = [round(mean(Dp(c3.plant('T2' if kind != 'S1' else 'S1', 1.0, 400 + s, ncheck=nc), pairs)), 3) for s in range(3)]
        print(kind, pl[kind], flush=True)
    print('real', len(real), mean(real), ci, flush=True)
    dump('cycle3b.json', dict(n=len(real), D=mean(real), ci=ci, plants=pl))
    v = 'recurring checker supported' if ci[0] > max(pl['S1']) else ('no recurring checker (B)' if ci[1] < min(pl['T2']) else 'open (C)')
    row(OUT, 'PE-25.3b', "RECURRING CHECKER, looser match (more power): ordered same-stratum pairs, bases on rev i, rev j, obv i; D' = agree(rev i, rev j) - agree(obv i, rev j); stratum bootstrap; plants S1 (one hand per tablet), T2 (3 checkers per stratum), T2one (1 checker per stratum), 3 seeds",
        "n %d cells; D' %.3f (90%% CI %.3f to %.3f); plants S1 %s, T2 %s, T2one %s" % (len(real), mean(real), ci[0], ci[1], pl['S1'], pl['T2'], pl['T2one']), v)
