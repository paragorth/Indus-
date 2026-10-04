#!/usr/bin/env python3
"""LA-14 cycle 3, part T: targeted tests of the two number structures the contest flagged.
T1 magnitude coherence: mean within-tablet SD of log(1+n) over tablets with >= 3 numbers.
   Null G: numbers permuted across all tablets (positions kept). Null K: permuted only among numbers that
   follow the same context class (the logogram base just before, or 'word', or 'other') AND the same site.
T2 descending order: share of strictly descending pairs among unequal adjacent numbers on a tablet
   (numbers after KU-RO, KI-RO, PO-TO-KU-RO and LB to-so/to-sa dropped, so totals cannot drive it).
   Null: numbers permuted within each tablet. Both on LA (all, HT only, non-HT) and LB.
Planted controls: LA with every tablet's numbers replaced by global random draws (T1 must vanish), and LA
with each tablet's numbers sorted descending with prob 0.3 (T2 must rise)."""
import math, random
from collections import defaultdict
import numpy as np
import la14_common as C

TOT = {'W:KU-RO', 'W:KI-RO', 'W:PO-TO-KU-RO', 'W:TO-SO', 'W:TO-SA'}


def entries(doc):
    """[(value, context)] for numbers on a tablet, totals dropped."""
    out = []; prev = 'other'; lastw = None
    for t in doc:
        if t.startswith('N:'):
            if lastw not in TOT: out.append((int(t[2:]), prev))
            continue
        if t.startswith('L:'): prev = t.split('+')[0]
        elif t.startswith('W:'): prev = 'word'; lastw = t
        elif t == 'NL': lastw = None
    return out


def t1(E):
    v = [np.std([math.log1p(x) for x, _ in e]) for e in E if len(e) >= 3]
    return float(np.mean(v)), len(v)


def t2(E):
    d = a = 0
    for e in E:
        for (x, _), (y, _) in zip(e, e[1:]):
            if y < x: d += 1
            elif y > x: a += 1
    return d / max(1, d + a), d + a


def perm_global(E, rng, sites=None, ctx=False):
    pool = defaultdict(list)
    for i, e in enumerate(E):
        for j, (x, c) in enumerate(e):
            pool[((sites[i] if sites else ''), c if ctx else '')].append(x)
    for k in pool: rng.shuffle(pool[k])
    out = []
    for i, e in enumerate(E):
        out.append([(pool[((sites[i] if sites else ''), c if ctx else '')].pop(), c) for x, c in e])
    return out


def perm_within(E, rng):
    out = []
    for e in E:
        v = [x for x, _ in e]; rng.shuffle(v); out.append([(x, c) for x, (_, c) in zip(v, e)])
    return out


def test(name, docs, sites, R=2000):
    rng = random.Random(1)
    E = [entries(d) for d in docs]
    o1, n1 = t1(E); o2, n2 = t2(E)
    g = [t1(perm_global(E, rng))[0] for _ in range(R // 4)]
    k = [t1(perm_global(E, rng, sites, True))[0] for _ in range(R // 4)]
    w = [t2(perm_within(E, rng))[0] for _ in range(R)]
    p1g = (1 + sum(x <= o1 for x in g)) / (1 + len(g)); p1k = (1 + sum(x <= o1 for x in k)) / (1 + len(k))
    p2 = (1 + sum(x >= o2 for x in w)) / (1 + len(w))
    return ('%-10s T1 sd(log n) %.3f on %d tablets | null G %.3f (P=%.4f) | null K site+context %.3f (P=%.4f) || '
            'T2 desc share %.3f of %d pairs | within-tablet shuffle %.3f+-%.3f (P=%.4f)'
            % (name, o1, n1, np.mean(g), p1g, np.mean(k), p1k, o2, n2, np.mean(w), np.std(w), p2))


if __name__ == '__main__':
    la, ids = C.load_la(); sla = [i[:2] for i in ids]
    lb = C.corpus('LB'); slb = [C.SITE.get(' '.join(d), '??') for d in lb]
    lines = [test('LA all', la, sla)]
    ht = [i for i, s in enumerate(sla) if s == 'HT']; nh = [i for i, s in enumerate(sla) if s != 'HT']
    lines.append(test('LA HT', [la[i] for i in ht], [sla[i] for i in ht]))
    lines.append(test('LA nonHT', [la[i] for i in nh], [sla[i] for i in nh]))
    lines.append(test('LB', lb, slb))
    rng = random.Random(7)
    allv = [t for d in la for t in d if t.startswith('N:')]
    pr = [[(rng.choice(allv) if t.startswith('N:') else t) for t in d] for d in la]
    lines.append(test('PLANT-rand', pr, sla))
    ps = []
    for d in la:
        if rng.random() < 0.3:
            idx = [i for i, t in enumerate(d) if t.startswith('N:')]
            v = sorted((d[i] for i in idx), key=lambda t: -int(t[2:])); d = list(d)
            for i, x in zip(idx, v): d[i] = x
        ps.append(d)
    lines.append(test('PLANT-sort', ps, sla))
    txt = '\n'.join(lines); print(txt)
    open(C.os.path.join(C.OUT, 'c3t_report.txt'), 'w').write(txt + '\n')
