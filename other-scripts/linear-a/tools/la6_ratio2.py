#!/usr/bin/env python3
"""LA-6 cycle 1b: what kind of ratio structure is there?

(a) Equality: share of commodity pairs in one entry with exactly equal quantities, against
    N1 (each commodity's quantities drawn from its own corpus-wide pool). LB doc level same.
(b) Cross-tablet fixed ratio after removing equal pairs (ratio 1): global HIT (distinct
    tablets within +-10% of one ratio), N1 null, 2,000 runs.
(c) Within-tablet rate: tablets where >=2 entries carry the same pair. Count pairs of
    entries on one tablet whose ratios agree within 10%; N1 null.
(d) Power: plant a fixed ratio (b = 0.5 a, rounded to the nearest 1/4) in a share s of the
    real LA pair entries; how often does the global cross-tablet HIT test reach P < 0.05?
"""
import sys, os, math, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from la6_common import la_entries, lb_docs
from la6_ratio import TOL

random.seed(61)
R = 2000


def la_multi():
    E = la_entries(use_frac='--nofrac' not in sys.argv)
    keep = [e for e in E if e['role'] in ('entry', 'head', 'post')]
    ents = [{c: float(v) for c, v in e['com'].items() if v > 0} for e in keep]
    docs = [e['doc'] for e in keep]
    pool = defaultdict(list)
    for e in ents:
        for c, v in e.items(): pool[c].append(v)
    multi = [(e, d) for e, d in zip(ents, docs) if len(e) >= 2]
    return multi, pool


def lb_multi():
    L = lb_docs()
    pool = defaultdict(list)
    for d in L:
        for c, v in d['com'].items():
            if v > 0: pool[c].append(float(v))
    multi = [({c: float(v) for c, v in d['com'].items() if v > 0}, d['id']) for d in L]
    return [m for m in multi if len(m[0]) >= 2], pool


def pairlist(multi, minn=4):
    P = defaultdict(list)
    for e, d in multi:
        cs = sorted(e)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                P[(cs[i], cs[j])].append((e[cs[i]], e[cs[j]], d))
    return {k: v for k, v in P.items() if len(v) >= minn}


def equality(P, pool):
    obs = sum(1 for v in P.values() for a, b, d in v if a == b)
    n = sum(len(v) for v in P.values())
    nl = []
    for _ in range(R):
        nl.append(sum(1 for (a, b), v in P.items() for _x in v if random.choice(pool[a]) == random.choice(pool[b])))
    return obs, n, sum(nl) / R, (sum(x >= obs for x in nl) + 1) / (R + 1)


def cross_hit(P, pool, drop_equal=True):
    def g(P2):
        tot = 0
        for k, v in P2.items():
            v2 = [(a, b, d) for a, b, d in v if not (drop_equal and a == b)]
            if len(v2) < 2: continue
            tot += hit([math.log(b / a) for a, b, d in v2], [d for *_x, d in v2])
        return tot
    obs = g(P)
    nl = []
    for _ in range(R):
        P2 = {k: [(random.choice(pool[k[0]]), random.choice(pool[k[1]]), d) for a, b, d in v] for k, v in P.items()}
        nl.append(g(P2))
    return obs, sum(nl) / R, (sum(x >= obs for x in nl) + 1) / (R + 1)


def hit(rs, docs):
    o = sorted(range(len(rs)), key=lambda i: rs[i])
    best, j = 0, 0
    for i in range(len(o)):
        while rs[o[i]] - rs[o[j]] > 2 * TOL: j += 1
        best = max(best, len({docs[o[k]] for k in range(j, i + 1)}))
    return best


def within(P, pool, drop_equal=False):
    def g(P2):
        agree = tot = 0
        for k, v in P2.items():
            byd = defaultdict(list)
            for a, b, d in v:
                if drop_equal and a == b: continue
                byd[d].append(math.log(b / a))
            for d, rs in byd.items():
                for i in range(len(rs)):
                    for j in range(i + 1, len(rs)):
                        tot += 1; agree += abs(rs[i] - rs[j]) <= TOL
        return agree, tot
    obs = g(P)
    nl = []
    for _ in range(R):
        P2 = {k: [(random.choice(pool[k[0]]), random.choice(pool[k[1]]), d) for a, b, d in v] for k, v in P.items()}
        nl.append(g(P2)[0])
    return obs, sum(nl) / R, (sum(x >= obs[0] for x in nl) + 1) / (R + 1)


def power(P, pool, share, runs=100, rr=300):
    hits = 0
    for _ in range(runs):
        P2 = {}
        for k, v in P.items():
            nv = []
            for a, b, d in v:
                if random.random() < share:
                    b = max(0.25, round(a * 0.5 * 4) / 4)
                nv.append((a, b, d))
            P2[k] = nv
        def g(Px):
            return sum(hit([math.log(b / a) for a, b, d in v if a != b], [d for a, b, d in v if a != b])
                       for v in Px.values() if sum(1 for a, b, d in v if a != b) >= 2)
        obs = g(P2)
        ge = 0
        for _r in range(rr):
            P3 = {k: [(random.choice(pool[k[0]]), random.choice(pool[k[1]]), d) for a, b, d in v] for k, v in P.items()}
            ge += g(P3) >= obs
        hits += (ge + 1) / (rr + 1) < 0.05
    return hits / runs


def main():
    out = ['# LA-6 cycle 1b: equality, cross-tablet and within-tablet ratios, power']
    for name, (multi, pool) in (('Linear A', la_multi()), ('Linear B doc-level (control)', lb_multi())):
        P = pairlist(multi)
        out.append(f'\n## {name}: {len(P)} pairs, {sum(len(v) for v in P.values())} pair-instances')
        o, n, m, p = equality(P, pool)
        out.append(f'(a) exactly equal quantities: {o}/{n} vs N1 {m:.1f}, P={p:.4f}')
        o, m, p = cross_hit(P, pool, True)
        out.append(f'(b) cross-tablet HIT sum, equal pairs removed: {o} vs N1 {m:.1f}, P={p:.4f}')
        o, m, p = cross_hit(P, pool, False)
        out.append(f'(b2) cross-tablet HIT sum, all pairs: {o} vs N1 {m:.1f}, P={p:.4f}')
        o, m, p = within(P, pool)
        out.append(f'(c) within-tablet agreeing entry pairs: {o[0]}/{o[1]} vs N1 {m:.1f}, P={p:.4f}')
        o, m, p = within(P, pool, True)
        out.append(f'(c2) within-tablet, equal pairs removed: {o[0]}/{o[1]} vs N1 {m:.1f}, P={p:.4f}')
        if name == 'Linear A':
            # list within-tablet agreements
            for k, v in P.items():
                byd = defaultdict(list)
                for a, b, d in v: byd[d].append((a, b))
                for d, l in byd.items():
                    for i in range(len(l)):
                        for j in range(i + 1, len(l)):
                            r1, r2 = l[i][1] / l[i][0], l[j][1] / l[j][0]
                            if abs(math.log(r1 / r2)) <= TOL:
                                out.append(f'    {d} {k[0]}:{k[1]} {l[i]} ~ {l[j]} (ratio {r1:.3g})')
            for s in ((0.25, 0.5) if '--nofrac' not in sys.argv else ()):
                out.append(f'(d) power: planted ratio 1:2 in {int(s*100)}% of LA pair-instances -> detected (P<0.05) in {power(P, pool, s):.2f} of 100')
    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(os.path.dirname(__file__), '..', 'data', 'la6_ratio2' + ('_nofrac' if '--nofrac' in sys.argv else '') + '.out'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
