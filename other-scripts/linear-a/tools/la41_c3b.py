#!/usr/bin/env python3
"""LA-41 cycle 3b: repairs and power. (i) Substitutions deduplicated by word pair (one vote per
distinct spelling pair), LB scored on the blind la21 LB rows (sign names upper-cased) and on LB
consonants, restricted to the non-formula families. (ii) Power of the la21-row scorer at Linear
A's real count (17 distinct substitutions): planted same-row substitutions at mixing rates
100/50/25 % (rest random partners), 200 draws each. (iii) Amount-ratio test with the HT 86/95
trio collapsed to one family and known-before (la24, la36) families removed."""
import sys, os, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la41_common import *
import la32_common as L32
from la41_c3 import fam_sets, subs_of, score_subs, lbcons, amount_test

OUT = os.path.join(HERE, '..', 'loops', 'la41_cycle3.txt')


def dedup(S):
    seen = set(); out = []
    for s in S:
        k = tuple(sorted((s[4], s[5])))
        if k not in seen: seen.add(k); out.append(s)
    return out


def main():
    rng = random.Random(4143)
    R21 = L32.la21_rows()
    rows = []
    LB = lb_docs(); SD = site_df(LB)
    rob, loose = fam_sets('LB')
    nonf = []
    for a, b in rob:
        mp = set_match(LB[a]['items'], LB[b]['items'])
        if mp and max(SD[LB[a]['site']][1][LB[a]['items'][i]['w']] for i, _ in mp) <= 8: nonf.append((a, b))
    for nm, F in (('robust all', rob), ('robust non-formula', nonf)):
        S = dedup(subs_of(LB, F))
        SU = [(x.upper(), y.upper()) + tuple(r) for x, y, *r in S]
        sc = score_subs(SU, R21['LB'], rng)
        cons = lbcons(S, rng)
        rows.append(f"| LA-41.3f | LB control, {nm} ({len(F)} pairs), substitutions deduplicated by word pair: blind la21 LB rows and LB consonants vs partner permutation. | {len(S)} distinct: {'; '.join(f'{x}->{y} ({w1}/{w2})' for x, y, a, b, w1, w2 in S[:30])}. la21 LB rows: {'n %d obs %.3f null %.3f P %.4f' % sc if sc else 'none'}; LB consonant: {'n %d obs %.2f null %.2f P %.4f' % cons if cons else 'none'}. | see verdict |")
    # LA dedup
    LA = la_docs()
    robA, looseA = fam_sets('LA')
    SA = dedup(subs_of(LA, looseA))
    sc = score_subs(SA, R21['LA'], rng)
    # power at n = len(SA)
    signs, M, _ = R21['LA']
    ix = {s: k for k, s in enumerate(signs)}
    E = [(x, y) for x, y, *_ in SA if x in ix and y in ix and x != y]
    n = len(E); pw = {}
    for mix in (1.0, 0.5, 0.25):
        hits = 0
        for r in range(200):
            ss = []
            for k in range(n):
                x = rng.choice(signs[:40])
                if rng.random() < mix:
                    order = np.argsort(-M[ix[x]]); y = rng.choice([signs[o] for o in order if signs[o] != x][:3])
                else:
                    y = rng.choice([s for s in signs if s != x])
                ss.append((x, y))
            res = score_subs(ss, R21['LA'], rng, R=500)
            hits += res[3] < 0.05
        pw[mix] = hits / 200
    rows.append(f"| LA-41.3g | LA loose families, substitutions deduplicated by word pair; blind la21 LA rows; power of the scorer at this n with planted same-row substitutions (each sign's top-3 row partners) mixed with random ones, 200 draws. | {len(SA)} distinct ({n} scorable): la21 rows {'n %d obs %.3f null %.3f P %.4f' % sc if sc else 'none'}. Power at n={n}: 100 % row-mates {pw[1.0]:.2f}, 50 % {pw[0.5]:.2f}, 25 % {pw[0.25]:.2f}. | see verdict |")
    # amount ratios, independent families
    keep = []
    seen86 = False
    for a, b in looseA:
        if {a[:4], b[:4]} == {'HT86', 'HT95'}:
            if seen86: continue
            seen86 = True
        keep.append((a, b))
    n1, k1, nm1, p1, det1 = amount_test(LA, keep, rng)
    K1 = {'HT9a', 'HT9b'}; K2 = {'HT86a', 'HT86b', 'HT95a', 'HT95b'}
    newf = [(a, b) for a, b in looseA if not ({a, b} <= K1 or {a, b} <= K2)]
    n2, k2, nm2, p2, det2 = amount_test(LA, newf, rng)
    rows.append(f"| LA-41.3h | LA amount-ratio test on independent families: HT 86/95 trio counted once; then with HT 9a/b and HT 86/95 (known from la24, la36) removed. Null as 3c. | Independent: {n1} families, {k1} ratio families vs null {nm1:.2f}, P {p1:.4f}. Without the known pairs: {n2} families, {k2} vs {nm2:.2f}, P {p2:.4f}; {'; '.join(f'{a}~{b} hits {h}: {pp}' for a, b, h, pp in det2)}. | see verdict |")
    for r in rows: wlog(OUT, r)
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
