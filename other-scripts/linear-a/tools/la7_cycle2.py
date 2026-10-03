#!/usr/bin/env python3
"""LA-7 cycle 2: (a) wider link window; (b) last sign instead of first (determinative-like slot);
(c) sign-by-commodity alignments with hypergeometric tails, BH correction, and a type-level check
(each word-commodity type counted once)."""
import os, sys
from collections import Counter, defaultdict
from scipy.stats import hypergeom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la7_common import *
import la7_test as T
T.N = 2000

lab, glyph = picture_codes(False)
labs, _ = picture_codes(True)
labab = {ab_number(glyph[s]): c for s, c in lab.items() if ab_number(glyph[s]) is not None}
fam = lambda P: [(s, k, COM_FAMILY[c], d, w) for s, k, c, d, w in P]
runs = [('LA window first', fam(la_pairs(mode='window')), lab),
        ('LA window first strict', fam(la_pairs(mode='window')), labs),
        ('LA adjacent last', fam(la_pairs(pos=-1)), lab),
        ('LA window last', fam(la_pairs(mode='window', pos=-1)), lab),
        ('LB adjacent last', lb_pairs(-1), labab)]
for name, P, L in runs:
    for sub, pr in list(T.subsets(P).items())[:2]:
        print('%-24s %-22s %s' % (name, sub, T.perm_test(pr, L)))

# (c) alignments: first sign of multi-sign words and single-sign words separately, window link
P = la_pairs(mode='window')
tests = []
for kind, sel in (('first', lambda k: k > 1), ('single', lambda k: k == 1)):
    for unit in ('tok', 'type'):
        rows = [(s, c, w) for s, k, c, d, w in P if sel(k)]
        if unit == 'type': rows = list({(w, c): (s, c, w) for s, c, w in rows}.values())
        N = len(rows); cc = Counter(c for _, c, _ in rows); sc = Counter(s for s, _, _ in rows)
        cell = Counter((s, c) for s, c, _ in rows)
        for s, n in sc.items():
            if n < 3: continue
            for c, K in cc.items():
                x = cell[(s, c)]
                if x < 2: continue
                p = hypergeom.sf(x - 1, N, K, n)
                tests.append((p, kind, unit, s, lab.get(s, '-'), c, COM_FAMILY[c], x, n, round(n * K / N, 2)))
tests.sort()
m = len(tests)
print('\n%d sign x commodity cells tested (n>=3 sign, x>=2)' % m)
print('p  BHq  kind unit sign class commodity family x/n expected  picture-consistent')
for i, t in enumerate(tests[:25]):
    q = min(1, t[0] * m / (i + 1))
    ok = t[6] in PRED.get(t[4], ())
    print('%.4f %.3f %s' % (t[0], q, t[1:]), 'YES' if ok else '')
