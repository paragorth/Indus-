#!/usr/bin/env python3
"""Test c2: 'conversion' tablets. Entries are plain counts (no capacity codes);
the final off-obverse line is a capacity (grain-type) amount. If the total is
count x fixed ration, then for the right unit ratios the per-head rate should be
a 'round' quantity (expressible by one or two unit signs) on many tablets.

Grid search over capacity ratios; score = number of tablets whose rate (total /
head count) equals k x (one unit sign) with k in 1..9 (a 'simple' rate).
Control: the same score with totals shuffled among tablets (200 shuffles),
which keeps the ratio grid's freedom but breaks the count/total pairing."""
import collections, itertools, json, os, random
from fractions import Fraction as F
from common import load, system_of, C_CODES, DATA

random.seed(5)
T = load()
cases = []
for t in T:
    off = [l for l in t['lines'] if l['surface'] != 'obverse' and l['numerals']]
    ent = [l for l in t['lines'] if l['surface'] == 'obverse' and l['numerals']]
    if not off or not ent:
        continue
    if not all(system_of(l['numerals']) == 'SDB' for l in ent):
        continue
    tot = off[-1]
    if system_of(tot['numerals']) != 'C':
        continue
    if any(n is None for l in ent + [tot] for n, _ in l['numerals']) or any(l['lacuna'] for l in ent + [tot]):
        continue
    heads = sum(n * {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600}.get(c, 0) for l in ent for n, c in l['numerals'])
    if heads == 0:
        continue
    cases.append({'id': t['id'], 'heads': heads, 'tot': [[n, c] for n, c in tot['numerals']],
                  'tot_signs': tot['signs']})
print('conversion tablets usable', len(cases))
for c in cases:
    print('  ', c['id'], c['heads'], '->', c['tot'], c['tot_signs'])


def values(r):
    r14, r39, r24, r30c, r30d, r39c = r
    V = {'N39C': F(1)}
    V['N30D'] = V['N39C'] * r39c
    V['N30C'] = V['N30D'] * r30d
    V['N24'] = V['N30C'] * r30c
    V['N39B'] = V['N24'] * r24
    V['N01'] = V['N39B'] * r39
    V['N14'] = V['N01'] * r14
    V['N45'] = V['N14'] * 10
    return V


def simple(rate, V):
    for u in ('N01', 'N39B', 'N24', 'N30C', 'N30D', 'N39C'):
        k = rate / V[u]
        if k.denominator == 1 and 1 <= k <= 9:
            return True
    return False


def score(cs, tots, V):
    ok = 0
    for c, tt in zip(cs, tots):
        try:
            g = sum(n * V[code] for n, code in tt)
        except KeyError:
            continue
        ok += simple(F(g) / c['heads'], V)
    return ok


grid = list(itertools.product((6, 10), (2, 3, 4, 5, 6, 10), (2, 3, 4, 5, 6), (2, 3, 4, 5, 6), (2, 3, 4, 5, 6), (2, 3, 4, 5, 6)))
best = []
tots = [c['tot'] for c in cases]
for r in grid:
    best.append((score(cases, tots, values(r)), r))
best.sort(key=lambda x: -x[0])
print('best ratio sets (score, (N14/N01, N01/N39B, N39B/N24, N24/N30C, N30C/N30D, N30D/N39C)):')
for b in best[:8]:
    print('  ', b)
top = best[0][0]
# control: max over the grid of the score with shuffled totals
ctrl = []
for _ in range(30):
    sh = tots[:]; random.shuffle(sh)
    ctrl.append(max(score(cases, sh, values(r)) for r in grid))
print('observed best %d of %d; shuffled-total best-over-grid: mean %.1f max %d' % (
    top, len(cases), sum(ctrl) / len(ctrl), max(ctrl)))
json.dump({'cases': cases, 'best': [(s, list(r)) for s, r in best[:20]], 'control': ctrl},
          open(os.path.join(DATA, 'res_c2_rations.json'), 'w'), indent=1)
