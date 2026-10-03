#!/usr/bin/env python3
"""LA-1 cycle 2: exact-sum test of the sub-unit assignment from la1_assign.py.

Sections: every KU-RO section with a total (tools/totals_test.py rule, 30 sections) plus the hand-split
HT 123+124a *308 column (8E + 8JE + 4A + 4E = 25H; commodity unknown, scored as dry).
Each quantity gets the value of its letters under a value set; dry/liquid family of an entry is taken
from its carried-over commodity (OLE, VIN = liquid; else dry).
Value sets:
  INT      integers only (fractions = 0)
  SITE     lineara.xyz conventional values (reference only)
  LB-ABS   letter -> assigned Linear B role, Linear B absolute values
           dry T 1/10, V 1/60, Z 1/240; liquid S 1/3, V 1/18, Z 1/72 (T in a liquid entry uses 1/10)
  LB-SCALE same role ratios (T = 6V, V = 4Z; S = 6V liquid), V's value free over the pool below,
           best scale kept (the null gets the same best-of-k search).
Null: each letter independently draws a value from POOL = unit fractions 1/2..1/16 and every sum of
two of them below 1 (10,000 draws). Statistic: sections exact; "extra" = exact minus INT exact.
"""
import sys, os, json, random, itertools
from fractions import Fraction as Fr
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import totals_test as TT
import la1_common as L

random.seed(23)
ASSIGN = json.load(open(os.path.join(L.D, 'la1_assign.json')))['cap']['assign']
LIQ_C = {'OLE', 'VIN'}
# annotate number tokens with carried-over commodity
for ins in TT.C:
    cur = None
    for t in ins['tokens']:
        if t['t'] == 'logo':
            b = t['v'].split('+')[0].lstrip('*')
            if b in L.LA_MAIN: cur = b
        elif t['t'] == 'word' and t['s'] == ['NI']: cur = 'NI'
        elif t['t'] == 'num': t['com'] = cur

secs = [s for s in TT.sections('KU-RO') if s['entries'] and s['tot']]
mk = lambda v, fr: {'t': 'num', 'v': v, 'frac': fr, 'com': None}
secs.append({'id': 'HT123+124a *308 (hand)', 'tot': mk(25, ['H']),
             'entries': [mk(8, ['E']), mk(8, ['JE']), mk(4, ['A']), mk(4, ['E'])]})
LETTERS = sorted({f for s in secs for q in s['entries'] + [s['tot']] for f in q['frac']})
FSEC = [s for s in secs if any(q['frac'] for q in s['entries'] + [s['tot']])]
print(f'{len(secs)} sections, {len(FSEC)} with fractions; letters in sections: {LETTERS}')

UF = [Fr(1, d) for d in range(2, 17)]
POOL = sorted(set(UF) | {a + b for a, b in itertools.combinations_with_replacement(UF, 2) if a + b < 1})
print('pool size', len(POOL))


def qval(q, valfn):
    return Fr(q['v']) + sum((valfn(f, q.get('com')) for f in q['frac']), Fr(0))


def score(valfn, sl=secs):
    ok = []
    for s in sl:
        if sum(qval(e, valfn) for e in s['entries']) == qval(s['tot'], valfn): ok.append(s['id'])
    return ok


INT = score(lambda f, c: Fr(0))
SITE = score(lambda f, c: TT.SITE.get(f, Fr(0)))
DRYV = {'T': Fr(1, 10), 'V': Fr(1, 60), 'Z': Fr(1, 240), 'S': Fr(1, 3)}
LIQV = {'T': Fr(1, 10), 'V': Fr(1, 18), 'Z': Fr(1, 72), 'S': Fr(1, 3)}
role = lambda f: ASSIGN.get(f, 'V')       # letters outside the assignment (rare) default to V
lbabs = lambda f, c: (LIQV if c in LIQ_C else DRYV)[role(f)]
LBABS = score(lbabs)
RATIO_D = {'T': 6, 'V': 1, 'Z': Fr(1, 4), 'S': 6}
RATIO_L = {'T': 6, 'V': 1, 'Z': Fr(1, 4), 'S': 6}


def lbscale(v):
    return lambda f, c: v * (RATIO_L if c in LIQ_C else RATIO_D)[role(f)]


scale_res = {v: score(lbscale(v)) for v in POOL}
best_v = max(scale_res, key=lambda v: len(scale_res[v]))
LBS = scale_res[best_v]
print(f'INT exact {len(INT)}/{len(secs)}: {INT}')
print(f'SITE exact {len(SITE)}: extra over INT {sorted(set(SITE) - set(INT))}')
print(f'LB-ABS exact {len(LBABS)}: extra {sorted(set(LBABS) - set(INT))}')
print(f'LB-SCALE best V = {best_v}: exact {len(LBS)}: extra {sorted(set(LBS) - set(INT))}; '
      f'scales giving any extra: {sum(1 for v in POOL if len(scale_res[v]) > len(INT))}/{len(POOL)}')

# null
K = len(POOL)
null1, nullk = [], []
for it in range(10000):
    vals = {f: random.choice(POOL) for f in LETTERS}
    null1.append(len(score(lambda f, c: vals[f], FSEC)))
nonf = len([s for s in INT if s not in [x['id'] for x in FSEC]])
ext = lambda ok: len([i for i in ok if i in {x['id'] for x in FSEC}])
e_abs, e_s = ext(LBABS), ext(LBS)
print(f'fraction sections exact: LB-ABS {e_abs}, LB-SCALE(best) {e_s}, SITE {ext(SITE)}, INT {ext(INT)}')
print(f'null (one random value set): mean {sum(null1)/len(null1):.3f}, P(>= LB-ABS {e_abs}) = {sum(v >= e_abs for v in null1)/len(null1):.3f}, '
      f'P(>= SITE {ext(SITE)}) = {sum(v >= ext(SITE) for v in null1)/len(null1):.3f}')
# best-of-K null for the scale search
bk = []
for it in range(300):
    best = 0
    for _ in range(K):
        vals = {f: random.choice(POOL) for f in LETTERS}
        best = max(best, len(score(lambda f, c: vals[f], FSEC)))
    bk.append(best)
print(f'best-of-{K} null: mean {sum(bk)/len(bk):.2f}, P(>= LB-SCALE {e_s}) = {sum(v >= e_s for v in bk)/len(bk):.3f}')
# per-section requirement under LB roles
print('per fraction section: entries - total under LB-ABS')
for s in FSEC:
    g = sum(qval(e, lbabs) for e in s['entries']) - qval(s['tot'], lbabs)
    com = Counter(q.get('com') for q in s['entries'])
    print(f"  {s['id']:28s} gap {str(g):>10s}  commodities {dict(com)}")
json.dump({'INT': INT, 'SITE': SITE, 'LBABS': LBABS, 'LBSCALE': [str(best_v), LBS], 'null_mean': sum(null1) / len(null1)},
          open(os.path.join(L.D, 'la1_totals.json'), 'w'), indent=1)
