"""pe59 cycle 3: freeze explicit outside-corpus predictions of the reading (rates measured on the held-out half
with null rates for each), hash them, and write data/pe59_frozen_predictions.json.
"""
import sys, os, json, math, random
from collections import Counter
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe59_lib import (READING, build_pe, split, pe_roles, pe_maps, tablet_type, ncls, value, line_role, sha, CK, DATA,
                      header_role)

T = build_pe()
tr, ho = split(T)
R = pe_roles()
cap, cnt = pe_maps()
cntm = cnt['sex2']


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 1)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def after_count(tabs, want_allot, mults=(60,)):
    k = n = 0
    for t in tabs:
        prev = None
        for l in t['lines']:
            if l['role'] != 'E':
                continue
            if prev is not None and prev['numclean'] and ncls(prev['nums']) == 'AMB' and l['numclean'] and l['signs'] \
                    and prev['signs'] and prev['signs'][-1] != l['signs'][-1]:
                is_allot = l['signs'][-1] == 'M288'
                ku = value(prev['nums'], cntm)
                if is_allot == want_allot and ku and ku <= 60:
                    v = value(l['nums'], cap)
                    if v is not None and ncls(l['nums']) in ('CAP', 'AMB'):
                        n += 1
                        k += v in [m * ku for m in mults]
            prev = l
    return k, n


out = {'reading_sha': json.load(open(os.path.join(DATA, 'pe59_reading_frozen.json')))['sha256'], 'predictions': []}
P = out['predictions']

# P1 unit sizes
lo, hi = READING['systems']['CAP']['litres_per_N39C']
units = {c: [round(v * lo, 3), round(v * hi, 3)] for c, v in READING['systems']['CAP']['values'].items() if v <= 21600}
P.append({'id': 'P1', 'grade': 'C', 'what': 'capacity unit sizes (litres), main scale',
          'values': units, 'alias_x9_12': {c: [round(v * 5, 2), round(v * 7, 2)] for c, v in
                                          READING['systems']['CAP']['values'].items() if v <= 120},
          'would_support': 'PE-period standard vessels (Susa III, Malyan Banesh, Yahya IVC) with capacity modes near '
                           '0.7 l (N39C), 2.8 l (N30C) and 8.4 l (N24), or at the x9-12 alias 5-7 l',
          'would_kill': 'standard vessel series with no mode in 0.5-1.0 l, 2-3.5 l or 4-9 l'})
P.append({'id': 'P2', 'grade': 'B-', 'what': 'standard allotment M288 = 60 N39C per unit (36-48 l main scale), '
          'i.e. one adult grain allotment; 2(N39B) 1(N24) per head',
          'would_support': 'see P3', 'would_kill': 'see P3'})
k, n = after_count(ho, True)
k0, n0 = after_count(ho, False)
k1, n1 = after_count(ho, True, (120,))
k10, n10 = after_count(ho, False, (120,))
P.append({'id': 'P3', 'grade': 'B', 'what': 'on new tablets, an M288 line directly after a count line of k <= 60 '
          'units holds exactly 60k N39C = 2k(N39B) k(N24) (1/2 N01 per unit)', 'heldout_rate': [k, n, wilson(k, n)],
          'null_other_signs_same_position': [k0, n0, wilson(k0, n0)],
          'demoted_alternative_120k': {'M288': [k1, n1], 'null': [k10, n10], 'note': 'the 1 N01 per unit variant is at or below the null; dropped'},
          'would_support': '>= 20% of such lines (>= 20 lines)', 'would_kill': '< 10% (the null rate is 8%)'})


def role_sys(tabs, role, test):
    k = n = 0
    for t in tabs:
        for l in t['lines']:
            if l['role'] == 'E' and l['numclean'] and l['signs'] and line_role(l['signs'], R) == role:
                n += 1; k += test(l, t)
    return k, n


def cap_ok(l, t):
    return ncls(l['nums']) == 'CAP' or (ncls(l['nums']) == 'AMB' and tablet_type(t) == 'CAPT')


k, n = role_sys(ho, 'MEASURED', cap_ok)
kb, nb = 0, 0
for t in ho:
    for l in t['lines']:
        if l['role'] == 'E' and l['numclean']:
            nb += 1; kb += cap_ok(l, t)
P.append({'id': 'P4', 'grade': 'B', 'what': 'entries ending in a MEASURED-class sign carry capacity numerals',
          'signs': sorted(R['sets']['MEASURED']), 'heldout_rate': [k, n, wilson(k, n)], 'base_rate_all_entries': [kb, nb, wilson(kb, nb)],
          'would_support': '>= 50%', 'would_kill': '<= base rate'})
k, n = role_sys(ho, 'COUNTED', lambda l, t: ncls(l['nums']) != 'CAP')
P.append({'id': 'P5', 'grade': 'A/B', 'what': 'entries ending in a COUNTED-class sign never carry capacity codes',
          'signs': sorted(R['sets']['COUNTED']), 'heldout_rate': [k, n, wilson(k, n)],
          'would_support': '>= 95% non-capacity', 'would_kill': '< 85%'})
d = json.load(open(os.path.join(CK, 'c1_PE.json')))
cl = d['closure']
P.append({'id': 'P6', 'grade': 'B', 'what': 'written totals (single numeric reverse line) equal the entry sum in the system '
          'predicted from the signs (capacity: N14 = 6 N01 = 720 N39C; counts: N14 = 10)',
          'heldout_rate': [cl['read'][0], cl['read'][1], wilson(cl['read'][0], cl['read'][1])],
          'null_random_totals_mean': cl['null']['read'][0],
          'would_support': '>= 10% of new clean totalled tablets', 'would_kill': '< 4%'})
# P7 total line has no dedicated sign
tot_signs = Counter(); ntot = 0
for t in T:
    for l in t['lines']:
        if l['role'] == 'T':
            ntot += 1
            tot_signs.update(set(l['signs']))
P.append({'id': 'P7', 'grade': 'A', 'what': 'no sign marks totals: no sign occurs on more than 25% of total lines',
          'max_sign_share_corpus': [tot_signs.most_common(1)[0][0], round(tot_signs.most_common(1)[0][1] / ntot, 3)],
          'would_kill': 'a sign on > 40% of new total lines'})
# P8 edge tag
m157 = [t for t in T if any(l['role'] == 'H' and l['signs'][:1] == ['M157'] for l in t['lines'])]
oth = [t for t in T if t not in m157]
e1 = sum(any(l['role'] == 'G' for l in t['lines']) for t in m157)
e0 = sum(any(l['role'] == 'G' for l in t['lines']) for t in oth)
P.append({'id': 'P8', 'grade': 'B', 'what': 'top-edge numeral tag (1(N34)) is commoner on M157-headed tablets and never a sum',
          'corpus': {'M157': [e1, len(m157)], 'other': [e0, len(oth)]},
          'would_support': 'rate on M157 tablets >= 2x other', 'would_kill': 'equal rates on >= 100 new tablets'})
P.append({'id': 'P9', 'grade': 'C', 'what': 'outposts (pe48 frozen file, sha 1946e391ffe27be8): new Yahya tablets carry M219, '
          'M056, M136 about 1 in 20 each; M044 Yahya-only', 'would_kill': '40+ new Yahya tablets with none of M219/M056/M136/M044'})
P.append({'id': 'P10', 'grade': 'C', 'what': 'the 89 untransliterated Tehran Susa tablets (pe22): a tablet whose entries end in '
          'MEASURED signs is a capacity account; one whose entries end in COUNTED signs is a count account; the decoder will '
          'gloss them with the same full-explanation rate as the held-out half (cycle 2) +- 10 points'})
out['sha256'] = sha(out['predictions'])
json.dump(out, open(os.path.join(DATA, 'pe59_frozen_predictions.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
