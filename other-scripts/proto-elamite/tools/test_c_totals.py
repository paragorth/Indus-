#!/usr/bin/env python3
"""Test c: do entries add up to the total?

Candidate total = the single numeric line off the obverse (reverse/top) on tablets
that have exactly one such line. Entries = all other numeric lines (obverse).
Only 'clean' tablets: no lacuna, no 'n' counts, no damage marks in numerals.

Step 1: code-wise check (no conversion needed): total count per code == sum.
Step 2: fit unit ratios by grid search over small integers, separately for
tablets with capacity-type codes (C) and without (plain N01/N14/N34/N45/N48).
Control: pair each tablet's entries with a different tablet's total (same class)
and count matches under the best-fit values."""
import collections, itertools, json, os, random, re
from common import load, C_CODES, DATA

random.seed(3)
import sys
STRICT = r'[\[#?]' if '--strict' in sys.argv else r'[\[?]'
T = load()


def clean_num(l):
    return not l['lacuna'] and not re.search(STRICT, l['raw'].split(',')[-1]) and all(
        n is not None for n, _ in l['numerals'])


cases = []
for t in T:
    off = [l for l in t['lines'] if l['surface'] != 'obverse' and l['numerals']]
    if len(off) != 1:
        continue
    ent = [l for l in t['lines'] if l['surface'] == 'obverse' and l['numerals']]
    if len(ent) < 2:
        continue
    if not all(clean_num(l) for l in ent + off):
        continue
    if any('...' in l['raw'] for l in t['lines']):
        continue
    s = collections.Counter(); tot = collections.Counter()
    for l in ent:
        for n, c in l['numerals']:
            s[c] += n
    for n, c in off[0]['numerals']:
        tot[c] += n
    codes = set(s) | set(tot)
    cls = 'C' if codes & C_CODES or any('@' in c for c in codes) else 'N'
    cases.append({'id': t['id'], 'sum': dict(s), 'tot': dict(tot), 'cls': cls,
                  'total_signs': off[0]['signs'], 'n_entries': len(ent)})
print('clean candidate tablets', len(cases), collections.Counter(c['cls'] for c in cases))
codewise = [c for c in cases if c['sum'] == c['tot']]
print('code-wise exact (no carrying needed):', len(codewise))


def value(counter, V):
    v = 0
    for c, n in counter.items():
        if c not in V:
            return None
        v += n * V[c]
    return v


def score(cs, V):
    ok = tot = 0
    for c in cs:
        a, b = value(c['sum'], V), value(c['tot'], V)
        if a is None or b is None:
            continue
        tot += 1; ok += (a == b)
    return ok, tot


# Non-capacity: N01=1, choose N14, N34, N45, N48
best = []
for r14 in (5, 6, 10, 12):
    for n34 in (6, 10, 12, 20):
        for n45 in (6, 10, 12, 20, 60, 100):
            V = {'N01': 1, 'N14': r14, 'N34': r14 * n34, 'N45': r14 * n45, 'N48': r14 * n34 * 60,
                 'N51': r14 * n34 * 2}
            ok, tot = score([c for c in cases if c['cls'] == 'N'], V)
            best.append((ok, tot, r14, n34, n45))
best.sort(reverse=True)
print('NON-CAPACITY fits (matches, testable, N14/N01, N34/N14, N45/N14):')
for b in best[:6]:
    print('  ', b)
# split S vs D: which tablets fit N34=60 and which fit N34=100
S = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N08': 0.5}
D = {'N01': 1, 'N14': 10, 'N34': 100, 'N45': 1000, 'N48': 10000, 'N08': 0.5}
SD = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 100, 'N48': 3600, 'N08': 0.5}  # best of a wider grid
nc = [c for c in cases if c['cls'] == 'N']
fitS = [c for c in nc if value(c['sum'], S) is not None and value(c['sum'], S) == value(c['tot'], S)]
fitD = [c for c in nc if value(c['sum'], D) is not None and value(c['sum'], D) == value(c['tot'], D)]
onlyS = [c for c in fitS if c not in fitD]; onlyD = [c for c in fitD if c not in fitS]
print('non-capacity: fit S (N34=60):', len(fitS), ' fit D (N34=100):', len(fitD),
      ' S-only:', len(onlyS), ' D-only:', len(onlyD), ' of', len(nc))
print('  S-only total signs:', [c['total_signs'] for c in onlyS])
print('  D-only total signs:', [c['total_signs'] for c in onlyD])

# Capacity: order N45 > N14 > N01 > N39B > N24 > N30C > N30D > N39C (from writing order)
cc = [c for c in cases if c['cls'] == 'C']
bestC = []
for r14, r45, r39, r24, r30c, r30d in itertools.product((5, 6, 10), (3, 6, 10), (2, 3, 4, 5, 6, 10), (2, 3, 4, 5, 6), (2, 3, 4, 5, 6), (2, 3, 4, 5)):
    u = r39 * r24 * r30c * r30d  # N30D = 1
    V = {'N30D': 1, 'N30C': r30d, 'N24': r30d * r30c, 'N39B': r30d * r30c * r24, 'N01': u,
         'N14': u * r14, 'N45': u * r14 * r45}
    ok, tot = score(cc, V)
    bestC.append((ok, tot, {'N14/N01': r14, 'N45/N14': r45, 'N01/N39B': r39, 'N39B/N24': r24, 'N24/N30C': r30c, 'N30C/N30D': r30d}))
bestC.sort(key=lambda x: -x[0])
print('CAPACITY fits (top 8):')
for b in bestC[:8]:
    print('  ', b)
# control: mismatched totals
bc = bestC[0][2]
u = bc['N01/N39B'] * bc['N39B/N24'] * bc['N24/N30C'] * bc['N30C/N30D']
VC = {'N30D': 1, 'N30C': bc['N30C/N30D'], 'N24': bc['N30C/N30D'] * bc['N24/N30C'],
      'N39B': bc['N30C/N30D'] * bc['N24/N30C'] * bc['N39B/N24'], 'N01': u, 'N14': u * bc['N14/N01'],
      'N45': u * bc['N14/N01'] * bc['N45/N14']}


def mismatch_rate(cs, V, reps=200):
    hits = n = 0
    for _ in range(reps):
        tots = [c['tot'] for c in cs]; random.shuffle(tots)
        for c, tt in zip(cs, tots):
            a, b = value(c['sum'], V), value(tt, V)
            if a is None or b is None:
                continue
            n += 1; hits += (a == b)
    return hits / max(n, 1)


okS, totS = score(nc, S)
okSD, totSD = score(nc, SD)
hi = [c for c in nc if {'N34', 'N45'} & (set(c['sum']) | set(c['tot']))]
print('wider grid best (N34=60, N45=100): %d/%d ; tablets involving N34/N45: %d, of which add up: %d' % (
    okSD, totSD, len(hi), score(hi, SD)[0]))
low = [c for c in nc if not ({'N34', 'N45'} & (set(c['sum']) | set(c['tot'])))]
print('tablets with only N01/N14(/N08): %d, add up: %d' % (len(low), score(low, S)[0]))
# conversion totals: entries carry no capacity codes, total carries capacity codes
conv = [c for c in cases if not (set(c['sum']) & C_CODES) and (set(c['tot']) & C_CODES)]
print('conversion-type totals (count entries -> capacity total):', len(conv))
for c in conv:
    print('   ', c['id'], c['sum'], '->', c['tot'], c['total_signs'])
okC, totC = score(cc, VC)
print('best S: %d/%d = %.2f ; shuffled-total control %.3f' % (okS, totS, okS / totS, mismatch_rate(nc, S)))
print('best C: %d/%d = %.2f ; shuffled-total control %.3f' % (okC, totC, okC / totC, mismatch_rate(cc, VC)))
# near misses: off by a single unit of some code
json.dump({'cases': cases, 'S': S, 'D': D, 'C_best': VC,
           'nonC_fits': best[:10], 'C_fits': [(a, b, c) for a, b, c in bestC[:10]]},
          open(os.path.join(DATA, 'res_c_totals.json'), 'w'), indent=1)
