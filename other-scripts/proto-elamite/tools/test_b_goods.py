#!/usr/bin/env python3
"""Test b: which entry-final signs (the sign touching the numeral) go with which
number system. Control: permute numerals across entries (global, and within tablet)."""
import collections, math, random, json, os
from common import load, entries, DATA

random.seed(2)
NPERM = 1000
T = load()
E = [e for e in entries(T) if e['signs'][-1] != 'x']
print('entries with readable final sign', len(E))
sysc = collections.Counter(e['system'] for e in E)
print('system counts', sysc.most_common())


def mi(pairs):
    n = len(pairs)
    a = collections.Counter(x for x, _ in pairs); b = collections.Counter(y for _, y in pairs)
    j = collections.Counter(pairs)
    return sum(c / n * math.log2(c * n / (a[x] * b[y])) for (x, y), c in j.items())


def keyed(e):
    return e['signs'][-1]


# restrict to final signs with n>=10 to keep MI estimates sane; others pooled
fc = collections.Counter(keyed(e) for e in E)
lab = [keyed(e) if fc[keyed(e)] >= 10 else 'OTHER' for e in E]
sy = [e['system'] for e in E]
obs = mi(list(zip(lab, sy)))
glob = []
for _ in range(NPERM):
    p = sy[:]; random.shuffle(p); glob.append(mi(list(zip(lab, p))))
# within-tablet permutation
bytab = collections.defaultdict(list)
for i, e in enumerate(E):
    bytab[e['tablet']].append(i)
within = []
for _ in range(NPERM):
    p = sy[:]
    for idx in bytab.values():
        v = [sy[i] for i in idx]; random.shuffle(v)
        for i, x in zip(idx, v):
            p[i] = x
    within.append(mi(list(zip(lab, p))))
print('MI(final sign; system) obs %.3f bits | global-perm mean %.3f max %.3f | within-tablet-perm mean %.3f max %.3f' % (
    obs, sum(glob) / NPERM, max(glob), sum(within) / NPERM, max(within)))

# per-sign profile: share in C (grain-type capacity) and B, versus baseline
base_c = sum(1 for s in sy if s in ('C', 'C*')) / len(sy)
rows = []
for s, n in fc.items():
    if n < 10:
        continue
    ss = [e['system'] for e in E if keyed(e) == s]
    cnt = collections.Counter(ss)
    top, tn = cnt.most_common(1)[0]
    c_share = (cnt['C'] + cnt['C*']) / n
    # binomial tail for C share vs baseline (normal approx)
    z = (c_share - base_c) / math.sqrt(base_c * (1 - base_c) / n)
    rows.append({'sign': s, 'n': n, 'profile': dict(cnt), 'top': top, 'top_share': round(tn / n, 2),
                 'C_share': round(c_share, 2), 'z_C': round(z, 1)})
rows.sort(key=lambda r: -r['n'])
print('baseline C share %.3f' % base_c)
print('sign | n | C share | z | profile')
for r in rows[:45]:
    print(r['sign'], r['n'], r['C_share'], r['z_C'], r['profile'])
strongC = [r for r in rows if r['C_share'] >= 0.9 and r['n'] >= 15]
strongN = [r for r in rows if r['C_share'] <= 0.02 and r['n'] >= 30]
print('signs >=90%% C (n>=15):', [(r['sign'], r['n']) for r in strongC])
print('signs <=2%% C (n>=30):', [(r['sign'], r['n']) for r in strongN])
# within-tablet control for the per-sign C share: purity of sign vs purity of its tablets
for r in strongC + strongN[:8]:
    s = r['sign']
    tabs = {e['tablet'] for e in E if keyed(e) == s}
    others = [e['system'] for e in E if e['tablet'] in tabs and keyed(e) != s]
    oc = sum(1 for x in others if x in ('C', 'C*'))
    r['C_share_other_entries_same_tablets'] = round(oc / len(others), 2) if others else None
    print('  %s: C share of OTHER entries on the same tablets = %s (n=%d), tablets=%d' % (
        s, r['C_share_other_entries_same_tablets'], len(others), len(tabs)))
json.dump({'mi_obs': obs, 'mi_glob_mean': sum(glob) / NPERM, 'mi_glob_max': max(glob),
           'mi_within_mean': sum(within) / NPERM, 'mi_within_max': max(within), 'rows': rows},
          open(os.path.join(DATA, 'res_b_goods.json'), 'w'), indent=1)
