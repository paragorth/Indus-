#!/usr/bin/env python3
"""Test a: entry structure. Length distribution and entry-initial / entry-final
sign preferences versus within-entry shuffles (1000 permutations)."""
import collections, random, json, os, sys
from common import load, entries, DATA

random.seed(1)
NPERM = 1000
T = load()
E = [e['signs'] for e in entries(T, require_clean=True)]
lens = collections.Counter(len(s) for s in E)
print('clean entries', len(E))
print('length distribution', sorted(lens.items()))
tot = sum(lens.values())
print('share len1 %.3f len2 %.3f len3 %.3f len>=4 %.3f' % (
    lens[1] / tot, lens[2] / tot, lens[3] / tot, sum(v for k, v in lens.items() if k >= 4) / tot))

M = [s for s in E if len(s) >= 2]
freq = collections.Counter(x for s in M for x in s)


def pos_counts(strings):
    ini, fin = collections.Counter(), collections.Counter()
    for s in strings:
        ini[s[0]] += 1
        fin[s[-1]] += 1
    return ini, fin


obs_i, obs_f = pos_counts(M)
common = [x for x, c in freq.items() if c >= 20]
ge_i = collections.Counter(); ge_f = collections.Counter()
sum_i = collections.Counter(); sum_f = collections.Counter()
sq_i = collections.Counter(); sq_f = collections.Counter()
for _ in range(NPERM):
    P = []
    for s in M:
        s = s[:]
        random.shuffle(s)
        P.append(s)
    pi, pf = pos_counts(P)
    for x in common:
        ge_i[x] += pi[x] >= obs_i[x]
        ge_f[x] += pf[x] >= obs_f[x]
        sum_i[x] += pi[x]; sum_f[x] += pf[x]
        sq_i[x] += pi[x] ** 2; sq_f[x] += pf[x] ** 2
rows = []
for x in common:
    mi = sum_i[x] / NPERM; mf = sum_f[x] / NPERM
    sdi = max((sq_i[x] / NPERM - mi * mi) ** .5, 1e-9)
    sdf = max((sq_f[x] / NPERM - mf * mf) ** .5, 1e-9)
    rows.append({'sign': x, 'n': freq[x], 'init': obs_i[x], 'init_exp': round(mi, 1),
                 'z_init': round((obs_i[x] - mi) / sdi, 1),
                 'final': obs_f[x], 'final_exp': round(mf, 1),
                 'z_final': round((obs_f[x] - mf) / sdf, 1)})
nsig_i = sum(1 for r in rows if r['z_init'] >= 3)
nsig_f = sum(1 for r in rows if r['z_final'] >= 3)
print('signs with n>=20 in multi-sign entries:', len(rows))
print('initial-biased (z>=3):', nsig_i, ' final-biased (z>=3):', nsig_f)
print('TOP INITIAL'); [print(r) for r in sorted(rows, key=lambda r: -r['z_init'])[:12]]
print('TOP FINAL'); [print(r) for r in sorted(rows, key=lambda r: -r['z_final'])[:12]]
# share of multi-sign entries that end in a final-biased sign / start in initial-biased
fin_set = {r['sign'] for r in rows if r['z_final'] >= 3}
ini_set = {r['sign'] for r in rows if r['z_init'] >= 3}
print('share of multi-sign entries ending in a final-biased sign %.3f' % (sum(s[-1] in fin_set for s in M) / len(M)))
print('share starting with an initial-biased sign %.3f' % (sum(s[0] in ini_set for s in M) / len(M)))
# overlap: signs both initial and final biased
print('both:', ini_set & fin_set)
json.dump({'lengths': dict(lens), 'rows': rows}, open(os.path.join(DATA, 'res_a_slots.json'), 'w'), indent=1)
