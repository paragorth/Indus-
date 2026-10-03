#!/usr/bin/env python3
"""PE-1.4: what recurs? (a) Tablet pairs sharing >=2 designations: near-duplicate
texts or shared rosters? Jaccard of full entry-string sets vs random tablet
pairs of matched size. (b) Composition: share of recurring (cross-tablet)
designation types that begin with a prefix-slot sign or contain a class sign,
vs one-off types; permutation control (labels shuffled over types, matched on
length). (c) Spread: cross-volume and cross-site recurrence vs a placement null
(designations permuted over slots)."""
import json, random
from pe1_lib import *
T = load(); meta = tablet_meta(T)
DE = designation_entries(T, minlen=2)
tabs = collections.defaultdict(set)
for e in DE: tabs[e['des']].add(e['tablet'])
rec = {d for d, ts in tabs.items() if len(ts) >= 2}
res = {}
# (a)
tp = collections.defaultdict(set)
for d in rec:
    ts = sorted(tabs[d])
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)): tp[(ts[i], ts[j])].add(d)
multi = {k: v for k, v in tp.items() if len(v) >= 2}
Eall = entries(T)
strs = collections.defaultdict(set)
for e in Eall: strs[e['tablet']].add(tuple(e['signs']))
def jac(a, b):
    A, B = strs[a], strs[b]; return len(A & B) / max(1, len(A | B))
rows = []
for (a, b), ds in sorted(multi.items(), key=lambda kv: -len(kv[1])):
    rows.append({'pair': [a, b], 'shared_des': [' '.join(x) for x in ds], 'jaccard_all_entries': round(jac(a, b), 3),
                 'n_entries': [len(strs[a]), len(strs[b])], 'vols': [meta[a]['vol'], meta[b]['vol']],
                 'headers': [meta[a]['header_full'], meta[b]['header_full']]})
res['pairs_sharing_2plus'] = rows
print('tablet pairs sharing >=2 recurring designations:', len(rows))
for r in rows: print('  ', r['pair'], r['shared_des'], 'J=%.2f' % r['jaccard_all_entries'], r['n_entries'], r['vols'], r['headers'])
share_pairs_multi = sum(len(v) for v in multi.values()) / sum(len(v) for v in tp.values())
res['share_of_cross_links_in_multi_pairs'] = share_pairs_multi
print('share of designation cross-links carried by pairs sharing >=2: %.3f' % share_pairs_multi)
# (b)
types = list(tabs)
def comp(ds):
    n = len(ds)
    return {'prefix_initial': sum(d[0] in PREFIX for d in ds) / n,
            'contains_class': sum(any(s in CLASS for s in d) for d in ds) / n,
            'len2': sum(len(d) == 2 for d in ds) / n}
ro = comp([d for d in types if d in rec]); oo = comp([d for d in types if d not in rec])
rng = random.Random(4)
bylen = collections.defaultdict(list)
for d in types: bylen[len(d)].append(d)
nl = []
for _ in range(2000):
    samp = []
    for L, ds in bylen.items():
        k = sum(1 for d in ds if d in rec)
        samp += rng.sample(ds, k)
    nl.append(comp(samp))
res['composition'] = {'recurring': ro, 'one_off': oo}
for k in ['prefix_initial', 'contains_class']:
    res['composition'][k + '|len_matched_null'] = zp(ro[k], [x[k] for x in nl])
    print(k, 'recurring %.3f one-off %.3f' % (ro[k], oo[k]), res['composition'][k + '|len_matched_null'])
# (c) spread
def spread(D):
    by = collections.defaultdict(set)
    for t, d in D: by[d].add(t)
    xv = xs = n = 0
    for d, ts in by.items():
        if len(ts) < 2: continue
        n += 1
        xv += len({meta[t]['vol'] for t in ts}) >= 2
        xs += len({meta[t]['prov'] for t in ts}) >= 2
    return {'n_rec': n, 'cross_vol': xv, 'cross_site': xs, 'cross_vol_share': xv / max(1, n)}
D = [(e['tablet'], e['des']) for e in DE]
o = spread(D); nl = []
for _ in range(1000):
    ds = [d for _, d in D]; rng.shuffle(ds); nl.append(spread([(t, d) for (t, _), d in zip(D, ds)]))
res['spread'] = {'obs': o}
for k in ['cross_vol_share', 'cross_site']:
    res['spread'][k] = zp(o[k], [x[k] for x in nl]); print(k, res['spread'][k])
xs = [(' '.join(d), sorted({meta[t]['prov'] for t in tabs[d]})) for d in rec if len({meta[t]['prov'] for t in tabs[d]}) >= 2]
res['cross_site_list'] = xs; print('cross-site recurring:', xs)
nonsusa = sum(1 for e in DE if not meta[e['tablet']]['prov'].startswith('Susa'))
res['non_susa_designation_tokens'] = nonsusa; print('non-Susa designation tokens', nonsusa)
json.dump(res, open(os.path.join(DATA, 'pe1_c4_whatrecurs.json'), 'w'), indent=1, default=str)
