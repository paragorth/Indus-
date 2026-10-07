#!/usr/bin/env python3
"""LA-84 cycle 1b: kill tests for the worn-group deficit and the frozen re-reading prediction.
(1) within-document control: worn vs unworn groups on the SAME documents, worn labels permuted within document
    and length class (10,000 permutations);  (2) site split: the worn deficit at HT and outside HT separately,
    each against its own matched-read null;  (3) publication split (<= 1950 vs later, la78 years);
(4) freeze: every worn group that is near-only (no exact match elsewhere, a one-sign neighbour elsewhere) with its
    neighbours -> data/la84_frozen_c1.json + .sha256.
"""
import sys, os, json, random, hashlib, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la84_common as C
import la78_common as L78

T, LEX = C.load()
LX = C.Lex(LEX)
YEAR = {d['id']: d['year'] for d in L78.load()}
rng = random.Random(841)


def fam(t):
    c = LX.count(t['s'], t['doc'])
    return (1 if c > 0 else 0), (1 if c == 0 and LX.near(t['s'], t['doc']) else 0)


for t in T:
    t['ex'], t['nr'] = fam(t)

out = {}
# (1) within document
bydoc = collections.defaultdict(list)
for t in T:
    if t['cls'] in ('READ', 'WORN'):
        bydoc[t['doc']].append(t)
mixed = {d: v for d, v in bydoc.items() if any(x['cls'] == 'WORN' for x in v) and any(x['cls'] == 'READ' for x in v)}
groups = []
for d, v in mixed.items():
    bl = collections.defaultdict(list)
    for x in v:
        bl[x['stratum'][2]].append(x)
    for L, xs in bl.items():
        if any(x['cls'] == 'WORN' for x in xs) and any(x['cls'] == 'READ' for x in xs):
            groups.append(xs)
W = [x for g in groups for x in g if x['cls'] == 'WORN']
R = [x for g in groups for x in g if x['cls'] == 'READ']


def diff(groups, labels=None):
    w = [];  r = []
    for gi, xs in enumerate(groups):
        lab = labels[gi] if labels else [x['cls'] for x in xs]
        for x, l in zip(xs, lab):
            (w if l == 'WORN' else r).append(x)
    return (np.mean([x['ex'] for x in w]) - np.mean([x['ex'] for x in r]),
            np.mean([x['nr'] for x in w]) - np.mean([x['nr'] for x in r]))


obs = diff(groups)
null = []
for _ in range(10000):
    labs = []
    for xs in groups:
        l = [x['cls'] for x in xs]
        rng.shuffle(l)
        labs.append(l)
    null.append(diff(groups, labs))
null = np.array(null)
out['within_doc'] = dict(n_docs=len(mixed), n_groups=len(groups), n_worn=len(W), n_read=len(R),
                         d_exact=obs[0], d_near=obs[1],
                         P_exact_le=float((null[:, 0] <= obs[0]).mean()), P_near_ge=float((null[:, 1] >= obs[1]).mean()))


# (2)/(3) matched null by split
def split_test(pred):
    worn = [t for t in T if t['cls'] == 'WORN' and pred(t)]
    st = collections.Counter(t['stratum'] for t in worn)
    pool = collections.defaultdict(list)
    for t in T:
        if t['cls'] == 'READ':
            pool[t['stratum']].append(t)
    o = (np.mean([t['ex'] for t in worn]), np.mean([t['nr'] for t in worn]))
    nn = []
    for _ in range(5000):
        s = [rng.choice(pool[k]) for k, n in st.items() for _ in range(n) if pool[k]]
        nn.append((np.mean([t['ex'] for t in s]), np.mean([t['nr'] for t in s])))
    nn = np.array(nn)
    return dict(n=len(worn), exact=o[0], exact_null=float(nn[:, 0].mean()), near=o[1], near_null=float(nn[:, 1].mean()),
                P_exact_le=float((nn[:, 0] <= o[0]).mean()), P_near_ge=float((nn[:, 1] >= o[1]).mean()))


out['all'] = split_test(lambda t: True)
out['HT'] = split_test(lambda t: t['stratum'][0] == 'HT')
out['nonHT'] = split_test(lambda t: t['stratum'][0] != 'HT')
out['pub_le1950'] = split_test(lambda t: YEAR.get(t['doc'], 2000) <= 1950)
out['pub_gt1950'] = split_test(lambda t: YEAR.get(t['doc'], 2000) > 1950)

# (4) freeze near-only worn groups and their one-sign neighbours
fz = []
for t in T:
    if t['cls'] != 'WORN' or not t['nr']:
        continue
    w = t['s']; nbs = []
    for i in range(len(w)):
        for x, v in LX.nb.get((len(w), i, w[:i] + ('_',) + w[i + 1:]), ()):
            if v != w and LX.count(v, t['doc']) > 0:
                nbs.append(dict(pos=i, sign=x, group='-'.join(v), docs=LX.count(v, t['doc'])))
    fz.append(dict(doc=t['doc'], group='-'.join(w), neighbours=sorted(nbs, key=lambda z: -z['docs'])))
fz.sort(key=lambda z: (z['doc'], z['group']))
frozen = dict(loop='la84 cycle 1', date='2026-10-07',
              claim='worn sign-groups that have no exact match elsewhere but a one-sign neighbour were misread in '
                    'the edition more often than unworn groups',
              prediction='on autopsy or new photographs, >= 25 % of the listed groups change by one sign, and in '
                         '>= half of those changes the new reading is one of the listed neighbours; matched unworn '
                         'near-only groups change in <= 10 %',
              kill='< 10 % of the listed groups change, or the changes go to non-neighbour signs as often as to neighbours',
              groups=fz)
p = os.path.join(C.DATA, 'la84_frozen_c1.json')
s = json.dumps(frozen, sort_keys=True, indent=1, ensure_ascii=False)
open(p, 'w').write(s)
h = hashlib.sha256(s.encode()).hexdigest()
open(p.replace('.json', '.sha256'), 'w').write(h + '  la84_frozen_c1.json\n')
out['frozen'] = dict(n=len(fz), sha256=h)
json.dump(out, open(os.path.join(C.CK, 'c1b.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
