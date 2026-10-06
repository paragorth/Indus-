#!/usr/bin/env python3
"""la66 cycle-2 summary: per-feature descriptor / identifier classification against the null pool.
Rule (frozen): STABLE DESCRIPTOR = D in >= 4 of the 6 real LA runs (3 split families x tab/sys), one sign in
all of them, and D in at most 1 of the null runs.  IDENTIFIER = >= 6 occurrences on >= 3 documents, never D in a
real run, |median effect| < 0.1 in every real run where present."""
import os, json, glob
import numpy as np
from collections import defaultdict
from la66_lib import CK, sha

J = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in glob.glob(os.path.join(CK, 'c2_*.json')) if 'class' not in f}
real = [k for k in J if k.startswith('LA_')]
nulls = [k for k in J if k.startswith(('LAqdoc', 'LAqcom', 'LAwdoc', 'LAsent'))]
print('real', sorted(real)); print('nulls', len(nulls))
for k in sorted(J):
    sc = J[k]['scores']; D = [f for f, v in J[k]['tab'].items() if v['cls'] == 'D']
    print(f"{k:18s} D {len(D):3d} gC-ctx0 {sc['best_gC'] - sc['g_ctx0']:+.3f} gCn {sc['best_gCn']:+.3f}")
feats = set()
for k in real:
    feats |= set(J[k]['tab'])
rows = []
for f in feats:
    rd = [J[k]['tab'][f] for k in real if f in J[k]['tab']]
    dmeds = [v['med'] for v in rd if v['cls'] == 'D']
    nD = sum(1 for k in nulls if J[k]['tab'].get(f, {}).get('cls') == 'D')
    sg = set(np.sign(dmeds))
    occ = max(v['occ'] for v in rd); nd = max(v['ndoc'] for v in rd)
    stable = len(dmeds) >= 4 and len(sg) == 1 and nD <= 1
    ident = (not dmeds) and occ >= 6 and nd >= 3 and all(abs(v['med']) < 0.1 for v in rd)
    rows.append(dict(f=f, nDreal=len(dmeds), nDnull=nD, med=float(np.median([v['med'] for v in rd])),
                     dmed=float(np.median(dmeds)) if dmeds else None, occ=occ, ndoc=nd,
                     cls='DESC' if stable else 'IDENT' if ident else '-'))
rows.sort(key=lambda r: (-r['nDreal'], r['nDnull']))
print('\nfeatures D in >= 2 real runs:')
for r in rows:
    if r['nDreal'] >= 2:
        print(f"  {r['f']:16s} Dreal {r['nDreal']}/6 Dnull {r['nDnull']}/{len(nulls)} med {r['med']:+.2f} occ {r['occ']} docs {r['ndoc']} {r['cls']}")
# null calibration: how many features would pass the same rule in a null (leave-one-null-out as 'real' is not
# possible with one run per null; instead count null features D in >= 2 null runs)
nn = defaultdict(int)
for k in nulls:
    for f, v in J[k]['tab'].items():
        if v['cls'] == 'D':
            nn[f] += 1
print('null features D in >= 2 null runs:', sorted([(f, c) for f, c in nn.items() if c >= 2], key=lambda x: -x[1]))
desc = [r for r in rows if r['cls'] == 'DESC']
ident = sorted([r for r in rows if r['cls'] == 'IDENT'], key=lambda r: -r['occ'])
print('\nSTABLE DESCRIPTORS', [(r['f'], round(r['dmed'], 2)) for r in desc])
print('IDENTIFIERS', len(ident), [(r['f'], r['occ']) for r in ident[:40]])
# LB through the same filter (one LB run, three LB nulls)
lb = J.get('LB_tab_0'); lbn = [J[k] for k in J if k.startswith('LBqdoc')]
if lb:
    for f, t in lb['truth'].items():
        v = lb['tab'].get(f)
        nD = sum(1 for x in lbn if x['tab'].get(f, {}).get('cls') == 'D')
        print('LB truth', f, t, v and (round(v['med'], 2), v['cls']), 'null D', nD, '/', len(lbn))
    print('LB D', sum(1 for v in lb['tab'].values() if v['cls'] == 'D'), 'LB null D', [sum(1 for v in x['tab'].values() if v['cls'] == 'D') for x in lbn])
tot = J.get('LAtot_tab_0')
if tot:
    for f in ('W:KU-RO', 'W:PO-TO-KU-RO', 'W:KI-RO'):
        print('LAtot', f, tot['tab'].get(f))
cand = [r['f'] for r in rows if r['cls'] != 'DESC' and r['nDreal'] >= 3 and r['nDnull'] <= 1]
print('CANDIDATES (D in 3/6 real, <= 1 null)', cand)
json.dump(dict(rule=__doc__, descriptors=[r['f'] for r in desc], candidates=cand, table=rows,
               identifiers=[r['f'] for r in ident]), open(os.path.join(CK, 'c2_class.json'), 'w'), indent=0)
