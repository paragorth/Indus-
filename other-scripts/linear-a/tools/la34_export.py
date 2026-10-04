#!/usr/bin/env python3
"""la34 export (repo, small): per-occurrence scalar shape features (no images, no grids) and per-unit blind
writer clusters (arm M, scale-free, average linkage per site, k = ceil(units/4); grade C labels).
Writes data/la34/la34_occ_features.json and data/la34/la34_blind_hands.json"""
import os, json, math, collections, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from la34_common import load, CK, D
from la34_img import occ_table, unit_M, SCAL

occ, meta, cid = load()
feat = json.load(open(os.path.join(CK, 'feat.json')))
os.makedirs(os.path.join(D, 'la34'), exist_ok=True)
rows = []
for o in occ:
    f = feat.get(o['file'])
    if not f: continue
    rows.append({'doc': o['doc'], 'n': o['n'], 'unit': o['unit'], 'role': o['role'], 'code': o['code'], 'read': o['read'],
                 **{k: round(f[k], 4) for k in SCAL}})
json.dump({'source': 'SigLA document drawings (Salgarella & Castellan, CC BY-NC-SA), features computed by la34', 'features': SCAL,
           'rows': rows}, open(os.path.join(D, 'la34', 'la34_occ_features.json'), 'w'))
o_, names, Z, P, codes = occ_table(feat, scale_free=True, occ=occ)
nocc = collections.Counter(o['unit'] for o in o_)
U = sorted(u for u in nocc if nocc[u] >= 3)
DM = unit_M(o_, Z, P, codes, U, 'unit')
out = {}
sites = collections.defaultdict(list)
for i, u in enumerate(U): sites[meta[u]['site']].append(i)
for s, ix in sites.items():
    if len(ix) < 4:
        for i in ix: out[U[i]] = {'site': s, 'blind_hand': f'{s}-1', 'published': meta[U[i]]['scribe']}
        continue
    sub = DM[np.ix_(ix, ix)].copy(); fm = np.isfinite(sub); sub[~fm] = np.nanmax(sub[fm]); np.fill_diagonal(sub, 0)
    lab = fcluster(linkage(squareform((sub + sub.T) / 2, checks=False), 'average'), math.ceil(len(ix) / 4), 'maxclust')
    for i, l in zip(ix, lab): out[U[i]] = {'site': s, 'blind_hand': f'{s}-{l}', 'published': meta[U[i]]['scribe'], 'n_drawings': nocc[U[i]]}
json.dump({'note': 'grade C blind writer clusters; published = lineara.xyz scribe field, not used in clustering', 'units': out},
          open(os.path.join(D, 'la34', 'la34_blind_hands.json'), 'w'), indent=0, ensure_ascii=False)
print(len(rows), len(out))
