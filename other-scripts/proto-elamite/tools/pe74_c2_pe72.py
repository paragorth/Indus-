#!/usr/bin/env python3
"""pe74 cycle 2, item F: the frozen pe59 (v1) and pe72 (v2) readings scored on the same held-out half, with
the corpus filtered by restoration status.  The readings are NOT re-derived: v2 is loaded from
data/pe72_reading_frozen.json, v1 from pe59.  Twins as in pe72_c1 (ALL, EXT shuffled roles; V1ALL).
usage: python3 pe74_c2_pe72.py MODE [n_twins]     MODE in all | rd | r
Output: data/pe74_ckpt/<MODE>/c2_pe72.json
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe74_harness as H
MODE = sys.argv[1]
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ROOT = H.activate(MODE)
import pe59_lib as P
H.redirect_ck(P, 'pe59', copy=['pc_tabs.json', 'pc_admin_ids.json'])
import pe70_common as C70
H.redirect_ck(C70, 'pe70', copy=['pc.pkl', 'ur3.pkl'])
import pe72_lib as L
H.redirect_ck(L, 'pe72', copy=['pc_tabs72.json'])
import numpy as np

T = L.pe_tablets(); maps = P.pe_maps()
tr, ho = P.split(T)
fz = json.load(open(os.path.join(L.DATA, 'pe72_reading_frozen.json')))
r1, r2 = L.roles_v1(), L.roles_v2(fz['reading'])
out = {'mode': MODE, 'n_train': len(tr), 'n_heldout': len(ho), 'reading_sha': fz['sha256']}
C1 = L.Ctx('PE', r1, maps, tr); C2 = L.Ctx('PE', r2, maps, tr)
out['v1'] = L.summarise([L.decode(t, C1) for t in ho])
out['v2'] = L.summarise([L.decode(t, C2) for t in ho])
print(MODE, 'v1', out['v1']['dense'], 'v2', out['v2']['dense'], 'C3', out['v2']['C3'], '/', out['v2']['C3_n'], flush=True)
tw = {}
for kind in ('ALL', 'EXT', 'V1ALL'):
    for k in range(NT if kind != 'V1ALL' else NT // 2):
        rng = random.Random(P.seed('pe72PE%s%d' % (kind, k)))
        if kind == 'ALL':
            R = L.twin_roles(r2, T, rng, 'all')
            C = L.Ctx('PE', R, maps, tr, sealed_override=L.shuffle_sealed(T, rng))
        elif kind == 'EXT':
            R = L.twin_roles(r2, T, rng, 'ext', v1roles=r1)
            C = L.Ctx('PE', R, maps, tr, sealed_override=L.shuffle_sealed(T, rng))
        else:
            R = P.shuffle_roles(r1, T, rng); R['version'] = 'v1'
            C = L.Ctx('PE', R, maps, tr)
        tw.setdefault(kind, []).append(L.summarise([L.decode(t, C) for t in ho]))
out['twins'] = tw
for k, v in tw.items():
    d = [x['dense'] for x in v]
    out['twin_' + k] = {'dense_mean': float(np.mean(d)), 'dense_max': int(max(d)),
                        'p_v2_dense': (1 + sum(x >= out['v2']['dense'] for x in d)) / (1 + len(d)),
                        'C3_mean': float(np.mean([x['C3'] for x in v]))}
    print(MODE, k, out['twin_' + k], flush=True)
json.dump(out, open(os.path.join(ROOT, 'c2_pe72.json'), 'w'), indent=1, default=str)
