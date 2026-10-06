#!/usr/bin/env python3
"""summarise la66 cycle-1 checkpoints"""
import sys, os, json, glob
import numpy as np
from la66_lib import CK

pre = sys.argv[1] if len(sys.argv) > 1 else 'c1'
for fn in sorted(glob.glob(os.path.join(CK, pre + '_*.json'))):
    J = json.load(open(fn)); T = J['tab']; sc = J['scores']
    D = {k: v for k, v in T.items() if v['cls'] == 'D'}
    I = [k for k, v in T.items() if v['cls'] == 'I']
    line = (f"{os.path.basename(fn)[:-5]:28s} n {J['n']:5d} D {len(D):3d} I {len(I):3d} gC {sc['best_gC']:+.3f} "
            f"gCn {sc['best_gCn']:+.3f} ctx0 {sc.get('g_ctx0', 0):+.3f} add {sc['add_share']:.2f}")
    tr = J.get('truth', {})
    if 'map' in tr:
        mp = tr['map']
        line += ' | ' + ' '.join(f"{mp[k][2:]} {T[k]['med']:+.2f}{T[k]['cls']}" for k in mp if k in T)
    elif tr:
        hits = [(k, v, T.get(k)) for k, v in tr.items()]
        ok = sum(1 for k, v, t in hits if t and t['cls'] == 'D' and np.sign(t['med']) == np.sign(v))
        sg = sum(1 for k, v, t in hits if t and np.sign(t['med']) == np.sign(v))
        pres = sum(1 for k, v, t in hits if t)
        fp = sum(1 for k in D if k not in tr)
        line += f" | truth D+sign {ok}/{pres} sign {sg}/{pres} otherD {fp}"
        if 'plant' not in fn:
            line += ' ' + ' '.join(f"{k[2:]}:{t['med']:+.2f}{t['cls']}" for k, v, t in hits if t)
    line += ' || ' + ' '.join(f"{k}{v['med']:+.2f}" for k, v in sorted(D.items(), key=lambda x: -abs(x[1]['med']))[:12])
    print(line)
