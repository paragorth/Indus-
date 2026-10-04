"""Print cycle-3 results (3a position nulls, 3b line totals, 3c delta channel)."""
import os, sys, json, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v15_lib as V

D = os.path.join(V.RES, 'cycle3')
p = os.path.join(D, '3a.json')
if os.path.exists(p):
    r = json.load(open(p))
    for k, v in r.items():
        if k == 'gap_pos':
            print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'profile'})
            print('   comma rate by gap index:', ' '.join(f"{j}:{x[2]:.3f}" for j, x in v['profile'].items()))
        else:
            print(k, v)

for tag in ('3b', '3c'):
    by = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(D, tag + '_*.json'))):
        r = json.load(open(f))
        real = [x for x in r['runs'] if x['kind'] == 'real'][0]
        nul = np.array([x['test'] for x in r['runs'] if x['kind'] != 'real'])
        z = (real['test'] - nul.mean()) / (nul.std(ddof=1) + 1e-9)
        by[(r['src'], r['unit'])].append((z, real['test'] - nul.max(), r['lang'], real.get('k'), real['test'],
                                          float(nul.mean()), real['sample'][:40]))
    print(f'\n== {tag}')
    for k, rs in sorted(by.items()):
        rs.sort(key=lambda t: -t[0])
        zs = [t[0] for t in rs]; ms = [t[1] for t in rs]
        print(f"{k[0]:12s} {k[1]:7s} cells {len(rs):2d} max z {max(zs):+.1f} max margin {max(ms):+.3f} n z>3 {sum(z > 3 for z in zs)}"
              f" | best: {rs[0][2]} k={rs[0][3]} J {rs[0][4]:.3f} null {rs[0][5]:.3f} '{rs[0][6]}'")
