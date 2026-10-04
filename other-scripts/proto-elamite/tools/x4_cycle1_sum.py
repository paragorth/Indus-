"""X-4 cycle 1 summary table (means over seeds 0, 1)."""
import os, sys, json, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X

R = defaultdict(list)
for f in glob.glob(os.path.join(X.CK, 'c1', '*.json')):
    r = json.load(open(f)); R[(r['name'], r['budget'], r['cond'])].append(r)
CONDS = ['real', 'rand0', 'rand1', 'body', 'WBG', 'TRI', 'SLOT', 'CPV']
out = []
for budget in (3000, 8000):
    out.append(f'\n== budget {budget} words: word-probe survivors total/freq (per 1,000 probes), sign-probe survivors (/300), '
               f'ep1 excess bits, KN-3 direction z')
    out.append(f'{"corpus":7s} ' + ' '.join(f'{c:>14s}' for c in CONDS))
    for name in X.LIST + ['VOY'] + X.PROSE:
        if (name, budget, 'real') not in R: continue
        cells_w, cells_s = [], []
        for c in CONDS:
            rs = R.get((name, budget, c), [])
            if not rs: cells_w.append(f'{"-":>14s}'); cells_s.append(f'{"-":>14s}'); continue
            wt = np.mean([sum(r['wsurv'].values()) for r in rs]); wf = np.mean([r['wsurv'].get('freq', 0) for r in rs])
            ss = np.mean([r['ssurv'].get('sign', 0) for r in rs]); ep = np.mean([r['ep1_x'] for r in rs])
            kz = np.mean([r['kn_z'] for r in rs])
            cells_w.append(f'{wt:6.1f}/{wf:5.1f}  '); cells_s.append(f'{ss:4.1f} {ep:4.2f} {kz:+4.1f}')
        out.append(f'{name:7s} W ' + ' '.join(f'{x:>14s}' for x in cells_w))
        out.append(f'{"":7s} S ' + ' '.join(f'{x:>14s}' for x in cells_s))
txt = '\n'.join(out)
open(os.path.join(X.CK, 'c1_sum.txt'), 'w').write(txt)
print(txt)
