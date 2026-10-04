"""X-4 cycle 2 summary: median gap ratio (over 31 segmentation rules) per corpus x budget x condition,
mean over seeds; share of rules with gap ratio < 0.8; pair-shuffle ratio (conn(pair) / conn(own trigram))."""
import os, sys, json, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X

R = defaultdict(list)
for f in glob.glob(os.path.join(X.CK, 'c2', '*.json')):
    r = json.load(open(f)); R[(r['name'], r['budget'], r['seed'], r['cond'])] = r
out = []
summ = {}
for budget in (3000, 8000):
    out.append(f'\n== budget {budget}: median g_tri [IQR over rules] (share of rules < 0.8) | g_slot | conditions')
    for name in X.LIST + ['VOY'] + X.PROSE:
        cells = []
        for cond in ['real', 'pair', 'TRI', 'SLOT', 'CPV', 'WBG']:
            gs, ss, fr = [], [], []
            for seed in (0, 1):
                r = R.get((name, budget, seed, cond))
                if not r: continue
                if cond == 'pair':
                    rr = R.get((name, budget, seed, 'real'))
                    if not rr: continue
                    g = [v['conn'] / rr['rules'][k]['tri'] for k, v in r['rules'].items() if rr['rules'][k]['tri']]
                else:
                    g = [v['g_tri'] for v in r['rules'].values() if v['g_tri'] == v['g_tri']]
                    ss += [v['g_slot'] for v in r['rules'].values() if v['g_slot'] == v['g_slot']]
                gs.append(np.median(g)); fr.append(np.mean(np.array(g) < 0.8))
                if cond == 'real': summ.setdefault(f'{name}_{budget}', []).extend(g)
            if gs:
                cells.append(f'{cond}:{np.mean(gs):.2f}({np.mean(fr):.0%})' + (f'/s{np.median(ss):.2f}' if ss else ''))
        if cells: out.append(f'{name:7s} ' + '  '.join(cells))
txt = '\n'.join(out)
open(os.path.join(X.CK, 'c2_sum.txt'), 'w').write(txt)
print(txt)
