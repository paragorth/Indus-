"""X-4 cycle 2b summary: generator-relative gap ratio G = median-over-rules g(real) / median g(own TRI generator),
per seed (0-5); mean and sd over seeds."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X
R = {}
for f in glob.glob(os.path.join(X.CK, 'c2', '*.json')):
    r = json.load(open(f)); R[(r['name'], r['budget'], r['seed'], r['cond'])] = r
med = lambda r: float(np.median([v['g_tri'] for v in r['rules'].values() if v['g_tri'] == v['g_tri']]))
out = {}
for budget in (3000, 8000):
    print('== budget', budget)
    for name in X.LIST + ['VOY'] + X.PROSE:
        G, g = [], []
        for s in range(6):
            a, b = R.get((name, budget, s, 'real')), R.get((name, budget, s, 'TRI'))
            if a and b: G.append(med(a) / med(b)); g.append(med(a))
        if G:
            out[f'{name}_{budget}'] = (float(np.mean(G)), float(np.std(G)), len(G), float(np.mean(g)))
            print(f'{name:6s} g {np.mean(g):.3f}  G {np.mean(G):.3f} +- {np.std(G):.3f} (n {len(G)})  min {min(G):.3f} max {max(G):.3f}')
X.save('c2b_sum.json', out)
