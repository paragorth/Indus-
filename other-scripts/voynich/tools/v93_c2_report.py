import os, json, glob
import numpy as np
import v93_lib as L
R = {}
for f in sorted(glob.glob(os.path.join(L.CK, 'c2_*.json'))):
    r = json.load(open(f))
    if r.get('skip'): continue
    R[(r['name'], r['lv'], r['rep'])] = r
print('%-12s %-5s %-6s %5s %5s %7s %6s %6s' % ('corpus', 'lv', 'rep', 'n_tr', 'n_ok', 'heldZ', 'rate', 'zSE'))
for k, r in R.items():
    se = r['pair_held_sd'] / np.sqrt(len(r['top_held']))
    zz = (r['mean_held'] - r['pair_held_mean']) / se
    r['zz'] = zz
    print('%-12s %-5s %-6s %5d %5d %7.3f %6.2f %6.1f' % (k + (r['n_tr'], r['n_ok'], r['mean_held'], r['rep_rate'], zz)))
for k, r in R.items():
    if 'decoded' in r and k[1] == 'page':
        print(k, [(round(h, 1), d) for h, d in zip(r['top_held'], r['decoded'])][:20])
for k, r in R.items():
    if k[0] in ('ZL3b', 'IT2a'):
        print(k, [(round(h, 1), s) for h, s in zip(r['top_held'], r['sets'])][:15])
