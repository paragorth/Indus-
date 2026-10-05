"""v55: compare a cycle's real run with its nulls and planted controls. Usage: v55_report.py c1"""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v55_lib as V

pre = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(V.CK, f'{pre}_*.json'))):
    j = json.load(open(f)); c = j['cond']
    z = os.path.join(V.CK, f'{pre}_{c}.npz')
    if os.path.exists(z):
        r = np.load(z)
        full = r['full']
        years = np.arange(V.Y0, V.Y0 + full.shape[2])
        win = (years >= V.WINDOW[0]) & (years <= V.WINDOW[1])
        by = years[full.reshape(-1, full.shape[2]).argmax(1)]
        j['frac_best_in_window'] = float(((by >= V.WINDOW[0]) & (by <= V.WINDOW[1])).mean())
        rep_y = (((r['A'] >= 2) & (r['AB'] >= 2)) | ((r['B'] >= 2) & (r['BA'] >= 2))).sum((0, 1))
        j['rep_win_z'] = float((rep_y[win].mean() - rep_y[~win].mean()) / (rep_y[~win].std() + 1e-9))
        j['rep_peak_year'] = int(years[rep_y.argmax()])
        # strict replication: in-sample >=3 AND held-out >=3
        j['n_rep3'] = int((((r['A'] >= 3) & (r['AB'] >= 3)) | ((r['B'] >= 3) & (r['BA'] >= 3))).sum())
    rows[c] = j
keys = ['max_full', 'n_rep', 'n_rep3', 'hold', 'win_z', 'rep_win_z', 'frac_best_in_window', 'peak_year', 'rep_peak_year']
print('cond'.ljust(9), ' '.join(k[:10].rjust(10) for k in keys))
for c, j in rows.items():
    print(c.ljust(9), ' '.join(('%10.3f' % j.get(k, np.nan)) if isinstance(j.get(k), float) else str(j.get(k)).rjust(10) for k in keys))
nulls = [c for c in rows if c.startswith(('shuf', 'fake', 'gen'))]
if 'real' in rows and nulls:
    print('real percentile among', len(nulls), 'nulls:')
    for k in keys[:7]:
        v = rows['real'][k]; nv = [rows[c][k] for c in nulls]
        print(f'  {k}: real {v:.3f}  null mean {np.mean(nv):.3f} max {np.max(nv):.3f}  frac>=real {np.mean([x >= v for x in nv]):.2f}')
    print('real best:', rows['real'].get('best'))
