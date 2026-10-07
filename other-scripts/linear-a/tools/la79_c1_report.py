import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la79_common import CK
from la79_c1 import success
R = json.load(open(os.path.join(CK, 'c1_real.json')))
names = list(R['full'].keys())
out = {}
for nm in names:
    row = {'full': success(nm, R['full'][nm])}
    for kind in ('time', 'site', 'rand'):
        s = [success(nm, r['r'].get(nm)) for r in R['res'] if r['kind'] == kind]
        s2 = [x for x in s if x is not None]
        row[kind] = (sum(s2), len(s2))
    # LOSO = site splits where test is a single group
    lo = [r for r in R['res'] if r['kind'] == 'site' and r['split'].count('+') == 4]
    s = [success(nm, r['r'].get(nm)) for r in lo]
    row['loso'] = (sum(x for x in s if x), len([x for x in s if x is not None]))
    row['time_detail'] = [(r['split'], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in (r['r'].get(nm) or {}).items() if k != 'top'}) for r in R['res'] if r['kind'] == 'time']
    # median test z (or pct) on site splits vs rand
    key = 'z' if R['full'][nm].get('z') is not None else 'pct'
    for kind in ('site', 'rand'):
        v = [r['r'][nm][key] for r in R['res'] if r['kind'] == kind and nm in r['r'] and r['r'][nm].get(key) is not None]
        row['med_' + kind] = round(float(np.median(v)), 2) if v else None
    out[nm] = row
    print(nm, row['full'], 'time', row['time'], 'site', row['site'], 'loso', row['loso'], 'rand', row['rand'], 'med site/rand', row['med_site'], row['med_rand'])
    print('    ', row['time_detail'])
json.dump(out, open(os.path.join(CK, 'c1_summary.json'), 'w'))
