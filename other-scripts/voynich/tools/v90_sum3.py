import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L
for n in sys.argv[1:]:
    f = os.path.join(L.CK, 'c3_%s.json' % n)
    if not os.path.exists(f): continue
    r = json.load(open(f))
    for lev, rows in r['levels'].items():
        whole = [x for x in rows if len(x['cols']) == 7][0]
        prop = [x for x in rows if len(x['cols']) < 7 and len(x['cols']) >= 2 and x['n'] >= 50]
        prop.sort(key=lambda x: -x['z'])
        b = prop[0]
        comp = [x for x in rows if sorted(x['cols']) == sorted(set(range(7)) - set(b['cols']))]
        cz = comp[0]['z'] if comp else float('nan')
        print('%-8s %-5s whole z %5.1f (lift %.2f) | best proper %-22s z %5.1f lift %.2f n %4d, complement z %5.1f | #proper z>=5.3: %d/%d' % (
            n, lev, whole['z'], whole['lift'], '+'.join(L.FIELDS[c] for c in b['cols']), b['z'], b['lift'], b['n'], cz,
            sum(x['z'] >= 5.3 for x in prop), len(prop)))
