import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L
pre = sys.argv[1]
for n in sys.argv[2:]:
    f = os.path.join(L.CK, '%s_%s.json' % (pre, n))
    if not os.path.exists(f): continue
    r = json.load(open(f))
    zt = [x['test']['z'] for x in r['top']]
    calls = [x for x in r['top'] if x['test']['z'] >= 3 and x['test']['lift'] >= 1.3]
    print('%-8s wheels %5d  train cz top %.1f  test cz: max %.1f median %.1f, >=3: %d/20; best: %s' % (
        n, r['n_wheels'], r['top'][0]['cz_train'], max(zt), float(np.median(zt)), len(calls),
        ' | '.join('%s tr%.1f te%.1f L%.2f' % ('+'.join(x['fields']), x['cz_train'], x['test']['z'], x['test']['lift']) for x in sorted(r['top'], key=lambda x: -x['test']['z'])[:3])))
