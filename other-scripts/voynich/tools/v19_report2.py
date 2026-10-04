"""v19 cycle 2 report: matched-null pooled tests and cross-stretch (held-out) alphabet consistency."""
import json, os, glob
from v19_lib import RES

D = os.path.join(RES, 'c2')
print('== matched / block nulls (pooled = one order for all pages) ==')
print('%-28s %8s %8s %6s  %-10s %s' % ('cell', 'tau', 'null', 'z', 'pages z>4', 'order'))
for f in sorted(glob.glob(os.path.join(D, 'm_*.json'))):
    d = json.load(open(f)); p = d['pooled']
    rs = d.get('rows', [])
    s = '%d/%d' % (sum(r['z'] > 4 for r in rs), len(rs)) if rs else '-'
    print('%-28s %8.4f %8.4f %6.1f  %-10s %s' % (os.path.basename(f)[2:-5], p['tau'], p['nmean'], p['z'], s, p['order_s'][:70]))
print('\n== cross-stretch prediction: order learned on stretch A applied (fixed) to stretch B ==')
print('%-26s %4s %9s %9s %9s %9s %8s' % ('cell', 'n', 'mean tau', '+-se', 'shufB', 'nullord', 'LOO z'))
for f in sorted(glob.glob(os.path.join(D, 'x_*.json'))):
    d = json.load(open(f))
    if not d or not d.get('real'):
        continue
    r, n = d['real'], d['null_orders']
    print('%-26s %4d %9.4f %9.4f %9.4f %9.4f %8.1f' % (os.path.basename(f)[2:-5], d['n'], r['mean'], r['se'], r['shmean'], n['mean'], r['looz']))
