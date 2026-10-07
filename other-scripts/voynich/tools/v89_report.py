"""summarise a v89 jsonl: per target, rank of partner (if any) by held-out B among sources, decoy z, block-null zB."""
import sys, json, numpy as np
from collections import defaultdict
R = defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        d = json.loads(l); R[d['target']].append(d)
for t, rows in R.items():
    rows.sort(key=lambda d: -d['B'])
    Bs = np.array([d['B'] for d in rows])
    p = rows[0].get('partner')
    print('== %s (%d sources)' % (t, len(rows)))
    for i, d in enumerate(rows[:5]):
        others = np.delete(Bs, i)
        print('  %d %-20s A %.3f B %.3f zB(block) %5.1f  zDecoy %5.1f  map off %.1f span %.1f %s drop %d' % (
            i + 1, d['src'], d['A'], d['B'], d['zB'], (d['B'] - others.mean()) / (others.std() + 1e-9),
            d['a']['off'], d['a']['span'], d['a']['mode'], d['a']['drop']))
    if p:
        r = [d['src'] for d in rows].index(p) + 1 if p in [d['src'] for d in rows] else None
        print('  partner %s rank %s' % (p, r))
