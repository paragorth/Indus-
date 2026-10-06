"""v80 cycle 3 report: periodic recurrence of line types (transposed table), search-corrected."""
import json, os, glob
import numpy as np
from collections import Counter
import v80_lib as L
d = os.path.join(L.CK, 'c3')
out = {}
for fn in sorted(glob.glob(os.path.join(d, '*.json'))):
    nm = os.path.basename(fn)[:-5]
    if nm == 'report': continue
    R = json.load(open(fn))
    cand = []
    for hi, r in enumerate(R):
        for g in range(2, 14):
            cand.append((r['z0'][g], r['z1'][g], g, hi))
    cand.sort(reverse=True)
    top = cand[:10]
    best = max(top, key=lambda c: c[1])
    lagvote = Counter(c[2] for c in top)
    # replication rate: share of the top-50 discovery picks with held-out z > 2
    rep50 = np.mean([c[1] > 2 for c in cand[:50]])
    h = R[best[3]]
    out[nm] = dict(disc_max=top[0][0], hold_best=best[1], lag=best[2], lagvote=lagvote.most_common(3), rep50=float(rep50),
                   hyp=dict(rep=h['rep'], idf=h['idf'], pos=h['pos'], grp=h['grp']))
    print('%-10s disc max z %.1f | held-out best of top10 z %.1f at lag %d | top10 lags %s | top50 replicate(z>2) %.2f | %s' % (
        nm, top[0][0], best[1], best[2], lagvote.most_common(3), rep50, out[nm]['hyp']))
json.dump(out, open(os.path.join(d, 'report.json'), 'w'))
