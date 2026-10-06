"""v80 report for a cycle directory: search-corrected best table hypotheses per corpus."""
import sys, json, os, glob
import numpy as np
import v80_lib as L

d = os.path.join(L.CK, sys.argv[1] if len(sys.argv) > 1 else 'c1')
rows = []
for fn in sorted(f for f in glob.glob(os.path.join(d, '*.json')) if not f.endswith('report.json')):
    nm = os.path.basename(fn)[:-5]
    R = json.load(open(fn))
    if not isinstance(R, list): continue
    R.sort(key=lambda r: -r['disc'])
    top = R[:10]
    b = max(top, key=lambda r: r['total'])
    Ri = sorted(R, key=lambda r: -r.get('total_int', 0))
    # column-count read-off: best holdout total by C for align L/R (max over unit/rep), elbow at 90% of the max
    prof = {}
    for r in R:
        if r['align'] in ('L', 'R', 'X', 'XR'):
            k = (r['align'], r['C']); prof[k] = max(prof.get(k, -9), r['total'])
    best_al = max(('L', 'R', 'X', 'XR'), key=lambda a: max([v for (al, c), v in prof.items() if al == a] or [-9]))
    pc = sorted((c, v) for (al, c), v in prof.items() if al == best_al)
    mx = max(v for c, v in pc)
    elbow = min(c for c, v in pc if v >= 0.9 * mx) if mx > 0 else 0
    rows.append((nm, b['total'], np.median([r['total'] for r in top]), b['g_glob'], b['g_loc'], b['g_vert'],
                 b.get('total_int', 0), max(r.get('total_int', 0) for r in top), L.hyp_key(b), best_al, elbow))
    print('%-12s best10 %.4f med10 %.4f | glob %.4f loc %.4f vert %.4f | int %.4f (top10 max %.4f) | %s | colcount %s:%d' % rows[-1])
json.dump(rows, open(os.path.join(d, 'report.json'), 'w'))
