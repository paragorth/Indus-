"""v56 cycle-1 report: per corpus, held-out (unseen word types) robust z of the rules; family-wise null threshold
from the null corpora; recovery of the planted / Roman rules; survivors."""
import sys, os, json, glob
import numpy as np
CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'v56_ckpt')
TAG = sys.argv[1] if len(sys.argv) > 1 else 'c1'
NULLS = ['ZL_relab', 'ZL_markov', 'ZL_selfcit', 'CULP_names']
R = {}
for f in sorted(glob.glob(os.path.join(CK, '%s_*.json' % TAG))):
    o = json.load(open(f)); R[o['name']] = o
thr = max([max(r['z_te'] for r in R[n]['rows']) for n in NULLS if n in R] or [np.nan])
print('family-wise null threshold (max held-out robust z over all rules of all null corpora): %.2f' % thr)
for n, o in R.items():
    rows = o['rows']; zte = np.array([r['z_te'] for r in rows]); ztr = np.array([r['z_tr'] for r in rows])
    top = sorted(rows, key=lambda r: -r['z_tr'])[:20]
    cov = o.get('coverage', {})
    print('\n== %s: %d rules from %d configs visited of %d, %.2e rule evaluations' % (n, len(rows), cov.get('visited', 0),
          cov.get('total', 0), o['n_eval']))
    print('  held-out robust z: mean %.2f sd %.2f max %.2f; >3: %d; >thr: %d; top-20-by-train held-out mean %.2f' % (
        zte.mean(), zte.std(), zte.max(), (zte > 3).sum(), (zte > thr).sum(), np.mean([r['z_te'] for r in top])))
    best = sorted(rows, key=lambda r: -r['z_te'])[:4]
    for r in best:
        print('   te %.2f (plain %.2f) tr %.2f %s %s %s %s' % (r['z_te'], r.get('z_te_plain', 0), r['z_tr'], r['sel'][:28],
              r['feat'], r['mode'], dict(sorted(r['d'].items(), key=lambda x: -x[1])[:8])))
    if 'digits' in o:
        dig = o['digits']; hits = []
        for r in rows:
            if r['feat'][0] == 'pos' and r['feat'][2] == 7:
                ok = sum(r['d'].get(g, 0) == k for k, g in enumerate(dig)); hits.append((ok, r['z_te'], r['sel'], r['mode']))
        hits.sort(reverse=True)
        print('  planted rule: base-7 rules tried %d; best digit match %s' % (len(hits), hits[:3]))
    if 'roman' in o:
        rom = o['roman']; val = dict(i=1, v=5, x=10, l=50, c=100); hits = []
        for r in rows:
            if r['feat'][0] == 'count':
                ok = sum(r['d'].get(rom[c], 0) == val[c] for c in val); hits.append((ok, r['z_te'], r['sel'][:20], r['mode']))
        hits.sort(reverse=True)
        print('  Roman rule: best value match (of 5) %s' % hits[:3])
