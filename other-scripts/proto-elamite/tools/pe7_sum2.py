"""summarise pe7 cycle 2 / 3A checkpoints: per kind x label, share of replicates with
PS p < 0.05, median PS ratio (null/obs), mean z of distance and co-membership tests."""
import sys, os, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe7_common import CKPT
pref = sys.argv[1] if len(sys.argv) > 1 else 'c2'
R = {}
for p in sorted(glob.glob(os.path.join(CKPT, pref + '_*.json'))):
    k = os.path.basename(p)[len(pref) + 1:-5]
    kind, rep = k.rsplit('_', 1)
    d = json.load(open(p))
    for lab, v in d.items():
        if isinstance(v, dict) and 'PS' in v:
            R.setdefault((kind, lab), []).append(v)
        if lab == 'mantel_sb' and v:
            R.setdefault((kind, 'mantel'), []).append(v)
for (kind, lab), L in sorted(R.items()):
    if lab == 'mantel':
        print('%-10s %-9s n=%d rho %.3f null %.3f  p<.05 %d/%d' % (kind, lab, len(L), np.mean([x['rho'] for x in L]), np.mean([x['null'] for x in L]), sum(x['p'] < .05 for x in L), len(L)))
        continue
    ps = [x['PS'] for x in L]
    dz = [x['dist']['z'] for x in L if x.get('dist')]
    cz = [x['co']['z'] for x in L if x.get('co')]
    dd = [x['dist']['diff'] for x in L if x.get('dist')]
    pr = [x['dist']['pairs'] for x in L if x.get('dist')]
    print('%-10s %-9s n=%d PS p<.05 %d/%d ratio %.3f | dist z %.2f (p<.05 %d) diff %.2f bits pairs %.0f | co z %s' % (
        kind, lab, len(L), sum(x['p'] < .05 for x in ps), len(ps), np.median([x['ratio'] or 0 for x in ps]),
        np.mean(dz) if dz else np.nan, sum(x['dist']['p'] < .05 for x in L if x.get('dist')), np.mean(dd) if dd else np.nan,
        np.mean(pr) if pr else 0, ('%.2f' % np.mean(cz)) if cz else '-'))
