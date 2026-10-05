#!/usr/bin/env python3
"""LA-46 cycle-2 report: held-out prediction of residues found on half A, scored on half B."""
import sys, json, collections, statistics as st
from la46_common import *

MODE = sys.argv[1] if len(sys.argv) > 1 else 'mean'
for name in ['LA', 'W1', 'W2', 'SH1', 'PL1', 'LB1', 'LB2', 'UR1']:
    fn = os.path.join(CK, 'c2_%s_%s.json' % (MODE, name))
    if not os.path.exists(fn):
        continue
    o = json.load(open(fn))
    S = o['splits']
    a = [s['auc'] for s in S]; r = [s['auc_rand'] for s in S]
    d = [x - y for x, y in zip(a, r)]
    fam = collections.defaultdict(list)
    for s in S:
        for f, v in s['fam_auc'].items():
            fam[f].append(v)
    sd = st.stdev(d) if len(d) > 1 else float('nan')
    t = st.mean(d) / (sd / len(d) ** .5) if sd and sd == sd and sd > 0 else float('nan')
    print('%-4s n=%d nres %.1f  AUC %.3f (sd %.3f)  random-set %.3f  diff %+.3f t %.1f  planted %s' % (
        name, len(S), st.mean(s['nres'] for s in S), st.mean(a), st.stdev(a) if len(a) > 1 else 0, st.mean(r),
        st.mean(d), t, [s.get('planted_in_W') for s in S] if 'planted_in_W' in S[0] else ''))
    print('      family AUC:', ' '.join('%s %.3f(%d)' % (f, st.mean(v), len(v)) for f, v in sorted(fam.items())))
    agg = collections.defaultdict(lambda: [0, 0.0, 0])
    for s in S:
        for k, cr, cf in s['fire']:
            k = tuple(k); agg[k][0] += cr; agg[k][1] += cf; agg[k][2] += 1
    top = sorted(agg.items(), key=lambda x: -(x[1][0] - x[1][1]))[:12]
    print('      held-out excess (real B - forged B, summed over splits):',
          '; '.join('%s %s/%s %d/%.1f' % (k[0], k[1], k[2], v[0], v[1]) for k, v in top))
