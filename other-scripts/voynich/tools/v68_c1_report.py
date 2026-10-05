"""summarise v68 cycle 1."""
import json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v68_lib as V

R = [json.loads(l) for l in open(os.path.join(V.CK, 'c1_results.jsonl'))]
T = json.load(open(os.path.join(V.CK, 'c1_targets.json')))
by = defaultdict(list)
for r in R:
    by[r['src']].append(r)
tg = ['ZL', 'plant_chant', 'plant_la']
print('targets', {t: {k: round(T[t]['mean'][k], 3) for k in ['mi_lf', 'mi_ll', 'mi_j', 'adjrep', 'linerep', 'ttr', 'mwl', 'wpl']} for t in tg})
print('%-14s %4s ' % ('source', 'n') + ' '.join('%-22s' % ('best/p05 ' + t) for t in tg))
for s, rs in by.items():
    row = []
    for t in tg:
        d = np.array([r['d'][t] for r in rs])
        row.append('%.3f / %.3f' % (d.min(), np.percentile(d, 5)))
    print('%-14s %4d ' % (s, len(rs)) + ' '.join('%-22s' % x for x in row))
print()
print('feature reach: fraction of codes within the Voynich ZL band (|z|<1 on the chunk sd) per feature, and best-5 mean')
Z = T['ZL']['mean']
import v53_lib as L53
TZ = L53.target_profile([l['words'] for l in L53.load_voynich('ZL3b')])
sd = TZ['sd']
print('ZL chunk sd', {k: round(sd[k], 4) for k in ['mi_lf', 'mi_ll', 'mi_j', 'adjrep', 'linerep', 'ttr']})
for s, rs in by.items():
    out = []
    for k in ['mi_lf', 'mi_ll', 'mi_j', 'adjrep', 'linerep', 'ttr']:
        v = np.array([r['f'][k] for r in rs])
        out.append('%s %.2f [%.3f-%.3f]' % (k, np.mean(np.abs(v - Z[k]) < sd[k]), v.min(), v.max()))
    joint = np.mean([all(abs(r['f'][k] - Z[k]) < sd[k] for k in ['mi_lf', 'mi_ll', 'mi_j', 'adjrep']) for r in rs])
    print('%-14s joint4 %.3f | ' % (s, joint) + ' | '.join(out))
print()
for s, rs in by.items():
    b = sorted(rs, key=lambda r: r['d']['ZL'])[:3]
    for r in b:
        print(s, round(r['d']['ZL'], 3), r['enc']['rep'], r['enc']['group'], r['enc']['fold'], r['enc']['trunc'],
              {k: round(r['f'][k], 3) for k in ['mi_lf', 'mi_ll', 'mi_j', 'adjrep', 'ttr', 'mwl', 'wpl']})
