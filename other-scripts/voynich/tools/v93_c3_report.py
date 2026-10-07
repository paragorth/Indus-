import os, json, sys
import numpy as np
import v93_lib as L
NAMES = ['PE_DIAL', 'PE_SHUF', 'PE_MONO', 'CE_LA', 'PS_EN', 'ZL3b', 'IT2a', 'ZLshuf0', 'ZLshuf1', 'ITshuf0', 'SELFCIT', 'MK2', 'JUNC']
C = {}
for n in NAMES:
    p = os.path.join(L.CK, 'c3_%s.jsonl' % n)
    if os.path.exists(p): C[n] = {r['i']: r for r in map(json.loads, open(p)) if r['sel'] is not None and r['test'] is not None}
print('%-9s %5s %7s %7s %7s %7s' % ('corpus', 'n', 'maxSel', 'q99', 'top20test', 'med'))
for n, d in C.items():
    s = np.array([r['sel'] for r in d.values()])
    top = sorted(d.values(), key=lambda r: -r['sel'])[:20]
    print('%-9s %5d %7.2f %7.2f %7.2f %7.2f' % (n, len(d), s.max(), np.quantile(s, .99), np.median([r['test'] for r in top]), np.median(s)))
KZ = ['ZLshuf0', 'ZLshuf1', 'ITshuf0', 'SELFCIT', 'MK2', 'JUNC']
def surv(name, kills, thr=3.0):
    out = []
    for i, r in C[name].items():
        if not all(i in C[k] for k in kills): continue
        km = max(C[k][i]['sel'] for k in kills)
        if r['sel'] > thr and r['sel'] > km + 1:
            kt = max(C[k][i]['test'] for k in kills)
            out.append((r['sel'] - km, i, r, kt))
    return sorted(out, key=lambda t: -t[0])
for name, kills in [('PE_DIAL', ['PE_SHUF', 'PE_MONO', 'CE_LA', 'PS_EN']), ('ZL3b', KZ), ('IT2a', KZ)]:
    if name not in C: continue
    S = surv(name, [k for k in kills if k in C])
    held = [t for t in S[:20] if t[2]['test'] > max(2.0, t[3])]
    print('\n==', name, 'survivors', len(S), 'top20 held (test z > 2 and > kills):', len(held))
    for _, i, r, kt in S[:10]: print('  ', i, r['h'], 'sel %.2f test %.2f killtest %.2f' % (r['sel'], r['test'], kt))
if 'ZL3b' in C and 'IT2a' in C:
    both = [i for i in C['ZL3b'] if i in C['IT2a'] and all(i in C[k] for k in KZ)
            and min(C['ZL3b'][i]['sel'], C['ZL3b'][i]['test'], C['IT2a'][i]['sel'], C['IT2a'][i]['test']) > max(2.0, max(max(C[k][i]['sel'], C[k][i]['test']) for k in KZ))]
    print('\nZL and IT2a both splits > 2 and > all kills:', len(both), both[:10])
