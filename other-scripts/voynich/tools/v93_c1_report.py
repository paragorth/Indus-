"""v93 cycle 1 report: calibration, Voynich survivors vs kills, held-out test."""
import os, json, sys
from collections import defaultdict
import numpy as np
import v93_lib as L

def load(name):
    p = os.path.join(L.CK, 'c1_%s.jsonl' % name.replace('~', '_'))
    if not os.path.exists(p): return {}
    return {r['i']: r for r in map(json.loads, open(p)) if r['sel'] is not None}

C = {n: load(n) for n in ['PS_LINE', 'PS_INLINE', 'CE_PARA', 'PS_LINE~shuf', 'CE_PARA~shuf', 'PS_EN', 'PS_HE', 'CE_LA',
                          'ZL3b', 'IT2a', 'ZLshuf0', 'ZLshuf1', 'ITshuf0', 'SELFCIT', 'MK2', 'JUNC']}
fam = lambda r: r['h']['rule'][0]

print('== per corpus, per family: max |sel|, max |test| among hyps with |sel| top 5 ==')
for n, d in C.items():
    s = []
    for f in ['ALT', 'PARA', 'HALF', 'PAGEH', 'HLINE']:
        rs = [r for r in d.values() if fam(r) == f]
        if not rs: s.append('%s -' % f); continue
        top = sorted(rs, key=lambda r: -abs(r['sel']))[:5]
        s.append('%s sel %.3f test %.3f (n%d)' % (f, max(abs(r['sel']) for r in rs), np.median([abs(r['test']) for r in top]), len(rs)))
    print('%-14s' % n, ' | '.join(s))

def survivors(V, kills, pos=None, margin=0.0):
    out = []
    for i, r in V.items():
        k = [C[x][i] for x in kills if i in C[x]]
        if len(k) < len(kills): continue
        km = max(abs(x['sel']) for x in k)
        if abs(r['sel']) > km + margin and abs(r['sel']) > 0.03:
            kt = max(abs(x['test']) for x in k)
            out.append((abs(r['sel']) - km, i, r, km, kt))
    return sorted(out, key=lambda t: -t[0])

KZ = ['ZLshuf0', 'ZLshuf1', 'SELFCIT', 'MK2', 'JUNC']
for name, kills in [('ZL3b', KZ), ('IT2a', ['ITshuf0', 'SELFCIT', 'MK2', 'JUNC']),
                    ('PS_LINE', ['PS_LINE~shuf', 'PS_EN', 'PS_HE']), ('CE_PARA', ['CE_PARA~shuf', 'CE_LA']),
                    ('PS_INLINE', ['PS_EN', 'PS_HE']), ('PS_EN', ['PS_HE', 'CE_LA'])]:
    S = survivors(C[name], kills)
    rep = [(i, r['h']['rule'], r['h']['rep'], round(r['sel'], 3), round(r['test'], 3), round(kt, 3)) for _, i, r, km, kt in S[:20]]
    held = sum(1 for _, i, r, km, kt in S[:20] if np.sign(r['test']) == np.sign(r['sel']) and abs(r['test']) > kt)
    print('\n==', name, 'survivors', len(S), 'of', len(C[name]), '; top-20 held out (same sign, |test| > max kill |test|):', held)
    for x in rep[:12]: print('  ', x)
