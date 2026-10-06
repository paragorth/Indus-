#!/usr/bin/env python3
"""la66 cycle-2b summary: the stable-descriptor rule applied to full-design null corpora and to Linear B."""
import os, json, glob
import numpy as np
from la66_lib import CK

c2 = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in glob.glob(os.path.join(CK, 'c2_*.json')) if 'class' not in f}
nullpool = [v for k, v in c2.items() if k.startswith(('LAqdoc', 'LAqcom', 'LAwdoc', 'LAsent'))]
lbpool = [v for k, v in c2.items() if k.startswith('LBqdoc')]
b = {os.path.basename(f)[4:-5]: json.load(open(f)) for f in glob.glob(os.path.join(CK, 'c2b_*.json'))}
b['LB_tab_0'] = c2['LB_tab_0']


def apply(runs, pool, truth=None):
    feats = set().union(*[set(r['tab']) for r in runs])
    out = []
    for f in feats:
        rd = [r['tab'][f] for r in runs if f in r['tab']]
        dm = [v['med'] for v in rd if v['cls'] == 'D']
        nD = sum(1 for r in pool if r['tab'].get(f, {}).get('cls') == 'D')
        if len(dm) >= 4 and len(set(np.sign(dm))) == 1 and nD <= 1:
            out.append((f, round(float(np.median(dm)), 2), len(dm)))
        elif truth and f in truth:
            pass
    cand = [f for f in feats if sum(1 for r in runs if r['tab'].get(f, {}).get('cls') == 'D') >= 3]
    return out, len(cand)


for kind, pool in (('Nqcom', nullpool), ('Nwdoc', nullpool), ('LB', lbpool)):
    runs = [v for k, v in b.items() if k.startswith(kind + '_')]
    res, nc = apply(runs, pool)
    print(kind, 'runs', len(runs), 'stable', len(res), res[:30], 'D>=3/6:', nc)
    if kind == 'LB':
        tr = runs[0]['truth']
        for f, t in tr.items():
            ds = [(round(r['tab'][f]['med'], 2), r['tab'][f]['cls']) for r in runs if f in r['tab']]
            print('  truth', f, t, ds)
        print('  word gain over context', [round(r['scores']['best_gC'] - r['scores']['g_ctx0'], 3) for r in runs])
