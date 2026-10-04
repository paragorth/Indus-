#!/usr/bin/env python3
"""LA-45 report: stability across populations, truth recovery on controls, stable Linear A meanings.
Usage: la45_report.py TAG [corpus ...]"""
import sys, os, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C

TAG = sys.argv[1]
names = sys.argv[2:] or ['PLANT', 'UR3', 'LB', 'LA', 'LA_S1', 'LA_S2']
sys.argv = [sys.argv[0], '40', '6', TAG]
import la45_c1 as J

out = {}
for n in names:
    fs = sorted(glob.glob(os.path.join(C.CK, '%s_%s_*.json' % (TAG, n))))
    if not fs:
        continue
    runs = [json.load(open(f)) for f in fs]
    truth = None
    if n in ('PLANT', 'UR3', 'LB'):
        truth = J.corpus(n)[1]
    st = C.stability(runs, minc=3, truth=truth)
    st['nev'] = int(sum(r['nev'] for r in runs))
    st['npop'] = len(runs)
    modal = st.pop('modal')
    print('==', n, 'pops', len(runs), 'evals %.2fM' % (st['nev'] / 1e6))
    for k, v in st.items():
        if k not in ('nev', 'npop'):
            print('  ', k, v)
    if n.startswith('LA') or n == 'PLANT':
        cnt = collections.Counter()
        for r in runs:
            for w, c in r['tcnt'].items():
                cnt[w] += c
        stable = collections.defaultdict(list)
        for w, (nm, ag, npop) in modal.items():
            if npop >= len(runs) * 0.8 and ag >= 0.8 and cnt[w] / len(runs) >= 3:
                stable[nm].append((w, round(ag, 2), round(cnt[w] / len(runs), 1)))
        st['stable'] = {k: sorted(v, key=lambda x: -x[2]) for k, v in stable.items()}
        nst = sum(len(v) for v in stable.values())
        print('   stable (agree>=0.8, train count>=3):', nst, {k: len(v) for k, v in stable.items()})
        for k, v in st['stable'].items():
            print('     ', k, v[:14])
        for w in ['KU-RO', 'KI-RO', 'PO-TO-KU-RO', 'SA-RA₂', 'A-DU', 'KA-PA', 'L:GRA', 'L:VIN', 'L:OLE', 'L:CYP',
                  'L:VIR', 'L:OLIV', 'NI', '*304', '*308', 'TE', 'SI', 'KU-NI-SU', 'DA-ME', 'PA-I-TO', 'A-SA-SA-RA-ME']:
            if w in modal:
                print('       %-14s %s' % (w, modal[w]))
    out[n] = st
json.dump(out, open(os.path.join(C.CK, TAG + '_report.json'), 'w'), indent=1, ensure_ascii=False)
