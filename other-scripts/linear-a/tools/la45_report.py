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
import importlib
J = importlib.import_module('la45_c3' if TAG.startswith('c3') else 'la45_c2' if TAG.startswith('c2') else 'la45_c1')

out = {}
for n in names:
    import re as _re
    fs = sorted(f for f in glob.glob(os.path.join(C.CK, '%s_%s_*.json' % (TAG, n)))
                if _re.fullmatch(r'%s_%s_\d+\.json' % (TAG, n), os.path.basename(f)))
    if not fs:
        continue
    runs = [json.load(open(f)) for f in fs]
    truth = None
    if n.startswith(('PLANT', 'UR3', 'LB')):
        truth = J.corpus(n)[1]
    st = C.stability(runs, minc=3, truth=truth)
    st['nev'] = int(sum(r['nev'] for r in runs))
    st['npop'] = len(runs)
    modal = st.pop('modal')
    if 'refuted_frac' in runs[0]:
        st['refuted_frac'] = [round(r['refuted_frac'], 2) for r in runs]
        st['final_one'] = [round(r['final_one'], 3) for r in runs]
        st['final_n'] = [r['final_n'] for r in runs]
        crit = collections.defaultdict(list)
        for r in runs:
            for w, v in r['critique'].items():
                crit[w].append(v[0])
        st['_crit'] = {w: (sum(v), len(v)) for w, v in crit.items()}
        if truth:
            for cat in ('TOTAL', 'COMMODITY', 'VERB', 'PERSON', 'PLACE'):
                xs = [x for w, v in crit.items() if truth.get(w) == cat for x in v]
                if xs:
                    st['refuted_' + cat] = (sum(xs), len(xs))
    print('==', n, 'pops', len(runs), 'evals %.2fM' % (st['nev'] / 1e6))
    for k, v in st.items():
        if k not in ('nev', 'npop', '_crit'):
            print('  ', k, v)
    if n.startswith('LA') or n == 'PLANT':
        cnt = collections.Counter()
        for r in runs:
            for w, c in r['tcnt'].items():
                cnt[w] += c
        stable = collections.defaultdict(list)
        for w, (nm, ag, npop) in modal.items():
            if npop >= len(runs) * 0.8 and ag >= 0.8 and cnt[w] / len(runs) >= 3:
                cr = st.get('_crit', {}).get(w)
                stable[nm].append((w, round(ag, 2), round(cnt[w] / len(runs), 1)) + ((('ref %d/%d' % cr),) if cr else ()))
        st['stable'] = {k: sorted(v, key=lambda x: -x[2]) for k, v in stable.items()}
        nst = sum(len(v) for v in stable.values())
        print('   stable (agree>=0.8, train count>=3):', nst, {k: len(v) for k, v in stable.items()})
        for k, v in st['stable'].items():
            print('     ', k, v[:14])
        for w in ['KU-RO', 'KI-RO', 'PO-TO-KU-RO', 'SA-RA₂', 'A-DU', 'KA-PA', 'L:GRA', 'L:VIN', 'L:OLE', 'L:CYP',
                  'L:VIR', 'L:OLIV', 'NI', '*304', '*308', 'TE', 'SI', 'KU-NI-SU', 'DA-ME', 'PA-I-TO', 'A-SA-SA-RA-ME']:
            if w in modal:
                print('       %-14s %s %s' % (w, modal[w], st.get('_crit', {}).get(w, '')))
    out[n] = st
json.dump(out, open(os.path.join(C.CK, TAG + '_report.json'), 'w'), indent=1, ensure_ascii=False)
