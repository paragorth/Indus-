import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v67_lib as X
R = [json.loads(l) for l in open(os.path.join(X.CK, 'c3.jsonl'))]
from collections import defaultdict
by = defaultdict(list)
for r in R: by[r['corpus']].append(r)
for n, L in by.items():
    print(f'== {n} ({len(L)} fits)')
    for r in sorted(L, key=lambda r: (r['kinds'], r['K'])):
        a = r['real']['all']; t = r['twin']['all']
        print(f"  {r['kinds']:3s} K{r['K']:<4d} real MI {a['MI_z']:6.2f} ASYM {a['ASYM_z']:6.2f} REC {a['REC_z']:6.2f} REP {a['REP_ratio']:5.2f} | odd LANG {r['real']['o']['LANG']:6.2f} even {r['real']['e']['LANG']:6.2f} | twin MI {t['MI_z']:6.2f} ASYM {t['ASYM_z']:6.2f} REC {t['REC_z']:6.2f} REP {t['REP_ratio']:5.2f} | sec {r['real']['sec_nmi']:.3f}/{r['twin']['sec_nmi']:.3f}" + (f" nmix {r['real']['nmix']:.3f}" if 'nmix' in r['real'] else ''))
    A = np.array([[r['real']['all'][k] for k in ('MI_z', 'ASYM_z', 'REC_z', 'REP_ratio')] for r in L])
    T = np.array([[r['twin']['all'][k] for k in ('MI_z', 'ASYM_z', 'REC_z', 'REP_ratio')] for r in L])
    print('  MEAN real', A.mean(0).round(2), 'twin', T.mean(0).round(2))
