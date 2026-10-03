"""v8 cycle 2b: transcription robustness (IT2a) of the V-2.3 within-B 'ed' anti-correlation and the within-A results,
plus a mechanical check: does the sign survive when the B-score excludes ALL e-containing words (not just 'ed')?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr
from v8_lib import voynich_pages
import v8_cycle2 as c2  # noqa (runs cycle-2 on ZL; reuse functions)
for name in ('ZL3b', 'IT2a'):
    V = [p for p in voynich_pages(name=name) if p['lang'] in ('A', 'B')]
    lab = [p['lang'] for p in V]; strata = [(p['illus'], p['hand']) for p in V]
    F = c2.feats(V)
    for exname, ex in [('ed-words', lambda w, pos: 'ed' in w), ('all e-words', lambda w, pos: 'e' in w)]:
        Sx = c2.bscores(V, lab, ex)[:, 2]
        out = []
        for g in 'AB':
            idx = [i for i in range(len(V)) if lab[i] == g]; st = [strata[i] for i in idx]
            x = c2.demean(Sx[idx], st); y = c2.demean(np.array(F['ed share'])[idx], st)
            r = spearmanr(x, y); out.append(f'{g} {r.correlation:+.2f} (p={r.pvalue:.2g}, n={len(idx)})')
        print('ROBUST', name, 'excl', exname, 'ed share:', '; '.join(out))
