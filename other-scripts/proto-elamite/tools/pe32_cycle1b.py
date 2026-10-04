"""pe32 cycle 1b: is the C14 coherence more than 'they sit on the same tablets'?  N3 = frequency-matched 14-sets
drawn only from signs ending lines on the tablets that carry a standard M288 entry (the class-defining tablets);
the 'run' statistic of cycle 1 with N3 in place of N2.  Composite recomputed without the office family (pe15
input) and without the two families cycle 1's calibration showed to be anti-conservative (pos, left).
Planted control at full strength: 14 random signs sharing a left neighbour on 60% of their final uses."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe32_common import *  # noqa
import pe32_cycle1 as c1

if __name__ == '__main__':
    rng = np.random.default_rng(321)
    T = table()
    E = m288_events(T)
    deft = {e['tid'] for e in E if is_std(e) and e['x']}
    pool3 = sorted({l['fin'] for t in T if t['id'] in deft for l in t['L'][1:] if l['fin'] not in ('-', 'x')})
    out = {'n_def_tablets': len(deft), 'pool3': len(pool3)}
    fams = ['sys', 'size', 'hdr', 'next', 'herd', 'tabs']
    c1.FAMS_USED = fams
    out['C14_N3'] = c1.run(T, C14, 'C14 vs N3 (same-tablet pool)', E, rng, nr=2000, fams=fams, pool2=pool3)
    F0 = sign_features(T)
    univ = sorted(s for s, n in F0['fin_n'].items() if n >= 3 and s not in ('x', '-') and s not in C14)
    pl = freq_matched(rng, C14, univ, F0['fin_n'])
    out['plant60'] = c1.run(c1.plant(T, pl, rng, frac=0.6), pl, 'PLANT left 60%', E, rng, nr=1000,
                            fams=['left', 'sys', 'size', 'hdr', 'next', 'herd', 'tabs'], pool2=pool3)
    json.dump(out, open(os.path.join(CK, 'cycle1b.json'), 'w'), indent=1, default=str)
    print('done')
