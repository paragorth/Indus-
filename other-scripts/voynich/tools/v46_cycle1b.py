"""v46 cycle 1b: calibrate the pipeline on Hyginus with two count sources (Ptolemy = independent picture count;
'faithful' = the chapter's own total / listed stars = a picture drawn from the text) and two S1 forms
(rate; log count residualised on log length)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v46_lib as L
from v46_cycle1 import summarise, log, out
H = L.hyginus_units()
for mode in ('rate', 'resid'):
    L.S1MODE = mode
    summarise('HYGINUS S1=%s, faithful counts' % mode, [dict(u, n=u['stated']) for u in H])
    summarise('HYGINUS S1=%s, Ptolemy counts' % mode, H)
    rng = np.random.default_rng(5)
    summarise('HYGINUS S1=%s, faithful counts shuffled' % mode, [dict(u, n=v) for u, v in zip(H, rng.permutation([u['stated'] for u in H]))])
open(os.path.join(L.CK, 'c1b_hyg.log'), 'w').write('\n'.join(log))
