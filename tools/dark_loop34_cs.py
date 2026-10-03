"""S-DARK-34: class-sign planted control re-run (lookup on the 40 commonest signs; rarer first signs one class)."""
import sys
sys.argv = [sys.argv[0], '1'] + sys.argv[1:]
from dark_loop34 import *
seals = [o for o in OBJ if o['ot'] == 'SEAL']; train = [o for o in seals if o['big']]; sites = [o for o in seals if not o['big']]
log = open(SP + f'loop34_cs_{LV}.txt', 'w')
for t in ('first_sign', 'second_sign', 'penult_sign', 'last_sign', 'closer', 'has_closer', 'emblem', 'material', 'boss', 'shape'):
    r, _ = evaluate(t, train, {'sites': sites, 'im77': NEW if t not in ('material', 'boss', 'shape') else []}, use_facts=False, plant='classsign', report_feats=False)
    s = 'PLANT-CS40 ' + fmt(r); print(s); log.write(s + '\n'); log.flush()
