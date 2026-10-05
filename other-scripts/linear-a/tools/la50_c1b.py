#!/usr/bin/env python3
"""la50 cycle 1b: LA rows rerun after the fix that keeps the LA sign SA out of the flax class,
plus the planted p=0.8 control (10 reps) that the first run did not finish. Log: c1b.log."""
import json, la50_c1 as C
from la50_common import *
C.OUT = os.path.join(CK, 'c1b.log'); open(C.OUT, 'w').close()
la = load_la(); ht = [d for d in la if d['id'].startswith('HT')]
R = [C.analyse(nm, d, u) for nm, d in [('LA', la), ('LA-HT', ht)] for u in ('basket', 'tablet')]
P = C.planted(la, 0.8, reps=10)
json.dump({'results': R, 'planted08': P}, open(os.path.join(CK, 'c1b.json'), 'w'), indent=1)
