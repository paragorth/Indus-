"""v14 cycle 3b: noise tolerance of the full-resolution device test (no nulls; identification only).
Planted mixed dice corpus with 0/5/10/20% of tokens replaced by Voynich tokens."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from v14_lib import *
vt = [w for l in voy_lines('ZL3b') for w in l['words']]
out = {}
for frac in (0.0, 0.02, 0.05, 0.10, 0.20):
    L, truth = planted_dice(design='mixed', seed=21)
    r2 = random.Random(4)
    for l in L: l['words'] = [w if r2.random() >= frac else r2.choice(vt) for w in l['words']]
    m = learn_model(L, 4, 'planted_noise%d' % int(frac * 100))
    toks = parse(L, m)
    ids = [parsimony(slot_scores(c)) for c in slot_counts(toks, 4)]
    out[frac] = ids
    print(frac, truth, ids, m['cuts'], flush=True)
save('c3d_noise', out)
