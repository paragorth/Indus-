"""v14 cycle 3c: (i) noise tolerance with the clean slot grammar held fixed (separates cut drift from count noise);
(ii) within-word slot MI of the planted dice corpus as a function of the share of Voynich tokens mixed in."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from v14_lib import *
vt = [w for l in voy_lines('ZL3b') for w in l['words']]
m0 = learn_model(None, 4, 'planted_noise0')
out = {}
for frac in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
    L, truth = planted_dice(design='mixed', seed=21)
    r2 = random.Random(4)
    for l in L: l['words'] = [w if r2.random() >= frac else r2.choice(vt) for w in l['words']]
    toks = parse(L, m0)
    ids = [parsimony(slot_scores(c)) for c in slot_counts(toks, 4)]
    ind = independence(toks, 4, nperm=50)
    out[frac] = (ids, {k: round(v['mi'], 3) for k, v in ind.items()})
    print(frac, ids, out[frac][1], flush=True)
save('c3e_noise_fixedcuts', out)
