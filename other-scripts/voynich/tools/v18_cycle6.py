"""v18 cycle 6: does the Voynich pen-load coherence come only from repeated words?
Same test as cycle 3 (clean, content-residualised settings; page-swap surrogates) with
(a) identical-word pairs removed, (b) pairs with edit similarity >= 0.75 removed,
(c) only pairs at distance 3-4. Checkpoint: data/results/v18/c6.json
"""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v18_dips import *
from v18_lib import glyphs
from v18_cycle3 import pairs, simmats, coherence
from v18_cycle4 import voy_pages, model
from v18_cycle5 import clean_dips, short
RES = os.path.join(HERE, '..', 'data', 'results', 'v18')

C = Corpus(voy_pages(), glyphs, model())
ds = clean_dips(C, True)
maps = [page_swap_map(C, k) for k in range(1, C.npages)]
I, J, Dd = pairs(C); S = simmats(C, I, J)
w = np.array(C.word, dtype=object)
out = {}
for tag, keep in (('no_identical', w[I] != w[J]), ('no_close', S['edit'] < 0.75), ('dist34', Dd >= 3)):
    S2 = {k: v[keep] for k, v in S.items()}
    out[tag] = coherence(C, ds, maps, I[keep], J[keep], Dd[keep], S2)
    out[tag + '_n'] = int(keep.sum())
    print(tag, int(keep.sum()), short(out[tag]), flush=True)
out['identical_share'] = float((w[I] == w[J]).mean())
json.dump(out, open(os.path.join(RES, 'c6.json'), 'w'), indent=1, default=str)
print('done')
