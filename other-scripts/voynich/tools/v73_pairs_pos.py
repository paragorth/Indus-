"""v73: v67 pair test on fixed positional streams (first, second, third, last word of each line) for the Voynich
(ZL, IT) and generator text fitted to ZL."""
import numpy as np, v73_lib as L, v72_lib as V
from v73_pairs import test
P = V.voynich('ZL3b')
T = {'ZL3b': P, 'IT2a': V.voynich('IT2a'), 'SELFCIT': L.NULLS['SELFCIT'](P, 801), 'MK2': L.NULLS['MK2'](P, 802),
     'JUNC': L.NULLS['JUNC'](P, 803)}
out = {}
for t, pp in T.items():
    C = L.Corpus(pp, t); rng = np.random.default_rng(3); r = {}
    for nm, idx in [('first', C.line_start), ('second', C.line_start + np.minimum(1, C.llen[C.line_start] - 1)),
                    ('third', C.line_start + np.minimum(2, C.llen[C.line_start] - 1)), ('last', C.line_start + C.llen[C.line_start] - 1)]:
        r[nm] = test(C, idx, rng, 200)
    out[t] = r
    print(t, {k: (round(v['RECz'], 1), round(v['ASYMz'], 1)) for k, v in r.items()}, flush=True)
L.jsave('pairs_pos.json', out)
