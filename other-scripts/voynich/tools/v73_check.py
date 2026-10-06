import sys, numpy as np, v73_lib as L, v72_lib as V
from collections import Counter
P = V.voynich('ZL3b')
voc = Counter(w for p in P for l in p['lines'] for w in l['w'])
it = L.item_stream('antid')
for code in ('VOY', 'NOVEL'):
    co = L.code_items(it, P, code)
    print(code, list(zip(it[100:108], co[100:108])))
    print(' types', len(set(co)), 'mean len %.2f' % np.mean([len(x) for x in co]), 'in Voynich voc %.3f' % np.mean([c in voc for c in co]))
print('voy mean len %.2f' % np.mean([len(w) for p in P for l in p['lines'] for w in l['w']]))
pp, tr = L.plant(P, 'FREE', 'SELFCIT', 'VOY'); print(pp[5]['lines'][2]['w'], tr[:3])
