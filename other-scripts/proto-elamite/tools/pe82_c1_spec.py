"""pe82 cycle 1 specificity: is a marker's surprisal effect UID-like (present after every kind of surprise) or
one adjacency rule (one previous/next target)?  Within (tab, y, ns) strata, z of presence vs surprisal computed
separately for lines grouped by the context value that drives the surprise."""
import sys, json, collections, numpy as np
import pe82_common as pc, pe82_c1 as c1
corpus = sys.argv[1]; signs = set(sys.argv[2].split(',')); feats = sys.argv[3].split(',')
T = pc.load_pe() if corpus == 'PE' else pc.load_pc()
rows = c1.build_rows(T)
s = c1.surprisal(rows, feats)
x = np.array([bool(set(r['sg']) & signs) for r in rows])
allm = np.ones(len(rows), bool)
print('all', c1.Scorer(rows, s, allm).z(x))
for f in feats:
    vals = collections.Counter(r['f'][f] for r in rows)
    for v, n in vals.most_common(8):
        m = np.array([r['f'][f] == v for r in rows])
        z, k = c1.Scorer(rows, s, m).z(x)
        print(f, v, n, 'z', round(z, 2), 'present', k)
# by own target
for y, n in collections.Counter(r['y'] for r in rows).most_common(6):
    m = np.array([r['y'] == y for r in rows]); print('own', y, n, [round(v, 2) if isinstance(v, float) else v for v in c1.Scorer(rows, s, m).z(x)])
