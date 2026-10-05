#!/usr/bin/env python3
"""la51 cycle 3c: family-wise test of every single-class word-room link (all terms x 9 classes) against
the within-site room shuffle (N1) and the cross-site context shuffle (N2), max-z over all links per null
replicate (corrects for picking the best link after the fact). usage: la51_c3c.py la|lbsmall reps"""
import sys, json, os, numpy as np
import la51_common as L, la51_engine as E
corpus = sys.argv[1]; reps = int(sys.argv[2])
if corpus == 'la': M = L.build_matrices(L.load_la())
else:
    docs = L.load_lb(); rng = np.random.default_rng(1000)
    M = L.build_matrices([docs[i] for i in rng.choice(len(docs), 1541, replace=False)])
st = np.array([E.stratum(s) for s in M['support']]); allm = np.ones(len(st), bool)
z, k = E.cmh_z(M['X'], M['C'][M['dep_of']], st, allm)
mx = {'n1': [], 'n2': []}; per = {'n1': np.zeros_like(z), 'n2': np.zeros_like(z)}
for r in range(reps):
    rng = np.random.default_rng(4000 + r)
    z1, _ = E.cmh_z(M['X'], M['C'][E.null_within_site(M, rng)], st, allm)
    z2, _ = E.cmh_z(M['X'], E.null_cross_site(M, rng)[M['dep_of']], st, allm)
    mx['n1'].append(z1.max()); mx['n2'].append(z2.max())
    per['n1'] += z1 >= z; per['n2'] += z2 >= z
rows = []
for i, j in zip(*np.unravel_index(np.argsort(-z, axis=None)[:15], z.shape)):
    fw1 = (np.sum(np.array(mx['n1']) >= z[i, j]) + 1) / (reps + 1)
    fw2 = (np.sum(np.array(mx['n2']) >= z[i, j]) + 1) / (reps + 1)
    rows.append(dict(term=M['terms'][i], cls=L.CLASSES[j], z=float(z[i, j]), k=int(k[i, j]), n=int(M['X'][:, i].sum()),
                     p1=float((per['n1'][i, j] + 1) / (reps + 1)), p2=float((per['n2'][i, j] + 1) / (reps + 1)), fw1=float(fw1), fw2=float(fw2)))
    print(corpus, rows[-1], flush=True)
json.dump(rows, open(os.path.join(L.CK, f'c3c_{corpus}.json'), 'w'), indent=1)
