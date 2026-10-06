#!/usr/bin/env python3
"""PE-61: feature blocks for Proto-Elamite (3 partitions into LA-size draws) and its shuffles
(S1 global, S2 within-entry, S3 line order within tablet; NSHUF each, 1 partition), cached in pe61_ckpt/pe_feats.pkl."""
import os, sys, pickle, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe61_common as P

t0 = time.time()
pe = P.pe_docs()
print('PE docs', len(pe), 'tokens', P.L.ntok(pe), flush=True)
out = {'REAL': P.featurise(pe, 3, 'real')}
print('real blocks', len(out['REAL']), sum(len(b['X']) for b in out['REAL']), '%.0fs' % (time.time() - t0), flush=True)
for tag, fn in (('S1', P.shuf_global), ('S2', P.shuf_entry), ('S3', P.shuf_lines)):
    for j in range(P.NSHUF):
        d = fn(pe, random.Random(P.seed('pe61-%s-%d' % (tag, j))))
        out['%s_%d' % (tag, j)] = P.featurise(d, 1, 'real')
    print(tag, 'done %.0fs' % (time.time() - t0), flush=True)
pickle.dump(out, open(os.path.join(P.CK, 'pe_feats.pkl'), 'wb'))
print('saved')
