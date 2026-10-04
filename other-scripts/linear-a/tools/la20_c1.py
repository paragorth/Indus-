#!/usr/bin/env python3
"""LA-20 cycle 1: does Linear A have a pecking order? Held-out pair-order prediction for four
item types x four rankers, vs a null that shuffles order within every list (same pipeline).
Family-wise P from the max z over all configurations in each null replicate.
usage: la20_c1.py TYPES NREP  (e.g. 'WE' 200) ; writes la20_ckpt/c1_<T>.json"""
import sys, json, time
from la20_common import *

types, NREP = sys.argv[1], int(sys.argv[2])
docs = load_la()
for T in types:
    ords = [o for _, o in orders(docs, T)]
    fit = ['BT', 'ELO', 'POS'] + (['PL'] if T == 'L' else [])
    real = {m: list(cv_score(ords, FITTERS[m], seed=0)) for m in fit}
    nulls = {m: [] for m in fit}
    t0 = time.time()
    for r in range(NREP):
        s = shuffle_within(ords, random.Random(1000 + r))
        for m in fit: nulls[m].append(list(cv_score(s, FITTERS[m], seed=r + 1)))
        if r % 20 == 0:
            print(T, r, round(time.time() - t0), flush=True)
            json.dump({'real': real, 'nulls': nulls}, open(os.path.join(CK, 'c1_%s.json' % T), 'w'))
    json.dump({'real': real, 'nulls': nulls}, open(os.path.join(CK, 'c1_%s.json' % T), 'w'))
