#!/usr/bin/env python3
"""LA-57: build Linear-A-sized feature draws for every system (cached in data/la57_ckpt/feats.pkl).
Known systems: 4 draws each at Linear A token size; PLANT 3 draws; LA full + 20 type-shuffles (S1) +
20 order-shuffles (S2)."""
import os, sys, pickle, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la57_common as C

NDRAW = 4


def main():
    la = C.la_docs()
    n = C.ntok(la)
    out = {}
    for k in C.KNOWN:
        f, tr = C.LOADERS[k]
        docs = f()
        for j in range(NDRAW):
            rng = random.Random(C.seed('la57-draw-%s-%d' % (k, j)))
            if k == 'KH':     # every khipu with top cords (the only TOT truth) + random others up to LA size
                top = [x for x in docs if 'TOT' in x['occ']]
                d = top + C.draw([x for x in docs if 'TOT' not in x['occ']], n - C.ntok(top), rng)
            else:
                d = C.draw(docs, n, rng)
            T = tr(d) if tr else None
            X, types, labs, di = C.features(d, T)
            out[(k, j)] = dict(X=X, types=types, labs=labs, didx=di)
            print(k, j, len(d), X.shape, collections.Counter(l for l in labs if l), flush=True)
    for j in range(3):
        d, T = C.plant_docs(n, random.Random(C.seed('la57-plant-%d' % j)))
        X, types, labs, di = C.features(d, T)
        out[('PLANT', j)] = dict(X=X, types=types, labs=labs, didx=di)
        print('PLANT', j, X.shape, collections.Counter(l for l in labs if l), flush=True)
    X, types, labs, di = C.features(la)
    out[('LA', 0)] = dict(X=X, types=types, labs=labs, didx=di)
    for j in range(20):
        for tag, fn in (('S1', C.shuffle_types), ('S2', C.shuffle_order)):
            d = fn(la, random.Random(C.seed('la57-%s-%d' % (tag, j))))
            X, types, labs, di = C.features(d)
            out[('LA_' + tag, j)] = dict(X=X, types=types, labs=labs, didx=di)
    pickle.dump(out, open(os.path.join(C.CK, 'feats.pkl'), 'wb'))
    print('saved', len(out))


if __name__ == '__main__':
    main()
