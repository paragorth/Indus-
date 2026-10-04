#!/usr/bin/env python3
"""LA-33: simulate archives (scheduled / Poisson / mixed) and store their summary statistics.
usage: la33_sims.py N_PER_REGIME SEED OUTNAME   (2 worker processes at most)"""
import sys, math, json, os
import numpy as np
from multiprocessing import Pool
from la33_common import sim_archive, feats, CK, FEAT


def one(args):
    seed, regime = args
    rng = np.random.default_rng(seed)
    T = int(math.exp(rng.uniform(math.log(8), math.log(250))))
    docs, lab, w = sim_archive(T, regime, rng)
    f = feats(docs, rng)
    return [regime, float(w), float(np.mean(lab))] + [float(x) for x in f]


if __name__ == '__main__':
    N, seed, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    jobs = [(seed * 10_000_000 + i * 3 + k, r) for i in range(N) for k, r in enumerate('SPM')]
    with Pool(2) as p:
        rows = p.map(one, jobs, chunksize=50)
    json.dump({'feat': FEAT, 'rows': rows}, open(os.path.join(CK, name + '.json'), 'w'))
    print('done', len(rows))
