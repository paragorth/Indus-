"""pe6: reference table for ABC.  Prior draws -> simulated corpora -> statistic vectors.
Checkpointed in chunks: data/pe6_bank/chunk_<k>.npz (skips chunks already written).
Usage: python3 pe6_bank.py N_CHUNKS [CHUNK=500] [WORKERS=2]
"""
import os, sys, time, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe6_common import draw_prior, simulate, stats, to_unbounded, PNAMES, PEDATA

OUT = os.path.join(PEDATA, 'pe6_bank')


def one(seed):
    th = draw_prior(random.Random(seed * 7919 + 1))
    C, _ = simulate(th, seed)
    return to_unbounded(th), np.array([th[p] for p in PNAMES], float), stats(C)


def main():
    nch = int(sys.argv[1]); CH = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    os.makedirs(OUT, exist_ok=True)
    with Pool(W) as pool:
        for k in range(nch):
            fn = os.path.join(OUT, f'chunk_{k:03d}.npz')
            if os.path.exists(fn):
                continue
            t0 = time.time()
            seeds = list(range(k * CH + 1000, (k + 1) * CH + 1000))
            res = pool.map(one, seeds, chunksize=10)
            np.savez(fn, seeds=np.array(seeds), u=np.array([r[0] for r in res]),
                     raw=np.array([r[1] for r in res]), s=np.array([r[2] for r in res]))
            print(f'chunk {k} done {time.time() - t0:.0f}s', flush=True)


def load_bank():
    import glob
    fs = sorted(glob.glob(os.path.join(OUT, 'chunk_*.npz')))
    D = [np.load(f) for f in fs]
    return (np.concatenate([d['seeds'] for d in D]), np.concatenate([d['u'] for d in D]),
            np.concatenate([d['raw'] for d in D]), np.concatenate([d['s'] for d in D]))


if __name__ == '__main__':
    main()
