"""pe9 cycle 2: approximate Bayesian computation over bag and language models.

Reference table: N simulated corpora on the structure (tablets, strings per tablet,
string lengths, tablet order) of the observed corpus.  Prior:
  mode  uniform over LM, POLYA, BAG_WR, BAG_WOR, STAMP
  base  unigram or bigram (fitted on the observed corpus), tempered by gamma ~ U(0.7, 1.3)
  K ~ log-U(2, 300); c in {1,2,3,5,10}; eps ~ U(0, 1); theta ~ log-U(0.1, 300);
  session S in {1, 2, 4, 8} consecutive tablets sharing one bag / cache.
Summary statistics: pe9_common.stats (11).  Rejection ABC (distance on MAD-scaled
stats, accept nearest 0.1%).  Checkpoints: data/pe9_ckpt/c2_<corpus>_<chunk>.npz.
usage: pe9_cycle2.py CORPUS NSIMS
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe9_common import *  # noqa
from pe9_cycle1 import pe_fit_tablets  # noqa

CHUNK = 5000
CS = np.array([1, 2, 3, 5, 10], float)
SS = np.array([1, 2, 4, 8])


def corpus(name):
    C = load_corpora()
    if name == 'PE':
        return pe_fit_tablets(C)[0]
    return C[name]['tablets']


def chunk_job(arg):
    name, ci = arg
    fn = os.path.join(CKPT, 'c2_%s_%04d.npz' % (name, ci))
    if os.path.exists(fn):
        return fn
    E = Enc(corpus(name))
    allt = np.arange(E.T, dtype=np.int64)
    w0, B0 = fit_base(E.tok, E.ss, E.ts, allt, E.V)
    rng = np.random.RandomState(1000 + ci)
    P = np.zeros((CHUNK, 8))
    St = np.zeros((CHUNK, 11))
    sessc = {S: sess_flags(E.T, S) for S in SS}
    for r in range(CHUNK):
        mode = rng.randint(5)
        base = rng.randint(2)
        g = rng.uniform(0.7, 1.3)
        K = float(np.exp(rng.uniform(np.log(2), np.log(300))))
        c = CS[rng.randint(5)]
        eps = rng.uniform()
        th = float(np.exp(rng.uniform(np.log(0.1), np.log(300))))
        S = SS[rng.randint(4)]
        w = w0 ** g
        w /= w.sum()
        if base == 1:
            B = B0 ** g
            B /= B.sum(1, keepdims=True)
        else:
            B = B0
        tk = simulate(E.ss, E.ts, sessc[S], E.V, w, B, base, mode, K, c, eps, th, int(rng.randint(2**31 - 1)))
        St[r] = stats(tk, E.ss, E.ts, E.V, 15, r)
        P[r] = [mode, base, g, K, c, eps, th, S]
    np.savez(fn, P=P, S=St)
    return fn


def load_table(name):
    fs = sorted(f for f in os.listdir(CKPT) if f.startswith('c2_%s_' % name))
    P = np.concatenate([np.load(os.path.join(CKPT, f))['P'] for f in fs])
    S = np.concatenate([np.load(os.path.join(CKPT, f))['S'] for f in fs])
    return P, S


if __name__ == '__main__':
    name, N = sys.argv[1], int(sys.argv[2])
    nch = N // CHUNK
    t0 = time.time()
    with Pool(2) as pool:
        for i, fn in enumerate(pool.imap_unordered(chunk_job, [(name, c) for c in range(nch)])):
            print(name, i + 1, '/', nch, '%.0fs' % (time.time() - t0), flush=True)
