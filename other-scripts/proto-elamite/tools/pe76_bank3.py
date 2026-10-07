"""pe76 cycle 3 bank: through-corpus worlds (pe76_bank2.sim) scored by DIFFERENCE statistics:
stats(world) - stats(BLOCK shuffle of the same world), so within-tablet consistency and all margins cancel
and only between-tablet sharing of habits can inform the span.  usage: pe76_bank3.py <all|A|B> <seed> <nsim> [workers]"""
import os, sys, collections
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, multiprocessing as mp
import pe76_common as P, common
import pe76_bank2 as B2


def block(tokpairs, rng):
    by = collections.defaultdict(list)
    for i, s in tokpairs:
        by[common.base(s)].append((i, s))
    out = []
    for b, lst in by.items():
        per = collections.defaultdict(list)
        for i, s in lst:
            per[i].append(s)
        bysize = collections.defaultdict(list)
        for i, v in per.items():
            bysize[len(v)].append(i)
        for sz, tl in bysize.items():
            order = rng.permutation(len(tl))
            for i, j in zip(tl, order):
                out += [(i, s) for s in per[tl[j]]]
    return out


def dstats(tokpairs, n, rng, nb=1):
    s = B2.tok_stats(tokpairs, n, rng)
    sb = np.mean([B2.tok_stats(block(tokpairs, rng), n, rng) for _ in range(nb)], 0)
    return s - sb


def work(a):
    half, seed, nsim, out = a
    if os.path.exists(out):
        return out
    T = common.load(); n = len(T)
    keep = None if half == 'all' else B2.halves(T)[0 if half == 'A' else 1]
    toks = B2.token_table(T, keep)
    rng = np.random.default_rng(seed)
    TH, ST = [], []
    for _ in range(nsim):
        th = B2.draw(rng)
        ST.append(dstats(B2.sim(th, toks, n, rng), n, rng)); TH.append([th[k] for k in B2.KEYS])
    np.savez(out + '.tmp.npz', theta=np.array(TH), stats=np.array(ST)); os.replace(out + '.tmp.npz', out)
    return out


def load_bank3(half, seed=1):
    d = P.CKPT
    fs = sorted(f for f in os.listdir(d) if f.startswith('bank3_%s_s%d_' % (half, seed)) and f.endswith('.npz') and 'tmp' not in f)
    TH = np.concatenate([np.load(os.path.join(d, f))['theta'] for f in fs]); ST = np.concatenate([np.load(os.path.join(d, f))['stats'] for f in fs])
    ok = np.isfinite(ST).all(1)
    return TH[ok], ST[ok]


if __name__ == '__main__':
    half, seed, nsim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    chunk = 150
    jobs = [(half, seed * 100000 + c, chunk, os.path.join(P.CKPT, 'bank3_%s_s%d_%04d.npz' % (half, seed, c))) for c in range(nsim // chunk)]
    with mp.Pool(nw) as pool:
        for i, o in enumerate(pool.imap_unordered(work, jobs)):
            print(i, o, flush=True)
