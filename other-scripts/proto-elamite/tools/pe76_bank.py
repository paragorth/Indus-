"""Simulation bank for pe76.  usage: pe76_bank.py <markerset> <seed> <nsim> [workers]
markerset: var (all variant forms), varA / varB (variant forms of base signs in half A / B), num (numeral variants)."""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, multiprocessing as mp
import pe76_common as P, common


def marker_sets(T, which):
    if which == 'var':
        return P.variant_markers(T)
    if which in ('varA', 'varB'):
        names, sets = P.variant_markers(T)
        bases = sorted({common.base(n) for n in names})
        rng = np.random.default_rng(76)
        perm = rng.permutation(len(bases))
        half = {bases[i] for i in perm[:len(bases) // 2]} if which == 'varA' else {bases[i] for i in perm[len(bases) // 2:]}
        return P.variant_markers(T, keep_base=half)
    if which == 'num':
        return P.numeral_markers(T)
    raise ValueError(which)


def work(args):
    which, seed, n, out = args
    if os.path.exists(out):
        return out
    T = common.load(); size = P.tablet_sizes(T)
    _, sets = marker_sets(T, which); fq = [len(s) for s in sets]
    rng = np.random.default_rng(seed)
    TH, ST = [], []
    for _ in range(n):
        th = P.draw_theta(rng)
        X = P.to_matrix(P.simulate(th, size, fq, rng), len(T))
        TH.append([th[k] for k in P.THETA_KEYS]); ST.append(P.stats(X, rng))
    np.savez(out + '.tmp.npz', theta=np.array(TH), stats=np.array(ST)); os.replace(out + '.tmp.npz', out)
    return out


def load_bank(which, seed):
    d = P.CKPT
    fs = sorted(f for f in os.listdir(d) if f.startswith('bank_%s_s%d_' % (which, seed)) and f.endswith('.npz') and 'tmp' not in f)
    TH = np.concatenate([np.load(os.path.join(d, f))['theta'] for f in fs]); ST = np.concatenate([np.load(os.path.join(d, f))['stats'] for f in fs])
    ok = np.isfinite(ST).all(1)
    return TH[ok], ST[ok]


if __name__ == '__main__':
    which, seed, nsim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    chunk = 250
    jobs = [(which, seed * 100000 + c, chunk, os.path.join(P.CKPT, 'bank_%s_s%d_%04d.npz' % (which, seed, c))) for c in range(nsim // chunk)]
    with mp.Pool(nw) as pool:
        for i, o in enumerate(pool.imap_unordered(work, jobs)):
            print(i, o, flush=True)
