"""pe76 cycle 2 bank: simulate THROUGH the real corpus.  Every real tablet, base sign and token position is
kept; only the variant letter of each real variant token is re-drawn from a hidden world:
tablet time t ~ U(0, S careers); clerks with careers of length 1 (M concurrent); per base sign each variant
has a Gaussian fashion (centre ~ U(0,S), width W careers); a clerk is loyal to a favourite variant with
probability idio (per base, used with prob loyal); a share 'mix' of tokens ignores time (drawn from the
real variant proportions).  S -> 0 or mix -> 1 with idio -> 0 IS the variant shuffle.
usage: pe76_bank2.py <half: all|A|B> <seed> <nsim> [workers]"""
import os, sys, collections
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, multiprocessing as mp, scipy.sparse as sp
import pe76_common as P, common

KEYS = ['logS', 'logM', 'idio', 'loyal', 'logW', 'mix']


def halves(T):
    toks = token_table(T)
    bases = sorted(toks)
    rng = np.random.default_rng(76); perm = rng.permutation(len(bases))
    return {bases[i] for i in perm[:len(bases) // 2]}, {bases[i] for i in perm[len(bases) // 2:]}


def token_table(T, keep=None):
    toks = collections.defaultdict(list)
    for i, t in enumerate(T):
        for l in t['lines']:
            for s in l['signs']:
                if common.is_sign(s) and '~' in s and not s.startswith('|'):
                    b = common.base(s)
                    if keep is None or b in keep:
                        toks[b].append((i, s))
    return toks


def draw(rng):
    return dict(logS=rng.uniform(np.log(.1), np.log(30)), logM=rng.uniform(0, np.log(30)), idio=rng.uniform(0, 1),
                loyal=rng.uniform(.3, 1), logW=rng.uniform(np.log(.1), np.log(5)), mix=rng.uniform(0, 1))


def sim(th, toks, n, rng):
    S, M, W = np.exp(th['logS']), np.exp(th['logM']), np.exp(th['logW'])
    t = rng.uniform(0, S, n)
    ns = max(1, rng.poisson(M * (S + 1)))
    st = rng.uniform(-1, S, ns)
    act = (st[None, :] <= t[:, None]) & (t[:, None] < st[None, :] + 1)
    r = rng.random(act.shape) * act - (~act)
    scr = r.argmax(1)
    none = ~act.any(1)
    if none.any():
        scr[none] = np.abs(st[None, :] + .5 - t[none, None]).argmin(1)
    out = []
    for b, lst in toks.items():
        cnt = collections.Counter(s for _, s in lst)
        V = sorted(cnt); pi = np.array([cnt[v] for v in V], float); pi /= pi.sum()
        c = rng.uniform(0, S, len(V))
        fav = {}
        for (i, _) in lst:
            sc = scr[i]
            if sc not in fav:
                fav[sc] = rng.choice(len(V), p=pi) if rng.random() < th['idio'] else -1
            u = rng.random()
            if fav[sc] >= 0 and u < th['loyal']:
                k = fav[sc]
            elif rng.random() < th['mix']:
                k = rng.choice(len(V), p=pi)
            else:
                p = pi * np.exp(-(t[i] - c) ** 2 / (2 * W ** 2)) + 1e-3 * pi
                k = rng.choice(len(V), p=p / p.sum())
            out.append((i, V[k]))
    return out


def tok_stats(tokpairs, n, rng):
    """Geometry stats on filtered markers + content-controlled habit-coherence stats."""
    tab = collections.defaultdict(set)
    for i, s in tokpairs:
        tab[s].add(i)
    names = sorted(tab)
    sets = [tab[k] for k in names if 2 <= len(tab[k]) <= 60]
    g = P.stats(P.to_matrix(sets, n), rng)
    Xv = P.to_matrix([tab[k] for k in names], n)
    bmap = {}
    bcols = [bmap.setdefault(common.base(k), len(bmap)) for k in names]
    Bm = sp.csr_matrix((np.ones(len(names)), (np.arange(len(names)), bcols)), shape=(len(names), len(bmap)))
    Xb = (Xv @ Bm); Xb.data[:] = 1
    SV = sp.triu(Xv @ Xv.T, 1).tocsr(); SB = sp.triu(Xb @ Xb.T, 1).tocsr()
    r1 = SV.sum() / max(SB.sum(), 1)
    SB2 = SB.multiply(SB >= 2).tocsr()
    rows, cols = SB2.nonzero()
    sv = np.asarray(SV[rows, cols]).ravel()
    sb = np.asarray(SB2[rows, cols]).ravel()
    r2 = (sv >= 2).mean() if len(sv) else 0
    r3 = (sv >= 1).mean() if len(sv) else 0
    r4 = (sv / sb).mean() if len(sv) else 0
    # coherence: agreement on a second base given agreement on the first, minus base rate
    r5 = r2 / max(r3, 1e-9) - r1
    return np.r_[g, r1, r2, r3, r4, r5]


STAT_NAMES2 = P.STAT_NAMES + ['agree_rate', 'agree2_given_sb2', 'agree1_given_sb2', 'agree_frac_sb2', 'coherence']


def work(a):
    half, seed, nsim, out = a
    if os.path.exists(out):
        return out
    T = common.load(); n = len(T)
    keep = None if half == 'all' else halves(T)[0 if half == 'A' else 1]
    toks = token_table(T, keep)
    rng = np.random.default_rng(seed)
    TH, ST = [], []
    for _ in range(nsim):
        th = draw(rng)
        ST.append(tok_stats(sim(th, toks, n, rng), n, rng)); TH.append([th[k] for k in KEYS])
    np.savez(out + '.tmp.npz', theta=np.array(TH), stats=np.array(ST)); os.replace(out + '.tmp.npz', out)
    return out


def real_tokens(T, half='all'):
    keep = None if half == 'all' else halves(T)[0 if half == 'A' else 1]
    toks = token_table(T, keep)
    return [p for lst in toks.values() for p in lst]


def load_bank2(half, seed):
    d = P.CKPT
    fs = sorted(f for f in os.listdir(d) if f.startswith('bank2_%s_s%d_' % (half, seed)) and f.endswith('.npz') and 'tmp' not in f)
    TH = np.concatenate([np.load(os.path.join(d, f))['theta'] for f in fs]); ST = np.concatenate([np.load(os.path.join(d, f))['stats'] for f in fs])
    ok = np.isfinite(ST).all(1)
    return TH[ok], ST[ok]


if __name__ == '__main__':
    half, seed, nsim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    chunk = 200
    jobs = [(half, seed * 100000 + c, chunk, os.path.join(P.CKPT, 'bank2_%s_s%d_%04d.npz' % (half, seed, c))) for c in range(nsim // chunk)]
    with mp.Pool(nw) as pool:
        for i, o in enumerate(pool.imap_unordered(work, jobs)):
            print(i, o, flush=True)
