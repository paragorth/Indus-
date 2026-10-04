"""pe27 cycle 6: SIGN-PAIR RATES.  For every ordered pair of final signs (A, B) where a B line follows
an A line within 2 numeric lines on the same tablet, collect y_B / x_A (exact, any systems; capacity
in N39C units, counts in units).  A fixed exchange or ration rate tying A to B shows as one ratio
recurring on many tablets.  Statistic per (A, sysA, B, sysB, r): number of TABLETS with >= 1 such
pair.  Nulls (200 each): (i) B values re-dealt among all lines of the same (system, sign) across
tablets; (ii) the same, but only among tablets of the same value-scale bin (log2 of the tablet's
median count) so that tablet-level number size is kept.  Search-corrected max over all keys.
Controls: Ur III (count lines then grain lines), planted pair rate, equal-value artefact check
(pairs with y = x reported apart)."""
import json, sys, os, time
from math import gcd, log2
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import poisson
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe27_common import *  # noqa
from pe27_cycle4 import seqs  # noqa

NP = int(os.environ.get('NP', 200))
GAP = 2
t0 = time.time()


def scale_bin(v):
    m = float(np.median(v))
    return int(log2(max(m, 1)))


def keys_of(S, vals=None):
    H = Counter()
    for k, (tid, sy, v, fn, ln) in enumerate(S):
        vv = vals[k] if vals is not None else v
        ks = set()
        for j in range(1, len(vv)):
            for i in range(max(0, j - GAP), j):
                if fn[i] == '-' or fn[j] == '-':
                    continue
                x, y = vv[i], vv[j]
                if x < 1 or y < 1 or (x == 1 and y == 1):
                    continue
                g = gcd(x, y)
                ks.add((fn[i], sy[i], fn[j], sy[j], y // g, x // g))
        for kk in ks:
            H[kk] += 1
    return H


def shuffle(S, rng, strat):
    pools = defaultdict(list)
    for k, (tid, sy, v, fn, ln) in enumerate(S):
        b = strat[k]
        for i in range(len(v)):
            pools[(sy[i], fn[i], b)].append((k, i))
    out = [list(v) for _, _, v, _, _ in S]
    for locs in pools.values():
        if len(locs) < 2:
            continue
        vals = [S[k][2][i] for k, i in locs]
        p = rng.permutation(len(vals))
        for (k, i), j in zip(locs, p):
            out[k][i] = vals[j]
    return out


def run(S, name, nperm=NP, seed=0, minH=3, strat_kind='global'):
    rng = np.random.default_rng(seed)
    strat = [0] * len(S) if strat_kind == 'global' else [scale_bin(s[2]) for s in S]
    Hr = keys_of(S)
    Hp = [keys_of(S, shuffle(S, rng, strat)) for _ in range(nperm)]
    keys = {k for k, c in Hr.items() if c >= minH}
    for h in Hp:
        keys |= {k for k, c in h.items() if c >= minH}
    keys = list(keys)
    M = np.array([[h.get(k, 0) for k in keys] for h in Hp], float)
    tot = M.sum(0)
    real = np.array([Hr.get(k, 0) for k in keys], float)

    def sc(o, mu):
        return -np.log10(np.maximum(poisson.sf(o - 1, np.maximum(mu, 0.05)), 1e-300))
    s_real = np.where(real >= minH, sc(real, tot / nperm), 0)
    mx = np.array([np.where(M[i] >= minH, sc(M[i], (tot - M[i]) / (nperm - 1)), 0).max() for i in range(nperm)])
    order = np.argsort(-s_real)[:12]
    top = []
    for j in order:
        if s_real[j] <= 0:
            break
        k = keys[j]
        top.append({'A': k[0], 'sA': k[1], 'B': k[2], 'sB': k[3], 'r': f'{k[4]}/{k[5]}', 'H': int(real[j]),
                    'mu': round(float(tot[j] / nperm), 2), 'score': round(float(s_real[j]), 2),
                    'p_corr': float((1 + (mx >= s_real[j]).sum()) / (1 + nperm))})
    out = {'n_tab': len(S), 'null': strat_kind, 'n_keys': len(Hr), 'max_real': float(s_real.max()),
           'null_q95': float(np.quantile(mx, 0.95)), 'p_global': float((1 + (mx >= s_real.max()).sum()) / (1 + nperm)),
           'top': top}
    print(name, strat_kind, out['n_keys'], round(out['max_real'], 2), round(out['null_q95'], 2), out['p_global'],
          round(time.time() - t0), flush=True)
    for e in top[:8]:
        print('    ', e, flush=True)
    return out


if __name__ == '__main__':
    res = {}
    U = ur3_tablets()
    SU = seqs(U)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(SU), min(600, len(SU)), replace=False)
    res['ur3_600'] = run([SU[i] for i in idx], 'UR3_600', nperm=100)
    T = pe_tablets()
    S = seqs(T)
    # planted: a sign pair (count line, next capacity line) at rate 30 on 12 tablets
    Sp, n = [], 0
    rp = np.random.default_rng(2)
    for tid, sy, v, fn, ln in S:
        v, fn = list(v), list(fn)
        c = [j for j in range(1, len(v)) if sy[j] == 'CAP' and sy[j - 1] == 'CNT']
        if c and n < 12 and rp.random() < 0.08:
            j = c[0]
            fn[j - 1], fn[j] = 'PLA', 'PLB'
            v[j] = 30 * v[j - 1]
            n += 1
        Sp.append((tid, sy, v, fn, ln))
    res['plant12'] = run(Sp, 'PLANT12', nperm=100, strat_kind='scale')
    for sk in ('global', 'scale'):
        res['pe_' + sk] = run(S, 'PE', strat_kind=sk)
    json.dump(res, open(os.path.join(CK, 'cycle6.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
