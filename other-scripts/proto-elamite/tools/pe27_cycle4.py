"""pe27 cycle 4: COMPUTED LINES.  Is a line's number a fixed multiple r of the count (or the sum of
1-3 contiguous count lines) written just before it?  That is the in-tablet form of a bilingual
number: 'N workers ... their grain', 'herd of N ... N units of X'.
Statistic per (final sign s of the target line, target system, r): number of target lines with
value >= 2 for which some window of 1-3 contiguous count lines among the 6 numeric lines before it
sums to x >= 2 with y = r x.  Null: values re-dealt among all lines of the same system and final
sign across tablets (keeps fixed columns and each sign's numbers, kills the alignment between a line
and the count lines before it), 200 times; search-corrected over all
(s, system, r).  Controls: Ur III ('sze-bi' grain lines after counts), planted computed lines."""
import json, sys, os, time
from math import gcd
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import poisson
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe27_common import *  # noqa

NP = int(os.environ.get('NP', 200))
W, LBACK = 3, 6
t0 = time.time()


def seqs(tabs):
    """per tablet: list of (sys, int value, fin) in line order (only CNT and CAP lines)."""
    out = []
    for t in tabs:
        L = sorted([q for q in t['Q'] if q['sys'] in ('CNT', 'CAP')], key=lambda q: q['line'])
        if len(L) < 2:
            continue
        out.append((t['id'], [q['sys'] for q in L], [int(q['v']) for q in L], [q['fin'] for q in L],
                    [q['line'] for q in L]))
    return out


def hits(S, vals_override=None, detail=False):
    H = Counter()
    D = defaultdict(list)
    for k, (tid, sy, v, fn, ln) in enumerate(S):
        vv = vals_override[k] if vals_override is not None else v
        n = len(vv)
        for j in range(1, n):
            y = vv[j]
            if y < 2:
                continue
            keys = set()
            lo = max(0, j - LBACK)
            for a in range(j - 1, lo - 1, -1):
                s = 0
                for b in range(a, max(lo - 1, a - W), -1):   # window b..a
                    if sy[b] != 'CNT':
                        break
                    s += vv[b]
                    if s < 2:
                        continue
                    g = gcd(y, s)
                    keys.add((fn[j], sy[j], y // g, s // g))
            for kk in keys:
                H[kk] += 1
                if detail:
                    D[kk].append((tid, ln[j], y))
    return (H, D) if detail else H


def perm_vals(S, rng):
    """values re-dealt among all lines with the same (system, final sign) ACROSS tablets; each
    line keeps its position, each sign keeps its value distribution (fixed columns stay fixed)."""
    pools = defaultdict(list)
    for k, (tid, sy, v, fn, ln) in enumerate(S):
        for i in range(len(v)):
            pools[(sy[i], fn[i])].append((k, i))
    out = [list(v) for _, _, v, _, _ in S]
    for key, locs in pools.items():
        if len(locs) < 2:
            continue
        vals = [S[k][2][i] for k, i in locs]
        p = rng.permutation(len(vals))
        for (k, i), j in zip(locs, p):
            out[k][i] = vals[j]
    return out


def run(S, name, nperm=NP, seed=0, minH=3):
    rng = np.random.default_rng(seed)
    Hr, D = hits(S, detail=True)
    Hp = [hits(S, perm_vals(S, rng)) for _ in range(nperm)]
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
    order = np.argsort(-s_real)[:15]
    top = []
    for j in order:
        if s_real[j] <= 0:
            break
        k = keys[j]
        top.append({'sign': k[0], 'sys': k[1], 'r': f'{k[2]}/{k[3]}', 'H': int(real[j]),
                    'mu': round(float(tot[j] / nperm), 2), 'score': round(float(s_real[j]), 2),
                    'p_corr': float((1 + (mx >= s_real[j]).sum()) / (1 + nperm)),
                    'tabs': sorted({d[0] for d in D[k]})[:8], 'n_tabs': len({d[0] for d in D[k]})})
    out = {'n_tab': len(S), 'n_keys': len(keys), 'max_real': float(s_real.max()), 'null_q95': float(np.quantile(mx, 0.95)),
           'p_global': float((1 + (mx >= s_real.max()).sum()) / (1 + nperm)), 'top': top}
    print(name, out['n_tab'], out['n_keys'], round(out['max_real'], 2), round(out['null_q95'], 2), out['p_global'],
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
    res['ur3_600'] = run([SU[i] for i in idx], 'UR3_600')
    T = pe_tablets()
    S = seqs(T)
    # planted: on 4% of eligible PE tablets, one CAP line set to 30 x the preceding count line
    Sp = []
    rp = np.random.default_rng(2)
    npl = 0
    for tid, sy, v, fn, ln in S:
        v = list(v)
        c = [j for j in range(1, len(v)) if sy[j] == 'CAP' and sy[j - 1] == 'CNT']
        if c and rp.random() < 0.10:
            j = c[rp.integers(len(c))]
            v[j] = 30 * v[j - 1]
            fn = list(fn); fn[j] = 'PLANT'
            npl += 1
        Sp.append((tid, sy, v, fn, ln))
    res['plant'] = run(Sp, f'PLANT30_{npl}')
    res['pe'] = run(S, 'PE')
    # alternative value sets / drop ambiguous lines
    for cs, amb in (('alt_d', 'sign'), ('apriori', 'drop')):
        res[f'pe_{cs}_{amb}'] = run(seqs(pe_tablets(capset=cs, amb=amb)), f'PE_{cs}_{amb}', nperm=100)
    json.dump(res, open(os.path.join(CK, 'cycle4.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
