"""la63 cycle 3 summary: group erasure vs matched random word-sets.
usage: python3 la63_groups_sum.py CORPUS [CORPUS ...]
Per group, loss key k and model: excess = D_group - mean D_null (random same-count tokens, same docs);
the same for NRG matched random word-sets; diff = excess_group - mean excess_random. t over models.
Pseudo-group: random set 0 against sets 1.. (same statistic) -> null distribution of t for every
loss group; p = share of pseudo t >= t. Loss groups as in la63_sum (max t over their keys).
"""
import glob, json, math, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la63_lib as L
from la63_sum import GROUPS, KEYS


def exc(a):
    if a is None:
        return None
    s, n, ns, nn = a
    ok = [j for j in range(len(nn)) if nn[j] >= 1]
    if n < 2 or len(ok) < 2:
        return None
    return s / n - float(np.mean([ns[j] / nn[j] for j in ok]))


def analyse(name):
    F = [f for f in glob.glob(os.path.join(L.CK, 'grp_%s_*.json' % name))
         if re.match(r'^grp_%s_\d+\.json$' % re.escape(name), os.path.basename(f))]
    D = [json.load(open(f)) for f in F]
    diff = collections.defaultdict(lambda: collections.defaultdict(list))
    pse = collections.defaultdict(lambda: collections.defaultdict(list))
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    info = {}
    for d in D:
        for g, x in d['res'].items():
            info[g] = x['types']
            for k in KEYS:
                e = exc(x['real']['acc'].get(k))
                rs = [exc(r['acc'].get(k)) for r in x['rand']]
                rs = [v for v in rs if v is not None]
                if e is not None and len(rs) >= 2:
                    diff[g][k].append(e - float(np.mean(rs)))
                    raw[g][k].append(e)
                if len(rs) >= 3:
                    pse[g][k].append(rs[0] - float(np.mean(rs[1:])))

    def tstat(v):
        v = np.array(v)
        if len(v) < 4:
            return None
        return float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)) + 1e-9))

    def gsc(tab):
        out = {}
        for gname, ks in GROUPS.items():
            ts = [tstat(tab[k]) for k in ks if k in tab]
            ts = [t for t in ts if t is not None]
            out[gname] = max(ts) if ts else None
        return out

    T = {g: gsc(diff[g]) for g in diff}
    TP = {g: gsc(pse[g]) for g in pse}
    pool = {gn: np.sort([TP[g][gn] for g in TP if TP[g][gn] is not None]) for gn in GROUPS}

    def p(gn, t):
        arr = pool[gn]
        if t is None or not len(arr):
            return 1.0
        return (1 + len(arr) - np.searchsorted(arr, t, side='left')) / (1 + len(arr))

    P = {g: {gn: p(gn, T[g][gn]) for gn in GROUPS} for g in T}
    return dict(models=len(D), T=T, P=P, info=info, pool=pool,
                M={g: {k: float(np.mean(v)) for k, v in raw[g].items()} for g in raw},
                DIF={g: {k: float(np.mean(v)) for k, v in diff[g].items()} for g in diff})


if __name__ == '__main__':
    for name in sys.argv[1:]:
        R = analyse(name)
        n_pool = {gn: len(v) for gn, v in R['pool'].items()}
        print('%s: models %d, groups %d, pseudo-group pool %s' % (name, R['models'], len(R['T']), n_pool))
        # false rate of the decision rule on pseudo-groups (min p over loss groups < 0.01, leave-in)
        ps = []
        for gn, arr in R['pool'].items():
            pass
        order = sorted(R['T'], key=lambda g: min(R['P'][g].values()))
        for g in order:
            best = min(R['P'][g], key=R['P'][g].get)
            print('  %-10s n_types %3d  best %-6s p %.3f | %s' % (
                g, len(R['info'][g]), best, R['P'][g][best],
                ' '.join('%s t%+.1f' % (gn, t) for gn, t in R['T'][g].items() if t is not None)))
