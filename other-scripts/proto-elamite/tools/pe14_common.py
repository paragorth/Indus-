"""pe14 shared code: 'the tablet is woven' -- periodic structure in the order of units.

Each feature f turns a tablet's units into a category sequence (int >= 0 counted,
-1 not counted). Lag-p agreement A_p(f) = number of unit pairs (i, i+p) on the same
tablet with the same counted category.

Nulls (all keep each tablet's multiset of categories):
  shuffle : uniform within-tablet permutation (rho = 1)
  adj     : FIRST-ORDER ADJACENCY null, fitted per feature: units are drawn one by
            one without replacement, a unit of the previous unit's counted category
            has weight rho_f; rho_f is tuned so the null reproduces the observed lag-1
            agreement (rho < 1: the pe13 adjacent-avoidance dip; rho > 1: runs).
            This is a non-periodic model with the same capacity (one parameter) as a
            one-lag periodic model; any excess at lag p >= 2 beyond it is structure
            that adjacency alone cannot make.
"""
import json, math, os, random
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe14_ckpt')
LAGS = list(range(1, 13))
RHOS = np.exp(np.linspace(math.log(0.01), math.log(30), 15))


def units(name, unit):
    C = json.load(open(os.path.join(CK, 'units.json')))
    T = C[name]
    out = []
    for t in T:
        U = [u for u in t['u'] if not u['hdr']]
        if unit == 'entries':
            U = [u for u in U if u['num']]
        if len(U) >= 3:
            out.append({'id': t['id'], 'u': U})
    return out


def lenbin(n):
    return min(n, 5)


HDR_TOP = None


def features(T, topk_sign=60, topk_cat=20, with_num=True, extra=None):
    """Return dict name -> list (per tablet) of int category lists."""
    feats = {}
    sc = Counter(s for t in T for u in t['u'] for s in set(u['toks']))
    top_signs = [s for s, _ in sc.most_common(topk_sign)]

    def cat_feature(fn, k=topk_cat, drop=()):
        vals = Counter(fn(u) for t in T for u in t['u'])
        keep = [v for v, _ in vals.most_common() if v not in drop][:k]
        ix = {v: i for i, v in enumerate(keep)}
        return [[ix.get(fn(u), -1) for u in t['u']] for t in T]
    feats['first'] = cat_feature(lambda u: u['toks'][0])
    feats['last'] = cat_feature(lambda u: u['toks'][-1])
    feats['sys'] = cat_feature(lambda u: u['sys'])
    feats['lead'] = cat_feature(lambda u: u['lead'])
    feats['size'] = cat_feature(lambda u: u['size'])
    feats['len'] = cat_feature(lambda u: lenbin(len(u['toks'])))
    feats['sizeXlen'] = cat_feature(lambda u: (u['size'], lenbin(len(u['toks']))), k=40)
    if with_num:
        feats['num'] = cat_feature(lambda u: u['num'])
    for s in top_signs:
        feats['sign:' + s] = [[0 if s in u['toks'] else -1 for u in t['u']] for t in T]
    if extra:
        for k, fn in extra.items():
            feats[k] = cat_feature(fn)
    # drop features with < 2 counted values anywhere
    return {k: v for k, v in feats.items() if sum(1 for s in v for x in s if x >= 0) >= 20}


def lag_agree(seqs, lags=LAGS):
    """seqs: list of int lists. returns array len(lags)."""
    out = np.zeros(len(lags))
    for s in seqs:
        a = np.asarray(s)
        n = len(a)
        for j, p in enumerate(lags):
            if p >= n:
                break
            x, y = a[:-p], a[p:]
            out[j] += np.count_nonzero((x == y) & (x >= 0))
    return out


def sample_adj(s, rho, rng):
    """one null sequence for one tablet: draw without replacement, weight rho for a
    unit sharing the previous unit's counted category."""
    cnt = Counter(s)
    n = len(s)
    out = []
    prev = None
    for _ in range(n):
        keys = list(cnt)
        w = [cnt[k] * (rho if (k == prev and k >= 0) else 1.0) for k in keys]
        tot = sum(w)
        if tot <= 0:
            w = [cnt[k] for k in keys]
            tot = sum(w)
        r = rng.random() * tot
        acc = 0.0
        for k, wk in zip(keys, w):
            acc += wk
            if acc >= r:
                break
        out.append(k)
        cnt[k] -= 1
        if cnt[k] == 0:
            del cnt[k]
        prev = k
    return out


def _prep(seqs):
    useful = [s for s in seqs if sum(1 for x in s if x >= 0) >= 2]
    cats = sorted({x for s in useful for x in s})
    ix = {c: i for i, c in enumerate(cats)}
    N, C = len(useful), len(cats)
    cnt = np.zeros((N, C))
    for i, s in enumerate(useful):
        for x in s:
            cnt[i, ix[x]] += 1
    counted = np.array([c >= 0 for c in cats], float)
    lens = cnt.sum(1).astype(int)
    return cnt, counted, lens


def null_stats(seqs, rho, nsur, rng, prep=None):
    """array nsur x len(LAGS): agreement counts of the adjacency null (vectorised over
    tablets, longest first). rho = 1 gives the plain within-tablet shuffle."""
    cnt0, counted, lens = prep if prep is not None else _prep(seqs)
    order = np.argsort(-lens, kind='stable')
    cnt0, lens = cnt0[order], lens[order]
    N, C = cnt0.shape
    T = lens.max()
    nact = np.array([(lens > t).sum() for t in range(T)])
    nr = np.random.RandomState(rng.randrange(1 << 30))
    R = np.zeros((nsur, len(LAGS)))
    fac = np.where(counted > 0, rho, 1.0)
    for i in range(nsur):
        cnt = cnt0.copy()
        seq = np.full((N, T + 13), -1, int)
        prev = np.full(N, -1)
        for t in range(T):
            n = nact[t]
            w = cnt[:n].copy()
            pv = prev[:n]
            hp = np.nonzero(pv >= 0)[0]
            w[hp, pv[hp]] *= fac[pv[hp]]
            tot = w.sum(1)
            z = tot <= 0
            if z.any():
                w[z] = cnt[:n][z]
                tot = w.sum(1)
            u = nr.random_sample(n) * tot
            k = np.minimum((np.cumsum(w, 1) < u[:, None]).sum(1), C - 1)
            cnt[np.arange(n), k] -= 1
            seq[:n, t] = k
            prev[:n] = k
        cm = np.where(seq >= 0, np.where(counted[np.maximum(seq, 0)] > 0, seq, -1), -1)
        for j, p in enumerate(LAGS):
            x, y = cm[:, :-p], cm[:, p:]
            R[i, j] = np.count_nonzero((x == y) & (x >= 0))
    return R


def tune_rho(seqs, obs1, rng, nsur=8, prep=None):
    prep = prep if prep is not None else _prep(seqs)
    m = []
    for r in RHOS:
        m.append(null_stats(seqs, r, nsur, rng, prep)[:, 0].mean())
    m = np.array(m)
    # m increases with rho (monotone in expectation); interpolate on log scale
    lm = np.log(np.maximum(m, 1e-3))
    lo = math.log(max(obs1, 1e-3))
    if lo <= lm[0]:
        return float(RHOS[0])
    if lo >= lm[-1]:
        return float(RHOS[-1])
    # enforce monotone
    lm = np.maximum.accumulate(lm)
    return float(np.exp(np.interp(lo, lm, np.log(RHOS))))


def plant_weave(T, kind, frac, seed):
    """Reorder (never change) the units of a random fraction of tablets so that a
    feature cycles. P2LAST: alternate the tablet's commonest last sign with other
    entries. P3SIZE: cycle three size groups. P4FIRST: cycle the 4 commonest first signs."""
    rng = random.Random(seed)
    out = []
    planted = set()
    for t in T:
        U = list(t['u'])
        if len(U) >= 6 and rng.random() < frac:
            if kind == 'P2LAST':
                c = Counter(u['toks'][-1] for u in U).most_common(1)[0][0]
                a = [u for u in U if u['toks'][-1] == c]
                b = [u for u in U if u['toks'][-1] != c]
                groups = [a, b]
            elif kind == 'P3SIZE':
                srt = sorted(U, key=lambda u: (u['size'], rng.random()))
                k = len(srt) // 3
                groups = [srt[:k], srt[k:2 * k], srt[2 * k:]]
            elif kind == 'P4FIRST':
                cc = [c for c, _ in Counter(u['toks'][0] for u in U).most_common(4)]
                groups = [[u for u in U if u['toks'][0] == c] for c in cc]
                rest = [u for u in U if u['toks'][0] not in cc]
                groups[-1] = groups[-1] + rest
            for g in groups:
                rng.shuffle(g)
            NU = []
            while any(groups):
                for g in groups:
                    if g:
                        NU.append(g.pop())
            U = NU
            planted.add(t['id'])
        out.append({'id': t['id'], 'u': U})
    return out, planted
