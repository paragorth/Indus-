"""pe14 cycle 3: PAIRED RECORDS AND WOVEN TABLETS.
(a) Massive pair search: every ordered pair of items (a, b) (top 60 signs plus
    numeral pseudo-items: size bin, system) at lag 1 and lag 2: count of units with
    a followed (lag later, same tablet) by a unit with b. Null: the joint adjacency
    (DIP) model of pe14_cycle2 (units drawn one by one, weight rho^shared signs,
    rho tuned to the observed lag-1 sharing) -- so the pe13 dip alone cannot make a
    hit. FWER from the surrogates' own search maxima. Half A searches, half B tests.
(b) Per-tablet weave: for each tablet with >= 8 units, sharing at lag p (2-6)
    against that tablet's own DIP surrogates; min-over-p p-value corrected by the
    same min on surrogates. Planted tablets (PL_*) give the recovery rate.
usage: python3 pe14_cycle3.py <config> <half>
"""
import json, math, os, random, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import CK  # noqa
from pe14_cycle1 import corpus  # noqa

NS = 150
TOPK = 60


def items(u):
    s = set(u['toks'])
    s.add('#size%d' % u['size'])
    s.add('#sys:' + u['sys'])
    if 'num' in u:
        s.add('#num%d' % u['num'])
    return s


def prep(T):
    c = Counter(x for t in T for u in t['u'] for x in set(u['toks']))
    top = [x for x, _ in c.most_common(TOPK)]
    pc = Counter(x for t in T for u in t['u'] for x in items(u) if x.startswith('#'))
    top += [x for x, n in pc.items() if n >= 30]
    ix = {x: i for i, x in enumerate(top)}
    tabs = []
    for t in T:
        It = [items(u) for u in t['u']]
        X = np.zeros((len(It), len(top)), np.float32)
        for i, s in enumerate(It):
            for x in s:
                if x in ix:
                    X[i, ix[x]] = 1
        S = np.array([[len(a & b) for b in It] for a in It], float)  # shared items
        St = np.array([[len(set(u['toks']) & set(v['toks'])) for v in t['u']] for u in t['u']], float)
        tabs.append({'id': t['id'], 'X': X, 'S': S, 'St': St, 'n': len(It)})
    return top, tabs


def pair_counts(tabs, perms):
    L = tabs[0]['X'].shape[1]
    C = np.zeros((2, L, L))
    for tb, p in zip(tabs, perms):
        X = tb['X'][p]
        for j, lag in enumerate((1, 2)):
            if tb['n'] > lag:
                C[j] += X[:-lag].T @ X[lag:]
    return C


def tab_lag_share(tb, p, lags=range(1, 7)):
    S = tb['S'][np.ix_(p, p)]
    return np.array([np.trace(S, offset=l) if tb['n'] > l else 0 for l in lags])


def dip_perm(tb, rho, rng):
    n = tb['n']
    rem = list(range(n))
    first = rem.pop(rng.randrange(n))
    seq = [first]
    St = tb['St']
    while rem:
        w = rho ** St[seq[-1], rem]
        r = rng.random() * w.sum()
        j = int(np.searchsorted(np.cumsum(w), r))
        seq.append(rem.pop(min(j, len(rem) - 1)))
    return seq


def lag1(tabs, perms):
    return sum(np.trace(tb['St'][np.ix_(p, p)], offset=1) for tb, p in zip(tabs, perms))


def tune(tabs, rng):
    obs = lag1(tabs, [list(range(tb['n'])) for tb in tabs])
    lo, hi = 0.05, 8.0
    for _ in range(11):
        mid = math.sqrt(lo * hi)
        m = np.mean([lag1(tabs, [dip_perm(tb, mid, rng) for tb in tabs]) for _ in range(2)])
        lo, hi = (mid, hi) if m < obs else (lo, mid)
    return math.sqrt(lo * hi), obs


def main(cfg, half):
    T = corpus(cfg)
    if half != 'ALL':
        r = random.Random(1234)
        ids = sorted(t['id'] for t in T)
        r.shuffle(ids)
        A = set(ids[: len(ids) // 2])
        T = [t for t in T if (t['id'] in A) == (half == 'A')]
    T = [t for t in T if len(t['u']) >= 3]
    rng = random.Random(sum(map(ord, cfg + half)))
    top, tabs = prep(T)
    rho, obs1 = tune(tabs, rng)
    ident = [np.arange(tb['n']) for tb in tabs]
    C = pair_counts(tabs, ident)
    TL = np.array([tab_lag_share(tb, np.arange(tb['n'])) for tb in tabs])
    CS = np.zeros((NS,) + C.shape, np.float32)
    TLS = np.zeros((NS,) + TL.shape, np.float32)
    CU = np.zeros((40,) + C.shape, np.float32)
    for s in range(NS):
        perms = [np.array(dip_perm(tb, rho, rng)) for tb in tabs]
        CS[s] = pair_counts(tabs, perms)
        TLS[s] = [tab_lag_share(tb, p) for tb, p in zip(tabs, perms)]
        if s < 40:
            pu = [np.array(rng.sample(range(tb['n']), tb['n'])) for tb in tabs]
            CU[s] = pair_counts(tabs, pu)
        if s % 25 == 0:
            print(cfg, half, 'sur', s, flush=True)
    np.savez_compressed(os.path.join(CK, 'c3', f'{cfg}_{half}.npz'), C=C, CS=CS, CU=CU, TL=TL, TLS=TLS,
                        top=np.array(top), ids=np.array([tb['id'] for tb in tabs]),
                        n=np.array([tb['n'] for tb in tabs]), rho=rho, obs1=obs1)
    print(cfg, half, 'done rho %.3f' % rho)


if __name__ == '__main__':
    os.makedirs(os.path.join(CK, 'c3'), exist_ok=True)
    main(sys.argv[1], sys.argv[2])
