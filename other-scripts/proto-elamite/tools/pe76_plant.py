"""pe76 planted worlds built THROUGH the real corpus (a different generator from the ABC simulator):
real tablets, real base-sign tokens and positions; only the variant letter of each real variant token is
re-drawn from a hidden time (soft Gaussian fashions per variant) and from scribe idiolects (careers = 1)."""
import numpy as np, collections
import common


def variant_tokens(T):
    toks = collections.defaultdict(list)
    for i, t in enumerate(T):
        for l in t['lines']:
            for s in l['signs']:
                if common.is_sign(s) and '~' in s and not s.startswith('|'):
                    toks[common.base(s)].append((i, s))
    return toks


def plant(T, S, rng, M=8.0, idio=0.5, loyal=0.7, width=(0.3, 2.0), office_k=0, office_sets=None):
    n = len(T)
    t = rng.uniform(0, S, n)
    ns = max(1, rng.poisson(M * (S + 1)))
    st = rng.uniform(-1, S, ns)
    scr = np.empty(n, int)
    for i in range(n):
        act = np.where((st <= t[i]) & (t[i] < st + 1))[0]
        scr[i] = rng.choice(act) if len(act) else np.argmin(np.abs(st + .5 - t[i]))
    toks = variant_tokens(T)
    tab = collections.defaultdict(set)
    for b, lst in toks.items():
        cnt = collections.Counter(s for _, s in lst)
        V = sorted(cnt); pi = np.array([cnt[v] for v in V], float); pi /= pi.sum()
        c = rng.uniform(0, S, len(V)); w = rng.uniform(*width, len(V))
        fav = {}
        for (i, _) in lst:
            sc = scr[i]
            if sc not in fav:
                fav[sc] = rng.choice(len(V), p=pi) if rng.random() < idio else -1
            if fav[sc] >= 0 and rng.random() < loyal:
                k = fav[sc]
            else:
                p = pi * np.exp(-(t[i] - c) ** 2 / (2 * w ** 2)) + 1e-3 * pi
                k = rng.choice(len(V), p=p / p.sum())
            tab[V[k]].add(i)
    names = sorted(k for k, v in tab.items() if 2 <= len(v) <= 60)
    return [tab[k] for k in names], t
