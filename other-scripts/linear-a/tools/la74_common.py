#!/usr/bin/env python3
"""la74: quantities as fingerprints of a physical vessel set (pouring simulator).

A hypothesis H is a set of k measuring vessels (sizes v_1 > ... > v_k, in units of the integer unit U),
mapped to k fraction signs, plus pouring habits: share u of remainders that are continuous
(uniform) vs 'intended' (c x one vessel), continue-probability g after each used vessel, and
round-up threshold rho for the last remainder.  Simulating H gives a distribution over vessel
strings (largest first).  Real fraction strings are scored with
    P_V(s) = (1-e) P_H(s) + e P_NV(s)
where P_NV is a fitted no-vessel model (plain random fractions: number of distinct signs, unigram
signs, canonical order, geometric repeats).  Gain = mean log P_V - mean log P_NV on held-out data.
"""
import json, math, itertools, os, random
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la74_ckpt')


def load_strings(statuses=('read', 'damaged')):
    d = json.load(open(os.path.join(DATA, 'corpus_ra.json')))
    out = []
    for doc in d:
        for t in doc['tokens']:
            if t['t'] == 'num' and t['frac'] and t['st'] in statuses:
                out.append(dict(doc=doc['id'], site=doc['site'], sup=doc['support'],
                                v=t['v'], s=tuple(t['frac'])))
    return out


def vocab(strings, minc=3):
    c = Counter(x for r in strings for x in r)
    return sorted([k for k, n in c.items() if n >= minc], key=lambda k: -c[k])


def norm(s, V):
    return tuple(x if x in V else 'o' for x in s)

# ---------------------------------------------------------------- no-vessel model
class NoVessel:
    def __init__(self, train, V):
        self.V = list(V) + ['o']
        c = Counter(x for s in train for x in set(s))
        self.p = {k: (c[k] + 0.5) for k in self.V}
        z = sum(self.p.values()); self.p = {k: v / z for k, v in self.p.items()}
        Lc = Counter(len(set(s)) for s in train)
        self.PL = {L: (Lc[L] + 0.5) / (len(train) + 2.0) for L in (1, 2, 3)}
        tot = sum(self.PL.values()); self.PL = {k: v / tot for k, v in self.PL.items()}
        # repeats: extra copies geometric
        ext = sum(len(s) - len(set(s)) for s in train); nd = sum(len(set(s)) for s in train)
        self.q = (ext + 0.5) / (ext + nd + 1.0)       # P(another copy)
        # canonical order via pairwise wins (Borda)
        w = Counter()
        for s in train:
            seen = []
            for x in s:
                if x not in seen: seen.append(x)
            for i in range(len(seen)):
                for j in range(i + 1, len(seen)):
                    w[seen[i]] += 1; w[seen[j]] -= 1
        self.rank = {k: -w[k] + 1e-3 * i for i, k in enumerate(self.V)}
        # order fidelity
        ok = tot2 = 0
        for s in train:
            seen = []
            for x in s:
                if x not in seen: seen.append(x)
            if len(seen) > 1:
                tot2 += 1; ok += seen == sorted(seen, key=lambda k: self.rank[k])
        self.eps = (tot2 - ok + 0.5) / (tot2 + 1.0)

    def logp(self, s):
        seen = []; cnt = Counter(s)
        for x in s:
            if x not in seen: seen.append(x)
        # contiguous blocks required (repeats adjacent)
        blocks = [x for i, x in enumerate(s) if i == 0 or s[i - 1] != x]
        if len(blocks) != len(seen): return math.log(1e-9)
        L = len(seen)
        if L > 3: return math.log(1e-9)
        lp = math.log(self.PL[L])
        tot = 0.0
        for perm in itertools.permutations(seen):
            pr = 1.0; rem = 1.0
            for x in perm:
                pr *= self.p[x] / rem; rem -= self.p[x]
            tot += pr
        lp += math.log(tot)
        if L > 1:
            can = sorted(seen, key=lambda k: self.rank[k])
            nf = math.factorial(L)
            lp += math.log(1 - self.eps) if seen == can else math.log(self.eps / (nf - 1))
        for x in seen:
            lp += (cnt[x] - 1) * math.log(self.q) + math.log(1 - self.q)
        return lp

# ---------------------------------------------------------------- vessel simulator
def sample_H(rng, kmin=2, kmax=6):
    k = int(rng.integers(kmin, kmax + 1))
    v1 = math.exp(rng.uniform(math.log(0.15), math.log(0.95)))
    sizes = [v1]
    for _ in range(k - 1):
        sizes.append(sizes[-1] / math.exp(rng.uniform(math.log(1.25), math.log(10))))
    return dict(k=k, sizes=sizes, u=float(rng.uniform(0, 1)), g=float(rng.uniform(0, 1)),
                rho=float(rng.uniform(0.3, 1.0)), cmax=int(rng.integers(1, 5)))


def simulate(H, N, rng):
    """returns Counter over tuples of vessel indices (with repeats), fraction-present draws only."""
    v = np.array(H['sizes']); k = H['k']
    cont = rng.random(N) < H['u']
    R = rng.random(N)
    vi = rng.integers(0, k, N); cc = rng.integers(1, H['cmax'] + 1, N)
    R = np.where(cont, R, np.minimum(cc * v[vi], 0.999))
    rem = R.copy(); counts = np.zeros((N, k), dtype=np.int64)
    active = np.ones(N, bool); used = np.zeros(N, bool)
    for i in range(k):
        c = np.floor(rem / v[i] + 1e-9).astype(np.int64)
        c = np.where(active, c, 0)
        counts[:, i] = c; rem = rem - c * v[i]
        nz = c > 0
        # after using a vessel, continue with prob g
        stop = nz & (rng.random(N) > H['g'])
        newly = active & stop
        # round-up habit on stopping: if remainder >= rho * next vessel... use next vessel once
        if i + 1 < k:
            ru = newly & (rem >= H['rho'] * v[i + 1]) & (rem > 1e-9)
            counts[:, i + 1] += ru.astype(np.int64)
        active &= ~stop
        used |= nz
    # end: remainder vs last vessel
    ru = active & (rem >= H['rho'] * v[k - 1]) & (rem > 1e-9)
    counts[:, k - 1] += ru.astype(np.int64)
    keep = counts.sum(1) > 0
    counts = counts[keep]
    C = Counter()
    for row in counts.tolist():
        C[tuple(row)] += 1
    n = int(keep.sum())
    return {key: c / n for key, c in C.items()}


def string_to_counts(s, m_inv, k):
    """s (sign tuple) -> vessel count tuple if writable by the mapped set in size order, else None"""
    cnt = [0] * k; last = -1
    for x in s:
        i = m_inv.get(x)
        if i is None or i < last: return None
        cnt[i] += 1; last = i
    return tuple(cnt)


def ll_strings(dist, mapping, strings, lnv, e):
    k = len(mapping); m_inv = {x: i for i, x in enumerate(mapping)}
    tot = 0.0
    for s, l0 in zip(strings, lnv):
        key = string_to_counts(s, m_inv, k)
        ph = dist.get(key, 0.0) if key is not None else 0.0
        tot += math.log((1 - e) * ph + e * math.exp(l0))
    return tot

EGRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95, 1.0)


def fit_mapping(dist, k, strings, lnv, signs, rng, iters=150):
    """hill-climb an injective mapping vessel->sign and e on train strings."""
    # initial: rank vessel single-use frequency vs sign frequency
    use = np.zeros(k)
    for key, p in dist.items():
        for i, c in enumerate(key):
            if c: use[i] += p
    order = list(np.argsort(-use))
    mapping = [None] * k
    for r, i in enumerate(order):
        mapping[i] = signs[r] if r < len(signs) else signs[-1]
    def score(mp):
        return max((ll_strings(dist, mp, strings, lnv, e), e) for e in EGRID)
    best, be = score(mapping)
    for it in range(iters):
        mp = list(mapping)
        if rng.random() < 0.5 and k > 1:
            a, b = rng.choice(k, 2, replace=False); mp[a], mp[b] = mp[b], mp[a]
        else:
            a = int(rng.integers(k)); cand = [x for x in signs if x not in mp]
            if not cand: continue
            mp[a] = cand[int(rng.integers(len(cand)))]
        sc, e = score(mp)
        if sc > best:
            best, be, mapping = sc, e, mp
    return mapping, best, be


def nv_sample(nv, n, rng):
    out = []
    Ls = list(nv.PL); pL = np.array([nv.PL[l] for l in Ls])
    V = nv.V; pv = np.array([nv.p[x] for x in V])
    for _ in range(n):
        L = Ls[rng.choice(len(Ls), p=pL)]
        idx = rng.choice(len(V), size=L, replace=False, p=pv)
        seen = [V[i] for i in idx]
        if L > 1:
            can = sorted(seen, key=lambda k: nv.rank[k])
            if rng.random() > nv.eps: seen = can
            else:
                perms = [list(p) for p in itertools.permutations(seen) if list(p) != can]
                seen = perms[rng.integers(len(perms))]
        s = []
        for x in seen:
            c = 1
            while rng.random() < nv.q: c += 1
            s += [x] * c
        out.append(tuple(s))
    return out


def dist_sample(dist, mapping, n, rng):
    keys = list(dist); p = np.array([dist[k] for k in keys]); p /= p.sum()
    out = []
    for i in rng.choice(len(keys), size=n, p=p):
        s = []
        for j, c in enumerate(keys[i]): s += [mapping[j]] * c
        out.append(tuple(s))
    return out


def shuffle_signs(strings, rng):
    toks = [x for s in strings for x in s]; rng.shuffle(toks)
    out = []; i = 0
    for s in strings:
        out.append(tuple(toks[i:i + len(s)])); i += len(s)
    return out
