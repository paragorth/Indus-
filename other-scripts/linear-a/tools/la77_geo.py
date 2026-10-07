"""LA-77 phylogeography engine: trees (from random mutation models) x random rooting rules ->
directed transmissions between sites; distance decay and site outflow; planted worlds.
Deposit phases are NOT read here.
"""
import json, collections, time
import numpy as np
from numba import njit
from la77_common import *
from la77_c1 import fam_arrays, score_all, total_opp, KI


@njit(cache=True)
def trees_one(R, sizes, starts, w, sig, pnet, seed):
    np.random.seed(seed)
    keep = np.zeros(R.shape[0], np.bool_)
    par = np.zeros(sizes.max() + 1, np.int64)
    for f in range(sizes.shape[0]):
        s0, s1 = starts[f], starts[f + 1]
        ne = s1 - s0
        cost = np.empty(ne)
        for e in range(ne):
            cost[e] = -w[R[s0 + e, 3]] + sig * np.random.randn()
        order = np.argsort(cost)
        for a in range(sizes[f]):
            par[a] = a
        for e in order:
            u = R[s0 + e, 1]; v = R[s0 + e, 2]
            while par[u] != u:
                u = par[u]
            while par[v] != v:
                v = par[v]
            if u != v:
                par[u] = v; keep[s0 + e] = True
            elif np.random.rand() < pnet:
                keep[s0 + e] = True
    return keep


def tree_dists(R, keep, fams, starts):
    """per family: all-pairs hop distance in the kept graph (BFS)."""
    out = []
    for f, mem in enumerate(fams):
        n = len(mem)
        adj = [[] for _ in range(n)]
        for e in range(starts[f], starts[f + 1]):
            if keep[e]:
                a, b = R[e, 1], R[e, 2]; adj[a].append(b); adj[b].append(a)
        D = np.full((n, n), 99, np.int16)
        for s in range(n):
            D[s, s] = 0; q = [s]
            for x in q:
                for y in adj[x]:
                    if D[s, y] == 99:
                        D[s, y] = D[s, x] + 1; q.append(y)
        out.append(D)
    return out


class Engine:
    """Fixed families + H phylogeny hypotheses; evaluates site assignments."""

    def __init__(self, types, H, rng, codes, ntop_pool=20000):
        self.types = types
        E, F = families(types)
        self.F = F
        R, sizes, starts = fam_arrays(types, E, F)
        self.R, self.sizes, self.starts = R, sizes, starts
        # global type index of each edge endpoint
        self.gu = np.array([F[R[e, 0]][R[e, 1]] for e in range(len(R))], np.int64)
        self.gv = np.array([F[R[e, 0]][R[e, 2]] for e in range(len(R))], np.int64)
        self.cls = R[:, 3] if len(R) else np.zeros(0, np.int64)
        logO = total_opp(types)
        # tree models: best H of ntop_pool random (w, sigma, pnet) on ALL families' mutation score
        W = rng.normal(0, 1.5, (ntop_pool, 6)); SIG = rng.uniform(0, 2, ntop_pool); PN = rng.uniform(0, .3, ntop_pool)
        if len(F):
            s, _ = score_all(R, sizes, starts, np.arange(len(F)), W, SIG, PN, logO, int(rng.integers(1 << 30)))
            top = np.argsort(-s)[:H]
        else:
            top = np.arange(H)
        self.H = H
        self.keeps, self.dists = [], []
        for h in top:
            k = trees_one(R, sizes, starts, W[h], SIG[h], PN[h], int(rng.integers(1 << 30))) if len(R) else np.zeros(0, bool)
            self.keeps.append(k); self.dists.append(tree_dists(R, k, F, starts))
        # random rooting rules: root score = a*log(tokens) + b*nsites + c*length + d*noise
        self.root_coef = rng.normal(0, 1, (H, 4))
        self.root_noise = rng.normal(0, 1, (H, len(types)))
        self.lens = np.array([len(t) for t in types], float)
        self.codes = codes
        self.fam_of_edge = R[:, 0] if len(R) else np.zeros(0, np.int64)

    def X(self, docs):
        ci = {c: i for i, c in enumerate(self.codes)}
        ti = {t: i for i, t in enumerate(self.types)}
        X = np.zeros((len(self.types), len(self.codes)))
        for d in docs:
            for w in d['words']:
                if w in ti and d['site'] in ci:
                    X[ti[w], ci[d['site']]] += 1
        return X

    def transmissions(self, X, hyps=None, famsel=None):
        """M[h] summed: directed site x site weight (parent site -> child site)."""
        S = X.shape[1]
        P = X / np.maximum(X.sum(1, keepdims=True), 1)
        logtok = np.log1p(X.sum(1)); nsite = (X > 0).sum(1)
        Mtot = np.zeros((S, S))
        hyps = list(range(self.H)) if hyps is None else list(hyps)
        fsel = None if famsel is None else set(famsel)
        for h in hyps:
            a, b, c, d = self.root_coef[h]
            rs = a * logtok + b * nsite + c * self.lens + d * self.root_noise[h]
            keep = self.keeps[h]
            par, chi = [], []
            for f, mem in enumerate(self.F):
                if fsel is not None and f not in fsel:
                    continue
                r = int(np.argmax(rs[mem]))
                D = self.dists[h][f]
                for e in range(self.starts[f], self.starts[f + 1]):
                    if not keep[e]:
                        continue
                    u, v = self.R[e, 1], self.R[e, 2]
                    if D[r, u] <= D[r, v]:
                        par.append(mem[u]); chi.append(mem[v])
                    else:
                        par.append(mem[v]); chi.append(mem[u])
            if par:
                Mtot += P[par].T @ P[chi]
        return Mtot / max(len(hyps), 1)


def stats(M, Dlog, mask):
    """local share, mean log-distance of cross-site transmissions, net outflow per site."""
    tot = M.sum()
    off = M * mask
    loc = np.trace(M) / max(tot, 1e-12)
    mld = (off * Dlog).sum() / max(off.sum(), 1e-12)
    O = (M - M.T).sum(1)
    return loc, mld, O


# ------------------------------------------------------------------ planted worlds
def plant_world(docs, codes, km, rng, L=60.0, q=0.35, mu=0.35, N0=220, coin=25, prof=None, times=None, zexp=0.9):
    """Words born at the first site, copied to later-activated sites with prob q*exp(-km/L) from
    the nearest earlier holder, mutated on copy with prob mu (edit class from prof)."""
    sig = collections.Counter(s for d in docs for w in d['words'] for s in w)
    sg = list(sig); pw = np.array([sig[s] for s in sg], float); pw /= pw.sum()
    lens = [len(w) for d in docs for w in d['words']]
    S = len(codes)
    order = rng.permutation(S) if times is None else np.argsort(times)
    t = np.empty(S); t[order] = np.arange(S)
    prof = prof if prof is not None else np.array([.1, .1, .3, .05, .1, .35])

    def newword():
        n = lens[rng.integers(len(lens))]
        return tuple(sg[i] for i in rng.choice(len(sg), n, p=pw))

    def mutate(w):
        k = rng.choice(6, p=prof); op, pos = KCLS[k]
        w = list(w); n = len(w)
        if op == 'sub':
            if n < 3:
                return tuple(w)
            i = 0 if pos == 'ini' else (n - 1 if pos == 'fin' else rng.integers(1, n - 1))
            w[i] = sg[rng.choice(len(sg), p=pw)]
        else:
            s = sg[rng.choice(len(sg), p=pw)]
            if rng.random() < .5 and n >= 3:     # deletion
                i = 0 if pos == 'ini' else (n - 1 if pos == 'fin' else rng.integers(1, n - 1))
                del w[i]
            else:
                i = 0 if pos == 'ini' else (n if pos == 'fin' else rng.integers(1, n))
                w.insert(i, s)
        return tuple(w)
    lex = {}
    for k, s in enumerate(order):
        L_s = [newword() for _ in range(N0 if k == 0 else coin)]
        if k > 0:
            prev = order[:k]
            pool = {}
            for p in prev:
                for w in lex[p]:
                    dd = km[s, p]
                    if w not in pool or dd < pool[w]:
                        pool[w] = dd
            for w, dd in pool.items():
                if rng.random() < q * np.exp(-dd / L):
                    L_s.append(mutate(w) if rng.random() < mu else w)
        lex[s] = L_s
    ci = {c: i for i, c in enumerate(codes)}
    zipf = {s: np.arange(1, len(lex[s]) + 1) ** -zexp for s in lex}
    for s in zipf:
        zipf[s] /= zipf[s].sum()
        rng.shuffle(lex[s])
    out = []
    for d in docs:
        s = ci[d['site']]
        ws = [lex[s][rng.choice(len(lex[s]), p=zipf[s])] for _ in d['words']]
        out.append(dict(d, words=ws))
    return out, t
