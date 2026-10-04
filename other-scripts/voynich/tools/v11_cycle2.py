"""v11 cycle 2: is the learned space a geometry, and do lines move through it like walks?
Uses cycle-1 embeddings (dist2, fitted on train pages) and evaluates on held-out pages.
 (a) lag-displacement curve R(k) = mean sq distance at lag k / mean sq distance of random test-token pairs, k=1..6;
     null: words shuffled within each held-out line (embedding fixed), 200x.
 (b) model-free graph geometry on train counts (top 300 types, symmetric PPMI kNN graph, k=8):
     clustering coefficient vs degree-preserving rewiring; Isomap variance share in top 2/3 dims; ball growth.
 (c) walk statistics in the 2D map: step-length tail (share of steps > median random-pair distance = 'teleports'),
     return-to-start (last word of a line nearer the first than the line's midpoint word).
 (d) composition test: lag-2 held-out gain predicted by squaring the fitted lag-1 dist2 / bilin16 kernels
     vs a direct lag-2 bilin16 fit."""
import sys, os, time, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_lib as L
import numpy as np
from multiprocessing import Pool
from collections import deque

CORPORA = ['Planted-grid-30', 'Planted-grid-60', 'Planted-graph-30', 'Voynich-ZL', 'Voynich-IT', 'Latin-Isidore',
           'Italian-Manzoni', 'Spanish-Cervantes', 'Shuffle-line', 'SelfCitation', 'Voynich-A', 'Voynich-B']
LN2 = np.log(2)


def lag_curve(lines, idx, X, K=6):
    num = np.zeros(K + 1); den = np.zeros(K + 1)
    toks = []
    for l in lines:
        ids = [idx.get(w, -1) for w in l['words']]
        for i, a in enumerate(ids):
            if a < 0: continue
            toks.append(a)
            for k in range(1, K + 1):
                if i + k < len(ids) and ids[i + k] >= 0:
                    b = ids[i + k]; num[k] += ((X[a] - X[b]) ** 2).sum(); den[k] += 1
    toks = np.array(toks); rng = np.random.default_rng(0)
    a = rng.choice(toks, 20000); b = rng.choice(toks, 20000)
    base = ((X[a] - X[b]) ** 2).sum(1).mean()
    return (num[1:] / np.maximum(den[1:], 1) / base).tolist(), base


def walk_stats(lines, idx, X, base):
    med = np.sqrt(base)  # rms random-pair distance
    steps, ret, retn = [], 0, 0
    for l in lines:
        ids = [idx.get(w, -1) for w in l['words']]
        for a, b in zip(ids, ids[1:]):
            if a >= 0 and b >= 0: steps.append(np.sqrt(((X[a] - X[b]) ** 2).sum()))
        v = [i for i in ids if i >= 0]
        if len(v) >= 5:
            mid = v[len(v) // 2]
            ret += ((X[v[-1]] - X[v[0]]) ** 2).sum() < ((X[mid] - X[v[0]]) ** 2).sum(); retn += 1
    s = np.array(steps)
    return {'teleport_share': float((s > med).mean()), 'step_med_over_rms': float(np.median(s) / med),
            'step_p90_over_rms': float(np.percentile(s, 90) / med), 'return_share': ret / max(retn, 1)}


def ppmi_graph(C, top=300, k=8):
    A = (C + C.T)[:top, :top].astype(float); np.fill_diagonal(A, 0)
    r = A.sum(1) + 1e-9; E = np.outer(r, r) / r.sum()
    with np.errstate(divide='ignore'):
        P = np.where(A >= 2, np.maximum(np.log(A / E), 0), 0)
    G = np.zeros_like(P, bool)
    for i in range(top):
        nn = np.argsort(-P[i])[:k]
        G[i, nn[P[i, nn] > 0]] = True
    return G | G.T


def clustering(G):
    Gi = G.astype(float); deg = Gi.sum(1)
    tri = np.einsum('ij,jk,ki->i', Gi, Gi, Gi) / 2
    poss = deg * (deg - 1) / 2
    return float(tri.sum() / max(poss.sum(), 1))


def rewire(G, rng, nswap=None):
    edges = [tuple(e) for e in np.argwhere(np.triu(G, 1))]
    E = set(edges); el = list(edges); m = len(el)
    for _ in range(nswap or 10 * m):
        i, j = rng.integers(m, size=2)
        (a, b), (c, d) = el[i], el[j]
        if len({a, b, c, d}) < 4: continue
        n1, n2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if n1 in E or n2 in E: continue
        E.discard(el[i]); E.discard(el[j]); E.add(n1); E.add(n2); el[i], el[j] = n1, n2
    H = np.zeros_like(G)
    for a, b in E: H[a, b] = H[b, a] = True
    return H


def bfs_all(G):
    n = len(G); nb = [np.flatnonzero(G[i]) for i in range(n)]; D = np.full((n, n), np.inf)
    for s in range(n):
        D[s, s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if D[s, v] == np.inf: D[s, v] = D[s, u] + 1; q.append(v)
    return D


def isomap_share(G):
    D = bfs_all(G)
    # largest component
    comp = np.isfinite(D[np.argmax(np.isfinite(D).sum(1))]); D = D[np.ix_(comp, comp)]
    n = len(D); J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J; w = np.sort(np.linalg.eigvalsh(B))[::-1]
    pos = w[w > 0]
    growth = [float((D <= r).sum(1).mean()) for r in (1, 2, 3, 4)]
    return {'iso_top2': float(pos[:2].sum() / pos.sum()), 'iso_top3': float(pos[:3].sum() / pos.sum()),
            'comp_size': int(n), 'diam': float(D.max()), 'ball': growth}


def job(name):
    ck = f'c2_{name}.json'
    r = L.jload(ck)
    if r: return r
    t = time.time(); lines, truth = L.corpus(name)
    out = {'corpus': name, 'folds': []}
    for fold in (0, 1):
        c1 = L.jload(f'c1_{name}_{fold}.json'); voc = c1['vocab']; idx = {w: i for i, w in enumerate(voc)}
        X = np.array(c1['emb']['2']); tr, te = L.split(lines, fold)
        R, base = lag_curve(te, idx, X)
        rng = random.Random(fold); null = []
        for _ in range(200):
            sh = [dict(l, words=rng.sample(l['words'], len(l['words']))) for l in te]
            null.append(lag_curve(sh, idx, X)[0])
        null = np.array(null)
        z = ((np.array(R) - null.mean(0)) / null.std(0)).tolist()
        ws = walk_stats(te, idx, X, base)
        nullw = [walk_stats([dict(l, words=rng.sample(l['words'], len(l['words']))) for l in te], idx, X, base) for _ in range(20)]
        ws_null = {k: float(np.mean([w[k] for w in nullw])) for k in ws}
        # composition test (d)
        C1 = L.bigram_counts(tr, idx, 1); C2tr = L.bigram_counts(tr, idx, 2); C2te = L.bigram_counts(te, idx, 2)
        u2 = L.fit(C2tr, 'uni', 0, iters=300); b2 = L.test_ll(u2, C2te)
        f2 = L.fit(C2tr, 'bilin', 16, iters=800); direct = (L.test_ll(f2, C2te) - b2) / C2te.sum() / LN2
        comp = {}
        for kind, d in (('dist', 2), ('bilin', 16)):
            init = L.spectral_init(C1, 2) if kind == 'dist' else None
            f1 = L.fit(C1, kind, d, iters=800, init=init)
            P2 = f1['probs'] @ f1['probs']
            comp[f'{kind}{d}'] = float(((C2te * np.log(P2)).sum() - b2) / C2te.sum() / LN2)
        G = ppmi_graph(C1)
        cl = clustering(G); grng = np.random.default_rng(fold)
        cl_null = [clustering(rewire(G, grng)) for _ in range(10)]
        iso = isomap_share(G)
        iso_null = isomap_share(rewire(G, grng))
        out['folds'].append({'fold': fold, 'R': R, 'R_null': null.mean(0).tolist(), 'R_z': z, 'walk': ws, 'walk_null': ws_null,
                             'lag2_direct_bilin16': direct, 'lag2_composed': comp,
                             'clustering': cl, 'clustering_null': float(np.mean(cl_null)), 'clustering_null_sd': float(np.std(cl_null)),
                             'iso': iso, 'iso_rewired': iso_null})
    out['secs'] = time.time() - t
    L.jdump(out, ck)
    return out


if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(job, CORPORA):
            f = r['folds'][0]
            print(r['corpus'], 'R', [round(x, 3) for x in f['R']], 'z', [round(x, 1) for x in f['R_z'][:3]],
                  'walk', {k: round(v, 3) for k, v in f['walk'].items()},
                  'lag2 direct %.3f comp %s' % (f['lag2_direct_bilin16'], {k: round(v, 3) for k, v in f['lag2_composed'].items()}),
                  'clust %.3f null %.3f' % (f['clustering'], f['clustering_null']),
                  'iso2 %.2f iso3 %.2f (rewired %.2f)' % (f['iso']['iso_top2'], f['iso']['iso_top3'], f['iso_rewired']['iso_top2']),
                  'ball', [round(x, 1) for x in f['iso']['ball']], flush=True)
