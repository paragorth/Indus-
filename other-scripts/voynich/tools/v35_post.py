"""v35 post-hoc controls on cached behaviour/shape matrices (no new behaviour fits):
  top-K   Mantel restricted to the K most frequent units (K = 23, the Voynich v25 inventory), 2,000 permutations;
  rand-K  mean r over 300 random K-unit subsets (size-matched effect size, no p);
  subset  Mantel on a named subset (e.g. Tengwar without tehtar, Cree without finals).
usage: python3 v35_post.py name [name ...]  -> prints JSON lines, caches post_<name>.pkl"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v35_lib as V, v28_lib as X, v25_lib as L

MODELS = ('ppmi', 'svd', 'potts')


def sub(M, ix):
    return M[np.ix_(ix, ix)]


def subsets(name, c):
    A = c['alph']; h = c.get('hand') or {}
    out = {}
    if name == 'tengwar':
        out['no_tehtar'] = [i for i, g in enumerate(A) if 'MARK' not in h.get(g, {})]
    if name == 'cree':
        out['no_finals'] = [i for i, g in enumerate(A) if 'SMALL' not in h.get(g, {}) and
                            any(k.startswith('ORI_') for k in h.get(g, {}))]
    if name.startswith('copiale'):
        marks = {'CIRC', 'DOT', 'UNDER'}
        out['no_marked'] = [i for i, g in enumerate(A) if not (marks & set(h.get(g, {})))]
    return out


def post(name, K=23, nsub=300, nperm=2000, seed=11):
    fn = f'post_{name}.pkl'
    r = X.load(fn)
    if r is not None:
        return r
    rng = np.random.default_rng(seed)
    c = V.build(name)
    B = X.load(f'beh_{name}.pkl'); S = X.load(f'sims_{name}.pkl')
    n = len(c['alph'])
    order = np.argsort(-B['_freq'])
    res = {}
    keys = [k for k in S if k.startswith('img:') or k == 'hand']
    for k in keys:
        for m in MODELS:
            if n > K:
                ix = np.sort(order[:K])
                r0, p0, _, _ = L.mantel(sub(S[k], ix), sub(B[m], ix), nperm=nperm, rng=rng)
                rs = []
                for _ in range(nsub):
                    jx = np.sort(rng.choice(n, K, replace=False))
                    rs.append(L.spearman_vec(L.triu(sub(S[k], jx)), L.triu(sub(B[m], jx))))
                res[('topK', k, m)] = (r0, p0)
                res[('randK', k, m)] = (float(np.mean(rs)), float(np.std(rs)))
            for sname, ix in subsets(name, c).items():
                ix = np.array(ix)
                r1, p1, _, _ = L.mantel(sub(S[k], ix), sub(B[m], ix), nperm=nperm, rng=rng)
                res[(sname, k, m)] = (r1, p1, len(ix))
    X.save(fn, res)
    return res


if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:]
    with Pool(2) as P:
        for n, res in zip(names, P.imap(post, names)):
            for k, v in sorted(res.items()):
                print(n, k, tuple(round(x, 4) if isinstance(x, float) else x for x in v), flush=True)
