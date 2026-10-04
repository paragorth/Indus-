"""LA-31 statistics: site x site sharing z-scores (document-label permutation nulls) and
Mantel / partial Mantel tests against travel-time matrices."""
import numpy as np
from scipy.stats import rankdata
from la31_common import SITES, load_docs, euclid  # noqa

LAYERS = ('W', 'S', 'L', 'E')


def doc_matrices(docs, codes, rare_skip=40):
    """Per document: sets of word / rare-sign / logogram types and entry feature sums."""
    import collections
    sc = collections.Counter(s for d in docs for s in d['signs'])
    common = {s for s, _ in sc.most_common(rare_skip)}
    vocab = {}
    def ids(xs, tag):
        return sorted({vocab.setdefault((tag, x), len(vocab)) for x in xs})
    rows = []
    for d in docs:
        rows.append(dict(W=ids(d['words'], 'W'), S=ids([s for s in d['signs'] if s not in common], 'S'),
                         L=ids(d['logos'], 'L')))
    tag = np.array([t for t, _ in sorted(vocab, key=lambda k: vocab[k])]) if vocab else np.array([])
    keys = [None] * len(vocab)
    for k, v in vocab.items():
        keys[v] = k
    tags = np.array([k[0] for k in keys])
    from scipy.sparse import csr_matrix
    r, c = [], []
    for i, row in enumerate(rows):
        for L in ('W', 'S', 'L'):
            r += [i] * len(row[L]); c += row[L]
    M = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(docs), len(vocab)))
    F = np.array([d['feat'] for d in docs])
    return M, tags, F


def site_stats(M, tags, F, lab, K):
    """Shared type counts per layer (K x K) and entry-structure similarity."""
    from scipy.sparse import csr_matrix
    A = csr_matrix((np.ones(len(lab), np.float32), (lab, np.arange(len(lab)))), shape=(K, len(lab)))
    B = (A @ M)
    B.data = np.minimum(B.data, 1)
    out = {}
    for L in ('W', 'S', 'L'):
        Bl = B[:, np.nonzero(tags == L)[0]]
        out[L] = (Bl @ Bl.T).toarray()
    Fs = A @ F
    # ratios: nums/word, frac/num, logos/word, words/line, word->num rate, mean word length
    eps = 1.0
    R = np.c_[Fs[:, 1] / (Fs[:, 0] + eps), Fs[:, 2] / (Fs[:, 1] + eps), Fs[:, 3] / (Fs[:, 0] + eps),
              Fs[:, 0] / (Fs[:, 4] + eps), Fs[:, 5] / (Fs[:, 0] + eps), Fs[:, 6] / (Fs[:, 7] + eps)]
    out['E'] = -np.sqrt((((R[:, None] - R[None]) / (R.std(0) + 1e-9)) ** 2).sum(-1))
    return out


def sharing_z(docs, codes, nperm=500, strat=True, rng=None, M=None, tags=None, F=None):
    rng = rng or np.random.default_rng(0)
    K = len(codes); ci = {c: i for i, c in enumerate(codes)}
    if M is None:
        M, tags, F = doc_matrices(docs, codes)
    lab = np.array([ci[d['site']] for d in docs])
    obs = site_stats(M, tags, F, lab, K)
    grp = np.array([d['support'] if strat else '' for d in docs])
    acc = {L: [] for L in obs}
    for _ in range(nperm):
        l2 = lab.copy()
        for g in np.unique(grp):
            ix = np.nonzero(grp == g)[0]
            l2[ix] = rng.permutation(lab[ix])
        s = site_stats(M, tags, F, l2, K)
        for L in s:
            acc[L].append(s[L])
    Z = {}
    for L in obs:
        a = np.array(acc[L]); mu = a.mean(0); sd = a.std(0)
        Z[L] = np.where(sd > 0, (obs[L] - mu) / np.maximum(sd, 1e-9), 0.0)
    return Z, obs


def upper(X, sel=None):
    K = X.shape[0]
    iu = np.triu_indices(K, 1)
    v = X[iu]
    return v if sel is None else v[sel]


def mantel(Z, P, nperm=5000, rng=None, covar=None, pairmask=None):
    """Spearman Mantel between Z (similarity) and P (predictor: closeness), site permutation null.
    covar: matrix regressed out of both (partial Mantel on ranks, Smouse-style permuting Z)."""
    rng = rng or np.random.default_rng(1)
    K = Z.shape[0]
    iu = np.triu_indices(K, 1)
    m = np.ones(len(iu[0]), bool) if pairmask is None else pairmask
    def prep(X):
        return rankdata(X[iu][m])
    p = prep(P)
    if covar is not None:
        c = prep(covar); C = np.c_[np.ones_like(c), c]
        res = lambda y: y - C @ np.linalg.lstsq(C, y, rcond=None)[0]  # noqa
        p = res(p)
    else:
        res = lambda y: y  # noqa
    def r_of(Zm):
        z = res(prep(Zm))
        return np.corrcoef(z, p)[0, 1]
    r0 = r_of(Z)
    if not np.isfinite(r0):
        return float('nan'), 1.0, np.zeros(1)
    null = np.empty(nperm)
    for k in range(nperm):
        q = rng.permutation(K)
        null[k] = r_of(Z[np.ix_(q, q)])
    return r0, (np.sum(null >= r0) + 1) / (nperm + 1), null
