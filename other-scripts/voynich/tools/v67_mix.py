"""v67 learned line compressor: mixture of multinomials with ONE latent token per line.
Each word contributes features (prefix-2, suffix-2, glyph bigrams, whole word); the line's words are emitted
i.i.d. from the line's latent class. The MAP class of each line is its token."""
import numpy as np
from collections import Counter


def feats(w, kinds):
    f = []
    if 'p' in kinds: f.append('p:' + w[:2])
    if 's' in kinds: f.append('s:' + w[-2:])
    if 'w' in kinds: f.append('w:' + w)
    if 'b' in kinds: f += ['b:' + w[i:i + 2] for i in range(len(w) - 1)]
    return f


def matrix(C, kinds, minc=3):
    cnt = Counter(x for L in C for w in L['words'] for x in feats(w, kinds))
    voc = {x: i for i, x in enumerate(k for k, v in cnt.items() if v >= minc)}
    M = np.zeros((len(C), len(voc)), dtype=np.float32)
    for i, L in enumerate(C):
        for w in L['words']:
            for x in feats(w, kinds):
                j = voc.get(x)
                if j is not None: M[i, j] += 1
    return M


def fit(M, K, seed=0, iters=60, alpha=0.1):
    rng = np.random.default_rng(seed)
    n, F = M.shape
    R = rng.dirichlet(np.ones(K), size=n).astype(np.float32)
    for _ in range(iters):
        phi = R.T @ M + alpha
        phi /= phi.sum(1, keepdims=True)
        pi = R.sum(0) + 1.0; pi /= pi.sum()
        LL = M @ np.log(phi).T + np.log(pi)
        LL -= LL.max(1, keepdims=True)
        R = np.exp(LL); R /= R.sum(1, keepdims=True)
    return R.argmax(1)
