"""pe14 hidden-state models of a woven tablet (batched EM, numpy).

Observation per unit = tuple of categorical features (independent given the state).
Models with K states:
  MIX  : states iid each unit (a flat mixture; no order)
  FREE : full K x K transition matrix (non-periodic, MORE capacity than CYC)
  CYC  : cyclic weave, state k -> k+1 mod K with prob 1-eps, else uniform; phase of
         the first unit unknown (uniform start): a period-K weave
Fitted by EM with many random restarts; scored by held-out log-likelihood per unit.
"""
import numpy as np


def pack(seqs, nf):
    """seqs: list of list of tuples -> X (N, T, nf) int, M (N, T) bool"""
    N = len(seqs)
    T = max(len(s) for s in seqs)
    X = np.zeros((N, T, nf), int)
    M = np.zeros((N, T), bool)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
        M[i, :len(s)] = True
    return X, M


def logB(X, M, th):
    """th: list over features of (K, V_f) log-probs -> (N, T, K)"""
    L = 0
    for f, t in enumerate(th):
        L = L + t[:, X[:, :, f]].transpose(1, 2, 0)
    return np.where(M[:, :, None], L, 0.0)


def fwd_bwd(lb, M, A, pi, need_post=True):
    N, T, K = lb.shape
    mx = lb.max(2, keepdims=True)
    B = np.exp(lb - mx)
    ll = mx[:, :, 0].sum(1) * 0  # accumulate
    al = np.zeros((N, T, K))
    c = np.zeros((N, T))
    a = pi[None, :] * B[:, 0]
    c[:, 0] = a.sum(1)
    al[:, 0] = a / c[:, :1]
    for t in range(1, T):
        a = (al[:, t - 1] @ A) * B[:, t]
        s = a.sum(1)
        m = M[:, t]
        a = np.where(m[:, None], a / np.where(s > 0, s, 1)[:, None], al[:, t - 1])
        al[:, t] = a
        c[:, t] = np.where(m, s, 1.0)
    ll = (np.log(c) + np.where(M, mx[:, :, 0], 0)).sum(1)
    if not need_post:
        return ll, None, None
    be = np.ones((N, T, K))
    xi = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        m = M[:, t + 1]
        bb = B[:, t + 1] * be[:, t + 1]
        b = (bb @ A.T) / c[:, t + 1][:, None]
        be[:, t] = np.where(m[:, None], b, 1.0)
        x = al[:, t][:, :, None] * A[None] * (bb / c[:, t + 1][:, None])[:, None, :]
        xi += x[m].sum(0)
    g = al * be
    g = g / g.sum(2, keepdims=True)
    g = np.where(M[:, :, None], g, 0)
    return ll, g, xi


def trans(kind, K, P):
    if kind == 'MIX':
        return np.tile(P['pi'], (K, 1)), P['pi']
    if kind == 'FREE':
        return P['A'], P['pi']
    e = P['eps']
    A = (1 - e) * np.roll(np.eye(K), 1, axis=1) + e / K
    return A, np.full(K, 1.0 / K)


def fit(X, M, V, kind, K, rng, iters=30, alpha=0.1):
    nf = len(V)
    th = [np.log(rng.dirichlet(np.ones(v), K)) for v in V]
    P = {'pi': rng.dirichlet(np.ones(K)), 'A': rng.dirichlet(np.ones(K), K), 'eps': rng.uniform(0.05, 0.5)}
    last = -np.inf
    for it in range(iters):
        A, pi = trans(kind, K, P)
        lb = logB(X, M, th)
        ll, g, xi = fwd_bwd(lb, M, A, pi)
        tot = ll.sum()
        # M step
        for f in range(nf):
            cnt = np.zeros((K, V[f]))
            for v in range(V[f]):
                cnt[:, v] = g[X[:, :, f] == v].sum(0)
            th[f] = np.log((cnt + alpha) / (cnt + alpha).sum(1, keepdims=True))
        if kind == 'MIX':
            P['pi'] = (g[M].sum(0) + 1) / (g[M].sum() + K)
        elif kind == 'FREE':
            P['pi'] = (g[:, 0].sum(0) + 1) / (g[:, 0].sum() + K)
            P['A'] = (xi + 0.5) / (xi + 0.5).sum(1, keepdims=True)
        else:
            on = (xi * np.roll(np.eye(K), 1, axis=1)).sum()
            # E[non-cyclic] = eps*(K-1)/K * n ; solve eps
            n = xi.sum()
            off = n - on
            P['eps'] = float(np.clip(off / n * K / max(K - 1, 1), 1e-3, 0.999)) if n > 0 else 0.5
        if tot - last < 1e-3 * abs(tot) * 1e-3:
            break
        last = tot
    A, pi = trans(kind, K, P)
    return {'th': th, 'A': A, 'pi': pi, 'eps': P.get('eps'), 'train_ll': last}


def score(X, M, model):
    lb = logB(X, M, model['th'])
    ll, _, _ = fwd_bwd(lb, M, model['A'], model['pi'], need_post=False)
    return ll.sum()
