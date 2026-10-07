"""v95 cycle 1: THE WRITER MEASURED SOMETHING.  Word value = additive weighted glyph count (non-positional notation).
Momentum statistic (5%-trimmed) lambda = E[(2nd difference)^2] / E[(1st difference)^2] (forward and backward steps pooled) along a series of values (iid 3, random walk 2,
smooth process < 1); G = 1 - lambda.  Gram matrices of difference vectors make each hypothesis a small generalised eigenproblem.
usage: python3 v95_c1.py CORPUS [NHYP]
"""
import sys, os, json, random, time
import numpy as np
from scipy.linalg import eigh
import v95_lib as L

SERIES = ['IN'] + ['%s_%s' % (t, s) for t in ('C0', 'C1', 'C2', 'CL', 'SUM', 'MEAN') for s in ('para', 'page')]
NTWIN = 4
NH_DEFAULT = 3000


def unit_series(pages, key):
    """list of value-vector sequences (each a list of F-vectors) for a series key."""
    out = []
    if key == 'IN':
        for p in pages:
            for l in p['lines']:
                if len(l['w']) >= 3: out.append([L.wvec(w) for w in l['w']])
        return out
    t, scope = key.split('_')
    for p in pages:
        cur = []
        for l in p['lines']:
            if scope == 'para' and l['ps'] and cur:
                out.append(cur); cur = []
            ws = l['w']
            if t in ('C0', 'C1', 'C2'):
                k = int(t[1])
                if len(ws) <= k:
                    if cur: out.append(cur)
                    cur = []; continue
                cur.append(L.wvec(ws[k]))
            elif t == 'CL':
                cur.append(L.wvec(ws[-1]))
            else:
                v = sum(L.wvec(w) for w in ws)
                cur.append(v if t == 'SUM' else v / len(ws))
        if cur: out.append(cur)
    return out


def grams(pages):
    """{(series, half): (A, B, n)} with A = sum d2 d2^T, B = sum d1 d1^T over consecutive triplets."""
    res = {}
    for key in SERIES:
        for h in (0, 1):
            sub = [p for p in pages if L.half(p['id']) == h]
            D1 = []; D2 = []
            for seq in unit_series(sub, key):
                if len(seq) < 3: continue
                X = np.array(seq)
                d1 = X[1:] - X[:-1]; d2 = d1[1:] - d1[:-1]
                D1.append(np.stack([d1[1:], d1[:-1]], 1)); D2.append(d2)   # forward and backward steps per triplet
            if not D1:
                res[(key, h)] = (None, None, 0, None, None); continue
            D1 = np.vstack(D1).astype(np.float32); D2 = np.vstack(D2).astype(np.float32)
            A = D2.T.astype(float) @ D2; B = (D1[:, 0].T.astype(float) @ D1[:, 0] + D1[:, 1].T.astype(float) @ D1[:, 1]) / 2
            res[(key, h)] = (D2, D1, len(D2), A, B)
    return res


def hypotheses(n, seed=95):
    rng = random.Random(seed); H = []
    gl = list(range(L.K - 1))
    for key in SERIES:                          # the full-alphabet fits first (pooled and position-split)
        H.append(dict(i=len(H), key=key, groups=[[pos * L.K + g for pos in range(3)] for g in gl], mode='fit', vals=[]))
        H.append(dict(i=len(H), key=key, groups=[[pos * L.K + g] for g in gl for pos in range(3)], mode='fit', vals=[]))
    for i in range(len(H), n):
        key = rng.choice(SERIES)
        m = rng.randint(2, 16)
        gs = sorted(rng.sample(gl, m))
        split = rng.random() < 0.4
        mode = 'fit' if rng.random() < 0.6 else 'guess'
        if split:
            cols = sorted(rng.sample([pos * L.K + g for g in gs for pos in range(3)], min(3 * m, rng.randint(m, 3 * m))))
            groups = [[c] for c in cols]
        else:
            groups = [[pos * L.K + g for pos in range(3)] for g in gs]
        vals = [rng.choice([1, 5, 10, 50, 100]) for _ in groups]
        H.append(dict(i=i, key=key, groups=groups, mode=mode, vals=vals))
    return H


def proj(groups):
    P = np.zeros((L.F, len(groups)))
    for j, g in enumerate(groups):
        for c in g: P[c, j] = 1
    return P


def lam(A, B, c):
    b = c @ B @ c
    return (c @ A @ c) / b if b > 1e-9 else np.nan


def _solve(a, b):
    s, Vb = np.linalg.eigh(b)                  # whiten in the principal subspace of the step covariance
    k = s > 1e-3 * max(s.max(), 1e-12)
    if k.sum() < 1: return None
    W = Vb[:, k] / np.sqrt(s[k])
    w, U = np.linalg.eigh(W.T @ a @ W)
    return W @ U[:, 0]


def robust_G(D2, D1, cf, trim=0.05):
    """1 - lambda with the largest 5% of squared second and first differences trimmed (wrap-arounds, misreadings)."""
    d2 = (D2 @ cf) ** 2; d1 = np.r_[(D1[:, 0] @ cf) ** 2, (D1[:, 1] @ cf) ** 2]
    q2 = np.sort(d2)[:max(1, int(len(d2) * (1 - trim)))]; q1 = np.sort(d1)[:max(1, int(len(d1) * (1 - trim)))]
    return 1 - q2.mean() / q1.mean() if q1.mean() > 1e-9 else np.nan


def evaluate(G, hyp):
    P = proj(hyp['groups'])
    D2a, D1a, n0, A0, B0 = G[(hyp['key'], 0)]; D2b, D1b, n1, A1, B1 = G[(hyp['key'], 1)]
    if n0 < 30 or n1 < 30: return None
    if hyp['mode'] == 'fit':
        a = P.T @ A0 @ P; b = P.T @ B0 @ P
        c = _solve(a, b)
        if c is None: return None
        # one robust refit: remove the triplets with the largest 5% residuals under the first fit
        cf = (P @ c).astype(np.float32)
        r = (D2a @ cf) ** 2 + ((D1a[:, 0] @ cf) ** 2 + (D1a[:, 1] @ cf) ** 2) / 2
        idx = np.argsort(r)[-max(1, len(r) // 20):]
        X2 = D2a[idx] @ P; Xf = D1a[idx, 0] @ P; Xb = D1a[idx, 1] @ P
        c2 = _solve(a - X2.T @ X2, b - (Xf.T @ Xf + Xb.T @ Xb) / 2)
        if c2 is not None: c = c2
    else:
        c = np.array(hyp['vals'], float)
    cf = (P @ c).astype(np.float32)
    return (robust_G(D2a, D1a, cf), robust_G(D2b, D1b, cf), c)


def run(name, nh):
    out = os.path.join(L.CK, 'c1_%s.json' % name)
    if os.path.exists(out): return
    t0 = time.time()
    C = L.corpus(name)
    variants = [('real', C)] + [('tw%d' % s, L.twin(C, 9500 + s)) for s in range(NTWIN)]
    GR = {v: grams(c) for v, c in variants}
    H = hypotheses(nh)
    rows = []
    for h in H:
        r = {'i': h['i']}
        for v, _ in variants:
            e = evaluate(GR[v], h)
            r[v] = None if e is None else [float(e[0]), float(e[1])]
            if v == 'real' and e is not None: r['c'] = [float('%.6g' % x) for x in e[2]]
        rows.append(r)
    json.dump(dict(name=name, nh=nh, rows=rows, sec=time.time() - t0), open(out, 'w'))
    print(name, 'done', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
