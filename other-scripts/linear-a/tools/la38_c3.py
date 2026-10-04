#!/usr/bin/env python3
"""LA-38 cycle 3: which consonant rows, vowel columns and signs carry the signal?
Score = SEL composite (mean of SEL measure z's, z taken against 5,000 R3 relabelings of that corpus).
  row / column test: the k signs of one row (column) each swap their full (C, V) value with a random sign outside it
                     (2,000 draws); P = share of draws scoring >= the hypothesis.
  sign support:      the sign swaps its value with every other sign in turn; support = share of swaps that lower the score.
Calibration: LB drawn at LA size (true values: rows and columns should mostly pass) and planted errors
(LB-at-LA-size with 6 random value swaps = 12 wrong signs: support should rank the wrong signs low)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la38_common as L
from la38_c1 import corpus
from multiprocessing import Pool

SEL = json.load(open(os.path.join(L.CK, 'sel.json')))


def scorer(c, seed):
    null = L.null_measures(c, 'R3', 5000, seed)
    mu = {k: null[k].mean() for k in SEL}; sd = {k: null[k].std() for k in SEL}

    def f(Cn, Vn):
        m = L.measures(c, Cn, Vn)
        return np.mean([(m[k] - mu[k]) / sd[k] for k in SEL], 0)
    return f


def group_test(c, f, C0, V0, grp, g, rng, N=2000):
    ii = np.where(grp == g)[0]; oo = np.where(grp != g)[0]
    k = len(ii)
    if k == 0 or len(oo) < k:
        return None
    Cn = np.broadcast_to(C0, (N, len(C0))).copy(); Vn = np.broadcast_to(V0, (N, len(V0))).copy()
    for n in range(N):
        tg = rng.choice(oo, k, replace=False)
        Cn[n, ii], Cn[n, tg] = C0[tg], C0[ii]
        Vn[n, ii], Vn[n, tg] = V0[tg], V0[ii]
    obs = f(C0[None], V0[None])[0]
    nul = np.concatenate([f(Cn[b:b + 1000], Vn[b:b + 1000]) for b in range(0, N, 1000)])
    return dict(k=int(k), obs=float(obs), null=float(nul.mean()), p=float(((nul >= obs).sum() + 1) / (N + 1)))


def sign_support(c, f, C0, V0):
    S = len(C0); obs = f(C0[None], V0[None])[0]
    out = []
    for s in range(S):
        Cn = np.broadcast_to(C0, (S, S)).copy(); Vn = np.broadcast_to(V0, (S, S)).copy()
        t = np.arange(S)
        Cn[t, s], Cn[t, t] = C0[t], C0[s]
        Vn[t, s], Vn[t, t] = V0[t], V0[s]
        sc = np.delete(f(Cn, Vn), s)
        out.append(float((sc < obs).mean()))
    return out


def analyse(args):
    tag, plant, seed = args
    c = corpus(tag)
    rng = np.random.default_rng(seed)
    C0, V0 = c.C0.copy(), c.V0.copy()
    wrong = []
    if plant:
        # swap values within 6 random pairs of signs that differ in both C and V
        cand = rng.permutation(len(C0))
        used = set()
        for a in cand:
            if len(wrong) >= 12 or a in used:
                continue
            for b in rng.permutation(len(C0)):
                if b != a and b not in used and C0[a] != C0[b] and V0[a] != V0[b]:
                    C0[a], C0[b] = C0[b], C0[a]; V0[a], V0[b] = V0[b], V0[a]
                    used |= {a, b}; wrong += [int(a), int(b)]; break
    f = scorer(c, seed)
    rows = {c.Clab[g] or '(V)': group_test(c, f, C0, V0, C0, g, rng) for g in np.unique(C0)}
    cols = {c.Vlab[g]: group_test(c, f, C0, V0, V0, g, rng) for g in np.unique(V0)}
    sup = sign_support(c, f, C0, V0)
    signs = {c.signs[i]: dict(val=(c.Clab[C0[i]] + c.Vlab[V0[i]]), n=int(c.u[i]), support=sup[i], wrong=i in wrong)
             for i in range(len(C0))}
    return dict(tag=tag, plant=plant, rows=rows, cols=cols, signs=signs)


if __name__ == '__main__':
    stage = sys.argv[1]
    t = time.time()
    if stage == 'calib':
        jobs = [(f'LBd{d}', False, 30 + d) for d in range(4)] + [(f'LBd{d}', True, 40 + d) for d in range(6)]
    elif stage == 'la':
        jobs = [('LA', False, 99)] + [('LA', True, 100 + d) for d in range(4)]
    with Pool(2) as p:
        res = p.map(analyse, jobs, chunksize=1)
    json.dump(res, open(os.path.join(L.CK, f'c3_{stage}.json'), 'w'), default=float)
    print('done', stage, round(time.time() - t))
