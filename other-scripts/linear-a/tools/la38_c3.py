#!/usr/bin/env python3
"""LA-38 cycle 3: which consonant rows, vowel columns and signs carry the signal?
Two scores, each the mean z (against 5,000 relabelings of that corpus) of the measures that passed on full LB:
  SC = ocpC, ocpP, vinit (consonant-sensitive; R3 / R2b set);  SV = comp (vowel-sensitive; R2a set).
  row test (SC):    the k signs of one consonant row each swap their full (C, V) value with a random sign outside it
                    (2,000 draws); P = share of draws scoring >= the hypothesis.
  column test (SV): the same for one vowel column.
  C-support (SC):   the sign exchanges its consonant with each sign of the same vowel column (e.g. RO <-> KO, giving
                    KO, RO); share of exchanges that lower SC.
  V-support (SV):   the sign exchanges its vowel with each sign of the same consonant row; share that lower SV.
  (Exchanges keep all values distinct; a first version that swapped with any sign created homophones, which the
   plug-in compressibility rewards: fixed before any LA result was read.)
Calibration: LB drawn at LA size (true values: rows and columns should mostly pass) and planted errors
(LB-at-LA-size with 6 random value swaps = 12 wrong signs: support should rank the wrong signs low)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la38_common as L
from la38_c1 import corpus
from multiprocessing import Pool

SC = ['ocpC', 'ocpP', 'vinit']; SV = ['comp']


def scorer(c, seed, keys, tier, base):
    null = L.null_measures(c, tier, 5000, seed, base=base)
    mu = {k: null[k].mean() for k in keys}; sd = {k: null[k].std() for k in keys}

    def f(Cn, Vn):
        m = L.measures(c, Cn, Vn)
        return np.mean([(m[k] - mu[k]) / sd[k] for k in keys], 0)
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


def sign_support(c, f, C0, V0, which):
    S = len(C0); obs = f(C0[None], V0[None])[0]
    out = []
    for s in range(S):
        X0 = C0 if which == 'C' else V0
        other = V0 if which == 'C' else C0
        # exchange with a sign of the same column (C) or row (V): values stay distinct (no homophones created)
        t = np.where((X0 != X0[s]) & (other == other[s]))[0]
        if len(t) == 0:
            out.append(float('nan')); continue
        X = np.broadcast_to(X0, (len(t), S)).copy()
        X[np.arange(len(t)), s] = X0[t]; X[np.arange(len(t)), t] = X0[s]
        sc = f(X, np.broadcast_to(V0, X.shape)) if which == 'C' else f(np.broadcast_to(C0, X.shape), X)
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
    fC = scorer(c, seed, SC, 'R3', (C0, V0)); fV = scorer(c, seed, SV, 'R2a', (C0, V0))
    rows = {c.Clab[g] or '(V)': group_test(c, fC, C0, V0, C0, g, rng) for g in np.unique(C0)}
    cols = {c.Vlab[g]: group_test(c, fV, C0, V0, V0, g, rng) for g in np.unique(V0)}
    supC = sign_support(c, fC, C0, V0, 'C'); supV = sign_support(c, fV, C0, V0, 'V')
    signs = {c.signs[i]: dict(val=(c.Clab[C0[i]] + c.Vlab[V0[i]]), n=int(c.u[i]), supC=supC[i], supV=supV[i],
                              wrong=i in wrong) for i in range(len(C0))}
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
