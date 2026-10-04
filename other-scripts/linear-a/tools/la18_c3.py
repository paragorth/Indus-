#!/usr/bin/env python3
"""LA-18 cycle 3: do ENDINGS avoid each other on a tablet? (case harmony / slot exclusivity of endings)
If a tablet is written 'in one case', the final signs of its words should segregate: words ending in X and
words ending in Y (alternative endings of one slot) should avoid each other on tablets, endings should
self-attract. Statistic per ending pair (x, y): number of co-present type pairs on a tablet whose last signs
are x and y, vs the curveball null (site strata; word frequencies and tablet sizes fixed; endings travel with
their words). Then an annealed partition of the top endings into g classes maximising within-class excess
(segregation), calibrated by the same anneal on null samples.
Controls: LB full and LA-sized LB; LA with endings re-assigned at random across types (kills ending identity,
keeps incidence); planted case harmony in LA (a new final sign added to every word of 10-20 % of tablets).
usage: la18_c3.py lb | la
"""
import sys, json, random, collections, itertools
import numpy as np
from la18_common import *


def ending_pairs_count(sets, end, E):
    """matrix C[x, y] (x <= y) of co-present type pairs on a tablet by endings; E = list of tracked endings"""
    k = len(E) + 1; ix = {e: i for i, e in enumerate(E)}
    C = np.zeros((k, k), np.int32)
    for s in sets:
        es = [ix.get(end[w], k - 1) for w in s]
        cnt = np.bincount(es, minlength=k)
        C += np.outer(cnt, cnt)
        C[np.diag_indices(k)] -= cnt  # unordered pairs of distinct types: diag n(n-1), off 2*n_x*n_y counted twice
    return C


def run(S, site, end, K, nsamp, seed, ngrp=(2, 3, 4), nanneal=200):
    ec = collections.Counter(end[w] for s in S for w in s)
    E = [e for e, _ in ec.most_common(K)]
    obs = ending_pairs_count(S, end, E)
    cb = Curveball(S, site, seed)
    N = np.array([ending_pairs_count(cur, end, E) for cur in cb.samples(nsamp)], float)
    mu = N.mean(0); sd = N.std(0) + 1e-9
    k = len(E)
    Z = (obs - mu) / sd
    # homogeneity: total same-ending pairs (tracked endings only)
    diag = lambda M: np.trace(M[..., :k, :k], axis1=-2, axis2=-1) if M.ndim == 3 else np.trace(M[:k, :k])
    Hn = np.array([np.trace(m[:k, :k]) for m in N]); Ho = np.trace(obs[:k, :k])
    H = dict(O=float(Ho), E=round(float(Hn.mean()), 1), z=round(float((Ho - Hn.mean()) / (Hn.std() + 1e-9)), 2),
             p_hi=round(float(((Hn >= Ho).sum() + 1) / (nsamp + 1)), 4), p_lo=round(float(((Hn <= Ho).sum() + 1) / (nsamp + 1)), 4))
    # most avoiding / attracting off-diagonal ending pairs, FWER by max |z| over null rows
    Zn = (N - mu) / sd
    iu = np.triu_indices(k, 1)
    mxn = np.abs(Zn[:, iu[0], iu[1]]).max(1)
    off = sorted([(float(Z[i, j]), E[i], E[j], int(obs[i, j]), round(float(mu[i, j]), 1)) for i, j in zip(*iu)])
    pairs = [dict(x=a, y=b, z=round(z, 2), O=o // 2, E=round(e / 2, 1), fwer=round(float(((mxn >= abs(z)).sum() + 1) / (nsamp + 1)), 4))
             for z, a, b, o, e in off[:8] + off[-8:]]
    dz = [dict(x=E[i], z=round(float(Z[i, i]), 2), O=int(obs[i, i]) // 2, E=round(float(mu[i, i]) / 2, 1)) for i in range(k)]
    # annealed partition: maximise sum over within-class OFF-diagonal pairs of (O - mu) / sqrt(sum var)
    Vu = (sd ** 2)[iu]
    def score_vec(M, lab):
        lab = np.asarray(lab); same = (lab[iu[0]] == lab[iu[1]])
        return float(((M - mu)[iu][same]).sum() / np.sqrt(Vu[same].sum() + 1e-9))
    def anneal(M, g, rnd, steps=3000, restarts=3):
        """identical search effort for the real matrix and every null matrix (v1 bug: nulls got less)"""
        best = (-1e9, None)
        for _r in range(restarts):
            lab = [rnd.randrange(g) for _ in range(k)]; cur = score_vec(M, lab)
            for t in range(steps):
                T = 1.0 * (1 - t / steps) + 0.01
                i = rnd.randrange(k); old = lab[i]; lab[i] = rnd.randrange(g)
                new = score_vec(M, lab)
                if new >= cur or rnd.random() < np.exp((new - cur) / T): cur = new
                else: lab[i] = old
                if cur > best[0]: best = (cur, lab[:])
        return best
    rnd = random.Random(seed)
    part = {}
    for g in ngrp:
        b = anneal(obs, g, rnd)
        nb = []
        for r in range(nanneal):
            nb.append(anneal(N[r], g, rnd)[0])
        nb = np.array(nb)
        part[g] = dict(score=round(float(b[0]), 2), null_mean=round(float(nb.mean()), 2), null_max=round(float(nb.max()), 2),
                       p=round(float(((nb >= b[0]).sum() + 1) / (len(nb) + 1)), 4),
                       classes=[[E[i] for i in range(k) if b[1][i] == c] for c in range(g)])
    return dict(K=k, endings=E, homogeneity=H, diag=dz, pairs=pairs, partition=part)


def ends(S):
    return {w: w[-1] for s in S for w in s}


def main():
    job = sys.argv[1]; out = {}
    if job == 'lb':
        docs = lb_corpus(); S, site, _ = units(docs)
        r = run(S, site, ends(S), 15, 500, 31, nanneal=200); out['LB_full'] = r; dump(out, 'c3_lb.json')
        print('LB', r['homogeneity'], {g: (v['score'], v['p']) for g, v in r['partition'].items()}, flush=True)
        subs = []
        for rep in range(6):
            rr = random.Random(3100 + rep); idx = rr.sample(range(len(S)), 245)
            S2 = [S[i] for i in idx]; s2 = [site[i] for i in idx]
            r2 = run(S2, s2, ends(S2), 12, 500, 3200 + rep, nanneal=200)
            subs.append(r2); print('LB245', rep, r2['homogeneity'], {g: (v['score'], v['p']) for g, v in r2['partition'].items()}, flush=True)
        out['LB245'] = subs; dump(out, 'c3_lb.json')
    else:
        docs = la_corpus(); S, site, _ = units(docs)
        r = run(S, site, ends(S), 12, 1000, 41); out['LA'] = r; dump(out, 'c3_la.json')
        print('LA', r['homogeneity'], {g: (v['score'], v['p'], v['classes']) for g, v in r['partition'].items()}, flush=True)
        print(r['pairs'], flush=True)
        neg = []
        for rep in range(5):
            rr = random.Random(4100 + rep)
            e0 = ends(S); ws = list(e0); vals = [e0[w] for w in ws]; rr.shuffle(vals)
            r2 = run(S, site, dict(zip(ws, vals)), 12, 500, 4200 + rep, nanneal=200)
            neg.append(r2); print('neg', rep, r2['homogeneity'], {g: (v['score'], v['p']) for g, v in r2['partition'].items()}, flush=True)
        out['LA_endshuf'] = neg; dump(out, 'c3_la.json')
        pl = []
        for rep, frac in enumerate([0.1, 0.1, 0.2, 0.2]):
            rr = random.Random(4300 + rep)
            pick = set(rr.sample(range(len(S)), int(frac * len(S))))
            S2 = [frozenset(w + ('#C',) for w in s) if i in pick else s for i, s in enumerate(S)]
            r2 = run(S2, site, ends(S2), 12, 500, 4400 + rep, nanneal=200)
            hc = [d for d in r2['diag'] if d['x'] == '#C']
            pl.append(dict(frac=frac, res=r2, planted_diag=hc)); print('plant', frac, r2['homogeneity'], hc,
                                                                      {g: (v['score'], v['p']) for g, v in r2['partition'].items()}, flush=True)
        out['LA_planted'] = pl; dump(out, 'c3_la.json')


if __name__ == '__main__':
    main()
