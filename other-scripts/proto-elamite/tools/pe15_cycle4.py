"""pe15 cycle 4: massive random office guessing, with reserved tablets.

Sign web (PE_SIGN; controls UR3_SIGN cut to PE size, LB_SIGN).  20% of tablets are
reserved.  On the other 80%, 5 tablet folds score an "office partition" g of the
consumers by held-out bits of P(r | c) = (n_{g(c),r} + a*pop_r) / (n_{g(c)} + a),
a in {1,10}: an office profile over raw resources (no resource blocks).
Hypotheses: 4,000 uniformly random partitions (K 2-8), 60 greedy-climbed partitions
(K 2-5, random start, single-node moves for 2 sweeps on the training folds), and the cycle-2
SBM consumer blocks.  The top 20 by fold score are re-tested on the reserved tablets
against DEG (popularity) and a consumer's own history.  Control: the whole search on
the same corpus with resources shuffled within tablet (family-wise null for the best
gain), and on the event-re-paired corpus.
"""
import json, os, sys, random
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe15_common import *
from pe15_cycle2 import shuffle_global, shuffle_tablet

NRAND = int(os.environ.get('NRAND', 4000))
NCLIMB = int(os.environ.get('NCLIMB', 30))


def prep(ev, seed):
    tabs = sorted({e[2] for e in ev})
    random.Random(seed).shuffle(tabs)
    res = set(tabs[:len(tabs) // 5])
    fit = [e for e in ev if e[2] not in res]
    rv = [e for e in ev if e[2] in res]
    W, C, R = matrix(fit, min_c=2, min_r=1)
    ci = {c: i for i, c in enumerate(C)}; ri = {r: i for i, r in enumerate(R)}
    ft = sorted({e[2] for e in fit})
    random.Random(seed + 1).shuffle(ft)
    fo = {t: i % 5 for i, t in enumerate(ft)}
    folds = []
    for f in range(5):
        Wtr = np.zeros_like(W); te = []
        for c, r, t in fit:
            if c in ci and r in ri:
                if fo[t] == f:
                    te.append((ci[c], ri[r]))
                else:
                    Wtr[ci[c], ri[r]] += 1
        folds.append((Wtr.astype(float), np.array(te)))
    Wall = W.astype(float)
    rte = np.array([(ci[c], ri[r]) for c, r, t in rv if c in ci and r in ri])
    return C, R, folds, Wall, rte


def score(W, te, g, K, a):
    pop = (W.sum(0) + 0.1) / (W.sum() + 0.1 * W.shape[1])
    O = np.zeros((K, W.shape[0])); O[g, np.arange(W.shape[0])] = 1
    G = O @ W
    P = (G + a * pop) / (G.sum(1, keepdims=True) + a)
    return -np.log2(P[g[te[:, 0]], te[:, 1]]).mean()


def cv(folds, g, K):
    return min(np.mean([score(W, te, g, K, a) for W, te in folds]) for a in (1, 10))


def baselines(W, te):
    pop = (W.sum(0) + 0.1) / (W.sum() + 0.1 * W.shape[1])
    deg = -np.log2(pop[te[:, 1]]).mean()
    best = 9e9
    for c in (1, 3, 10, 30):
        H = (W + c * pop) / (W.sum(1, keepdims=True) + c)
        best = min(best, -np.log2(H[te[:, 0], te[:, 1]]).mean())
    return deg, best


def climb(folds, n, K, rng, sweeps=3):
    g = rng.integers(0, K, n)
    cur = cv(folds, g, K)
    for s in range(sweeps):
        for i in rng.permutation(n):
            old = g[i]
            bestk, bestv = old, cur
            for k in range(K):
                if k == old:
                    continue
                g[i] = k
                v = cv(folds, g, K)
                if v < bestv - 1e-9:
                    bestk, bestv = k, v
            g[i] = bestk
            cur = bestv
    return g, cur


def job(args):
    name, ev, kind, seed, sbm_part = args
    rng = np.random.default_rng(seed)
    if kind == 'glob':
        ev = shuffle_global(ev, random.Random(seed))
    elif kind == 'tab':
        ev = shuffle_tablet(ev, random.Random(seed))
    C, R, folds, W, rte = prep(ev, 2026)
    n = len(C)
    deg_cv = np.mean([baselines(Wf, te)[0] for Wf, te in folds])
    hyps = []
    for h in range(NRAND):
        K = int(rng.integers(2, 9))
        g = rng.integers(0, K, n)
        hyps.append(('rand', K, g, cv(folds, g, K)))
    for h in range(NCLIMB):
        K = int(rng.integers(2, 6))
        g, v = climb(folds, n, K, rng, sweeps=2)
        hyps.append(('climb', K, g, v))
    if sbm_part is not None and kind == 'real':
        g = np.array([sbm_part.get(c, 0) for c in C])
        K = int(g.max()) + 1
        hyps.append(('sbm', K, g, cv(folds, g, K)))
    hyps.sort(key=lambda x: x[3])
    deg_r, hist_r = baselines(W, rte)
    top = []
    for kind_h, K, g, v in hyps[:20]:
        rr = min(score(W, rte, g, K, a) for a in (1, 10))
        top.append({'type': kind_h, 'K': K, 'cv': float(v), 'reserved': float(rr)})
    sb = [h for h in hyps if h[0] == 'sbm']
    sbm_res = None
    if sb:
        kind_h, K, g, v = sb[0]
        sbm_res = {'cv': float(v), 'reserved': float(min(score(W, rte, g, K, a) for a in (1, 10))),
                   'rank': [h[0] for h in hyps].index('sbm') + 1}
    rand_cv = [h[3] for h in hyps if h[0] == 'rand']
    out = {'net': name, 'kind': kind, 'seed': seed, 'n_cons': n, 'deg_cv': float(deg_cv),
           'best_cv_gain': float(deg_cv - hyps[0][3]), 'best_type': hyps[0][0], 'best_K': hyps[0][1],
           'rand_gain_median': float(deg_cv - np.median(rand_cv)), 'rand_gain_max': float(deg_cv - min(rand_cv)),
           'deg_reserved': float(deg_r), 'hist_reserved': float(hist_r),
           'top20_reserved_gain_mean': float(np.mean([deg_r - t['reserved'] for t in top])),
           'top1_reserved_gain': float(deg_r - top[0]['reserved']), 'sbm': sbm_res, 'top': top,
           'best_partition': {C[i]: int(hyps[0][2][i]) for i in range(n)}}
    print('%-9s %-4s s%d n %d | CV gain over DEG: best %.3f (%s K%d), random max %.3f median %.3f | reserved: DEG %.3f HIST %.3f top1 gain %+.3f top20 mean %+.3f %s' % (
        name, kind, seed, n, out['best_cv_gain'], out['best_type'], out['best_K'], out['rand_gain_max'], out['rand_gain_median'],
        deg_r, hist_r, out['top1_reserved_gain'], out['top20_reserved_gain_mean'],
        ('| SBM rank %d cv gain %.3f reserved gain %+.3f' % (sbm_res['rank'], deg_cv - sbm_res['cv'], deg_r - sbm_res['reserved'])) if sbm_res else ''), flush=True)
    with open(os.path.join(CK, 'c4.jsonl'), 'a') as f:
        f.write(json.dumps(out) + '\n')
    return out


if __name__ == '__main__':
    D = load_nets()['nets']
    sbm = {}
    for r in json.load(open(os.path.join(CK, 'c2.json'))):
        if r['kind'] == 'real':
            sbm[r['net']] = dict(zip(r['blocks']['C'], r['blocks']['bc']))
    nets = {'PE_SIGN': [tuple(e) for e in D['PE_SIGN']],
            'UR3_SIGN': [tuple(e) for e in subsample_tablets(D['UR3_SIGN'], len(D['PE_SIGN']), 0)],
            'LB_SIGN': [tuple(e) for e in D['LB_SIGN']],
            'PLANT_MOD': planted(480, 41, 6700, 5, 0)[0]}
    jobs = []
    for nm, ev in nets.items():
        for kind in ('real', 'tab', 'glob'):
            jobs.append((nm, ev, kind, 5, sbm.get(nm)))
    done = set()
    if os.path.exists(os.path.join(CK, 'c4.jsonl')):
        for l in open(os.path.join(CK, 'c4.jsonl')):
            x = json.loads(l); done.add((x['net'], x['kind']))
    skip = set(os.environ.get('SKIP', '').split(','))
    jobs = [j for j in jobs if (j[0], j[2]) not in done and j[0] not in skip]
    with Pool(2) as p:
        p.map(job, jobs, chunksize=1)
