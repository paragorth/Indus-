#!/usr/bin/env python3
"""LA-37 cycle 4: MERGE AND PREDICT (massive random guessing).
If two signs are doublets, writing them as ONE class loses nothing and pools their evidence, so a class-bigram
model that merges them predicts held-out words at least as well as the unmerged model. Each hypothesis = a set of
4 random sign pairs merged. Score = held-out log-likelihood gain (bits per 1,000 sign tokens) of the ORIGINAL signs:
P(s | prev) = P(class(s) | class(prev)) * P(s | class(s)), add-k smoothed, word edges as '#'.
Design: documents split into 3 parts by hash: A (search), B (re-test), C (final). 20,000 random hypotheses on A
(trained on the other two parts, scored on A); top 200 re-tested on B; survivors' pairs counted.
Single-pair gains (all pairs, 5-fold CV) are also computed; pair marginal = mean gain of hypotheses containing it.
Controls: Linear B (pair gains vs LB labels, AUC), Linear B at LA size (5 draws), planted token-doublet in LA,
re-test of the top 200 against 200 random hypotheses on B.
"""
import os, sys, json, collections, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import la37_common as K

LOG = os.path.join(K.CK, 'c4.log')


def encode(units, alph):
    ix = {s: i + 1 for i, s in enumerate(alph)}  # 0 = '#', len+1 = rare '?'
    q = len(alph) + 1
    return [np.array([0] + [ix.get(s, q) for s in r['w']] + [0]) for r in units], q + 1


def bigram_counts(seqs, n):
    M = np.zeros((n, n))
    for s in seqs:
        np.add.at(M, (s[:-1], s[1:]), 1)
    return M


def loglik(Mtr, test_pairs, cls, k=0.1):
    """class bigram + within-class unigram; Mtr = sign bigram counts (train); test_pairs = (prev, cur) arrays."""
    n = Mtr.shape[0]; nc = cls.max() + 1
    Cm = np.zeros((n, nc)); np.add.at(Cm.T, cls, Mtr.T)       # prev sign x cur class
    CC = np.zeros((nc, nc)); np.add.at(CC, cls, Cm)            # prev class x cur class
    Pcc = (CC + k) / (CC.sum(1, keepdims=True) + k * nc)
    uni = Mtr.sum(0) + 0.5
    cu = np.zeros(nc); np.add.at(cu, cls, uni)
    Ps = uni / cu[cls]
    p, c = test_pairs
    return float(np.log2(Pcc[cls[p], cls[c]] * Ps[c]).sum())


def folds(units, nf, seed=0):
    rng = np.random.default_rng(seed)
    docs = sorted({r['doc'] for r in units})
    f = dict(zip(docs, rng.integers(0, nf, len(docs))))
    return np.array([f[r['doc']] for r in units])


def pair_gains(units, alph, nf=5, seed=0):
    seqs, n = encode(units, alph)
    fo = folds(units, nf, seed)
    iu = np.triu_indices(len(alph), 1)
    G = np.zeros(len(iu[0])); ntest = 0
    for f in range(nf):
        tr = bigram_counts([s for s, g in zip(seqs, fo) if g != f], n)
        te = [s for s, g in zip(seqs, fo) if g == f]
        tp = (np.concatenate([s[:-1] for s in te]), np.concatenate([s[1:] for s in te]))
        tp = (tp[0][tp[1] != 0], tp[1][tp[1] != 0])  # score signs, not word ends
        ntest += len(tp[0])
        cls0 = np.arange(n)
        base = loglik(tr, tp, cls0)
        for j, (a, b) in enumerate(zip(*iu)):
            cls = cls0.copy(); cls[b + 1] = a + 1
            G[j] += loglik(tr, tp, cls) - base
    return iu, G / ntest * 1000


def hyp_gain(tr, tp, n, pairs):
    cls = np.arange(n)
    for a, b in pairs:
        cls[cls == cls[b + 1]] = cls[a + 1]
    _, cls = np.unique(cls, return_inverse=True)
    return loglik(tr, tp, cls)


def random_search(units, alph, nh=20000, k=4, seed=0, top=200):
    rng = np.random.default_rng(seed)
    seqs, n = encode(units, alph)
    part = folds(units, 3, seed + 99)
    def split(f):
        tr = bigram_counts([s for s, g in zip(seqs, part) if g != f], n)
        te = [s for s, g in zip(seqs, part) if g == f]
        tp = (np.concatenate([s[:-1] for s in te]), np.concatenate([s[1:] for s in te]))
        m = tp[1] != 0
        return tr, (tp[0][m], tp[1][m])
    trA, tpA = split(0); trB, tpB = split(1)
    bA = loglik(trA, tpA, np.arange(n)); bB = loglik(trB, tpB, np.arange(n))
    m = len(alph)
    H = [[tuple(sorted(rng.choice(m, 2, replace=False))) for _ in range(k)] for _ in range(nh)]
    gA = np.array([(hyp_gain(trA, tpA, n, h) - bA) / len(tpA[0]) * 1000 for h in H])
    best = np.argsort(-gA)[:top]
    rand = rng.choice(nh, top, replace=False)
    gBbest = np.array([(hyp_gain(trB, tpB, n, H[i]) - bB) / len(tpB[0]) * 1000 for i in best])
    gBrand = np.array([(hyp_gain(trB, tpB, n, H[i]) - bB) / len(tpB[0]) * 1000 for i in rand])
    # pair marginal on A
    marg = collections.defaultdict(list)
    for h, g in zip(H, gA):
        for p in h:
            marg[p].append(g)
    surv = collections.Counter(p for i, gb in zip(best, gBbest) if gb > np.median(gBrand) for p in H[i])
    return dict(gA_best=float(gA[best].mean()), gA_all=float(gA.mean()), gB_best=float(gBbest.mean()),
                gB_rand=float(gBrand.mean()), p_retest=float(((gBrand[:, None] >= gBbest[None, :]).mean())),
                surv=[(alph[a], alph[b], c) for (a, b), c in surv.most_common(15)])


def lb_eval_gain(alph, iu, G):
    return K.lb_eval(dict(alph=alph, iu=iu, T=G))


def job(arg):
    kind, seed = arg
    rng = np.random.default_rng(seed)
    if kind == 'LBfull':
        u = K.lb_units(); al, c = K.alphabet(u, 10)
    elif kind == 'LBdraw':
        u = K.lb_draw(K.lb_units(), 3918, rng); al, c = K.alphabet(u, 8)
    elif kind.startswith('PL'):
        host = kind.split('|')[1]
        u = K.plant(K.la_units(), host, 0.4, 'token', rng); al, c = K.alphabet(u, 8)
    else:
        u = K.la_units(); al, c = K.alphabet(u, 8)
    iu, G = pair_gains(u, al, seed=seed)
    out = dict(kind=kind, seed=seed, alph=al, G=G.tolist())
    if kind.startswith('LB'):
        e = lb_eval_gain(al, iu, G)
        out['eval'] = {g: e[g] for g in ('doublet', 'sameC', 'sameV')}; out['top20'] = dict(e['top20']); out['base'] = e['base']
        out['top20_pairs'] = e['top20_pairs']
    if kind.startswith('PL'):
        host = kind.split('|')[1]
        j = [k for k, (a, b) in enumerate(zip(*iu)) if {al[a], al[b]} == {host, 'X*'}][0]
        out['planted_pct'] = float((G < G[j]).mean()); out['planted_rank'] = int((G > G[j]).sum()) + 1
    if kind in ('LA', 'LBfull'):
        out['rs'] = random_search(u, al, seed=seed)
    return out


if __name__ == '__main__':
    t0 = time.time()
    jobs = [('LA', 0), ('LBfull', 0)] + [('LBdraw', s) for s in range(5)] + [(f'PL|{h}', 50 + i) for i, h in enumerate(['KA', 'SI', 'TA', 'NA', 'DA', 'RE'])]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(K.CK, 'c4.json'), 'w'), default=str)
    K.log(LOG, f'done {time.time() - t0:.0f}s')
