#!/usr/bin/env python3
"""LA-37 cycle 2b: replication flag (replaces the within-word-shuffle z, which failed the planted control in cycle 1).
A pair is flagged if its all3 score is in the top 2 % of all pairs in BOTH halves of a random document split,
in >= 50 % of 20 splits. Chance level: pairs that replicate when the second half's pair labels are permuted.
Calibration: Linear B full and 10 LA-sized draws (precision = share of flagged pairs that are doublet / same C /
same V, against base rate); within-word-shuffled LA (kill); planted LA doublets (token, group, type; 6 hosts each).
"""
import os, sys, json, collections, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import la37_common as K

LOG = os.path.join(K.CK, 'c2b.log')


def rep(units, alph, nsplit=20, top=0.02, seed=0):
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(len(alph), 1)
    docs = sorted({r['doc'] for r in units})
    hits = np.zeros(len(iu[0])); chance = []
    for s in range(nsplit):
        h = dict(zip(docs, rng.integers(0, 2, len(docs))))
        Ts = []
        for g in (0, 1):
            u = [r for r in units if h[r['doc']] == g]
            Ts.append(K.all3(K.stats(u, alph, K.doc_halves(u, s)), iu))
        t1 = Ts[0] >= np.quantile(Ts[0], 1 - top); t2 = Ts[1] >= np.quantile(Ts[1], 1 - top)
        hits += t1 & t2
        chance.append(int((t1 & rng.permutation(t2)).sum()))
    return iu, hits / nsplit, float(np.mean(chance))


def job(arg):
    kind, seed = arg
    rng = np.random.default_rng(seed)
    host = None
    if kind == 'LBfull':
        u = K.lb_units(); fmin = 10
    elif kind == 'LBdraw':
        u = K.lb_draw(K.lb_units(), 3918, rng); fmin = 8
    elif kind == 'LA':
        u = K.la_units(); fmin = 8
    elif kind == 'LAsh':
        u = K.shuffle_within(K.la_units(), rng); fmin = 8
    else:
        _, host, mode = kind.split('|')
        u = K.plant(K.la_units(), host, 0.4, mode, rng); fmin = 8
    al, c = K.alphabet(u, fmin)
    iu, r, ch = rep(u, al, seed=seed)
    flag = r >= 0.5
    out = dict(kind=kind, seed=seed, nflag=int(flag.sum()), chance_per_split=ch, n_rep_any=int((r > 0).sum()),
               flagged=[(al[iu[0][j]], al[iu[1][j]], float(r[j])) for j in np.where(flag)[0]])
    if kind.startswith('LB'):
        labs = np.array([K.lb_label(al[a], al[b]) for a, b in zip(*iu)], dtype=object)
        rel = np.isin(labs, ['doublet', 'sameC', 'sameV']); ok = labs != None
        out['prec'] = float(rel[flag & ok].mean()) if (flag & ok).sum() else None
        out['base'] = float(rel[ok].mean())
        out['flag_lab'] = dict(collections.Counter(labs[flag]))
        out['ndoub'] = int((labs == 'doublet').sum())
    if host:
        if 'X*' in al:
            j = [k for k, (a, b) in enumerate(zip(*iu)) if {al[a], al[b]} == {host, 'X*'}][0]
            out['planted_rep'] = float(r[j])
        else:
            out['planted_rep'] = None
    if kind == 'LA':
        out['all'] = r.tolist(); out['alph'] = al
    return out


if __name__ == '__main__':
    t0 = time.time()
    hosts = ['KA', 'SI', 'TA', 'NA', 'DA', 'RE']
    jobs = [('LA', 0), ('LAsh', 1), ('LAsh', 2), ('LBfull', 0)] + [('LBdraw', s) for s in range(10)]
    jobs += [(f'PL|{h}|{m}', 200 + i) for i, h in enumerate(hosts) for m in ('token', 'group', 'type')]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(K.CK, 'c2b.json'), 'w'), default=str)
    K.log(LOG, f'done {time.time() - t0:.0f}s')
