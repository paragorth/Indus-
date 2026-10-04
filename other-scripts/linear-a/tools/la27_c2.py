#!/usr/bin/env python3
"""LA-27 cycle 2: do Linear A integers heap on the physical 1:8:60 ladder (unit 61 g, mina 8
units, talent 60 minas ~ oxhide ingot) beyond decimal heaping, better than 10^4 random ladders?

Score = held-out (5-fold by document) log-likelihood gain per number of the ladder model over
the base model (log-normal + decimal heaping). Datasets: all LA integers; per commodity label;
the fraction-bearing 'weighed-goods' set (labels whose numbers carry fractions >= 25 %).
Controls: planted heaping on (8,60) and on (7,30) in LA-sized samples from the fitted base;
numbers jittered by +-1-2 (kills divisibility, keeps magnitude); split-half selection of the
best random ladder on half A and re-test on half B.
"""
import json, math, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la27_common import *

NL = int(os.environ.get('NL', 10000))
RL = random_ladders(NL)
PHYS = {'P1 1:8:60': (8, 60), 'P2 1:4:120': (4, 120)}

def null_scan(sc, cache=None):
    # ladders whose steps above the largest value in the data are identical get one score
    cache = {} if cache is None else cache
    keyc = {}
    for L in set(RL):
        k = tuple(s for s in steps(L) if s <= sc.vmax)
        if k not in keyc: keyc[k] = sc.gain(L)
        cache[L] = keyc[k]
    return np.array([cache[L] for L in RL]), cache

def evaluate(vals, docs, name):
    fold = folds_by_doc([(d,) for d in docs])
    sc = Scorer(vals, fold)
    g, cache = null_scan(sc)
    res = dict(name=name, n=len(vals))
    for k, L in PHYS.items():
        gp = sc.gain(L)
        res[k] = dict(gain=round(gp, 5), p=float((g >= gp).mean()))
    best = max(cache, key=cache.get)
    res['best_random'] = dict(ladder=best, gain=round(cache[best], 5))
    res['r1_top5'] = [(L, round(v, 5)) for L, v in sorted(cache.items(), key=lambda t: -t[1])[:5]]
    res['null_q95'] = round(float(np.quantile(g, .95)), 5)
    return res, cache

def sample_base(vals, n, rng):
    lb = fit_base(counts(vals)); p = np.exp(lb - lb.max()); p /= p.sum()
    return rng.choice(N, size=n, p=p)

def plant(vals, docs, ladder, rate, rng):
    x = sample_base(vals, len(vals), rng)
    r1 = ladder[0]
    m = rng.random(len(x)) < rate
    x[m] = np.maximum(r1, np.round(x[m] / r1) * r1).astype(int)
    return x

if __name__ == '__main__':
    rng = np.random.default_rng(2727)
    R = la_numbers()
    R = [r for r in R if 1 <= r[2] <= NMAX]
    out = {}
    lab = Counter(r[1] for r in R)
    R0 = la_numbers()
    lab0 = Counter(r[1] for r in R0)
    fr_share = {l: np.mean([bool(r[3]) for r in R0 if r[1] == l]) for l in lab0}
    lab = Counter({l: lab0[l] for l in lab0 if l in lab})
    weighed = sorted(l for l in lab if l != 'W' and fr_share[l] >= .25)
    sets = {'ALL': R, 'LOGO': [r for r in R if r[1] != 'W'],
            'WEIGHED(' + ','.join(weighed) + ')': [r for r in R if r[1] in weighed]}
    for l in ('W', 'OLE', 'GRA', 'CYP', 'VIN', 'VIR'):
        sets[l] = [r for r in R if r[1] == l]
    out['fraction_share'] = {l: round(float(v), 2) for l, v in fr_share.items() if lab[l] >= 5}
    PART = os.environ.get('PART', 'AB')
    allv = np.array([r[2] for r in R]); docs = [r[0] for r in R]
    SKIP = set(os.environ.get('SKIP', '').split(','))
    for name, rr in (sets.items() if 'A' in PART else []):
        if name.split('(')[0] in SKIP: continue
        res, _ = evaluate([r[2] for r in rr], [r[0] for r in rr], name)
        out[name] = res; print(json.dumps(res), flush=True)
    # jitter control on ALL
    jit = []
    for k in (range(2) if 'A' in PART else []):
        v = allv + rng.choice([-2, -1, 1, 2], len(allv)); v = np.clip(v, 1, NMAX)
        res, _ = evaluate(v, docs, 'jitter%d' % k); jit.append(res); print(json.dumps(res), flush=True)
    out['jitter'] = jit
    # planted controls (LA-sized, ALL)
    pl = []
    if 'B' not in PART:
        json.dump(out, open(os.path.join(CK, 'c2%s.json' % PART), 'w'), indent=1, default=str); sys.exit()
    for L, rate in [((8, 60), .15), ((8, 60), .05), ((7, 30), .15), ((7, 30), .05)]:
        for k in range(2):
            v = plant(allv, docs, L, rate, rng)
            res, cache = evaluate(v, docs, 'plant%s@%.2f' % (L, rate))
            gp = cache.get(L) if L in cache else Scorer(v, folds_by_doc([(d,) for d in docs])).gain(L)
            g = np.array([cache[x] for x in RL])
            pl.append(dict(ladder=L, rate=rate, p_planted=float((g >= gp).mean()),
                           best=res['best_random']['ladder'], r1_hit=res['best_random']['ladder'][0] == L[0]))
            print(json.dumps(pl[-1]), flush=True)
    out['planted'] = pl
    # split-half: select best ladder on half A, re-test on half B
    dd = sorted(set(docs)); rng.shuffle(dd); A = set(dd[::2])
    ia = [i for i, d in enumerate(docs) if d in A]; ib = [i for i, d in enumerate(docs) if d not in A]
    resA, cA = evaluate(allv[ia], [docs[i] for i in ia], 'half A')
    resB, cB = evaluate(allv[ib], [docs[i] for i in ib], 'half B')
    bA = resA['best_random']['ladder']
    gB = np.array([cB[x] for x in RL])
    out['split'] = dict(bestA=bA, gainA=resA['best_random']['gain'], gainB=round(cB[bA], 5),
                        pB=float((gB >= cB[bA]).mean()), bestB=resB['best_random']['ladder'])
    print(json.dumps(out['split']))
    json.dump(out, open(os.path.join(CK, 'c2%s.json' % PART), 'w'), indent=1, default=str)
