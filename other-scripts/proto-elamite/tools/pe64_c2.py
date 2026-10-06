#!/usr/bin/env python3
"""pe64 cycle 2: the ladder and the signs on each rung.
usage: pe64_c2.py CORPUS VARIANT
Rungs = rates validated on held-out halves in cycle 1 (AB and/or BA; p_swap <= 0.005, kB >= 2; 'grid' or 'top' selector, so the rules are named).
(1) Ladder: is the rung set small and in simple ratios beyond chance?  Statistic S = share of rung pairs
    whose ratio is p/q with p, q <= 4 (p != q) or 6; null = random rung sets of the same size drawn from the
    tablet-weighted distribution of all rates the same rules produce on 50 swap-null corpora.
(2) Signs per rung: for every count-line sign on a rung's events, tablets k vs 200 swap-null corpora
    (same rules, same rate).  Rung specificity: the sign's rung vs the sign's other rungs.
Output: data/pe64_ckpt/c2_<CORPUS>_<VARIANT>.json
"""
import sys, os, json, random, math
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import numpy as np
import pe64_lib as L

C, V = sys.argv[1], sys.argv[2]
NNULL = int(sys.argv[3]) if len(sys.argv) > 3 else 200
rng = random.Random(L.seed('pe64c2' + C + V))
T = L.load(C)
if V == 'small':
    T = [T[i] for i in sorted(random.Random(L.seed('pe64c1' + C + V + 'AB')).sample(range(len(T)), 120))]
    # note: the AB and BA runs draw different subsamples; we use the AB one for sign association
elif V == 'plant':
    T, truth = L.plant_ladder(L.null_swap(T, random.Random(7)), random.Random(L.seed('pe64c1' + C + V + 'AB')),
                              rungs=(20, 40, 80), per=10)
elif V == 'nullreal':
    T = L.null_swap(T, random.Random(11))

val = defaultdict(list)       # rate -> rules
dirs = defaultdict(set)
ntests = {}
for D in ('AB', 'BA'):
    fn = os.path.join(L.CK, 'c1_%s_%s_%s.json' % (C, V, D))
    if not os.path.exists(fn):
        continue
    d = json.load(open(fn))
    ntests[D] = d['n_tests']
    for x in d['tests']:
        if x['p_swap'] <= 0.005 and x['kB'] >= 2 and x['sel'] != 'union':
            for r in x['rules']:
                rule = (r[0], r[1], r[2], r[3], r[4], tuple(r[5]), r[6])
                if rule not in val[x['rate']]:
                    val[x['rate']].append(rule)
            dirs[x['rate']].add(D)
rungs = sorted(val)
print(C, V, 'rungs', rungs, {r: sorted(dirs[r]) for r in rungs}, flush=True)

P = L.PairTable(T)
cache = {}


def events(Pp, cache):
    ev = {}
    for r in rungs:
        rk = int(round(r * 1000))
        idx = np.unique(np.concatenate([L.rule_mask(Pp, ru, cache) for ru in val[r]])) if val[r] else np.zeros(0, int)
        ev[r] = idx[Pp.rkey[idx] == rk] if len(idx) else idx
    return ev


def sign_tabs(Pp, ev):
    out = {}
    for r, idx in ev.items():
        d = defaultdict(set)
        for k in idx:
            for s in Pp.ssig[k]:
                d[s].add(int(Pp.tab[k]))
        out[r] = {s: len(v) for s, v in d.items()}
    return out


ev = events(P, cache)
st = sign_tabs(P, ev)
ntab = {r: len(np.unique(P.tab[ev[r]])) for r in rungs}

# nulls
nullst = []
pool = Counter()
for z in range(NNULL):
    Pn = P.revalue(L.null_swap(T, random.Random(9000 + z)))
    cn = {}
    nullst.append(sign_tabs(Pn, events(Pn, cn)))
    if z < 50:
        allidx = np.unique(np.concatenate([L.rule_mask(Pn, ru, cn) for r in rungs for ru in val[r]])) if rungs else []
        tr = {(int(Pn.tab[k]), round(float(Pn.rate[k]), 6)) for k in allidx if Pn.rate[k] > 0}
        for _, rt in tr:
            pool[rt] += 1

sig = []
for r in rungs:
    for s, k in st[r].items():
        if k < 2:
            continue
        nk = np.array([ns[r].get(s, 0) for ns in nullst])
        p = (1 + (nk >= k).sum()) / (NNULL + 1)
        sig.append({'rung': r, 'sign': s, 'k': k, 'null_mean': float(nk.mean()), 'p': float(p)})
sig.sort(key=lambda x: (x['p'], -x['k']))
ntest_sig = len(sig)


# rung specificity among significant signs: share of the sign's rung tablets on this rung
def ratio_simple(a, b):
    f = Fr(a).limit_denominator(1000) / Fr(b).limit_denominator(1000)
    if f < 1:
        f = 1 / f
    return f != 1 and f.numerator <= 6 and f.denominator <= 4 and f in {Fr(2), Fr(3), Fr(4), Fr(6), Fr(3, 2),
                                                                        Fr(4, 3), Fr(5, 2), Fr(5), Fr(5, 3), Fr(5, 4)}


def S(R):
    pr = [(a, b) for i, a in enumerate(R) for b in R[i + 1:]]
    return sum(ratio_simple(a, b) for a, b in pr) / max(len(pr), 1)


ladder = {}
if len(rungs) >= 2:
    s_real = S(rungs)
    keys = list(pool)
    w = np.array([pool[k] for k in keys], float)
    w /= w.sum()
    sims = []
    for _ in range(5000):
        R = set()
        while len(R) < len(rungs):
            R.add(keys[np.random.choice(len(keys), p=w)])
            if len(R) >= len(keys):
                break
        sims.append(S(sorted(R)))
    sims = np.array(sims)
    ladder = {'S_real': s_real, 'S_null_mean': float(sims.mean()), 'p': float((1 + (sims >= s_real).sum()) / 5001),
              'n_rungs': len(rungs), 'pool_size': len(keys)}
out = {'corpus': C, 'variant': V, 'rungs': rungs, 'dirs': {r: sorted(dirs[r]) for r in rungs},
       'n_rules': {r: len(val[r]) for r in rungs}, 'ntab': ntab, 'ladder': ladder, 'ntests_c1': ntests,
       'signs': sig[:200], 'n_sign_tests': ntest_sig,
       'rules': {r: [list(ru[:5]) + [list(ru[5]), ru[6]] for ru in val[r]] for r in rungs}}
if V == 'plant':
    out['truth'] = truth
json.dump(out, open(os.path.join(L.CK, 'c2_%s_%s.json' % (C, V)), 'w'), default=str)
print('ladder', ladder)
for x in sig[:25]:
    print(x)
