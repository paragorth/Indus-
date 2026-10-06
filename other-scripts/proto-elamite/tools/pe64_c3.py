#!/usr/bin/env python3
"""pe64 cycle 3: the ladder learnt on one half PREDICTS capacities on the other half.
usage: pe64_c3.py CORPUS VARIANT DIRECTION
From half A only: rungs = every rate with an A-survivor rule (cycle 1 'union' selectors); rung events on A;
marker signs = count-line signs on >= 3 A-tablets of a rung's events with p <= 0.01 against 100 swap nulls of A.
On half B: every count line carrying a marker sign of rung r predicts that a capacity line within 5 lines
holds count x r.  Hits vs 200 swap nulls of B (same lines, same predictions) and vs the same lines scored
with every OTHER rung's rate (rung specificity).
Output: data/pe64_ckpt/c3_<CORPUS>_<VARIANT>_<DIR>.json
"""
import sys, os, json, random
from collections import defaultdict
import numpy as np
import pe64_lib as L

C, V, D = sys.argv[1], sys.argv[2], sys.argv[3]
rng = random.Random(L.seed('pe64c1' + C + V + D))
T = L.load(C)
truth = None
if V == 'small':
    T = [T[i] for i in sorted(rng.sample(range(len(T)), 120))]
elif V.startswith('plant'):
    T, truth = L.plant_ladder(L.null_swap(T, random.Random(7)), rng, rungs=(20, 40, 80), per=int(V[5:] or 10))
elif V.startswith('nullreal'):
    T = L.null_swap(T, random.Random(int(V[8:] or 11)))
A, B = L.split(T, 'pe64' + C)
if D == 'BA':
    A, B = B, A
TA = [T[i] for i in A]
TB = [T[i] for i in B]
c1 = json.load(open(os.path.join(L.CK, 'c1_%s_%s_%s.json' % (C, V, D))))
rules = {x['rate']: [(r[0], r[1], r[2], r[3], r[4], tuple(r[5]), r[6]) for r in x['rules']]
         for x in c1['tests'] if x['sel'] == 'union'}
rungs = sorted(rules)
PA = L.PairTable(TA)


def rung_sign_tabs(Pp):
    cc = {}
    out = {}
    for r in rungs:
        idx = np.unique(np.concatenate([L.rule_mask(Pp, ru, cc) for ru in rules[r]]))
        idx = idx[Pp.rkey[idx] == int(round(r * 1000))] if len(idx) else idx
        d = defaultdict(set)
        for k in idx:
            for s in Pp.ssig[k]:
                d[s].add(int(Pp.tab[k]))
        out[r] = {s: len(v) for s, v in d.items()}
    return out


real = rung_sign_tabs(PA)
nulls = [rung_sign_tabs(PA.revalue(L.null_swap(TA, random.Random(7000 + z)))) for z in range(100)]
markers = {}   # sign -> rung (a sign that marks several rungs keeps the one with most tablets)
mk_detail = []
for r in rungs:
    for s, k in real[r].items():
        if k < 3:
            continue
        nk = np.array([n[r].get(s, 0) for n in nulls])
        p = (1 + (nk >= k).sum()) / 101
        if p <= 0.01:
            mk_detail.append((s, r, k, float(nk.mean()), float(p)))
best = {}
for s, r, k, nm, p in mk_detail:
    if s not in best or k - nm > best[s][1]:
        best[s] = (r, k - nm)
markers = {s: v[0] for s, v in best.items()}
print(C, V, D, 'rungs', len(rungs), 'markers', len(markers), flush=True)


def score(TT, rate_of):
    hits = n = 0
    for t in TT:
        Ls = t['lines']
        for i, l in enumerate(Ls):
            if not l['cnt']:
                continue
            ms = [s for s in l['s'] if s in markers]
            if not ms:
                continue
            r = rate_of(markers[ms[0]])
            if r is None:
                continue
            n += 1
            want = l['cnt'] * r
            for j in range(max(0, i - 5), min(len(Ls), i + 6)):
                if j != i and Ls[j]['cap'] is not None and not (l['amb'] and Ls[j]['amb'] and j == i) \
                        and abs(Ls[j]['cap'] - want) < 1e-6:
                    hits += 1
                    break
    return hits, n


h, n = score(TB, lambda r: r)
nh = np.array([score(L.null_swap(TB, random.Random(8000 + z)), lambda r: r)[0] for z in range(200)])
p = (1 + (nh >= h).sum()) / 201
other = []
for z in range(50):
    rr = random.Random(z)
    other.append(score(TB, lambda r: rr.choice([x for x in rungs if x != r]) if len(rungs) > 1 else None)[0])
out = {'corpus': C, 'variant': V, 'dir': D, 'n_rungs': len(rungs), 'markers': markers,
       'marker_detail': mk_detail, 'pred_lines': n, 'hits': h, 'null_mean': float(nh.mean()), 'p': float(p),
       'other_rung_mean': float(np.mean(other)), 'p_other': float((1 + sum(o >= h for o in other)) / 51),
       'truth': truth}
json.dump(out, open(os.path.join(L.CK, 'c3_%s_%s_%s.json' % (C, V, D)), 'w'), default=str)
print('pred lines', n, 'hits', h, 'swap null', nh.mean(), 'p', p, 'other-rung', np.mean(other), out['p_other'])
