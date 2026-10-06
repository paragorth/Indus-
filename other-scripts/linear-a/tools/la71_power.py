#!/usr/bin/env python3
"""la71 cycle 2: is a weaker result in the read-only version a loss of power or a change of effect?
la10 consonant avoidance (non-HT word types): draw random subsets of the rd-version types with the size of
the read-only set (100 draws); for each, the observed same-consonant rate and its P against 200 sign
re-dealings.  If the read-only rate sits inside the subset distribution, the loss is power."""
import os, sys, json, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la71_parse as P
from la67_c2 import cons


def types(ver, ht):
    C = P.load(ver); out = set()
    for d in C:
        if (d['site'] == 'Haghia Triada') != ht: continue
        for t in d['tokens']:
            if t['t'] == 'word' and len(t['s']) > 1: out.add(tuple(t['s']))
    return [list(x) for x in out]


def rate(ws):
    a = n = 0
    for w in ws:
        for x, y in zip(w, w[1:]):
            cx, cy = cons(x), cons(y)
            if not cx or not cy: continue
            n += 1; a += cx == cy
    return a / n


def pglob(ws, rng, k=200):
    o = rate(ws); allsig = [s for w in ws for s in w]; nl = []
    for _ in range(k):
        rng.shuffle(allsig); it = iter(allsig); nl.append(rate([[next(it) for _ in w] for w in ws]))
    return o, (np.sum(np.array(nl) <= o) + 1) / (k + 1)


rng = random.Random(7102)
res = {}
for ht in (False, True):
    rd = types('rd', ht); rdo = types('read', ht)
    o_read, p_read = pglob(rdo, rng)
    sub = [pglob(rng.sample(rd, len(rdo)), rng) for _ in range(100)]
    obs = np.array([s[0] for s in sub]); ps = np.array([s[1] for s in sub])
    res['HT' if ht else 'nonHT'] = dict(n_rd=len(rd), n_read=len(rdo), read_rate=o_read, read_P=float(p_read),
                                         sub_rate_mean=float(obs.mean()), sub_rate_share_ge_read=float((obs >= o_read).mean()),
                                         sub_power=float((ps <= 0.05).mean()))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(P.CK, 'c2_power_la10.json'), 'w'), indent=1)
