#!/usr/bin/env python3
"""X-2 cycle 3e: is the triangle consistency more than 'measured vs counted' class agreement?
Null: replace each PE partner Y by a random PE class sign Y' whose Ur III modal partner has the
same unit class (measured = grain, flour, oil, beer, bread; counted = everything else). 20,000 draws."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, random
from collections import Counter, defaultdict
import numpy as np
import x2_common as X

MEAS = {'sze', 'zi3', 'dabin', 'esza', 'ziz2', 'gig', 'i3', 'kasz', 'ninda', 'eša'}
G = json.load(open(X.GOLD_FILE))['gold']
r3 = [json.loads(l) for l in open(os.path.join(X.DX, 'c3_worlds.jsonl'))]
r2 = [json.loads(l) for l in open(os.path.join(X.DX, 'c2_worlds.jsonl'))]


def modal(d):
    c = Counter(d); t, n = c.most_common(1)[0]
    return t, n / sum(c.values())


def pooled(name, m):
    agg = defaultdict(Counter)
    for r in r3:
        if r['name'] == name:
            for s, d in r['res'][m]['top'].items():
                agg[s].update(d)
    return {s: modal(d)[0] for s, d in agg.items()}


res = {}
for m in ('prof', 'joint', 'flood'):
    lape = [r for r in r2 if r['name'] == 'lape'][0]['res'][m]['top']
    pairs = [(s, modal(d)[0]) for s, d in lape.items() if sum(d.values()) >= 500]
    la_lb, pe_ur = pooled('la_lb', m), pooled('pe_ur', m)
    cls = {y: (u in MEAS) for y, u in pe_ur.items()}
    pool = defaultdict(list)
    for y, c in cls.items():
        pool[c].append(y)

    def cons(x, y):
        return x in la_lb and y in pe_ur and pe_ur[y] in G.get(la_lb[x], [])
    real = sum(cons(x, y) for x, y in pairs)
    rng = random.Random(7)
    null = np.array([sum(cons(x, rng.choice(pool[cls[y]])) for x, y in pairs if y in cls) for _ in range(20000)])
    # also: class agreement of the chain itself (LB partner class vs Ur III partner class)
    lbmeas = {'GRA', 'HORD', 'FAR', 'OLE', 'VIN', 'NI', 'OLIV', 'CYP', 'AROM', 'ME±RI'}
    agree = [((la_lb.get(x) in lbmeas) == cls.get(y)) for x, y in pairs if x in la_lb and y in cls]
    res[m] = {'consistent': int(real), 'class_null_mean': float(null.mean()),
              'P_class': float((1 + (null >= real).sum()) / (1 + len(null))),
              'chain_class_agreement': float(np.mean(agree)), 'n': len(agree)}
    print(m, res[m])
json.dump(res, open(os.path.join(X.DX, 'c3_class.json'), 'w'), indent=1)
