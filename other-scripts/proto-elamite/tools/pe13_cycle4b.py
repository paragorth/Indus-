"""pe13 cycle 4b: calibrate the canonical-order tests (the cycle-4 planted ORDER was too weak to
be seen) and print PE's early / late signs.
Planted ORDER2: 60% of tablets sorted by a hidden global rank of their first sign, noise 0.05.
Real positive control for the random sign-set search: Ur III Drehem (known animal order)."""
import json, os, random, sys
import numpy as np
from collections import Counter
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe13_common as C  # noqa
import pe13_cycle4 as Q  # noqa
from pe13_cycle2 import variant  # noqa

if __name__ == '__main__':
    CO = C.corpora()
    ent = variant(CO['PE'], 'ENT')
    uni = Counter(x for t in ent for l in t['lines'] for x in l['toks'])
    with Pool(2) as p:
        p.map(Q.job_ab, [('PL_ORDER2', Q.plant_order(ent, 81, uni, 0.6, 0.05), 50),
                         ('PL_ORDER3', Q.plant_order(ent, 82, uni, 0.3, 0.05), 50)], chunksize=1)
        Q.search('PL_ORDER2', Q.plant_order(ent, 83, uni, 0.6, 0.05), p)
        ur = random.Random(5).sample(CO['UR3'], 1000)
        Q.search('UR3', ur, p)
    # PE early / late signs (relative first position, signs on >= 25 tablets)
    pos = {}
    for t in ent:
        f = Q.firsts(t)
        L = max(len(t['lines']) - 1, 1)
        for s, i in f.items():
            pos.setdefault(s, []).append(i / L)
    rng = random.Random(1)
    rows = []
    for s, v in pos.items():
        if len(v) < 25:
            continue
        rows.append((np.mean(v), len(v), s))
    rows.sort()
    # null spread of means for n tablets: shuffle-based sd from uniform positions
    print('EARLY:', [(s, round(m, 2), n) for m, n, s in rows[:10]])
    print('LATE :', [(s, round(m, 2), n) for m, n, s in rows[-10:]])
    sh = C.shuffle_lines(ent, rng)
    pos2 = {}
    for t in sh:
        f = Q.firsts(t)
        L = max(len(t['lines']) - 1, 1)
        for s, i in f.items():
            pos2.setdefault(s, []).append(i / L)
    r2 = sorted(np.mean(v) for s, v in pos2.items() if len(v) >= 25)
    print('shuffled range of means: %.2f - %.2f; real %.2f - %.2f' % (r2[0], r2[-1], rows[0][0], rows[-1][0]))
