#!/usr/bin/env python3
"""X-2 cycle 3: a second known-answer control and cycle consistency (triangulation).

(a) LA <-> LB: Linear A and Linear B logograms with the same conventional label (GRA, OLE, VIN,
    OLIV, VIR, NI, CYP, AROM, CAP...) are the same commodities by script descent. The pipeline,
    blind to the labels, should pair them. Real vs shuffled worlds, BT resamples each.
(b) PE <-> UR3: PE vs Ur III (no gold; needed for the triangle).
(c) Split halves LA x PE: random halves of the LA and PE tablets, modal partner per half;
    concordance between halves, real vs shuffled.
(d) Triangle: an LA commodity X paired with a PE class sign Y is 'cycle-consistent' when X's
    LB partner (from a) and Y's Ur III partner (from b) form a gold LB <-> Ur III pair.
Checkpoint: data/x2/c3_worlds.jsonl; 2 workers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time, zlib
from multiprocessing import Pool
import numpy as np
import x2_common as X
import x2_cycle2 as C2

W = int(os.environ.get('X2_W', 30))
BT = int(os.environ.get('X2_BT', 100))
NSPLIT = int(os.environ.get('X2_NSPLIT', 60))
CK = os.path.join(X.DX, 'c3_worlds.jsonl')


def task(args):
    name, k = args
    rng = np.random.default_rng(zlib.crc32(('c3|%s|%d' % (name, k)).encode()))
    shuf = name.endswith('_s')
    base = name[:-2] if shuf else name
    if base == 'la_lb':
        A, B = C2.data('LA'), C2.data('LB')
    elif base == 'pe_ur':
        UR = C2.data('UR3')
        A, B = C2.data('PE'), [UR[i] for i in rng.choice(len(UR), 3000, replace=False)]
    elif base == 'la_ur':
        UR = C2.data('UR3')
        A, B = C2.data('LA'), [UR[i] for i in rng.choice(len(UR), 3000, replace=False)]
    elif base == 'split':
        LA, PE = C2.data('LA'), C2.data('PE')
        ia = rng.permutation(len(LA)); ip = rng.permutation(len(PE))
        h = []
        for half in (0, 1):
            A = [LA[i] for i in ia[half::2]]; B = [PE[i] for i in ip[half::2]]
            if shuf:
                A = X.shuffle_all(A, rng); B = X.shuffle_all(B, rng)
            o = C2.world(A, B, BT // 2, rng, keep_sim=False)
            h.append({m: {'top': o[m]['top'], 'wtop': o[m]['wtop']} for m in o})
        return {'name': name, 'k': k, 'halves': h}
    if shuf:
        A = X.shuffle_all(A, rng); B = X.shuffle_all(B, rng)
    o = C2.world(A, B, BT, rng, keep_sim=False)
    return {'name': name, 'k': k, 'res': {m: {'top': o[m]['top'], 'btop': o[m]['btop'], 'wtop': o[m]['wtop']} for m in o}}


def main():
    done = set()
    if os.path.exists(CK):
        for l in open(CK):
            d = json.loads(l); done.add((d['name'], d['k']))
    tasks = []
    for k in range(max(W, NSPLIT)):
        if k < W:
            tasks += ([('la_lb', k)] if k < 5 else []) + [('la_lb_s', k)] + ([('pe_ur', k), ('la_ur', k)] if k < 15 else [])
        if k < NSPLIT:
            tasks += [('split', k), ('split_s', k)]
    tasks = [t for t in tasks if t not in done]
    print('tasks', len(tasks), file=sys.stderr)
    t0 = time.time()
    with Pool(2) as P, open(CK, 'a') as f:
        for i, r in enumerate(P.imap_unordered(task, tasks)):
            f.write(json.dumps(r) + '\n'); f.flush()
            print(i, r['name'], r['k'], round(time.time() - t0), file=sys.stderr)


if __name__ == '__main__':
    main()
