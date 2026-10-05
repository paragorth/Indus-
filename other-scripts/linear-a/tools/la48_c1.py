#!/usr/bin/env python3
"""LA-48 cycle 1: do quantities conserve across documents?

For each dataset, search document direction labels d in {IN, OUT, STOCK} (random starts + simulated
annealing; every proposal is one random hypothesis) to maximise the number of exactly balanced
pass-through nodes (K2: two-term balances, i.e. the same amount in and out; K3: three or more terms,
i.e. genuine aggregation A + B = C). The same search is run on nulls:
  QSHUF  quantities permuted among entries of the same site x commodity across documents
  POIS   integer parts resampled from Poisson
  RAND   the score of 20,000 random label vectors (no search) = label-shuffle baseline
Datasets: Linear A (all sites), Linear B KN and PY (DAMOS), Ur III Puzrish-Dagan at Linear A size
(393 documents, 5 draws) and at 2,000 documents, planted conserved economies (survival 1.0, 0.5, 0.2).
"""
import sys, os, json, time, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la48_common import *

STEPS = int(os.environ.get('LA48_STEPS', 150000))
RESTARTS = 4
NNULL = int(os.environ.get('LA48_NNULL', 8))


def run_one(args):
    name, docs, kind, rep = args
    rs = random.Random(seed('%s-%s-%d' % (name, kind, rep)))
    if kind == 'QSHUF':
        docs = shuffle_quantities(docs, rs)
    elif kind == 'POIS':
        docs = poisson_quantities(docs, rs)
    B = build(docs)
    best, bd = search(B, 0.0, 1.0, 3.0, restarts=RESTARTS, steps=STEPS, sd=seed(name + kind) % 10000 + rep)
    comp = components(B, bd)
    out = {'name': name, 'kind': kind, 'rep': rep, 'score': best, 'K2': int(comp[:, 1].sum()),
           'K3': int(comp[:, 2].sum())}
    bc = balance_census(B, bd)
    out['PAIR'] = bc['PAIR']; out['AGG'] = bc['AGG']
    if kind == 'REAL':
        rnd = random_scores(len(B['names']), B['nd'], B['nptr'], B['o_doc'], B['o_slot'], B['o_val'],
                            0.0, 1.0, 3.0, 20000, True, 7)
        out['rand_mean'] = float(rnd.mean()); out['rand_max'] = float(rnd.max())
        # which documents/nodes balance
        bal = [B['names'][n] for n in range(len(B['names'])) if comp[n, 1] or comp[n, 2]]
        out['balanced'] = [list(map(str, b)) + ['K3' if comp[B['names'].index(b), 2] else 'K2'] for b in bal][:200]
        out['d'] = bd.tolist()
        out['ids'] = [d['id'] for d in docs]
    return out


def datasets():
    L = la_docs()
    qpool = [e[2] for d in L for e in d['entries']]
    DS = [('LA', L)]
    LB = lb_docs()
    DS.append(('LB_KN', [d for d in LB if d['site'] == 'KN']))
    DS.append(('LB_PY', [d for d in LB if d['site'] == 'PY']))
    for r in range(5):
        DS.append(('UR_n393_%d' % r, ur3_docs(393, random.Random(seed('urdraw%d' % r)))))
    DS.append(('UR_n2000', ur3_docs(2000, random.Random(seed('urdraw2000')))))
    for sv in (1.0, 0.5, 0.2):
        DS.append(('PL_s%.1f' % sv, planted_docs(393, random.Random(seed('pl%.1f' % sv)), qpool, survival=sv)))
    for sv in (1.0, 0.3):
        DS.append(('PLAGG_s%.1f' % sv, planted_docs(393, random.Random(seed('plagg%.1f' % sv)), qpool, survival=sv, agg=3)))
    return DS


if __name__ == '__main__':
    DS = datasets()
    jobs = []
    for name, docs in DS:
        jobs.append((name, docs, 'REAL', 0))
        nn = NNULL if not name.startswith('UR') else (5 if name.endswith('_0') else 3)
        for r in range(nn):
            jobs.append((name, docs, 'QSHUF', r))
        for r in range(3):
            jobs.append((name, docs, 'POIS', r))
    fn = os.path.join(CK, 'c1_results.jsonl')
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            x = json.loads(l); done.add((x['name'], x['kind'], x['rep']))
    jobs = [j for j in jobs if (j[0], j[2], j[3]) not in done]
    print('jobs', len(jobs), flush=True)
    t0 = time.time()
    with Pool(2) as p:
        for out in p.imap_unordered(run_one, jobs):
            with open(fn, 'a') as f:
                f.write(json.dumps(out) + '\n')
            print('%.0fs' % (time.time() - t0), out['name'], out['kind'], out['rep'], out['score'], out['K2'], out['K3'],
                  out.get('rand_mean', ''), flush=True)
