#!/usr/bin/env python3
"""LA-45 cycle 1: calibration and first Linear A run.
Corpora (all at Linear A size, ~5,245 tokens incl. numbers): PLANT (made-up administration with known meanings),
UR3 (Ur III admin texts reduced to opaque word ids + numbers), LB (KN+PY reduced to opaque ids + numbers), LA,
LA_S1 (type identities shuffled over slots), LA_S2 (tokens shuffled inside documents).
6 independent populations per corpus (own split, own seed), 8 Proposers x 6 Critics, 40 rounds.
Usage: la45_c1.py [rounds] [npop]"""
import sys, os, json, random, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NPOP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TAG = sys.argv[3] if len(sys.argv) > 3 else 'c1'


def corpus(name):
    la = C.la_docs()
    n = C.ntok(la)
    if name == 'LA':
        return la, None
    if name == 'LA_S1':
        return C.shuffle_types(la, random.Random(C.seed('la45-s1'))), None
    if name == 'LA_S2':
        return C.shuffle_order(la, random.Random(C.seed('la45-s2'))), None
    if name == 'PLANT':
        return C.planted_docs(n, random.Random(C.seed('la45-plant')))
    if name == 'UR3':
        d = C.sample_size(C.ur3_docs_all(), n, random.Random(C.seed('la45-ur3')))
        return d, C.ur_truth(d)
    if name == 'LB':
        d = C.sample_size(C.lb_docs_all(), n, random.Random(C.seed('la45-lb')))
        return d, C.lb_truth(d)


def job(args):
    name, k = args
    fn = os.path.join(C.CK, '%s_%s_%d.json' % (TAG, name, k))
    if os.path.exists(fn):
        return fn
    docs, truth = corpus(name)
    B = C.build(docs)
    t = time.time()
    r = C.run_population(docs, B, C.seed('la45-%s-%s-%d' % (TAG, name, k)), rounds=ROUNDS)
    r['secs'] = time.time() - t
    r['corpus'] = name
    json.dump(r, open(fn, 'w'))
    with open(os.path.join(C.CK, TAG + '.log'), 'a') as f:
        f.write('%s %d K=%d gain=%.3f contra=%.3f nev=%d %.0fs\n' % (name, k, r['K'], r['final_gain'],
                                                                   r['final_contra'], r['nev'], r['secs']))
    return fn


if __name__ == '__main__':
    # warm caches in the parent
    C.ur3_docs_all(); C.lb_docs_all()
    names = ['PLANT', 'UR3', 'LB', 'LA', 'LA_S1', 'LA_S2']
    jobs = [(n, k) for k in range(NPOP) for n in names]
    with Pool(2) as p:
        for fn in p.imap_unordered(job, jobs):
            print(fn, flush=True)
