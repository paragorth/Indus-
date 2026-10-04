#!/usr/bin/env python3
"""LA-45 cycle 3: do settled meanings predict NEW FINDS? Time split.
Linear A: Proposers train on GORILA vols 1-3 (published 1976-79, 4,319 tokens); the Critic's vault is the documents
with no publication link (535 tokens); the final test is everything published later (GORILA 4-5 1982-85 and
post-1985 papers; 391 tokens, mostly sites other than Hagia Triada).
Controls at the same sizes: LB (train KN+PY, vault KN+PY, final Thebes = a later find), Ur III (train Umma +
Puzrish-Dagan + Girsu, vault same pool, final Ur + Nippur), PLANT (random), LA_TIME_S1 (type-shuffled LA).
Usage: la45_c3.py [rounds] [npop] [tag]"""
import sys, os, json, random, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NPOP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TAG = sys.argv[3] if len(sys.argv) > 3 else 'c3'
TR, VA, FI = ('G1', 'G2', 'G3'), ('blank',), ('G4', 'G5', 'post')


def corpus(name):
    la = C.la_docs_dedup()
    n = [sum(len(d['toks']) for d in la if d['pub'] in g) for g in (TR, VA, FI)]
    if name in ('LA_TIME', 'LA_TIME_S1'):
        idx = [[i for i, d in enumerate(la) if d['pub'] in g] for g in (TR, VA, FI)]
        docs = la if name == 'LA_TIME' else C.shuffle_types(la, random.Random(C.seed('la45-c3-s1')))
        return docs, None, idx
    if name == 'PLANT_TIME':
        d, tr = C.planted_docs(sum(n), random.Random(C.seed('la45-c3-plant')))
        rng = random.Random(4); idx = list(range(len(d))); rng.shuffle(idx)
        parts, k = [], 0
        for want in n:
            p, c = [], 0
            while k < len(idx) and c < want:
                p.append(idx[k]); c += len(d[idx[k]]['toks']); k += 1
            parts.append(p)
        return d, tr, parts
    if name == 'LB_TIME':
        L = C.lb_docs_all(); rng = random.Random(C.seed('la45-c3-lb'))
        pool = C.sample_size(L, n[0] + n[1], rng)
        a, c = [], 0
        for d in pool:
            (a if c < n[0] else []).append(d); c += len(d['toks'])
        b = pool[len(a):]
        th = C.sample_size(C.lb_docs_site('TH'), n[2], rng)
        d = a + b + th
        return d, C.lb_truth(d), [list(range(len(a))), list(range(len(a), len(a) + len(b))),
                                  list(range(len(a) + len(b), len(d)))]
    if name == 'UR3_TIME':
        U = C.ur3_docs_all(); rng = random.Random(C.seed('la45-c3-ur'))
        pool = C.sample_size([x for x in U if x['site'] in ('Umma', 'Puzriš-Dagan', 'Girsu')], n[0] + n[1], rng)
        a, c = [], 0
        for d in pool:
            (a if c < n[0] else []).append(d); c += len(d['toks'])
        b = pool[len(a):]
        f = C.sample_size([x for x in U if x['site'] in ('Ur', 'Nippur')], n[2], rng)
        d = a + b + f
        return d, C.ur_truth(d), [list(range(len(a))), list(range(len(a), len(a) + len(b))),
                                  list(range(len(a) + len(b), len(d)))]


def job(args):
    name, k = args
    fn = os.path.join(C.CK, '%s_%s_%d.json' % (TAG, name, k))
    if os.path.exists(fn):
        return fn
    docs, truth, (tr, va, fi) = corpus(name)
    B = C.build(docs)
    t = time.time()
    r = C.run_population_split(docs, B, C.seed('la45-%s-%s-%d' % (TAG, name, k)), tr, None, rounds=ROUNDS,
                               fixed=(va, fi))
    r['secs'] = time.time() - t; r['corpus'] = name
    json.dump(r, open(fn, 'w'))
    with open(os.path.join(C.CK, TAG + '.log'), 'a') as f:
        f.write('%s %d K=%d gain=%.3f one=%.3f contra=%.3f refuted=%.2f nfinal=%d nev=%d %.0fs\n' % (
            name, k, r['K'], r['final_gain'], r['final_one'], r['final_contra'], r['refuted_frac'], r['final_n'],
            r['nev'], r['secs']))
    return fn


if __name__ == '__main__':
    names = ['PLANT_TIME', 'UR3_TIME', 'LB_TIME', 'LA_TIME', 'LA_TIME_S1']
    jobs = [(n, k) for k in range(NPOP) for n in names]
    with Pool(2) as p:
        for fn in p.imap_unordered(job, jobs):
            print(fn, flush=True)
