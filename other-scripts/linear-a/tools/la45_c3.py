#!/usr/bin/env python3
"""LA-45 cycle 3: TABLETS ONLY, matched size, with order-shuffled twins.
Cycle 2 found that Linear A's meanings carry almost no word-order information, but LA is half one-word nodules.
Here every corpus is lists only, at the size of the Linear A tablets (433 tablets + lames, 3,700 tokens):
LA_TAB, LB_TAB (KN+PY), UR3_TAB, PLANT_TAB (planted lists, no sealings), each also as X_S2 (tokens shuffled inside
each document), plus LA_TAB_S1 (type identities shuffled). Same game as cycle 1 (random document split,
12 Proposers x 6 Critics, 120 rounds, 6 populations per corpus).
Decision rule fixed in advance: word order carries meaning in a corpus if its final-test gain exceeds its S2 twin's
in >= 5 of 6 population pairs AND the mean drop is >= 0.3 bits/occurrence.
Usage: la45_c3.py [rounds] [npop] [tag] [corpora]"""
import sys, os, json, random, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
NPOP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TAG = sys.argv[3] if len(sys.argv) > 3 else 'c3'
TABS = ('Tablet', 'Lames (short thin tablet)')


def corpus(name):
    if name.endswith('_S2'):
        d, tr = corpus(name[:-3])
        return C.shuffle_order(d, random.Random(C.seed('la45-c3-s2-' + name))), tr
    la = [d for d in C.la_docs() if d['support'] in TABS]
    n = C.ntok(la)
    if name == 'LA_TAB':
        return la, None
    if name == 'LA_TAB_S1':
        return C.shuffle_types(la, random.Random(C.seed('la45-c3-s1'))), None
    if name == 'PLANT_TAB':
        d, tr = C.planted_docs(3 * n, random.Random(C.seed('la45-c3-plant')))
        d = [x for x in d if len(x['toks']) > 2]
        out, c = [], 0
        for x in d:
            if c >= n: break
            out.append(x); c += len(x['toks'])
        return out, tr
    if name == 'UR3_TAB':
        d = C.sample_size(C.ur3_docs_all(), n, random.Random(C.seed('la45-c3-ur3')))
        return d, C.ur_truth(d)
    if name == 'LB_TAB':
        d = C.sample_size(C.lb_docs_all(), n, random.Random(C.seed('la45-c3-lb')))
        return d, C.lb_truth(d)


def job(args):
    name, k = args
    fn = os.path.join(C.CK, '%s_%s_%d.json' % (TAG, name, k))
    if os.path.exists(fn):
        return fn
    docs, truth = corpus(name)
    B = C.build(docs)
    t = time.time()
    # S2 twins share the split seed with their real corpus so the same documents are held out
    base = name[:-3] if name.endswith('_S2') else name
    r = C.run_population(docs, B, C.seed('la45-%s-%s-%d' % (TAG, base, k)), rounds=ROUNDS)
    r['secs'] = time.time() - t; r['corpus'] = name
    json.dump(r, open(fn, 'w'))
    with open(os.path.join(C.CK, TAG + '.log'), 'a') as f:
        f.write('%s %d K=%d gain=%.3f contra=%.3f nev=%d %.0fs\n' % (name, k, r['K'], r['final_gain'],
                                                                   r['final_contra'], r['nev'], r['secs']))
    return fn


if __name__ == '__main__':
    names = ['PLANT_TAB', 'UR3_TAB', 'LB_TAB', 'LA_TAB', 'PLANT_TAB_S2', 'UR3_TAB_S2', 'LB_TAB_S2', 'LA_TAB_S2',
             'LA_TAB_S1']
    if len(sys.argv) > 4:
        names = sys.argv[4].split(',')
    jobs = [(n, k) for k in range(NPOP) for n in names]
    with Pool(2) as p:
        for fn in p.imap_unordered(job, jobs):
            print(fn, flush=True)
