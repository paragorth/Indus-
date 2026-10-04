#!/usr/bin/env python3
"""LA-45 cycle 2: the meanings must survive at OTHER SITES, and the Critic attacks single words.
Proposers train only on one site (Linear A: Hagia Triada); the Critic's vault and the final test are random halves
of the other sites. After the game, a targeted Critic moves each committed word (train count >= 5) to every other
meaning and REFUTES the word's meaning if any move raises the out-of-site score.
Controls: LB (train KN, test PY), Ur III (train Umma, test Puzrish-Dagan + Girsu), PLANT (random 'site'),
LA_S1 (type-shuffled LA, same site split). Sizes match Linear A's split (train ~ HT tokens, rest ~ non-HT tokens).
Usage: la45_c2.py [rounds] [npop] [tag]"""
import sys, os, json, random, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NPOP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TAG = sys.argv[3] if len(sys.argv) > 3 else 'c2'


def corpus(name):
    la = C.la_docs_dedup()
    ht = [i for i, d in enumerate(la) if d['site'] == 'Haghia Triada']
    rest = [i for i, d in enumerate(la) if d['site'] != 'Haghia Triada']
    ntr = sum(len(la[i]['toks']) for i in ht); nre = sum(len(la[i]['toks']) for i in rest)
    if name == 'LA_SITE':
        return la, None, ht, rest
    if name == 'LA_SITE_S1':
        return C.shuffle_types(la, random.Random(C.seed('la45-c2-s1'))), None, ht, rest
    if name == 'PLANT_SITE':
        d, tr = C.planted_docs(ntr + nre, random.Random(C.seed('la45-c2-plant')))
        idx = list(range(len(d))); random.Random(3).shuffle(idx)
        cut, c = 0, 0
        for k, i in enumerate(idx):
            c += len(d[i]['toks'])
            if c >= ntr:
                cut = k + 1; break
        return d, tr, idx[:cut], idx[cut:]
    if name == 'LB_SITE':
        L = C.lb_docs_all()
        a = C.sample_size([d for d in L if d['site'] == 'KN'], ntr, random.Random(C.seed('la45-c2-kn')))
        b = C.sample_size([d for d in L if d['site'] == 'PY'], nre, random.Random(C.seed('la45-c2-py')))
        d = a + b
        return d, C.lb_truth(d), list(range(len(a))), list(range(len(a), len(d)))
    if name == 'UR3_SITE':
        U = C.ur3_docs_all()
        a = C.sample_size([d for d in U if d['site'] == 'Umma'], ntr, random.Random(C.seed('la45-c2-umma')))
        b = C.sample_size([d for d in U if d['site'] in ('Puzriš-Dagan', 'Girsu')], nre,
                          random.Random(C.seed('la45-c2-pd')))
        d = a + b
        return d, C.ur_truth(d), list(range(len(a))), list(range(len(a), len(d)))


def job(args):
    name, k = args
    fn = os.path.join(C.CK, '%s_%s_%d.json' % (TAG, name, k))
    if os.path.exists(fn):
        return fn
    docs, truth, tr, rest = corpus(name)
    B = C.build(docs)
    t = time.time()
    r = C.run_population_split(docs, B, C.seed('la45-%s-%s-%d' % (TAG, name, k)), tr, rest, rounds=ROUNDS)
    r['secs'] = time.time() - t; r['corpus'] = name
    json.dump(r, open(fn, 'w'))
    with open(os.path.join(C.CK, TAG + '.log'), 'a') as f:
        f.write('%s %d K=%d gain=%.3f one=%.3f contra=%.3f refuted=%.2f nev=%d %.0fs\n' % (
            name, k, r['K'], r['final_gain'], r['final_one'], r['final_contra'], r['refuted_frac'], r['nev'], r['secs']))
    return fn


if __name__ == '__main__':
    names = ['PLANT_SITE', 'UR3_SITE', 'LB_SITE', 'LA_SITE', 'LA_SITE_S1']
    jobs = [(n, k) for k in range(NPOP) for n in names]
    with Pool(2) as p:
        for fn in p.imap_unordered(job, jobs):
            print(fn, flush=True)
