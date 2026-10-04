#!/usr/bin/env python3
"""LA-40 cycle 1: global role inference (tempered Gibbs, many chains) on Linear A tablets,
with (i) a word-shuffle null run through the identical search, (ii) Linear B KN+PY at
Linear A size with known word classes (scored only), (iii) planted roles in Linear A.
Output: data/la40_ckpt/c1.json and a summary printed to stdout.
"""
import collections, json, os, random, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la40_common as C

STEPS = int(os.environ.get('STEPS', 1_000_000))
NCH = int(os.environ.get('NCH', 24))
NNULL = int(os.environ.get('NNULL', 6))
NLB = int(os.environ.get('NLB', 6))


def plant(docs, rng):
    docs = [dict(d, lines=[list(ln) for ln in d['lines']]) for d in docs]
    truth = {}
    # T: KU-RO renamed in a random half of its documents
    kd = [i for i, d in enumerate(docs) if any(t == ('W', 'KU-RO') for ln in d['lines'] for t in ln)]
    for i in rng.sample(kd, len(kd) // 2):
        docs[i]['lines'] = [[('W', 'PLANT-T') if t == ('W', 'KU-RO') else t for t in ln] for ln in docs[i]['lines']]
    truth['PLANT-T'] = 'T'
    # H: a new heading word written first on 10 multi-line tablets
    md = [i for i, d in enumerate(docs) if len(d['lines']) >= 3]
    for i in rng.sample(md, 10):
        docs[i]['lines'][0].insert(0, ('W', 'PLANT-H'))
    truth['PLANT-H'] = 'H'
    # C: a new word written directly before a logogram on 8 lines
    ll = [(i, li, j) for i, d in enumerate(docs) for li, ln in enumerate(d['lines']) for j, t in enumerate(ln)
          if t[0] == 'L' and j > 0 and ln[j - 1][0] in ('W', 'W1')]
    for i, li, j in sorted(rng.sample(ll, 8), key=lambda x: -x[2]):
        docs[i]['lines'][li].insert(j, ('W', 'PLANT-C'))
    truth['PLANT-C'] = 'C'
    # E: a new entry word replacing one counted entry head on 8 list tablets (expected: the role of its neighbours)
    el = []
    for i, d in enumerate(docs):
        heads = [(li, ln) for li, ln in enumerate(d['lines']) if len(ln) >= 2 and ln[0][0] == 'W' and ln[1][0] == 'N'
                 and ln[0][1] not in ('KU-RO', 'PLANT-T', 'KI-RO', 'PO-TO-KU-RO')]
        if len(heads) >= 4:
            el.append((i, rng.choice(heads)[0]))
    emap = {}
    for i, li in rng.sample(el, min(8, len(el))):
        emap[i] = (li, docs[i]['lines'][li][0][1])
        docs[i]['lines'][li][0] = ('W', 'PLANT-E')
    return docs, truth, emap


def job(args):
    kind, k = args
    rng = random.Random(C.seed('la40c1-%s-%d' % (kind, k)))
    if kind == 'LA':
        docs = C.la_docs()
    elif kind == 'NULL':
        docs = C.shuffle_words(C.la_docs(), rng)
    elif kind == 'PLANT':
        docs, truth, emap = plant(C.la_docs(), rng)
    elif kind == 'LB':
        LA = C.la_docs(); ntok = sum(1 for d in LA for ln in d['lines'] for t in ln if C.is_word(t))
        docs = C.lb_sample(C.lb_docs_all(), ntok, rng)
    elif kind == 'LBFULL':
        docs = C.lb_docs_all()
    elif kind == 'LBNULL':
        LA = C.la_docs(); ntok = sum(1 for d in LA for ln in d['lines'] for t in ln if C.is_word(t))
        docs = C.shuffle_words(C.lb_sample(C.lb_docs_all(), ntok, random.Random(C.seed('la40c1-LB-%d' % k))), rng)
    B = C.build(docs)
    nch = NCH if kind == 'LA' else max(8, NCH // 2)
    S = C.run_chains(B, nchains=nch, steps=STEPS, seed0=C.seed(kind) % 1000 + k)
    mode, freq, agree, Pm = C.summarize(S)
    out = dict(kind=kind, k=k, types=B['types'], nocc=B['nocc'].tolist(), ndocs=B['ndocs'].tolist(),
               mode=[C.ROLES[m] for m in mode], freq=freq.round(4).tolist(), agree=agree.round(4).tolist(),
               P=Pm.round(4).tolist(), nchains=nch, steps=STEPS)
    if kind == 'PLANT':
        out['truth'] = truth
        # expected role of PLANT-E: modal role of the other counted heads on those tablets
        out['emap'] = {str(i): v for i, v in emap.items()}
        exp = []
        for i, (li, orig) in emap.items():
            hs = [ln[0][1] for lj, ln in enumerate(docs[i]['lines']) if lj != li and len(ln) >= 2 and C.is_word(ln[0]) and ln[1][0] == 'N']
            rs = [out['mode'][B['ti'][w]] for w in hs if w in B['ti']]
            if rs: exp.append(collections.Counter(rs).most_common(1)[0][0])
        out['E_expected'] = exp
    return out


def main():
    jobs = [('LBFULL', 0), ('LA', 0), ('PLANT', 0), ('PLANT', 1)] + [('NULL', k) for k in range(NNULL)] + \
           [('LB', k) for k in range(NLB)] + [('LBNULL', k) for k in range(3)]
    t = time.time()
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(C.CK, 'c1.json'), 'w'))
    print('done %.0fs' % (time.time() - t))


if __name__ == '__main__':
    main()
