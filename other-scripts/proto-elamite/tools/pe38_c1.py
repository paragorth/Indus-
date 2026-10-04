#!/usr/bin/env python3
"""pe38 cycle 1: global role inference for all PE sign types at once (tempered Gibbs, many chains) with
pre-registered accounting rules and the line type-checker. Same search on
  PE real; PE nulls (signs shuffled across entry slots x4, across all slots x2); planted roles in PE x2;
  proto-cuneiform at PE size x4 (+ shuffled x2) and Ur III at PE size x4 (+ shuffled x2), scored on known roles.
Output: data/pe38_ckpt/c1.json
"""
import collections, json, os, random, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe38_common as C

STEPS = int(os.environ.get('STEPS', 300_000))
NCH = int(os.environ.get('NCH', 16))


def plant(docs, rng):
    docs = [dict(d, lines=[dict(l, toks=list(l['toks'])) for l in d['lines']]) for d in docs]
    truth = {}
    tl = [(i, li) for i, d in enumerate(docs) for li, l in enumerate(d['lines']) if l['kind'] == 3]
    for i, li in rng.sample(tl, len(tl) // 2):
        docs[i]['lines'][li]['toks'].insert(0, 'PLANT-T')
    truth['PLANT-T'] = 'TOT'
    hd = [i for i, d in enumerate(docs) if d['lines'][0]['kind'] == 0 and d['lines'][0]['toks'] and len(d['lines']) >= 3]
    for i in rng.sample(hd, 12):
        docs[i]['lines'][0]['toks'].insert(0, 'PLANT-H')
    truth['PLANT-H'] = 'HDR'
    vl = [(i, li) for i, d in enumerate(docs) for li, l in enumerate(d['lines']) if l['kind'] in (0, 1, 3) and l['toks']]
    for i, li in rng.sample(vl, 15):
        docs[i]['lines'][li]['toks'].append('PLANT-V')
    truth['PLANT-V'] = 'VRB'
    el = []
    for i, d in enumerate(docs):
        heads = [li for li, l in enumerate(d['lines']) if l['kind'] == 2 and len(l['toks']) == 1 and l['toks'][0]]
        if len(heads) >= 4:
            el.append((i, rng.choice(heads)))
    emap = {}
    for i, li in rng.sample(el, min(10, len(el))):
        emap[i] = (li, docs[i]['lines'][li]['toks'][0])
        docs[i]['lines'][li]['toks'][0] = 'PLANT-E'
    return docs, truth, emap


def job(args):
    kind, k = args
    rng = random.Random(C.seed('pe38c1-%s-%d' % (kind, k)))
    PE = C.pe_docs()
    n = C.ntok(PE)
    lab, truth, emap = None, None, None
    if kind == 'PE':
        docs = PE
    elif kind == 'NULLE':
        docs = C.shuffle_entries(PE, rng, 'entry')
    elif kind == 'NULLA':
        docs = C.shuffle_entries(PE, rng, 'all')
    elif kind == 'PLANT':
        docs, truth, emap = plant(PE, rng)
    elif kind.startswith('PC'):
        docs = C.sample_size(C.pc_docs(), n, random.Random(C.seed('pe38-pc-%d' % k)))
        lab = C.PC_LAB
        if kind == 'PCNULL':
            docs = C.shuffle_entries(docs, rng, 'entry')
    elif kind.startswith('UR'):
        docs = C.sample_size(C.ur3_docs_all(), n, random.Random(C.seed('pe38-ur-%d' % k)))
        lab = C.UR_LAB
        if kind == 'URNULL':
            docs = C.shuffle_entries(docs, rng, 'entry')
    B = C.build(docs)
    cost = C.rule_cost(B)
    nch = NCH if kind == 'PE' else max(8, NCH // 2)
    S, viol = C.run_chains(B, cost, nchains=nch, steps=STEPS, seed0=C.seed(kind) % 1000 + k)
    frc, mode, freq, agree, P = C.forced(B, S)
    out = dict(kind=kind, k=k, ntypes=len(B['types']), ntok=C.ntok(docs), nlines=B['nL'], viol=viol,
               forced=frc, mode={w: C.ROLES[mode[t]] for t, w in enumerate(B['types'])},
               freq={w: round(float(freq[t]), 3) for t, w in enumerate(B['types'])},
               nocc={w: int(B['nocc'][t]) for t, w in enumerate(B['types'])})
    if lab:
        out['score'] = C.score_labels(B, frc, mode, lab)
    if truth:
        out['truth'] = truth
        exp = []
        for i, (li, orig) in emap.items():
            hs = [l['toks'][0] for lj, l in enumerate(docs[i]['lines']) if lj != li and l['kind'] == 2 and len(l['toks']) == 1]
            rs = [out['mode'][w] for w in hs if w in out['mode']]
            if rs:
                exp.append(collections.Counter(rs).most_common(1)[0][0])
        out['E_expected'] = exp
    print(kind, k, 'done', len(frc), flush=True)
    return out


def main():
    jobs = [('PE', 0), ('PLANT', 0), ('PLANT', 1)] + [('NULLE', k) for k in range(4)] + [('NULLA', k) for k in range(2)] + \
           [('PC', k) for k in range(4)] + [('PCNULL', k) for k in range(2)] + [('UR', k) for k in range(4)] + \
           [('URNULL', k) for k in range(2)]
    t = time.time()
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(C.CK, 'c1.json'), 'w'))
    print('done %.0fs' % (time.time() - t))


if __name__ == '__main__':
    main()
