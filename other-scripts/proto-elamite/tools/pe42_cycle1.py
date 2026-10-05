"""pe42 cycle 1: which mental-arithmetic error families explain the failing
totals better than chance?  PE, proto-cuneiform (PC) and Mesopotamian count
lists (MES, cdli dump; called UR3 in the code).  Nulls: noise (real
discrepancy of another tablet added to this tablet's entries), totals
reassigned within class and size bin, entries re-dealt.  Calibration: one
family planted at a time into closing tablets (recovery + enrichment)."""
import json, random, sys, collections
from multiprocessing import Pool
from pe42_common import *

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NULLS = {'noise': null_noise, 'totals': null_totals, 'shuffle': null_shuffle_entries}
PLANT_FAMS = ['OMIT', 'DOUBLE', 'SYS', 'CONV', 'CARRY', 'XCARRY', 'NOCARRY', 'DIGIT', 'FOREIGN']


def job(args):
    name, kind, seed = args
    rng = random.Random(seed)
    C = load_cases(name); corp = CORP[name]
    R0 = evaluate(C, corp, foreign=False)
    if kind in NULLS:
        C2 = NULLS[kind](C, corp, rng, R0)
        return (name, kind, seed, dict(fam_counts(evaluate(C2, corp))))
    fam = kind.split(':')[1]
    P, truth = plant(C, corp, rng, fam, R0, nmax=30)
    R = evaluate(P, corp)
    hit = sum(r['expl'].get(fam, False) or (fam == 'OMIT' and (r['expl'].get('OMITH') or r['expl'].get('OMITO')))
              for r in R if not r['closes'])
    # noise-null on the planted set: is the planted family enriched?
    Pn = null_noise(P, corp, rng, None)
    Rn = evaluate(Pn, corp)
    return (name, kind, seed, {'n': len(P), 'hit': hit, 'real': dict(fam_counts(R)), 'noise': dict(fam_counts(Rn))})


if __name__ == '__main__':
    jobs = []
    for name in ('PE', 'PC', 'UR3'):
        load_cases(name)
        for kind in NULLS:
            jobs += [(name, kind, s) for s in range(NREP)]
        for f in PLANT_FAMS:
            jobs += [(name, 'plant:' + f, 1000 + s) for s in range(5)]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=4)
    real = {}
    for name in ('PE', 'PC', 'UR3'):
        C = load_cases(name); R = evaluate(C, CORP[name])
        real[name] = {'counts': dict(fam_counts(R)),
                      'detail': [{'id': r['id'], 'expl': [f for f, b in r['expl'].items() if b], 'det': r['det']}
                                 for r in R if not r['closes']]}
    json.dump({'real': real, 'res': res}, open(os.path.join(CK, 'c1.json'), 'w'), default=str)
    print('done', len(res))
