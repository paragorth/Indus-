#!/usr/bin/env python3
"""LA-58 driver: corpus specs, simulation banks (2 workers max), ABC-RF fits.

  python3 la58_run.py bank <corpus> <nworld> [chunk]     corpus in LA, LB, UR, LASHUF<i> (same nk as LA)
  python3 la58_run.py fit <corpus> <bankname>
"""
import sys, os, json, time, random, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la58_common import *

NTARGET = 1634   # Linear A documents at the 12 panel sites


def corpus(name):
    """Returns (docs, K, site names, tabonly)."""
    if name == 'LA':
        d = la_docs(); return d, len(LA_SITES), LA_ABBR, 0
    if name.startswith('LASHUF'):
        d = la_docs(); rng = random.Random(seed(name))
        sites = [x['site'] for x in d]; rng.shuffle(sites)
        return [dict(x, site=s) for x, s in zip(d, sites)], len(LA_SITES), LA_ABBR, 0
    if name == 'LAIB':   # LM IB horizon only (one synchronic political map)
        d = la_docs(horizon={'LMIB'})
        present = sorted(set(x['site'] for x in d), key=lambda k: -sum(1 for x in d if x['site'] == k))
        present = [k for k in present if sum(1 for x in d if x['site'] == k) >= 5]
        m = {k: i for i, k in enumerate(present)}
        return [dict(x, site=m[x['site']]) for x in d if x['site'] in m], len(present), [LA_ABBR[k] for k in present], 0
    if name == 'LB':
        d = lb_docs_all(); rng = random.Random(seed('la58-lb'))
        return thin(d, NTARGET, rng, len(LB_SITES)), len(LB_SITES), LB_SITES, 0
    if name == 'UR':
        d = ur_docs_all(); rng = random.Random(seed('la58-ur'))
        return thin(d, NTARGET, rng, len(UR_SITES)), len(UR_SITES), ['UMMA', 'PD', 'GIRSU', 'UR', 'NIPPUR', 'GARSANA', 'IRISAGRIG'], 1
    raise ValueError(name)


def nk_of(docs, K):
    return [sum(1 for x in docs if x['site'] == k) for k in range(K)]


def _job(a):
    K, nk, n, sd, tabonly = a
    return simulate(K, nk, n, sd, tabonly=tabonly)


def bank(name, nworld, chunk=2000, tag=None):
    docs, K, ab, tabonly = corpus(name)
    nk = nk_of(docs, K)
    tag = tag or name
    out = os.path.join(CK, 'bank_' + tag)
    os.makedirs(out, exist_ok=True)
    jobs = []
    for c in range(nworld // chunk):
        fn = os.path.join(out, 'c%04d.npy' % c)
        if not os.path.exists(fn):
            jobs.append((c, (K, nk, chunk, seed('la58-%s-%d' % (tag, c)), tabonly)))
    t0 = time.time()
    with Pool(2) as P:
        for (c, _), (S, T) in zip(jobs, P.imap(_job, [j for _, j in jobs])):
            np.save(os.path.join(out, 'c%04d.npy' % c), np.hstack([S, T]).astype(np.float32))
            print(tag, c, '%.0fs' % (time.time() - t0), flush=True)
    json.dump(dict(K=K, nk=nk, abbr=ab, tabonly=tabonly, nstat=nstat(K), ntheta=ntheta(K)),
              open(os.path.join(out, 'meta.json'), 'w'))


def load_bank(tag):
    out = os.path.join(CK, 'bank_' + tag)
    meta = json.load(open(os.path.join(out, 'meta.json')))
    A = np.vstack([np.load(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f.endswith('.npy')])
    return A[:, :meta['nstat']].astype(np.float64), A[:, meta['nstat']:].astype(np.float64), meta


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        bank(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 2000)
