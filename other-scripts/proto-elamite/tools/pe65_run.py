#!/usr/bin/env python3
"""PE-65 driver: simulation bank (2 workers max).

  python3 pe65_run.py bank <tag> <nworld> [chunk] [force_type]
One bank serves PE and both controls: all corpora are drawn at the same PE shape (units nk).
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe65_common import *


def _job(a):
    n, sd, ft = a
    return simulate(n, sd, force_type=ft)


def bank(tag, nworld, chunk=2000, ft=-1):
    out = os.path.join(CK, 'bank_' + tag)
    os.makedirs(out, exist_ok=True)
    jobs = []
    for c in range(nworld // chunk):
        fn = os.path.join(out, 'c%04d.npy' % c)
        if not os.path.exists(fn):
            jobs.append((c, (chunk, seed('pe65-%s-%d' % (tag, c)), ft)))
    t0 = time.time()
    with Pool(2) as P:
        for (c, _), (S, T) in zip(jobs, P.imap(_job, [j for _, j in jobs])):
            np.save(os.path.join(out, 'c%04d.npy' % c), S.astype(np.float16)); np.save(os.path.join(out, 't%04d.npy' % c), T.astype(np.float32))
            print(tag, c, '%.0fs' % (time.time() - t0), flush=True)
    json.dump(dict(nk=NK_PE(), units=UNITS, sites=SITES, nstat=nstat(), ntheta=ntheta()),
              open(os.path.join(out, 'meta.json'), 'w'))


def load_bank(tag):
    out = os.path.join(CK, 'bank_' + tag)
    meta = json.load(open(os.path.join(out, 'meta.json')))
    fs = sorted(f for f in os.listdir(out) if f.startswith('c') and f.endswith('.npy'))
    S = np.vstack([np.load(os.path.join(out, f)) for f in fs]).astype(np.float64)
    T = np.vstack([np.load(os.path.join(out, 't' + f[1:])) for f in fs]).astype(np.float64)
    return S, T, meta


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        bank(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 2000,
             int(sys.argv[5]) if len(sys.argv) > 5 else -1)
