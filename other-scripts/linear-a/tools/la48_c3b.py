#!/usr/bin/env python3
"""LA-48 cycle 3b: power check. Linear B KN and PY and Ur III subsampled to Linear A size (393 documents with
entries, 3 draws each); COPY / PHYS hits vs 6 quantity-shuffled runs per draw."""
import sys, os, json, time, random
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la48_common import *
import la48_c3 as C3
C3.RESTARTS = 2; C3.STEPS = 80000

if __name__ == '__main__':
    LB = lb_docs()
    DS = []
    for site in ('KN', 'PY'):
        pool = [d for d in LB if d['site'] == site]
        for r in range(3):
            rs = random.Random(seed('c3b%s%d' % (site, r))); x = list(pool); rs.shuffle(x)
            DS.append(('LB_%s_n393_%d' % (site, r), x[:393]))
    jobs = []
    for name, docs in DS:
        jobs.append((name, docs, 'REAL', 0))
        for r in range(6):
            jobs.append((name, docs, 'QSHUF', r))
    fn = os.path.join(CK, 'c3b_results.jsonl')
    with Pool(2) as p:
        for out in p.imap_unordered(C3.run_one, jobs):
            with open(fn, 'a') as f:
                f.write(json.dumps(out) + '\n')
            print(out['name'], out['kind'], out['rep'], out['elig'], out['copy'], out['phys'], flush=True)
