#!/usr/bin/env python3
"""la66 cycle 2b: calibration of the stable-descriptor rule.  The full real design (3 split families x tab/sys)
is repeated on two single null corpora (quantities shuffled within commodity; word bundles re-dealt within
documents) and on Linear B at LA size, so the rule's false-positive count and its power are measured on the
same footing as Linear A."""
import sys, os, json, time
from multiprocessing import Pool
from la66_lib import *
import la66_c2 as c2

S, M = 12, 2000


def run(job):
    kind, mode, fam = job
    name = f'c2b_{kind}_{mode}_{fam}'
    fn = os.path.join(CK, name + '.json')
    if os.path.exists(fn):
        return name
    t0 = time.time()
    L = la_rows()
    if kind == 'Nqcom':
        rows, truth = null_q_com(L, 2000), {}
    elif kind == 'Nwdoc':
        rows, truth = null_words_doc(L, 2000), {}
    elif kind == 'LB':
        rows, truth = thin(lb_rows(), len(L), 7), dict(c2.LBT)
    D, R = run_corpus(rows, mode, S, M, seed0=1000 + 100 * fam)
    T = summarise(D, R)
    json.dump(dict(job=job, n=D.n, scores=scores(R), truth=truth, tab=T, secs=time.time() - t0), open(fn, 'w'))
    return name


JOBS = [(k, m, f) for k in ('Nqcom', 'Nwdoc', 'LB') for f in range(3) for m in ('tab', 'sys')]
JOBS = [j for j in JOBS if not (j[0] == 'LB' and j[1] == 'tab' and j[2] == 0)]   # = c2_LB_tab_0

if __name__ == '__main__':
    with Pool(2) as p:
        for name in p.imap_unordered(run, JOBS):
            print(name, flush=True)
