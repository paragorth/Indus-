#!/usr/bin/env python3
"""la66 cycle 2: per-feature null calibration.  Real LA under three split families x two modes, against a
pool of null corpora (quantities shuffled within document / within commodity, word bundles re-dealt within
documents, signs shuffled across the words of an entry), plus Linear B at LA size through the same filter
and LA with totals kept (KU-RO as an internal positive control).  S=12 splits x M=800 models per job."""
import sys, os, json, time
from multiprocessing import Pool
from la66_lib import *
import la66_c1 as c1

S, M = 12, 800
LBT = c1.LB_TRUTH


def corpus(kind, seed):
    L = la_rows()
    if kind == 'LA':
        return L, {}
    if kind == 'LAtot':
        return la_rows(totals=True), {'W:KU-RO': 1, 'W:PO-TO-KU-RO': 1}
    if kind == 'LAqdoc':
        return null_q_doc(L, 1000 + seed), {}
    if kind == 'LAqcom':
        return null_q_com(L, 1000 + seed), {}
    if kind == 'LAwdoc':
        return null_words_doc(L, 1000 + seed), {}
    if kind == 'LAsent':
        return null_signs_entry(L, 1000 + seed), {}
    B = thin(lb_rows(), len(L), 7)
    if kind == 'LB':
        return B, dict(LBT)
    if kind == 'LBqdoc':
        return null_q_doc(B, 1000 + seed), dict(LBT)
    raise ValueError(kind)


def run(job):
    kind, mode, seed = job
    name = f'c2_{kind}_{mode}_{seed}'
    fn = os.path.join(CK, name + '.json')
    if os.path.exists(fn):
        return name
    t0 = time.time()
    rows, truth = corpus(kind, seed)
    D, R = run_corpus(rows, mode, S, M, seed0=1000 + 100 * seed)
    T = summarise(D, R)
    json.dump(dict(job=job, n=D.n, nfeat=len(D.fnames), scores=scores(R), truth=truth, tab=T,
                   secs=time.time() - t0), open(fn, 'w'))
    return name


JOBS = ([('LA', m, s) for s in range(3) for m in ('tab', 'sys')] + [('LAtot', 'tab', 0), ('LB', 'tab', 0)] +
        [('LAqdoc', 'tab', s) for s in range(4)] + [('LAqcom', 'sys', s) for s in range(3)] +
        [('LAwdoc', 'tab', s) for s in range(3)] + [('LAsent', 'tab', s) for s in range(2)] +
        [('LAqcom', 'tab', s) for s in range(2)] + [('LBqdoc', 'tab', s) for s in range(3)])

if __name__ == '__main__':
    with Pool(2) as p:
        for name in p.imap_unordered(run, JOBS):
            print(name, flush=True)
