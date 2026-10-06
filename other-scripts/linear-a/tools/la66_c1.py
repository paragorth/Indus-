#!/usr/bin/env python3
"""la66 cycle 1: calibration of the word-weight engine.  LA real / nulls / planted; Linear B at LA size;
Ur III herds (opaque) and Ur III per-head rations (opaque) at LA size.  Results: data/la66_ckpt/c1_<job>.json"""
import sys, os, json, time
from multiprocessing import Pool
from la66_lib import *

S, M = int(os.environ.get('LA66_S', 10)), int(os.environ.get('LA66_M', 1200))
LB_TRUTH = {'W:o': -1, 'W:o-pe-ro': -1, 'W:ko-wa': -1, 'W:ko-wo': -1, 'W:to-so': 1, 'W:to-so-de': 1}
UR_TRUTH = {'W:niga': -1}


def corpus(job):
    kind, mode, seed = job
    L = la_rows()
    nL = len(L)
    if kind == 'LA':
        return L, {}
    if kind == 'LAqcom':
        return null_q_com(L, seed), {}
    if kind == 'LAqdoc':
        return null_q_doc(L, seed), {}
    if kind == 'LAwdoc':
        return null_words_doc(L, seed), {}
    if kind == 'LAsent':
        return null_signs_entry(L, seed), {}
    if kind == 'LAplant':
        P, eff = plant(L, seed)
        return P, eff
    if kind.startswith('LB'):
        B = thin(lb_rows(), nL, seed)
        if kind == 'LBqdoc':
            B = null_q_doc(B, seed)
        return B, {k: v for k, v in LB_TRUTH.items()}
    if kind == 'URherd':
        H = thin(ur_herd_rows(), nL, seed)
        O, mp = opaque(H, seed)
        return O, {mp[k]: v for k, v in UR_TRUTH.items() if k in mp}
    if kind == 'URrat':
        U = thin(ur_ration_rows(False), nL, seed)
        O, mp = opaque(U, seed)
        return O, {'map': {v: k for k, v in mp.items() if k in ('W:gurusz', 'W:geme2', 'W:dumu', 'W:erin2')}}
    raise ValueError(kind)


def run(job):
    kind, mode, seed = job
    name = f'c1_{kind}_{mode}_{seed}'
    fn = os.path.join(CK, name + '.json')
    if os.path.exists(fn):
        return name
    t0 = time.time()
    rows, truth = corpus(job)
    D, R = run_corpus(rows, mode, S, M, seed0=100 * seed)
    T = summarise(D, R)
    out = dict(job=job, n=D.n, nfeat=len(D.fnames), scores=scores(R), truth=truth,
               tab={k: v for k, v in T.items() if v['cls'] != '-' or v['occ'] >= 8 or k in truth},
               secs=time.time() - t0)
    json.dump(out, open(fn, 'w'))
    return name


JOBS = ([('LA', m, 0) for m in ('tab', 'sys')] +
        [('LAplant', 'tab', s) for s in range(4)] + [('LAplant', 'sys', s) for s in range(2)] +
        [('LB', 'tab', s) for s in range(3)] + [('LB', 'sys', 0), ('LBqdoc', 'tab', 0)] +
        [('URherd', 'tab', s) for s in range(2)] + [('URherd', 'sys', 0)] +
        [('URrat', 'sys', s) for s in range(2)] + [('URrat', 'tab', 0)] +
        [('LAqcom', 'tab', s) for s in range(3)] + [('LAqcom', 'sys', s) for s in range(2)] +
        [('LAqdoc', 'tab', s) for s in range(2)] + [('LAwdoc', 'tab', s) for s in range(3)] +
        [('LAwdoc', 'sys', 0)] + [('LAsent', 'tab', s) for s in range(2)])

if __name__ == '__main__':
    lb_rows()
    with Pool(2) as p:
        for name in p.imap_unordered(run, JOBS):
            print(name, flush=True)
