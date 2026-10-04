#!/usr/bin/env python3
"""LA-43 cycle 2b: the cycle-1 single splits again, restricted to words written only with valued signs
(the la38 convention), 10,000 relabelings per tier, with LB drawn at the same (valued-only) sizes as control
(10 draws, 500 relabelings). Decided after cycle 2 showed that valued-only LOSO behaves differently from all-word LOSO:
this is a follow-up, so its P values are reported as such (not pre-registered in cycle 1)."""
import sys, os, json, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la43_common as M
import la43_c1 as C1
from multiprocessing import Pool

TIERS = ['R2a', 'R2b', 'R3']


def valued(us):
    return [u for u in us if len(u['w']) >= 2 and all(M.L38.cv_of(s) is not None for s in u['w'])]


def make(tag):
    if tag in ('LAV_SITE', 'LAV_TIME'):
        tr, te = C1.la_split(tag[4:])
        return valued(tr), valued(te)
    kind, d = tag.split('_')[1], int(tag.split('_')[2])
    tr, te = make('LAV_' + kind)
    ntr = sum(len(u['w']) for u in tr); nte = sum(len(u['w']) for u in te)
    L = M.lb_units()
    return (M.draw_docs([u for u in L if u['site'] == 'KN'], ntr, 500 + d),
            M.draw_docs([u for u in L if u['site'] == 'PY'], nte, 600 + d))


def job(a):
    tag, tier, N, seed = a
    tr, te = make(tag)
    E = M.Experiment(tr, te)
    Pt, wt, Pn, wn, h = E.frozen()
    st = E.test.score(Pt); sn = E.test.score(Pn)
    res = M.summarise(sn, st, E.null(tier, N, seed))
    return dict(tag=tag, tier=tier, N=N, hash=h, ntr=E.ntr, ntypes=E.nte_types, res=res)


def main():
    out_p = os.path.join(M.CK, 'c2b.json')
    done = json.load(open(out_p)) if os.path.exists(out_p) else []
    have = {(d['tag'], d['tier']) for d in done}
    jobs = []
    for d in range(10):
        for kind in ('SITE', 'TIME'):
            for k, t in enumerate(TIERS):
                jobs.append((f'LBV_{kind}_{d}', t, 500, 7000 + 10 * d + k))
    for i, tag in enumerate(['LAV_SITE', 'LAV_TIME']):
        for k, t in enumerate(TIERS):
            jobs.append((tag, t, 10000, 7500 + 10 * i + k))
    fz = os.path.join(M.CK, 'c2b_freeze.json')
    if not os.path.exists(fz):
        F = {}
        for tag in sorted({j[0] for j in jobs}):
            tr, te = make(tag); E = M.Experiment(tr, te)
            F[tag] = dict(pred=E.frozen()[4], test=E.test.digest())
        json.dump(F, open(fz, 'w'), indent=1)
        print('frozen', hashlib.sha256(open(fz, 'rb').read()).hexdigest()[:16], flush=True)
    jobs = [j for j in jobs if (j[0], j[1]) not in have]
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            done.append(r); json.dump(done, open(out_p, 'w'))
            print(r['tag'], r['tier'], r['ntr'], r['ntypes'], M.fmt(r['res'], ['bits', 'mask', 'auc']), flush=True)


if __name__ == '__main__':
    main()
