#!/usr/bin/env python3
"""LA-9 cycle 2: ABC with millions of simulated scribes.

Runs (each checkpointed to data/la9/abc_<tag>.npz):
  LA_real          Linear A, 4M draws
  LA_shufK         Linear A with KU-RO totals permuted across sections (K = 1..4), 1M each
  LB_real          Linear B to-so sections, 2M (truth known)
  UR3_subK         Ur III, LA-sized subsample (22 integer-only + 8 fraction sections), 1M each, K = 1..4
  UR3_big          Ur III, 300 random sections, 1M
  PLANT_<mech>     synthetic totals made on the LA entries with a planted mechanism and planted
                   values (lineara.xyz-like set), 1M each: does ABC give back the mechanism?
Summary written to data/la9/c2.out (json lines).
"""
import os, json, sys, copy
import numpy as np
from multiprocessing import Pool
from la9_common import *
from la9_engine import prep
from la9_abc import *

SITE = {'A': '1/6', 'E': '1/4', 'F': '1/8', 'H': '1/6', 'J': '1/2', 'JE': '3/4', 'L2': '3/20'}


def get(name):
    if name == 'LA': secs, truth = load_la(), None
    elif name == 'LB': secs, truth = load_lb(), LB_TRUTH
    else:
        import pickle; secs, truth = pickle.load(open(os.path.join(OUT, 'ur3_sections.pkl'), 'rb')), UR_TRUTH
    letters, P = pack(secs)
    for s in P: prep(s, len(letters))
    return letters, P, truth


def job(args):
    tag, name, ndraws, seed, mod = args
    letters, P, truth = get(name)
    rng = np.random.default_rng(1000 + seed)
    planted = None
    if mod and mod.startswith('shuf'):
        perm = rng.permutation(len(P))
        tots = [(P[i]['ta'], P[i]['ct'].copy()) for i in perm]
        for s, (ta, ct) in zip(P, tots): s['ta'] = ta; s['ct'] = ct
    elif mod in ('sub', 'big'):
        n_int, n_frac, = (22, 8) if mod == 'sub' else (100, 200)
        ints = [s for s in P if s['C'].sum() == 0 and s['ct'].sum() == 0]
        fr = [s for s in P if s['C'].sum() > 0]
        pick = [ints[i] for i in rng.choice(len(ints), min(n_int, len(ints)), replace=False)] + \
               [fr[i] for i in rng.choice(len(fr), n_frac, replace=False)]
        P = pick
    elif mod and mod.startswith('plant_'):
        mech = mod[6:]
        p = np.full(NMS, 0.0); p[MECH_SIM.index('exact')] = 0.3; p[MECH_SIM.index('lacuna')] = 0.3
        p[MECH_SIM.index(mech)] += 0.4
        vid = np.array([[gi(SITE[l]) for l in letters]])
        pool = setup(P, len(letters))
        ti, tc = simulate(P, len(letters), pool, vid, p[None, :], np.array([0.6]), rng, raw=True)
        for k, s in enumerate(P): s['ta'] = int(ti[k, 0]); s['ct'] = tc[k, 0]
        planted = {'mech': mech, 'values': SITE}
        truth = {l: tuple(int(x) for x in SITE[l].split('/')) for l in letters}
    res = abc(P, len(letters), ndraws, seed, tag=tag)
    out = describe(res, letters, truth)
    out.update({'tag': tag, 'n_sections': len(P), 'n_frac_sections': sum(1 for s in P if s['C'].sum() > 0),
                'ndraws': ndraws, 'planted': planted})
    return out


if __name__ == '__main__':
    jobs = [('LA_real', 'LA', 4_000_000, 1, None), ('LB_real', 'LB', 2_000_000, 2, None)]
    for k in range(1, 5): jobs.append((f'LA_shuf{k}', 'LA', 1_000_000, 10 + k, 'shuf'))
    for k in range(1, 5): jobs.append((f'UR3_sub{k}', 'UR3', 1_000_000, 20 + k, 'sub'))
    jobs.append(('UR3_big', 'UR3', 1_000_000, 30, 'big'))
    for m in ['carry', 'slip', 'tally', 'skip', 'misread', 'round']:
        jobs.append((f'PLANT_{m}', 'LA', 1_000_000, 40 + MECH_SIM.index(m), 'plant_' + m))
    if len(sys.argv) > 1: jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    fo = open(os.path.join(OUT, 'c2.out'), 'a')
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            print(json.dumps(r), file=fo); fo.flush()
            print(r['tag'], r['p_mech_post'], flush=True)
