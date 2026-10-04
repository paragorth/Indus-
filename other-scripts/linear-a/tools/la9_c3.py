#!/usr/bin/env python3
"""LA-9 cycle 3: power and null for the exact-likelihood (Gibbs) fit.

  UR3_lasize_K   Ur III, LA-sized subsamples (22 integer-only + 8 fraction sections), K = 1..8:
                 how often does the posterior mode hit the true barig / ban2 / sila3 value?
  UR3_errsize_K  same, but only sections that cannot balance under any grid values (errors only),
                 30 sections.
  LB_boot_K      Linear B, sections resampled with replacement (K = 1..4).
  LA_plant_K     synthetic totals on the real LA entries: lineara.xyz-like values planted, mechanism mix
                 exact 0.3, lacuna 0.3, the other 7 mechanisms 0.4 shared: can the fit get the values back?
  LA_shuf_K      Linear A with totals permuted across sections (K = 1..8): posterior
                 concentration and mechanism weights vs the real Linear A fit.
Statistic for concentration: mean over letters of the posterior mass of the modal grid value
(prior = 1/49 = 0.02), and the summed log-likelihood at the posterior mode.
Output data/la9/c3.out (json lines) and data/la9/c3_<tag>.json checkpoints.
"""
import os, sys, json, pickle
import numpy as np
from multiprocessing import Pool
from la9_common import *
from la9_engine import *
from la9_c1 import can_balance


def get(name):
    if name == 'LA': secs, truth = load_la(), None
    elif name == 'LB': secs, truth = load_lb(), LB_TRUTH
    else: secs, truth = pickle.load(open(os.path.join(OUT, 'ur3_sections.pkl'), 'rb')), UR_TRUTH
    letters, P = pack(secs)
    for s in P: prep(s, len(letters))
    return letters, P, truth


def job(args):
    tag, name, mod, seed = args
    fn = os.path.join(OUT, f'c3_{tag}.json')
    if os.path.exists(fn): return json.load(open(fn))
    letters, P, truth = get(name)
    rng = np.random.default_rng(500 + seed)
    if mod == 'lasize':
        ints = [s for s in P if s['C'].sum() == 0 and s['ct'].sum() == 0]
        fr = [s for s in P if s['C'].sum() > 0]
        P = [ints[i] for i in rng.choice(len(ints), 22, replace=False)] + [fr[i] for i in rng.choice(len(fr), 8, replace=False)]
    elif mod == 'errsize':
        err = [s for s in P if not can_balance(s, len(letters))]
        P = [err[i] for i in rng.choice(len(err), 30, replace=False)]
    elif mod == 'boot':
        P = [P[i] for i in rng.integers(0, len(P), len(P))]
    elif mod == 'shuf':
        perm = rng.permutation(len(P))
        tots = [(P[i]['ta'], P[i]['ct'].copy()) for i in perm]
        for s, (ta, ct) in zip(P, tots): s['ta'] = ta; s['ct'] = ct
    elif mod == 'plant':
        from la9_abc import simulate, setup, NMS, MECH_SIM
        SITE = {'A': '1/6', 'E': '1/4', 'F': '1/8', 'H': '1/6', 'J': '1/2', 'JE': '3/4', 'L2': '3/20'}
        p = np.full(NMS, 0.4 / (NMS - 2)); p[MECH_SIM.index('exact')] = 0.3; p[MECH_SIM.index('lacuna')] = 0.3
        vid = np.array([[gi(SITE[l]) for l in letters]])
        ti, tc = simulate(P, len(letters), setup(P, len(letters)), vid, p[None, :], np.array([0.6]), rng, raw=True)
        for k, s in enumerate(P): s['ta'] = int(ti[k, 0]); s['ct'] = tc[k, 0]
        truth = {l: tuple(int(x) for x in SITE[l].split('/')) for l in letters}
    res = [gibbs(P, len(letters), sweeps=400, burn=100, seed=seed * 10 + c) for c in range(2)]
    post = np.mean([r['post'] for r in res], 0); pm = np.mean([r['p'] for r in res], 0)
    used = sorted({l for s in P for l in s['letters_used']})
    mode = post.argmax(1)
    out = {'tag': tag, 'n_sections': len(P), 'letters': letters, 'used': [letters[l] for l in used],
           'p_mech': dict(zip(MECH, pm.round(4).tolist())),
           'conc': float(np.mean([post[l].max() for l in used])) if used else 0.0,
           'll_mode': loglik(P, len(letters), mode, pm),
           'mode': {letters[l]: GRID_LABEL[mode[l]] for l in used}}
    if truth:
        out['hit'] = {letters[l]: bool(GRID_LABEL[mode[l]] == '%d/%d' % truth[letters[l]]) for l in used}
        out['mass_truth'] = {letters[l]: round(float(post[l, gi('%d/%d' % truth[letters[l]])]), 3) for l in used}
    json.dump(out, open(fn, 'w'))
    return out


if __name__ == '__main__':
    jobs = []
    jobs.append(('LA_real', 'LA', None, 99))
    for k in range(1, 9): jobs.append((f'UR3_lasize_{k}', 'UR3', 'lasize', k))
    for k in range(1, 5): jobs.append((f'LA_plant_{k}', 'LA', 'plant', 80 + k))
    for k in range(1, 9): jobs.append((f'LA_shuf_{k}', 'LA', 'shuf', 60 + k))
    for k in range(1, 5): jobs.append((f'UR3_errsize_{k}', 'UR3', 'errsize', 20 + k))
    for k in range(1, 5): jobs.append((f'LB_boot_{k}', 'LB', 'boot', 40 + k))
    if len(sys.argv) > 1: jobs = [j for j in jobs if any(j[0].startswith(a) for a in sys.argv[1:])]
    fo = open(os.path.join(OUT, 'c3.out'), 'a')
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            print(json.dumps(r), file=fo); fo.flush()
            print(r['tag'], round(r['conc'], 3), r.get('hit'), r['mode'], flush=True)
