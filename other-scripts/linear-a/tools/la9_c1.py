#!/usr/bin/env python3
"""LA-9 cycle 1: discrepancy tables and exact-likelihood Gibbs fits on Linear A and two controls
with known sub-unit values (Linear B to-so sections; Ur III szunigin grain totals from CDLI).

Outputs data/la9/c1_*.json and data/la9/c1.out . Checkpoints each run to its own json.
"""
import json, os, sys, itertools, pickle, time
import numpy as np
from multiprocessing import Pool
from la9_common import *
from la9_engine import *

LA_SITE = {'J': '1/2', 'E': '1/4', 'F': '1/8', 'A': '1/6', 'H': '1/6', 'JE': '3/4', 'L2': '3/20'}
LA_BIN = {'J': '5/16', 'E': '3/8', 'F': '1/4', 'A': '1/4', 'H': '1/4', 'JE': '1/4', 'L2': '1/16'}


def datasets():
    la = load_la(); lb = load_lb()
    pk = os.path.join(OUT, 'ur3_sections.pkl')
    ur = pickle.load(open(pk, 'rb')) if os.path.exists(pk) else load_ur3()
    return {'LA': (la, None), 'LB': (lb, LB_TRUTH), 'UR3': (ur, UR_TRUTH)}


def packed(name, secs):
    letters, P = pack(secs)
    for s in P: prep(s, len(letters))
    return letters, P


def can_balance(sec, nletters):
    """Is there any grid assignment of the section's letters with total == sum?"""
    w = sec['ct'] - sec['C'].sum(0)
    K = (sec['asum'] - sec['ta']) * U
    ls = [l for l in range(nletters) if w[l] != 0]
    if not ls: return K == 0
    last = ls[-1]; rest = ls[:-1]
    gset = set(GRID.tolist())
    if not rest:
        return (K % w[last] == 0) and (K // w[last]) in gset
    combos = np.array(list(itertools.product(range(len(GRID)), repeat=len(rest))), dtype=np.int64) \
        if len(rest) <= 3 else None
    if combos is None:
        rng = np.random.default_rng(0); combos = rng.integers(0, len(GRID), (3_000_000, len(rest)))
    part = (GRID[combos] * w[rest]).sum(1)
    need = K - part
    ok = (need % w[last] == 0)
    vals = need[ok] // w[last]
    return bool(np.isin(vals, GRID).any())


def run_one(args):
    tag, name, keep, seeds, sweeps, BG = args
    fn = os.path.join(OUT, f'c1_{tag}.json')
    if os.path.exists(fn): return json.load(open(fn))
    D = datasets(); secs, truth = D[name]
    letters, P = packed(name, secs)
    if keep == 'errors':
        P = [s for s in P if not can_balance(s, len(letters))]
    res = []
    for sd in seeds:
        r = gibbs(P, len(letters), sweeps=sweeps, burn=sweeps // 4, seed=sd, BG=BG)
        res.append(r)
    post = np.mean([r['post'] for r in res], 0); pm = np.mean([r['p'] for r in res], 0)
    out = {'tag': tag, 'name': name, 'keep': keep, 'n_sections': len(P), 'letters': letters,
           'n_frac_sections': sum(1 for s in P if s['letters_used']),
           'p_mech': dict(zip(MECH, pm.round(4).tolist())), 'BG': BG,
           'chains_agree': float(np.mean([np.argmax(r['post'], 1).tolist() == np.argmax(res[0]['post'], 1).tolist() for r in res])),
           'letters_post': {}}
    for i, l in enumerate(letters):
        order = np.argsort(-post[i])[:5]
        d = {'top': [(GRID_LABEL[g], round(float(post[i, g]), 3)) for g in order]}
        if truth:
            t = gi('%d/%d' % truth[l]); d['truth'] = GRID_LABEL[t]
            d['mass_at_truth'] = round(float(post[i, t]), 3)
            d['rank_truth'] = int((post[i] > post[i, t]).sum()) + 1
        out['letters_post'][l] = d
    if truth:
        out['recovered'] = sum(1 for l in letters if out['letters_post'][l]['rank_truth'] == 1)
        out['n_letters'] = len(letters)
    json.dump(out, open(fn, 'w'), indent=1)
    return out


def table_la(fo):
    la = load_la(); letters, P = packed('LA', la)
    print('LINEAR A KU-RO sections:', len(P), 'letters', letters, file=fo)
    sets = {'integers-only': None, 'site (lineara.xyz)': LA_SITE, 'attack-1 binary': LA_BIN}
    for nm, vs in sets.items():
        if vs is None: V = np.zeros((1, len(letters)), dtype=np.int64)
        else: V = np.array([[GRID[gi(vs[l])] for l in letters]])
        ex = 0; rows = []
        for s in P:
            m = mech_probs(s, V, len(letters))[0]
            e = s['a'] * U + s['C'] @ V[0]; S = e.sum(); T = s['ta'] * U + s['ct'] @ V[0]
            expl = [MECH[k] for k in range(NM - 1) if m[k] > 1e-3]
            ex += m[0] > 0
            rows.append(f"   {s['id']:12s} sum {S / U:9.4f} total {T / U:9.4f} diff {(T - S) / U:+9.4f}  explained by: {','.join(expl) or '-'}")
        print(f'\n[{nm}] exact {ex}/{len(P)}', file=fo)
        for r in rows: print(r, file=fo)
    cb = [s['id'] for s in P if can_balance(s, len(letters))]
    print(f'\nsections that balance under SOME grid assignment ({len(GRID)} values per letter): {len(cb)}: {cb}', file=fo)


if __name__ == '__main__':
    fo = open(os.path.join(OUT, 'c1.out'), 'a')
    print('=' * 30, time.ctime(), file=fo)
    table_la(fo); fo.flush()
    jobs = []
    for name in ['UR3', 'LB', 'LA']:
        for keep in ['all', 'errors']:
            sw = 120 if name == 'UR3' else 600
            jobs.append((f'{name}_{keep}', name, keep, [1, 2] if name == 'UR3' else [1, 2, 3, 4], sw, 1e-3))
    jobs.append(('LA_all_BG1e-2', 'LA', 'all', [1, 2, 3, 4], 600, 1e-2))
    jobs.append(('LA_all_BG1e-4', 'LA', 'all', [1, 2, 3, 4], 600, 1e-4))
    with Pool(2) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            print(json.dumps(r), file=fo); fo.flush()
            print(r['tag'], r['n_sections'], r.get('recovered'), r['p_mech'], flush=True)
