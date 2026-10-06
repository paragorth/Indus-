#!/usr/bin/env python3
"""pe74 cycle 2 controls.

1. PLANTED: can the three-version audit kill a result that rests on restored / uncertain tokens?  A fake sign
   '|PLANT|' replaces one restored-or-uncertain sign on the closing line (last signed, unnumbered) of every tablet that
   has one (status kept), plus 12 READ copies on random non-closing unnumbered lines.  The pe71 closing-line test is run
   on PLANT in modes all / rd / r.  Expected: strong in all, gone in rd and r.
2. RANDOM THINNING: is a weakening under rd / r more than losing the same number of tokens at random?  For each mode,
   the same numbers of signs (to 'x' or removed) and numerals are dropped at random positions (50 draws); the pe42
   closers (A), pe71 closing formula (E) and the M157 header share (H2) are recomputed.  A real result that rests on
   the flagged tokens falls below the thinning distribution; one that loses only sample size sits inside it.
usage: python3 pe74_c2_ctrl.py -> data/pe74_ckpt/c2_ctrl.json
"""
import sys, os, json, random, copy, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pe74_parse as P
from common import base, is_sign

CK = os.path.join(P.DATA, 'pe74_ckpt')
M0 = P.load_marked()


def closing(t):
    L = t['lines']
    sl = [i for i, l in enumerate(L) if [s for s in l['signs'] if is_sign(s)]]
    if not sl or L[sl[-1]]['numerals']:
        return None
    return sl[-1]


def slot_test(T, targets, seed=7123, filt=lambda t: True):
    ps = []; obs = 0
    for t in T:
        if not filt(t):
            continue
        k = closing(t)
        L = t['lines']
        bl = [i for i, l in enumerate(L) if [s for s in l['signs'] if is_sign(s)] and not l['numerals']]
        for i, l in enumerate(L):
            for s in set(l['signs']):
                if s in targets:
                    obs += i == k
                    ps.append((k is not None) / len(bl) if bl else 0)
    ps = np.array(ps)
    if not len(ps):
        return dict(n=0, closing=0, expected=0, p=1.0)
    sim = (np.random.default_rng(seed).random((20000, len(ps))) < ps).sum(1)
    return dict(n=len(ps), closing=int(obs), expected=round(float(ps.sum()), 2), p=float((1 + (sim >= obs).sum()) / 20001))


def pe42(T):
    def nv(nums, n14):
        v = 0
        for n, s in nums:
            if s == 'N01':
                v += n
            elif s == 'N14':
                v += n14 * n
            else:
                return None
        return v
    cs = []
    for t in T:
        L = t['lines']
        ob = [l for l in L if l['surface'] == 'obverse' and l['numerals']]
        rv = [l for l in L if l['surface'] == 'reverse' and l['numerals']]
        if len(ob) < 2 or len(rv) != 1 or any(l['lacuna'] for l in L if l['surface'] == 'obverse'):
            continue
        if any(l.get('n_dropped_nums') for l in ob + rv):
            continue
        alln = [x for l in ob + rv for x in l['numerals']]
        if not alln or any(s not in ('N01', 'N14') for _, s in alln) or any(n is None for n, _ in alln):
            continue
        cs.append(([l['numerals'] for l in ob], rv[0]['numerals']))
    k = 0
    for ent, tot in cs:
        if sum(nv(e, 6) for e in ent) == nv(tot, 6) and sum(nv(e, 10) for e in ent) != nv(tot, 10):
            k += 1
    return dict(tablets=len(cs), closers=k)


def h2(T):
    heads = []
    for t in T:
        L = t['lines']
        if L and not L[0]['numerals']:
            h = [base(s) for s in L[0]['signs'] if is_sign(s)]
            if h:
                heads.append(h[0])
    return dict(headers=len(heads), m157=sum(h == 'M157' for h in heads), share=round(sum(h == 'M157' for h in heads) / len(heads), 3))


OUT = {}
# ---------------------------------------------------------------- 1 planted
rng = random.Random(7401)
T = copy.deepcopy(M0)
planted = 0
for t in T:
    k = closing(t)
    if k is None:
        continue
    l = t['lines'][k]
    idx = [i for i, (s, x) in enumerate(zip(l['signs'], l['sign_status'])) if is_sign(s) and x in ('uncertain', 'restored')]
    if idx:
        l['signs'][rng.choice(idx)] = '|PLANT|'
        planted += 1
cand = [(t, i) for t in T for i, l in enumerate(t['lines']) if i != closing(t) and not l['numerals']
        and l['signs'] and all(x == 'read' for x in l['sign_status'])]
for t, i in rng.sample(cand, 12):
    l = t['lines'][i]
    j = rng.randrange(len(l['signs']))
    l['signs'][j] = '|PLANT|'
OUT['planted'] = {'closing_plants_on_flagged_tokens': planted, 'read_decoys': 12}
for mode in ('all', 'rd', 'r'):
    T2 = P.apply_mode(copy.deepcopy(T), mode)
    OUT['planted'][mode] = slot_test(T2, {'|PLANT|'})
print('planted', OUT['planted'], flush=True)

# ---------------------------------------------------------------- 2 random thinning
for mode in ('rd', 'r'):
    keep = P.KEEP[mode]
    nx = sum(1 for t in M0 for l in t['lines'] for x in l['sign_status'] if x not in keep and x in ('damaged', 'uncertain'))
    nrm = sum(1 for t in M0 for l in t['lines'] for x in l['sign_status'] if x not in keep and x not in ('damaged', 'uncertain'))
    nn = sum(1 for t in M0 for l in t['lines'] for x in l['num_status'] if x not in keep)
    real = P.apply_mode(copy.deepcopy(M0), mode)
    res = {'real': {'A': pe42(real), 'E': slot_test(real, {'|M153+X|', '|M153+M342|'}), 'H2': h2(real)},
           'dropped': {'signs_to_x': nx, 'signs_removed': nrm, 'numerals': nn}, 'thin': []}
    sp = [(ti, li, si) for ti, t in enumerate(M0) for li, l in enumerate(t['lines']) for si in range(len(l['signs']))]
    npos = [(ti, li, ni) for ti, t in enumerate(M0) for li, l in enumerate(t['lines']) for ni in range(len(l['numerals']))]
    for d in range(50):
        r2 = random.Random(7500 + d + (0 if mode == 'rd' else 1000))
        T = copy.deepcopy(M0)
        for t in T:
            for l in t['lines']:
                l['sign_status'] = ['read'] * len(l['signs'])
                l['num_status'] = ['read'] * len(l['numerals'])
        pick = r2.sample(sp, nx + nrm)
        for j, (ti, li, si) in enumerate(pick):
            T[ti]['lines'][li]['sign_status'][si] = 'uncertain' if j < nx else 'restored'
        for ti, li, ni in r2.sample(npos, nn):
            T[ti]['lines'][li]['num_status'][ni] = 'uncertain'
        T = P.apply_mode(T, 'rd')
        res['thin'].append({'A': pe42(T), 'E': slot_test(T, {'|M153+X|', '|M153+M342|'}, seed=d), 'H2': h2(T)})
    th = res['thin']
    res['summary'] = {
        'A_closers_real': res['real']['A']['closers'], 'A_tablets_real': res['real']['A']['tablets'],
        'A_closers_thin_mean': float(np.mean([x['A']['closers'] for x in th])),
        'A_tablets_thin_mean': float(np.mean([x['A']['tablets'] for x in th])),
        'A_p_le': float(np.mean([x['A']['closers'] <= res['real']['A']['closers'] for x in th])),
        'E_real': (res['real']['E']['closing'], res['real']['E']['n']),
        'E_thin_mean': (float(np.mean([x['E']['closing'] for x in th])), float(np.mean([x['E']['n'] for x in th]))),
        'E_share_real': res['real']['E']['closing'] / max(1, res['real']['E']['n']),
        'E_share_thin': float(np.mean([x['E']['closing'] / max(1, x['E']['n']) for x in th])),
        'E_p_le_share': float(np.mean([x['E']['closing'] / max(1, x['E']['n']) <= res['real']['E']['closing'] / max(1, res['real']['E']['n']) for x in th])),
        'H2_real': res['real']['H2'], 'H2_thin_share_mean': float(np.mean([x['H2']['share'] for x in th])),
        'H2_p_le': float(np.mean([x['H2']['share'] <= res['real']['H2']['share'] for x in th]))}
    OUT['thin_' + mode] = res
    print(mode, json.dumps(res['summary']), flush=True)
json.dump(OUT, open(os.path.join(CK, 'c2_ctrl.json'), 'w'), indent=1, default=str)
