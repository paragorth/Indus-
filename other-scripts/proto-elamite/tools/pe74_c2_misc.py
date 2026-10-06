#!/usr/bin/env python3
"""pe74 cycle 2, items A, D, E, H2: small B results re-run per corpus mode (all / rd / r).

A  pe42 capacity closers: N01/N14-only tablets (>= 2 obverse entries, one reverse total, no obverse lacuna) that close
   with N14 = 6 N01 and not with N14 = 10; null = totals re-dealt among the same tablets (4,000x).  Code as pe73_c2.
   In rd / r a tablet is dropped if any used line lost a numeral (the filtered corpus marks it lacuna).
D  pe63 dossiers (frozen 8 series, pe72_lib.DOSSIERS): (i) header kept inside a series (share of members whose
   header string = the series mode, among members with a header); (ii) within-series sign-set Jaccard vs 200 draws
   of same-size groups of P-number neighbours (+-15) outside the series; (iii) M288 line after a count line inside
   the series at 60 N39C per unit (pe63 '6/6').
E  pe71 closing formula: |M153+X| / |M153+M342| tokens on the tablet's last signed line with no numerals, vs the
   within-tablet random-unnumbered-line expectation (Poisson-binomial, 100,000 sims); held out of MDP 06.
H2 test d header opener: M157 share of first-line (unnumbered) headers vs its share of entry-initial signs; z against
   1,000 draws of the same number of random entry-initial signs.
usage: python3 pe74_c2_misc.py MODE -> data/pe74_ckpt/<MODE>/c2_misc.json
"""
import sys, os, json, random, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe74_harness as H
MODE = sys.argv[1]
ROOT = H.activate(MODE)
import numpy as np
import common
from common import base, is_sign

T = common.load()
C = {t['id']: t for t in T}
OUT = {'mode': MODE}

# ------------------------------------------------------------------ A pe42
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


def cases():
    out = []
    for p, t in C.items():
        L = t['lines']
        ob = [l for l in L if l['surface'] == 'obverse' and l['numerals']]
        rv = [l for l in L if l['surface'] == 'reverse' and l['numerals']]
        if len(ob) < 2 or len(rv) != 1:
            continue
        if any(l['lacuna'] for l in L if l['surface'] == 'obverse'):
            continue
        if any(l.get('n_dropped_nums') for l in ob + rv):
            continue
        allnum = [x for l in ob + rv for x in l['numerals']]
        if not allnum or any(s not in ('N01', 'N14') for _, s in allnum) or any(n is None for n, _ in allnum):
            continue
        out.append((p, [l['numerals'] for l in ob], rv[0]['numerals']))
    return out


def closers(cs, perm=None):
    tots = [c[2] for c in cs] if perm is None else perm
    ids = []
    for (p, ent, _), tot in zip(cs, tots):
        s6 = sum(nv(e, 6) for e in ent); s10 = sum(nv(e, 10) for e in ent)
        if s6 == nv(tot, 6) and s10 != nv(tot, 10):
            ids.append(p)
    return len(ids), ids


cs = cases()
n, ids = closers(cs)
rng = random.Random(73); tots = [c[2] for c in cs]; ge = 0; mean = 0
for _ in range(4000):
    sh = tots[:]; rng.shuffle(sh); k = closers(cs, sh)[0]; mean += k; ge += k >= n
OUT['A_pe42'] = dict(tablets=len(cs), closers=n, ids=ids, null_mean=round(mean / 4000, 2), p=round((ge + 1) / 4001, 4))
print('A', OUT['A_pe42'], flush=True)

# ------------------------------------------------------------------ D dossiers
DOSS = {'D1': ['P008723', 'P008724', 'P008725', 'P008726', 'P008727', 'P008728', 'P008729', 'P008730', 'P008731'],
        'D2': ['P008796', 'P008797', 'P008798', 'P008799', 'P008800', 'P008801', 'P008802'],
        'D3': ['P009190', 'P009211', 'P009220', 'P009237', 'P009238', 'P009286', 'P009309'],
        'D4': ['P009056', 'P009137', 'P009138', 'P009140'],
        'D1b': ['P008717', 'P008718', 'P008719', 'P008720'],
        'D5': ['P008790', 'P008791', 'P008792', 'P008794'],
        'D6': ['P008100', 'P008125', 'P008193', 'P368479'],
        'D7': ['P393079', 'P393080', 'P393082']}   # = pe72_lib.DOSSIERS (pe63 B)
CAP = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720}
CNT = {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300}


def val(nums, m):
    v = 0
    for n_, c in nums:
        if c not in m or n_ is None:
            return None
        v += n_ * m[c]
    return v


def header(t):
    L = t['lines']
    if L and not L[0]['numerals']:
        h = tuple(base(s) for s in L[0]['signs'] if is_sign(s))
        return h or None
    return None


def sset(t):
    return {base(s) for l in t['lines'] for s in l['signs'] if is_sign(s)}


def jac(G):
    S = [sset(C[i]) for i in G if i in C]
    v = [len(a & b) / max(1, len(a | b)) for k, a in enumerate(S) for b in S[k + 1:]]
    return float(np.mean(v)) if v else float('nan')


ids_sorted = sorted(C, key=lambda p: int(p[1:]))
pos = {p: i for i, p in enumerate(ids_sorted)}
inany = {p for G in DOSS.values() for p in G}
rng = random.Random(7463)
D = {}
hdr_same = hdr_n = 0
rate_hit = rate_n = 0
for nm, G in DOSS.items():
    hs = [header(C[p]) for p in G if p in C]
    hs = [h for h in hs if h]
    mode_h = collections.Counter(hs).most_common(1)[0] if hs else (None, 0)
    hdr_same += mode_h[1]; hdr_n += len(hs)
    j = jac(G)
    null = []
    for _ in range(200):
        g2 = []
        for p in G:
            i = pos[p]
            cand = [ids_sorted[k] for k in range(max(0, i - 15), min(len(ids_sorted), i + 16))
                    if ids_sorted[k] not in inany and ids_sorted[k] not in g2]
            if cand:
                g2.append(rng.choice(cand))
        null.append(jac(g2))
    rh = []
    for p in G:
        L = C[p]['lines']
        for i, l in enumerate(L):
            if 'M288' in [base(s) for s in l['signs']] and i > 0 and l['numerals']:
                k = i - 1
                while k >= 0 and not L[k]['numerals']:
                    k -= 1
                if k < 0 or 'M288' in [base(s) for s in L[k]['signs']]:
                    continue
                c = val(L[k]['numerals'], CNT); m = val(l['numerals'], CAP)
                if c and m is not None and not l['lacuna'] and not L[k]['lacuna']:
                    rh.append((p, c, m, m == 60 * c))
    rate_hit += sum(x[3] for x in rh); rate_n += len(rh)
    D[nm] = dict(n=len(G), headers=len(hs), header_mode_share=(mode_h[1], len(hs)), jaccard=j,
                 null_mean=float(np.nanmean(null)), p=(1 + sum(x >= j for x in null)) / 201, rate60=rh)
OUT['D_dossiers'] = dict(series=D, header_kept=(hdr_same, hdr_n), rate60=(rate_hit, rate_n),
                         series_beating_neighbours=sum(1 for v in D.values() if v['p'] < 0.05))
print('D', {k: (v['header_mode_share'], round(v['jaccard'], 3), round(v['null_mean'], 3), round(v['p'], 3),
                [(x[0], x[1], x[2], x[3]) for x in v['rate60']]) for k, v in D.items()}, flush=True)
print('D summary', OUT['D_dossiers']['header_kept'], OUT['D_dossiers']['rate60'], OUT['D_dossiers']['series_beating_neighbours'], flush=True)

# ------------------------------------------------------------------ E pe71 closing formula
def closing(t):
    L = t['lines']
    sl = [i for i, l in enumerate(L) if [s for s in l['signs'] if is_sign(s)]]
    if not sl or L[sl[-1]]['numerals']:
        return None
    return sl[-1]


def m153_test(filt):
    ps = []; obs = 0
    for p, t in C.items():
        if not filt(t):
            continue
        k = closing(t)
        L = t['lines']
        bl = [i for i, l in enumerate(L) if [s for s in l['signs'] if is_sign(s)] and not l['numerals']]
        for i, l in enumerate(L):
            for s in set(l['signs']):
                if s in ('|M153+X|', '|M153+M342|'):
                    obs += i == k
                    ps.append((k is not None) / len(bl) if bl else 0)
    ps = np.array(ps)
    sim = (np.random.default_rng(7123).random((100000, len(ps))) < ps).sum(1)
    return dict(n=len(ps), closing=int(obs), expected=round(float(ps.sum()), 2), p=float((1 + (sim >= obs).sum()) / 100001))


OUT['E_pe71'] = dict(all=m153_test(lambda t: True), heldout_not_MDP06=m153_test(lambda t: t['volume'] != 'MDP 06'
                                                                                    if 'volume' in t else not t['designation'].startswith('MDP 06')))
print('E', OUT['E_pe71'], flush=True)

# ------------------------------------------------------------------ H2 test d (M157 opener)
heads = []; initials = []
for t in T:
    L = t['lines']
    if L and not L[0]['numerals']:
        h = [base(s) for s in L[0]['signs'] if is_sign(s)]
        if h:
            heads.append(h[0])
    for l in L[1:]:
        sg = [base(s) for s in l['signs'] if is_sign(s)]
        if sg and l['numerals']:
            initials.append(sg[0])
k = sum(h == 'M157' for h in heads)
r = np.random.default_rng(744)
draws = np.array([(r.choice(len(initials), len(heads)) == -1).sum() for _ in range(1)])
ini = np.array(initials)
null = np.array([(ini[r.choice(len(ini), len(heads))] == 'M157').sum() for _ in range(1000)])
OUT['H2_M157'] = dict(headers=len(heads), m157=k, share=round(k / len(heads), 3), entry_initial_share=round(float((ini == 'M157').mean()), 4),
                      null_mean=float(null.mean()), null_sd=float(null.std()), z=float((k - null.mean()) / max(1e-9, null.std())))
print('H2', OUT['H2_M157'], flush=True)
json.dump(OUT, open(os.path.join(ROOT, 'c2_misc.json'), 'w'), indent=1, default=str)
