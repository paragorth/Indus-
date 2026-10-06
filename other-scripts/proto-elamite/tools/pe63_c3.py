#!/usr/bin/env python3
"""pe63 cycle 3: do the dossier roles hold OUTSIDE the dossiers?
3a role transfer: signs that swap alone in dossiers (ID) vs signs that carry or change quantities (QB = COM
   variants + QTY cores). Out-of-dossier statistics per sign: adj (share of its occurrences directly next to
   the numeral: PE last sign of a numeral line, Ur III first word of a numeral line) and num (share on
   numeral lines). Prediction: QB > ID. Null: role labels permuted among labelled signs (5,000x).
   Calibration: Ur III Drehem + Umma (truth share of TIME/PERSON vs COMMODITY words among ID / QB tokens).
3b pe59 frozen classes vs dossier roles (permutation).
3c the D2 minimal pair: out of dossiers, after a counted line (n in count reading), does a line ending in
   M001 or M286 carry 48 or 60 N39C per unit (m / n in capacity units) more often than chance? Null: m
   permuted among all count->next-line pairs (10,000x). Reference: M288 and all other final signs.
3d M009 M003 out of dossiers: share with a numeral (kill line > 1/3) and share on tablets with M288.
"""
import os, json, random, re
from fractions import Fraction as Fr
from collections import Counter, defaultdict
import numpy as np
import pe63_common as C
import pe63_c2 as C2
from pe63_c2b import parse, CAP, CNT
import pe59_lib as L59

rng = random.Random(633)
res = json.load(open(os.path.join(C.CK, 'c2_res.json')))
OUT = {}


def labelled(corp):
    lab = defaultdict(Counter)
    for a in res[corp]['real']:
        for o in a['slots']:
            if o['role'] in ('ID',):
                for v in o['variants']:
                    for s in v.split():
                        lab[s]['ID'] += 1
            elif o['role'] in ('COM',):
                for v in o['variants']:
                    for s in v.split():
                        lab[s]['QB'] += 1
            elif o['role'] == 'QTY':
                for s in o['core']:
                    lab[s]['QB'] += 1
    return {s: c.most_common(1)[0][0] for s, c in lab.items() if len(c) == 1 or c.most_common(2)[0][1] > c.most_common(2)[1][1]}


def dossier_ids(corp):
    ids = set()
    for a in res[corp]['real']:
        for o in a['slots']:
            ids |= set(o['members'])
    return ids


def sign_stats(T, corp, excl):
    st = defaultdict(lambda: [0, 0, 0])   # occurrences, adjacent to numeral, on numeral line
    for t in T:
        if t['id'] in excl:
            continue
        for l in t['lines']:
            for k, s in enumerate(l['s']):
                st[s][0] += 1
                if l['sys'] is not None:
                    st[s][2] += 1
                    if (corp == 'PE' and k == len(l['s']) - 1) or (corp != 'PE' and k == 0):
                        st[s][1] += 1
    return st


def transfer(corp, T):
    lab = labelled(corp)
    st = sign_stats(T, corp, dossier_ids(corp))
    signs = [s for s in lab if st[s][0] >= 3]
    L = [lab[s] for s in signs]
    adj = np.array([st[s][1] / st[s][0] for s in signs]); num = np.array([st[s][2] / st[s][0] for s in signs])

    def gap(L, x):
        a = [v for v, l in zip(x, L) if l == 'QB']; b = [v for v, l in zip(x, L) if l == 'ID']
        return (np.mean(a) - np.mean(b)) if a and b else 0.0
    ga, gn = gap(L, adj), gap(L, num)
    na, nn = [], []
    for _ in range(5000):
        P = L[:]; rng.shuffle(P)
        na.append(gap(P, adj)); nn.append(gap(P, num))
    out = {'n_ID': L.count('ID'), 'n_QB': L.count('QB'), 'adj_ID': float(np.mean([v for v, l in zip(adj, L) if l == 'ID'] or [0])),
           'adj_QB': float(np.mean([v for v, l in zip(adj, L) if l == 'QB'] or [0])), 'gap_adj': ga,
           'p_adj': float(np.mean(np.array(na) >= ga)), 'gap_num': gn, 'p_num': float(np.mean(np.array(nn) >= gn)),
           'ID_signs': sorted(s for s, l in zip(signs, L) if l == 'ID'), 'QB_signs': sorted(s for s, l in zip(signs, L) if l == 'QB')}
    if corp != 'PE':
        # truth: the role a token takes in out-of-dossier lines
        tr = defaultdict(Counter)
        excl = dossier_ids(corp)
        for t in T:
            if t['id'] in excl:
                continue
            for l in t['lines']:
                r = C.ur3_role(l)
                for s in set(l['s']):
                    tr[s][r] += 1
        def share(group, roles):
            v = [sum(tr[s][r] for r in roles) / max(sum(tr[s].values()), 1) for s, l in zip(signs, L) if l == group]
            return float(np.mean(v)) if v else 0.0
        out['truth'] = {'ID_timeperson': share('ID', ('TIME', 'PERSON')), 'QB_timeperson': share('QB', ('TIME', 'PERSON')),
                        'ID_commodity': share('ID', ('COMMODITY',)), 'QB_commodity': share('QB', ('COMMODITY',))}
    return out


# ---------------------------------------------------------------- 3a
for corp in ['PE', 'DR', 'UM']:
    T = {'PE': C.pe_tabs, 'DR': lambda: C.ur3_tabs('Puzr', 1500, 'DR'),
         'UM': lambda: C.ur3_tabs('Umma', 1500, 'UM')}[corp]()
    OUT[corp] = transfer(corp, T)
    print(corp, json.dumps({k: v for k, v in OUT[corp].items() if 'signs' not in k}))
print('PE ID signs', OUT['PE']['ID_signs']); print('PE QB signs', OUT['PE']['QB_signs'])

# ---------------------------------------------------------------- 3b
R = L59.pe_roles()['sets']
QSET = R['MEASURED'] | R['COUNTED'] | R['ALLOT'] | R['FRACLINE']
NSET = R['PERSON'] | R['PREFIX']
lab = labelled('PE')
ls = [s for s in lab if s in QSET | NSET]


def agree(L):
    return sum((l == 'QB' and s in QSET and s not in NSET) or (l == 'ID' and s in NSET and s not in QSET) for s, l in zip(ls, L))


L = [lab[s] for s in ls]
g = agree(L)
nul = []
for _ in range(5000):
    P = L[:]; rng.shuffle(P); nul.append(agree(P))
OUT['pe59'] = {'n': len(ls), 'agree': g, 'null_mean': float(np.mean(nul)), 'p': float(np.mean(np.array(nul) >= g)),
               'rows': {s: [lab[s], 'Q' if s in QSET else '', 'N' if s in NSET else ''] for s in ls}}
print('pe59', json.dumps(OUT['pe59']))

# ---------------------------------------------------------------- 3c
T = C.pe_tabs()
excl = dossier_ids('PE')
pairs = []
for t in T:
    if t['id'] in excl:
        continue
    Ls = t['lines']
    for i in range(len(Ls) - 1):
        a, b = Ls[i], Ls[i + 1]
        if a['sys'] is None or b['sys'] is None or not b['s']:
            continue
        n = parse(a.get('raw'), CNT)
        m = parse(b.get('raw'), CAP)
        if n and m and n == int(n) and n >= 1:
            pairs.append((b['s'][-1], n, m, t['id']))
TARGET = {Fr(48), Fr(60)}


def hits(P, sign):
    return sum(1 for s, n, m, _ in P if (s == sign if sign != '*' else s not in ('M288', 'M001', 'M286')) and m / n in TARGET)


ms = [p[2] for p in pairs]
c3 = {}
for sign in ['M001', 'M286', 'M288', '*']:
    h = hits(pairs, sign)
    tot = sum(1 for p in pairs if (p[0] == sign if sign != '*' else p[0] not in ('M288', 'M001', 'M286')))
    nl = []
    for _ in range(10000 if sign != '*' else 2000):
        rng.shuffle(ms)
        nl.append(hits([(p[0], p[1], m, p[3]) for p, m in zip(pairs, ms)], sign))
    c3[sign] = {'pairs': tot, 'hits_48_60': h, 'null_mean': float(np.mean(nl)), 'p': float(np.mean(np.array(nl) >= h)),
                'examples': [(p[3], str(p[1]), str(p[2])) for p in pairs if (p[0] == sign) and p[2] / p[1] in TARGET][:8]}
    print('3c', sign, json.dumps(c3[sign]))
OUT['rates'] = c3

# ---------------------------------------------------------------- 3d
md = {'lines': 0, 'with_num': 0, 'tabs': 0, 'tabs_M288': 0, 'last_line': 0}
tabs_all_288 = 0
for t in T:
    if t['id'] in excl:
        continue
    has288 = any('M288' in l['s'] for l in t['lines'])
    tabs_all_288 += has288
    seen = False
    for k, l in enumerate(t['lines']):
        if 'M009' in l['s'] and 'M003' in l['s']:
            md['lines'] += 1; md['with_num'] += l['sys'] is not None
            md['last_line'] += (k == len(t['lines']) - 1)
            seen = True
    if seen:
        md['tabs'] += 1; md['tabs_M288'] += has288
md['base_rate_M288'] = tabs_all_288 / max(1, sum(t['id'] not in excl for t in T))
OUT['M009M003'] = md
print('3d', md)
json.dump(OUT, open(os.path.join(C.CK, 'c3_res.json'), 'w'), indent=1, default=str)
