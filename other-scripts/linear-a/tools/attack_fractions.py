#!/usr/bin/env python3
"""Joint search for Linear A fraction-sign values.

Inputs : data/corpus.json (tools/build_corpus.py), section finder from tools/totals_test.py.
Outputs: data/attack_fractions.json, data/attack_fractions_report.txt (written by hand from the printout).

Constraints used:
  order   every compound fraction (2+ letters on one quantity) gives "a written before b".
          Rule 'larger' = a > b strictly; rule 'smaller' = a < b strictly. Hard (0 violations).
  sum     every compound must add to < 1 (a whole unit would be written as an integer).
          Hard or soft (counted) depending on the run.
  totals  the 7 KU-RO sections with fractions (auto split, totals_test.py) plus one hand split
          column (HT 123+124a, *308 column: 8E 8JE 4A 4E = KU-RO 25H). Each balanced section scores 1.
Values  : pool of unit fractions 1/d, d in D, and simple non-unit fractions n/d, d <= 16.
Parsimony cost of a value n/d = log2(d) + 2*(n>1). A system's cost = sum over letters.
Families: binary (d = 2^k), sexagesimal (d | 60), duodecimal (d | 48), any.
Values are kept as integers over 960 (= lcm of all d).
"""
import json, os, sys, math, random, itertools, time
from fractions import Fraction as Fr
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import totals_test as TT

C = TT.C
U = 960
D_ALL = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 32, 48, 60, 64]
NONUNIT_D = {3, 4, 5, 6, 8, 10, 12, 16}
FAMILIES = {
    'binary': [d for d in D_ALL if d & (d - 1) == 0],
    'sexagesimal': [d for d in D_ALL if 60 % d == 0],
    'duodecimal': [d for d in D_ALL if 48 % d == 0],
    'any': D_ALL,
}
MAIN = {'GRA', 'VIN', 'OLE', 'OLIV', 'CYP', 'VIR', 'AROM', 'HIDE', 'NI'}
GROUP = {'GRA': 'dry/liquid', 'VIN': 'dry/liquid', 'OLE': 'dry/liquid', 'OLIV': 'dry/liquid',
         'AROM': 'dry/liquid', 'VIR': 'dry/liquid', 'CYP': 'CYP/NI', 'NI': 'CYP/NI', 'HIDE': 'HIDE', None: 'unknown'}
CONV = {'J': Fr(1, 2), 'E': Fr(1, 4), 'F': Fr(1, 8), 'K': Fr(1, 16), 'D': Fr(1, 5), 'B': Fr(1, 3),
        'A': Fr(1, 6), 'H': Fr(1, 6), 'JE': Fr(3, 4)}


def pool(family):
    out = {}
    for d in FAMILIES[family]:
        for n in range(1, d):
            if math.gcd(n, d) != 1: continue
            if n > 1 and d not in NONUNIT_D: continue
            f = Fr(n, d)
            c = math.log2(d) + (2 if n > 1 else 0)
            if f not in out or c < out[f]: out[f] = c
    return sorted(((int(f * U), c, f) for f, c in out.items()), key=lambda x: (x[1], -x[0]))


# ---------------------------------------------------------------- data
def compounds(corpus=C):
    rows = []
    for ins in corpus:
        cur = None
        for t in ins['tokens']:
            if t['t'] == 'logo':
                b = t['v'].split('+')[0].lstrip('*')
                if b in MAIN: cur = b
            elif t['t'] == 'word' and t['s'] == ['NI']:
                cur = 'NI'
            elif t['t'] == 'num' and len(t['frac']) >= 2:
                rows.append({'id': ins['id'], 'com': cur, 'grp': GROUP.get(cur, 'unknown'), 'f': list(t['frac']), 'v': t['v']})
    return rows


def lin(q):
    """quantity -> (integer part, Counter of letters)"""
    return q['v'], Counter(q['frac'])


def sections():
    ku = [s for s in TT.sections('KU-RO') if s['entries'] and s['tot']]
    fr = [s for s in ku if s['tot']['frac'] or any(e['frac'] for e in s['entries'])]
    out = []
    for k, s in enumerate(fr):
        i0 = sum(e['v'] for e in s['entries']) - s['tot']['v']
        co = Counter()
        for e in s['entries']: co.update(e['frac'])
        co.subtract(Counter(s['tot']['frac']))
        out.append({'id': s['id'] + ('' if [x['id'] for x in fr].count(s['id']) == 1 else f'#{[x["id"] for x in fr[:k + 1]].count(s["id"])}'),
                    'const': i0, 'coef': {l: c for l, c in co.items() if c}, 'auto': True,
                    'text': ' + '.join(f"{e['v']}{''.join(e['frac'])}" for e in s['entries']) + f" = {s['tot']['v']}{''.join(s['tot']['frac'])}"})
    # hand split column, HT 123+124a *308: 8E + 8JE + 4A + 4E = 25H
    out.append({'id': 'HT123+124a *308 column (hand split)', 'const': 24 - 25,
                'coef': {'E': 2, 'JE': 1, 'A': 1, 'H': -1}, 'auto': False, 'text': '8E + 8JE + 4A + 4E = 25H'})
    return out


def sec_balanced(s, val):
    tot = s['const'] * U + sum(c * val[l] for l, c in s['coef'].items())
    return tot == 0


# ---------------------------------------------------------------- solver (mixed-integer program, HiGHS)
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

SEC_GRP = {'HT9a': 'dry/liquid', 'HT13': 'dry/liquid', 'HT123+124a#1': 'dry/liquid', 'HT123+124a#2': 'dry/liquid'}


def pairs_of(comps):
    p = Counter()
    for c in comps:
        f = c['f']
        for a in range(len(f)):
            for b in range(a + 1, len(f)):
                if f[a] != f[b]: p[(f[a], f[b])] += 1
    return p


def is_cyclic(pairs):
    g = defaultdict(set); nodes = set()
    for a, b in pairs: g[a].add(b); nodes |= {a, b}
    indeg = Counter()
    for a in g:
        for b in g[a]: indeg[b] += 1
    q = [n for n in nodes if indeg[n] == 0]; k = 0
    while q:
        n = q.pop(); k += 1
        for b in g[n]:
            indeg[b] -= 1
            if indeg[b] == 0: q.append(b)
    return k < len(nodes)


def solve(comps, secs, direction, family='any', hard_sum=True, require=(), priority='sum',
          topk=1, by_group=False, objective='cost', fixed=None, time_limit=20, distinct=False):
    """Return up to topk systems ranked by priority:
         'sum' : fewest compounds >= 1, then most balanced sections, then lowest cost
         'bal' : most balanced, then fewest compounds >= 1, then lowest cost
       by_group: each commodity group gets its own copy of every letter.
       objective=('min'|'max', letter): bound search instead of cost."""
    P = pool(family)
    nv = len(P)
    lname = lambda l, g: f'{l}@{g}' if by_group else l
    comps2 = [dict(c, f=[lname(l, c['grp']) for l in c['f']]) for c in comps]
    secs2 = [dict(s, coef={lname(l, SEC_GRP.get(s['id'], 'unknown')): c for l, c in s['coef'].items()}) for s in secs]
    letters = sorted({l for c in comps2 for l in c['f']} | {l for s in secs2 for l in s['coef']})
    li = {l: i for i, l in enumerate(letters)}
    nL = len(letters)
    pairs = pairs_of(comps2)
    nx = nL * nv
    ns = 0 if hard_sum else len(comps2)
    nb = len(secs2)
    N = nx + ns + nb
    vals = np.array([p[0] for p in P], float)
    costs = np.array([p[1] for p in P], float)
    rows, lo, hi = [], [], []

    def row(): return np.zeros(N)
    def valrow(l, coef=1.0, r=None):
        r = row() if r is None else r
        r[li[l] * nv: li[l] * nv + nv] += coef * vals
        return r
    for l in letters:                       # one value per letter
        r = row(); r[li[l] * nv: li[l] * nv + nv] = 1; rows.append(r); lo.append(1); hi.append(1)
    for (a, b) in pairs:                    # strict order
        r = valrow(a, 1); r = valrow(b, -1, r)
        if direction == 'larger': rows.append(r); lo.append(1); hi.append(np.inf)
        else: rows.append(r); lo.append(-np.inf); hi.append(-1)
    for k, c in enumerate(comps2):          # compound < 1
        r = row()
        for l in c['f']: valrow(l, 1, r)
        if hard_sum: rows.append(r); lo.append(-np.inf); hi.append(U - 1)
        else:
            r[nx + k] = -4 * U; rows.append(r); lo.append(-np.inf); hi.append(U - 1)
    for k, s in enumerate(secs2):           # balanced indicator
        M = (abs(s['const']) + sum(abs(c) for c in s['coef'].values()) + 1) * U
        r = row()
        for l, c in s['coef'].items(): valrow(l, c, r)
        req = s['id'] in require
        r1 = r.copy(); r1[nx + ns + k] = M; rows.append(r1); lo.append(-np.inf); hi.append(-s['const'] * U + M)
        r2 = r.copy(); r2[nx + ns + k] = -M; rows.append(r2); lo.append(-s['const'] * U - M); hi.append(np.inf)
    if distinct:                            # different signs, different values
        for j in range(nv):
            r = row(); r[[li[l] * nv + j for l in letters]] = 1; rows.append(r); lo.append(0); hi.append(1)
    if fixed:
        for l, v in fixed.items():
            if l in li:
                r = valrow(l, 1); rows.append(r); lo.append(v); hi.append(v)
    lb = np.zeros(N); ub = np.ones(N)
    for k, s in enumerate(secs2):
        if s['id'] in require: lb[nx + ns + k] = 1
    obj = np.zeros(N)
    if objective == 'cost':
        for l in letters: obj[li[l] * nv: li[l] * nv + nv] = costs
        W1, W2 = (1e6, 1e4) if priority == 'sum' else (1e4, 1e6)
        obj[nx:nx + ns] = W1
        obj[nx + ns:] = -W2
    else:
        sgn, l = objective
        if l not in li: return []
        obj[li[l] * nv: li[l] * nv + nv] = vals * (1 if sgn == 'min' else -1)
    out = []
    A = list(rows); LO = list(lo); HI = list(hi)
    for _ in range(topk):
        res = milp(obj, constraints=LinearConstraint(np.array(A), LO, HI), integrality=np.ones(N),
                   bounds=Bounds(lb, ub), options={'time_limit': time_limit})
        if res.x is None: break
        x = np.round(res.x).astype(int)
        val = {}; cost = 0.0; chosen = []
        for l in letters:
            j = int(np.argmax(x[li[l] * nv: li[l] * nv + nv])); val[l] = int(vals[j]); cost += costs[j]
            chosen.append(li[l] * nv + j)
        sv = len([c for c in comps2 if sum(val[l] for l in c['f']) >= U])
        bal = [s['id'] for s in secs2 if s['coef'] and s['const'] * U + sum(c * val[l] for l, c in s['coef'].items()) == 0]
        out.append({'values': val, 'sum_violations': sv, 'balanced': bal, 'cost': round(cost, 2),
                    'optimal': res.status == 0})
        cut = np.zeros(N); cut[chosen] = 1
        A.append(cut); LO.append(-np.inf); HI.append(nL - 1)
    return out


def fmt(val):
    return {l: str(Fr(v, U)) for l, v in sorted(val.items(), key=lambda x: (-x[1], x[0]))}


def gap(s, val):
    return Fr(s['const'] * U + sum(c * val[l] for l, c in s['coef'].items()), U)


# ---------------------------------------------------------------- main
def main():
    random.seed(7)
    comps = compounds()
    secs = sections()
    R = {'n_compounds': len(comps),
         'sections': [{k: s[k] for k in ('id', 'text', 'const', 'coef', 'auto')} for s in secs]}
    print('compounds', len(comps))
    for s in secs: print('  ', s['id'], ':', s['text'], ' -> needs', s['coef'], '=', -s['const'])
    pairs = pairs_of(comps)
    R['pairs'] = {f'{a}>{b}': n for (a, b), n in pairs.items()}

    cv = {l: int(f * U) for l, f in CONV.items()}
    R['conventional'] = {}
    for d in ('larger', 'smaller'):
        v = sum(n for (a, b), n in pairs.items() if a in cv and b in cv and not (cv[a] > cv[b] if d == 'larger' else cv[a] < cv[b]))
        t = sum(n for (a, b), n in pairs.items() if a in cv and b in cv)
        R['conventional'][d] = f'{v}/{t}'
        print(f'conventional values, {d}-first violations: {v}/{t} pair instances')
    R['conventional']['sums_ge_1'] = [f"{c['id']} {'+'.join(c['f'])}" for c in comps if all(l in cv for l in c['f']) and sum(cv[l] for l in c['f']) >= U]
    R['conventional']['balanced'] = [s['id'] for s in secs if set(s['coef']) <= set(cv) and gap(s, cv) == 0]
    print('conventional:', R['conventional'])

    # ---- 1a. what can balance at all: each section alone, any family, both directions, sums soft
    R['per_section'] = {}
    for s in secs:
        for d in ('larger', 'smaller'):
            for hard in (True, False):
                r = solve(comps, secs, d, 'any', hard_sum=hard, require=[s['id']], priority='sum')
                if r: break
            R['per_section'][f"{s['id']}/{d}"] = ({'hard_sum': hard, 'sum_violations': r[0]['sum_violations'],
                                                   'values': fmt(r[0]['values'])} if r else 'impossible')
            print(f"  can {s['id']} balance ({d}-first)?", 'NO' if not r else
                  f"yes, compounds>=1: {r[0]['sum_violations']}")

    # ---- 1b. main grid: direction x family x priority
    R['grid'] = []
    for d in ('larger', 'smaller'):
        for fam, dist in [(f, False) for f in FAMILIES] + [(f, True) for f in FAMILIES]:
            for pri, hard in (('sum', True), ('bal', False)):
                r = solve(comps, secs, d, fam, hard_sum=hard, priority=pri, topk=10, distinct=dist)
                if not r:
                    R['grid'].append({'dir': d, 'family': fam, 'distinct': dist, 'priority': pri, 'feasible': False})
                    print(f"{d:7s} {fam:11s} distinct={dist} {pri}: infeasible"); continue
                for x in r: x['values_str'] = fmt(x['values'])
                R['grid'].append({'dir': d, 'family': fam, 'distinct': dist, 'priority': pri, 'feasible': True,
                                  'top': [{k: x[k] for k in ('values_str', 'sum_violations', 'balanced', 'cost', 'optimal')} for x in r]})
                b = r[0]
                print(f"{d:7s} {fam:11s} distinct={dist} {pri}: sum>=1 {b['sum_violations']} balanced {len(b['balanced'])} {b['balanced']} cost {b['cost']} {b['values_str']}")
                # pinned among top 10
                vs = defaultdict(set)
                for x in r:
                    for l, v in x['values'].items(): vs[l].add(v)
                pins = sorted(l for l, v in vs.items() if len(v) == 1)
                R['grid'][-1]['pinned_top10'] = {l: str(Fr(list(vs[l])[0], U)) for l in pins}
                R['grid'][-1]['range_top10'] = {l: [str(Fr(min(v), U)), str(Fr(max(v), U))] for l, v in vs.items()}
                print('     pinned in top 10:', R['grid'][-1]['pinned_top10'])

    # ---- 1c. bounds from order + sum<1 alone (any family)
    R['bounds'] = {}
    for d in ('larger', 'smaller'):
        bd = {}
        for l in sorted({l for c in comps for l in c['f']}):
            mn = solve(comps, [], d, 'any', objective=('min', l))
            mx = solve(comps, [], d, 'any', objective=('max', l))
            bd[l] = [str(Fr(mn[0]['values'][l], U)), str(Fr(mx[0]['values'][l], U))] if mn else None
        R['bounds'][d] = bd
        print('range allowed by order + sum<1,', d, bd)

    # ---- 1d. commodity-specific letters
    R['by_group'] = {}
    for d in ('larger', 'smaller'):
        for pri, hard in (('sum', True), ('bal', False)):
            r = solve(comps, secs, d, 'any', hard_sum=hard, priority=pri, by_group=True)
            g = solve(comps, secs, d, 'any', hard_sum=hard, priority=pri)
            if r and g:
                R['by_group'][f'{d}/{pri}'] = {
                    'grouped': {'sum_violations': r[0]['sum_violations'], 'balanced': r[0]['balanced'], 'cost': r[0]['cost'],
                                'n_params': len(r[0]['values']), 'values': fmt(r[0]['values'])},
                    'global': {'sum_violations': g[0]['sum_violations'], 'balanced': g[0]['balanced'], 'cost': g[0]['cost'],
                               'n_params': len(g[0]['values'])}}
                print('by commodity', d, pri, 'grouped:', r[0]['sum_violations'], r[0]['balanced'], len(r[0]['values']), 'params',
                      '| global:', g[0]['sum_violations'], g[0]['balanced'], len(g[0]['values']), 'params')

    # held-out (a): order learned on half the compounds predicts the other half? global vs by group
    hold = {'global': [0, 0, 0], 'by_commodity': [0, 0, 0]}
    for rep in range(500):
        idx = list(range(len(comps))); random.shuffle(idx)
        tr = [comps[i] for i in idx[:len(idx) // 2]]; te = [comps[i] for i in idx[len(idx) // 2:]]
        for mode in hold:
            for c in te:
                p = pairs_of(tr if mode == 'global' else [x for x in tr if x['grp'] == c['grp']])
                f = c['f']
                for a in range(len(f)):
                    for b in range(a + 1, len(f)):
                        if f[a] == f[b]: continue
                        if p[(f[a], f[b])] > p[(f[b], f[a])]: hold[mode][0] += 1
                        elif p[(f[b], f[a])] > p[(f[a], f[b])]: hold[mode][1] += 1
                        else: hold[mode][2] += 1
    R['heldout_order'] = {m: dict(zip(('right', 'wrong', 'no_prediction'), v)) for m, v in hold.items()}
    print('held-out order (500 half splits):', R['heldout_order'])
    # held-out (b): fit values on the other balanceable sections, predict the held-out one
    balanceable = [k.split('/')[0] for k, v in R['per_section'].items() if v != 'impossible' and k.endswith('/larger')]
    R['balanceable_larger'] = balanceable
    R['heldout_totals'] = []
    for d in ('larger', 'smaller'):
        for out_id in [s['id'] for s in secs]:
            rest = [s for s in secs if s['id'] != out_id]
            for grp in (False, True):
                r = solve(comps, rest, d, 'any', hard_sum=False, priority='bal', by_group=grp)
                if not r: continue
                v = r[0]['values']
                s = [x for x in secs if x['id'] == out_id][0]
                lname = (lambda l: f"{l}@{SEC_GRP.get(out_id, 'unknown')}") if grp else (lambda l: l)
                if not all(lname(l) in v for l in s['coef']):
                    ok = None
                else:
                    ok = s['const'] * U + sum(c * v[lname(l)] for l, c in s['coef'].items()) == 0
                R['heldout_totals'].append({'dir': d, 'held_out': out_id, 'by_group': grp, 'predicted_balanced': ok,
                                            'fit_balanced': r[0]['balanced']})
    for d in ('larger', 'smaller'):
        for grp in (False, True):
            xs = [x for x in R['heldout_totals'] if x['dir'] == d and x['by_group'] == grp]
            print(f'held-out totals {d} by_group={grp}: predicted balanced',
                  sum(1 for x in xs if x['predicted_balanced']), '/', len(xs),
                  [x['held_out'] for x in xs if x['predicted_balanced']])

    # ---- 2. control: relabel letters across all fraction tokens (counts kept)
    toks = [t for ins in C for t in ins['tokens'] if t['t'] == 'num' and t['frac']]
    allf = [f for t in toks for f in t['frac']]
    saved = [list(t['frac']) for t in toks]
    obs = {}
    for d in ('larger', 'smaller'):
        r = solve(comps, secs, d, 'any', hard_sum=True, priority='sum')
        obs[d] = {'feasible': bool(r), 'balanced': len(r[0]['balanced']) if r else None}
    NREP = 500
    stats = Counter(); balc = Counter()
    for rep in range(NREP):
        sh = allf[:]; random.shuffle(sh); k = 0
        for t in toks:
            n = len(t['frac']); t['frac'] = sh[k:k + n]; k += n
        cp = compounds(); sc = sections()
        cyc = is_cyclic(pairs_of(cp))
        stats['acyclic'] += not cyc
        any_ok = False
        for d in ('larger', 'smaller'):
            if cyc: continue
            r = solve(cp, sc, d, 'any', hard_sum=True, priority='sum', time_limit=5)
            if r:
                stats[f'feasible_{d}'] += 1; any_ok = True
                balc[(d, len(r[0]['balanced']))] += 1
        stats['feasible_either'] += any_ok
    for t, f in zip(toks, saved): t['frac'] = f
    R['control'] = {'reps': NREP, 'observed': obs, 'random': dict(stats),
                    'random_balanced_counts': {f'{a}/{b}': n for (a, b), n in balc.items()}}
    print('2. control:', R['control'])

    # ---- 3. predictions for unbalanced / damaged totals under the top systems
    R['predictions'] = {}
    for d in ('larger', 'smaller'):
        for fam in ('binary', 'sexagesimal', 'any'):
            r = solve(comps, secs, d, fam, hard_sum=True, priority='sum')
            if not r: continue
            v = r[0]['values']
            inv = defaultdict(list)
            for l, x in v.items(): inv[x].append(l)
            rows = []
            for s in secs:
                if not s['coef'] or not all(l in v for l in s['coef']): continue
                g = gap(s, v)
                # what fraction the written total would need (entries - integer total) and which letter has it
                tot_letters = [l for l, c in s['coef'].items() if c < 0]
                ent = s['const'] * U + sum(c * v[l] for l, c in s['coef'].items() if c > 0)
                need = Fr(ent, U)
                rows.append({'id': s['id'], 'text': s['text'], 'entries_minus_total': str(g),
                             'entries_minus_integer_total': str(need),
                             'letter_with_that_fraction': inv.get(int((need - int(need)) * U), []) if need > 0 else []})
            R['predictions'][f'{d}/{fam}'] = {'values': fmt(v), 'sections': rows}
            print(f'\n3. {d}/{fam}', fmt(v))
            for x in rows: print('    ', x)

    json.dump(R, open(os.path.join(HERE, '..', 'data', 'attack_fractions.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
