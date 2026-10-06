#!/usr/bin/env python3
"""pe63 cycle 2: inside each dossier, align the tablets line by line and type every variation as a
minimal pair.

Per dossier: reference = medoid tablet; every member aligned to it (Needleman-Wunsch on lines; score =
Jaccard of sign sets + 0.3 if both numeral or both header; gap -0.3). Per reference slot: core = signs in
>= 70% of aligned lines; variant = the rest. For each member pair present at the slot:
  sign change (variants differ), system change, scale change (|log2 ratio| >= log2 3), quantity change.
Slot roles:
  CONST     no sign change, no quantity change
  QTY       signs never change, the number does (a fixed item, variable amount)
  ID        sign changes; in >= 60% of sign-changing pairs neither system nor quantity changes
  COM       sign changes; in >= 50% of sign-changing pairs the system or scale (x3) changes
  STEP      sign changes, same system, quantity a function of the variant (>= 3 variants, consistent),
            sorted values form a constant ratio or constant difference (10% tolerance)
  MIX       anything else
  TOTAL     (numeral slot) value = sum of the other same-system numeral slots on >= 2 members
Controls: Ur III Drehem / Umma dossiers (truth role of each slot from the words: TIME / PERSON / COMMODITY),
planted PE dossiers (A=ID, B=COM, C=STEP), and a number-shuffle null (numbers permuted among members at
each slot) for every corpus.
usage: pe63_c2.py
"""
import os, json, math, random
from collections import Counter, defaultdict
import numpy as np
import pe63_common as C
from pe63_report1 import thresholds, components


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def align(R, X):
    n, m = len(R), len(X)
    D = np.zeros((n + 1, m + 1)); P = np.zeros((n + 1, m + 1), dtype=int)
    g = -0.3
    for i in range(1, n + 1):
        D[i, 0] = i * g; P[i, 0] = 1
    for j in range(1, m + 1):
        D[0, j] = j * g; P[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = jac(R[i - 1]['s'], X[j - 1]['s']) + (0.3 if (R[i - 1]['sys'] is None) == (X[j - 1]['sys'] is None) else -0.3)
            c = [D[i - 1, j - 1] + s - 0.4, D[i - 1, j] + g, D[i, j - 1] + g]
            k = int(np.argmax(c)); D[i, j] = c[k]; P[i, j] = k
    i, j, mp = n, m, {}
    while i > 0 or j > 0:
        k = P[i, j] if i > 0 and j > 0 else (1 if i > 0 else 2)
        if k == 0:
            mp[i - 1] = j - 1; i -= 1; j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return mp


def const_ratio(vals):
    v = sorted(vals)
    if len(v) < 3 or v[0] <= 0:
        return False
    r = [v[i + 1] / v[i] for i in range(len(v) - 1)]
    d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    okr = max(r) / min(r) <= 1.1 and min(r) > 1.05
    okd = min(d) > 0 and max(d) / min(d) <= 1.1
    return okr or okd


def analyse(dossier_tabs, rng=None, shuffle_nums=False):
    T = dossier_tabs
    n = len(T)
    sim = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                sim[a, b] = np.mean([max([jac(l['s'], m['s']) for m in T[b]['lines']] or [0]) for l in T[a]['lines']])
    ref = int(np.argmax(sim.sum(1)))
    R = T[ref]['lines']
    slots = defaultdict(dict)   # slot -> member -> line
    for k, t in enumerate(T):
        mp = {i: i for i in range(len(R))} if k == ref else align(R, t['lines'])
        for i, j in mp.items():
            slots[i][k] = t['lines'][j]
    out = []
    for i in range(len(R)):
        S = slots[i]
        if len(S) < 2:
            continue
        mem = sorted(S)
        cnt = Counter(s for k in mem for s in set(S[k]['s']))
        core = {s for s, c in cnt.items() if c >= 0.7 * len(mem)}
        var = {k: frozenset(set(S[k]['s']) - core) for k in mem}
        sysv = {k: S[k]['sys'] for k in mem}
        vv = {k: S[k]['v'] for k in mem}
        if shuffle_nums:
            ks = list(mem); perm = ks[:]; rng.shuffle(perm)
            sysv = {k: S[p]['sys'] for k, p in zip(ks, perm)}
            vv = {k: S[p]['v'] for k, p in zip(ks, perm)}
        sc = sy = scl = qc = 0
        alone = withu = 0
        anyq = False
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                x, y = mem[a], mem[b]
                sch = var[x] != var[y] and bool(var[x] | var[y])
                syc = sysv[x] != sysv[y]
                q = vv[x] is not None and vv[y] is not None
                qch = q and abs(vv[x] - vv[y]) > 1e-9
                scch = q and vv[x] > 0 and vv[y] > 0 and abs(math.log2(vv[x] / vv[y])) >= math.log2(3)
                anyq |= bool(qch or syc)
                if sch:
                    sc += 1
                    if syc or scch:
                        withu += 1
                    if not syc and not qch:
                        alone += 1
        numeral = sum(sysv[k] is not None for k in mem) >= 0.5 * len(mem)
        role = 'CONST'
        if sc == 0:
            role = 'QTY' if anyq else 'CONST'
        else:
            vmap = defaultdict(set)
            for k in mem:
                if vv[k] is not None:
                    vmap[var[k]].add(vv[k])
            consistent = all(len(s) == 1 for s in vmap.values())
            vals = [next(iter(s)) for s in vmap.values() if len(s) == 1]
            same_sys = len({sysv[k] for k in mem}) == 1
            if alone >= 0.6 * sc:
                role = 'ID'
            elif withu >= 0.5 * sc:
                role = 'COM'
            elif numeral and same_sys and consistent and len(vmap) >= 3 and len(set(vals)) == len(vals) and const_ratio(vals):
                role = 'STEP'
            else:
                role = 'MIX'
        out.append({'slot': i, 'n': len(mem), 'core': sorted(core),
                    'variants': sorted({' '.join(sorted(v)) for v in var.values()}),
                    'numeral': numeral, 'role': role, 'sc': sc, 'alone': alone, 'withu': withu,
                    'members': {T[k]['id']: {'s': S[k]['s'], 'sys': sysv[k], 'v': vv[k]} for k in mem}})
    # TOTAL detection
    for o in out:
        if not o['numeral']:
            continue
        hit = 0
        for tid, x in o['members'].items():
            if x['v'] is None:
                continue
            others = [p['members'][tid]['v'] for p in out if p is not o and tid in p['members']
                      and p['members'][tid]['sys'] == x['sys'] and p['members'][tid]['v'] is not None]
            if len(others) >= 2 and abs(sum(others) - x['v']) < 1e-6:
                hit += 1
        if hit >= 2:
            o['total'] = hit
    return {'ref': T[ref]['id'], 'slots': out}


def dossiers_for(corp, tag='real', s=0):
    fn = os.path.join(C.CK, 'c1_%s_%s_%d.json' % (corp, tag, s))
    R = json.load(open(fn))
    thr = thresholds(corp)
    _, comps = components(R['groups'], thr)
    if tag == 'plant':
        T = R['tabs']
    else:
        T = {'PE': C.pe_tabs, 'DR': lambda: C.ur3_tabs('Puzr', 1500, 'DR'),
             'UM': lambda: C.ur3_tabs('Umma', 1500, 'UM')}[corp]()
    byid = {t['id']: t for t in T}
    return [[byid[i] for i in d['ids']] for d in comps if 3 <= len(d['ids']) <= 40], R.get('truth')


def main():
    rng = random.Random(6302)
    res = {}
    for corp in ['PE', 'DR', 'UM']:
        D, _ = dossiers_for(corp)
        res[corp] = {'real': [analyse(d) for d in D],
                     'null': [analyse(d, rng, True) for d in D for _ in range(5)]}
        print(corp, len(D), Counter(o['role'] for a in res[corp]['real'] for o in a['slots']),
              'null', Counter(o['role'] for a in res[corp]['null'] for o in a['slots']))
    pl = []
    for s in (1, 2, 3):
        D, tr = dossiers_for('PE', 'plant', s)
        ids = set(tr['ids'])
        d = max(D, key=lambda d: len(ids & {t['id'] for t in d}))
        d = [t for t in d if t['id'] in ids]       # role reading on the planted members only
        a = analyse(d)
        ref = next(t for t in d if t['id'] == a['ref'])
        got = {o['slot']: o['role'] for o in a['slots']}
        pl.append({'seed': s, 'A_ID': got.get(tr['A']), 'B_COM': got.get(tr['B']), 'C_STEP': got.get(tr['C']),
                   'others': Counter(r for k, r in got.items() if k not in (tr['A'], tr['B'], tr['C']))})
    res['plant'] = pl
    print('plant', pl)
    json.dump(res, open(os.path.join(C.CK, 'c2_res.json'), 'w'))


if __name__ == '__main__':
    main()
