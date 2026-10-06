#!/usr/bin/env python3
"""la65 cycle 2: inside each dossier, align the copies entry by entry and type every variable slot by what
changes with it.
  slot = a reference entry and its aligned partners in the other members.
  ID   : the word(s) change, the logogram does not (number free)        -> identifier (person, place, period)
  COM  : the word(s) change together with the logogram base               -> commodity-bound word
  GEN  : the logogram changes, the word does not                          -> word not bound to a commodity
  STEP : number sequence along the series with a constant ratio or difference over >= 3 members
  RATE : two numeric slots whose ratio is constant over >= 3 members while both vary
  FRAC : fraction letters change together with the logogram (commodity-specific measure) vs without
Nulls: logogram-change flags permuted across aligned pairs of the same dossier (coupling null); numbers
permuted within slot (STEP/RATE null). Controls: planted dossiers (truth slots A=ID, B=COM, C=STEP),
Linear B draws (answer key used only for scoring: commodity-bound qualifiers vs place names and the rest).
"""
import os, sys, json, random, math, re
from collections import Counter, defaultdict
import numpy as np
import la65_common as K
from la65_c1 import corpus


def etype(e):
    return ('W' if e['w'] else '') + ('L' if e['l'] else '') + ('N' if e['n'] is not None else 'c')


def sim(a, b):
    s = 0.0
    wa, wb = set(a['w']), set(b['w'])
    la, lb = {K.lbase(x) for x in a['l']}, {K.lbase(x) for x in b['l']}
    s += 2.0 * len(wa & wb)
    s += 2.0 * len(la & lb)
    if a['n'] is not None and a['n'] == b['n']:
        s += 1.0
    if a['f'] and a['f'] == b['f']:
        s += 0.5
    if etype(a) == etype(b):
        s += 0.6
    return s


def align(R, X):
    """order-preserving global alignment (Needleman-Wunsch, gap 0); returns [(i, j)]"""
    n, m = len(R), len(X)
    S = np.zeros((n + 1, m + 1))
    B = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sc = sim(R[i - 1], X[j - 1])
            opts = (S[i - 1, j - 1] + (sc if sc >= 0.6 else -1.0), S[i - 1, j], S[i, j - 1])
            k = int(np.argmax(opts))
            S[i, j], B[i, j] = opts[k], k
    out = []
    i, j = n, m
    while i > 0 and j > 0:
        k = B[i, j]
        if k == 0:
            if sim(R[i - 1], X[j - 1]) >= 0.6:
                out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def const_ratio(vals):
    v = [x for x in vals if x is not None and x > 0]
    if len(v) < 3 or len(set(v)) < 3:
        return None
    d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    r = [v[i + 1] / v[i] for i in range(len(v) - 1)]
    if len(set(d)) == 1:
        return 'diff%+d' % d[0]
    if max(r) - min(r) < 1e-9:
        return 'ratio%.3g' % r[0]
    return None


def analyse(T, ids, perm_rng=None):
    """returns slot records and pair events for one dossier"""
    pos = {t['id']: t for t in T}
    M = [pos[i] for i in ids]
    M.sort(key=lambda t: (t['meta']['ord'] if t['meta']['ord'] >= 0 else 0, t['id']))
    ref = max(M, key=lambda t: len(t['ents']))
    R = ref['ents']
    slots = defaultdict(dict)                    # ref entry index -> member id -> entry
    for t in M:
        if t is ref:
            for i, e in enumerate(R):
                slots[i][t['id']] = e
            continue
        for i, j in align(R, t['ents']):
            slots[i][t['id']] = t['ents'][j]
    recs = []
    for i, S in sorted(slots.items()):
        if len(S) < 2:
            continue
        es = [S[t['id']] for t in M if t['id'] in S]
        ws = [' '.join(e['w']) for e in es]
        ls = ['|'.join(sorted({K.lbase(x) for x in e['l']})) for e in es]
        ns = [e['n'] for e in es]
        fs = [e['f'] for e in es]
        pairs = []
        for a in range(len(es)):
            for b in range(a + 1, len(es)):
                pairs.append({'wch': ws[a] != ws[b] and bool(ws[a]) and bool(ws[b]),
                              'lch': ls[a] != ls[b] and bool(ls[a]) and bool(ls[b]),
                              'fch': fs[a] != fs[b],
                              'nch': (ns[a] != ns[b]) if ns[a] is not None and ns[b] is not None else None,
                              'sc': (abs(math.log2(ns[b] / ns[a])) >= 1.5) if (ns[a] and ns[b]) else None,
                              'wa': ws[a], 'wb': ws[b], 'la': ls[a], 'lb': ls[b], 'fa': fs[a], 'fb': fs[b]})
        recs.append({'slot': i, 'n': len(es), 'mids': [t['id'] for t in M if t['id'] in S], 'ws': ws, 'ls': ls, 'ns': ns, 'fs': fs, 'pairs': pairs,
                     'step': const_ratio(ns) if len(set(ws)) == 1 else None})
    # RATE: two numeric slots with all members present, both varying, constant ratio
    rates = []
    full = [r for r in recs if r['n'] == len(M) and all(x for x in r['ns']) and len(set(r['ns'])) >= 2]
    for a in range(len(full)):
        for b in range(a + 1, len(full)):
            ra = [y / x for x, y in zip(full[a]['ns'], full[b]['ns'])]
            if len(M) >= 3 and max(ra) - min(ra) < 1e-9:
                rates.append((full[a]['slot'], full[b]['slot'], round(ra[0], 4)))
    return {'ids': [t['id'] for t in M], 'ref': ref['id'], 'recs': recs, 'rates': rates}


def slot_type(r):
    P = r['pairs']
    w = any(p['wch'] for p in P)
    l = any(p['lch'] for p in P)
    if w and l:
        cov = sum(p['wch'] and p['lch'] for p in P) / max(1, sum(p['wch'] or p['lch'] for p in P))
        return 'COM' if cov >= 0.5 else 'MIX'
    if w:
        return 'ID'
    if l:
        return 'GEN'
    if r['step']:
        return 'STEP'
    if any(p['nch'] for p in P):
        return 'NUM'
    return 'FIX'


def coupling(A, rng, reps=2000):
    """observed word-change & logo-change co-occurrence vs flags permuted within dossier"""
    obs = 0
    per = []
    for d in A:
        P = [p for r in d['recs'] for p in r['pairs'] if p['la'] or p['lb']]
        per.append(P)
        obs += sum(p['wch'] and p['lch'] for p in P)
    null = []
    for _ in range(reps):
        c = 0
        for P in per:
            f = [p['lch'] for p in P]
            rng.shuffle(f)
            c += sum(p['wch'] and x for p, x in zip(P, f))
        null.append(c)
    null = np.array(null)
    return obs, float(null.mean()), float((1 + (null >= obs).sum()) / (1 + reps))


def frac_test(A, rng, reps=2000):
    P = [p for d in A for r in d['recs'] for p in r['pairs'] if p['la'] and p['lb'] and (p['fa'] or p['fb'])]
    if not P:
        return None
    a = sum(p['fch'] for p in P if p['lch']), sum(1 for p in P if p['lch'])
    b = sum(p['fch'] for p in P if not p['lch']), sum(1 for p in P if not p['lch'])
    obs = (a[0] / a[1] if a[1] else 0) - (b[0] / b[1] if b[1] else 0)
    f = [p['lch'] for p in P]
    null = []
    for _ in range(reps):
        rng.shuffle(f)
        x = [p['fch'] for p, q in zip(P, f) if q]
        y = [p['fch'] for p, q in zip(P, f) if not q]
        null.append((np.mean(x) if x else 0) - (np.mean(y) if y else 0))
    null = np.array(null)
    return {'with_logo_change': a, 'without': b, 'diff': obs, 'p': float((1 + (null >= obs).sum()) / (1 + reps))}


def word_roles(A):
    """per word: votes from the slots in which it varies"""
    V = defaultdict(Counter)
    for d in A:
        for r in d['recs']:
            t = slot_type(r)
            if t in ('ID', 'COM', 'MIX'):
                for w in set(r['ws']):
                    for x in w.split():
                        V[x][t] += 1
            elif t in ('GEN', 'FIX', 'STEP', 'NUM'):
                for w in set(r['ws']):
                    for x in w.split():
                        V[x][t] += 1
    return V


def run(name, D, seed=0):
    T, meta = corpus(name)
    A = [analyse(T, d['ids']) for d in D if len(d['ids']) >= 2]
    rng = random.Random(seed)
    types = Counter(slot_type(r) for d in A for r in d['recs'])
    return T, A, types, coupling(A, rng), frac_test(A, rng), word_roles(A)


def main():
    R = json.load(open(os.path.join(K.CK, 'c1_dossiers.json')))
    out = {}
    lines = []
    # ---------------- planted
    ok = Counter()
    for p in R['PL']:
        T, meta = corpus(p['name'])
        pos = {t['id']: t for t in T}
        planted = [i for i in p['dossier'] if pos[i].get('planted')]
        if len(planted) < 2:
            lines.append('%s: dossier not recovered' % p['name'])
            continue
        a = analyse(T, planted)
        tr = pos[planted[0]]['truth']
        ref = pos[a['ref']]
        got = {}
        for r in a['recs']:
            got[r['slot']] = slot_type(r)
        # planted copies share the source layout, so slot index = source entry index
        res = {k: got.get(v) for k, v in tr.items()}
        exp = {'A': 'ID', 'B': ('COM', 'GEN'), 'C': 'STEP'}
        for k in 'ABC':
            ok[k] += res[k] in (exp[k] if isinstance(exp[k], tuple) else (exp[k],))
        others = Counter(t for s, t in got.items() if s not in tr.values())
        lines.append('%s: truth slots -> %s; other slots %s' % (p['name'], res, dict(others)))
        # COM requires the planted word to change too: B changes logogram only -> GEN is the right answer
    lines.append('planted roles right: A(ID) %d/5, B(logogram swap) %d/5, C(STEP) %d/5' % (ok['A'], ok['B'], ok['C']))
    out['planted'] = dict(ok)
    # ---------------- Linear B
    sys.path.insert(0, K.HERE)
    import la63_lib as L63
    key = {w: r for r, ws in L63.LB_KEY_W.items() for w in ws.split()}
    lbres = []
    for dd in range(3):
        name = 'LB%d' % dd
        if name not in R:
            continue
        T, A, types, cpl, fr, V = run(name, R[name]['dossiers'] + R[name + 'F']['dossiers'], dd)
        qual = [w for w in V if key.get(w) == 'QUAL']
        pla = [w for w in V if key.get(w) == 'PLA']
        rest = [w for w in V if w not in key]

        def comrate(ws):
            c = sum(V[w]['COM'] + V[w]['MIX'] + V[w]['GEN'] * 0 for w in ws)
            n = sum(V[w]['COM'] + V[w]['MIX'] + V[w]['ID'] for w in ws)
            return c, n
        lbres.append({'name': name, 'types': dict(types), 'coupling': cpl, 'frac': fr,
                      'qual': comrate(qual), 'pla': comrate(pla), 'rest': comrate(rest),
                      'id_words': sum(1 for w in V if V[w]['ID']), 'com_words': sum(1 for w in V if V[w]['COM'])})
        lines.append('%s: %d dossiers; slot types %s; word-logo coupling obs %d vs null %.1f (P %.4f); frac %s; '
                     'COM share of varying slots: QUAL %s, PLA %s, other words %s' % (
                         name, len(A), dict(types), cpl[0], cpl[1], cpl[2], fr, comrate(qual), comrate(pla), comrate(rest)))
    out['LB'] = lbres
    # ---------------- Linear A
    for name, key in [('LA', 'LA'), ('LAN', 'LANF')]:
        T, A, types, cpl, fr, V = run(name, R[key]['dossiers'], 7)
        out[name] = {'types': dict(types), 'coupling': cpl, 'frac': fr,
                     'roles': {w: dict(c) for w, c in V.items()},
                     'dossiers': [{'ids': a['ids'], 'ref': a['ref'], 'rates': a['rates'],
                                   'slots': [{'slot': r['slot'], 'type': slot_type(r), 'ws': r['ws'], 'ls': r['ls'],
                                              'ns': r['ns'], 'fs': r['fs'], 'step': r['step']} for r in a['recs']]}
                                  for a in A]}
        lines.append('%s: %d dossiers; slot types %s; word-logo coupling obs %d vs null %.1f (P %.4f); frac %s' % (
            name, len(A), dict(types), cpl[0], cpl[1], cpl[2], fr))
        if name == 'LAN':
            for a in A:
                vs = [(r['slot'], slot_type(r), r['ws'], r['ls'], r['ns'], r['fs']) for r in a['recs']
                      if slot_type(r) not in ('FIX',)]
                lines.append('  dossier %s (ref %s) rates %s' % (' '.join(a['ids']), a['ref'], a['rates']))
                for v in vs[:14]:
                    lines.append('     slot %d %s words %s logos %s nums %s fr %s' % v)
    json.dump(out, open(os.path.join(K.CK, 'c2_out.json'), 'w'))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
