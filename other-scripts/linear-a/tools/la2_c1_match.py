#!/usr/bin/env python3
"""Cycle 1: Linear A words matching Linear B Cretan place names (exact or one-sign variant).

Targets: (G) gazetteer of 15 LB places with known/probable locations; (B) 88 data-derived
Knossos place-like words (allative -de or ethnic -jo/-ja in DAMOS) incl. G.
Variant = one sign substituted, inserted or deleted; both words >= 3 signs.
Null 1: Linear A signs shuffled across Linear A types (lengths kept), 1000 runs.
Null 2: target lists replaced by random Knossos words of the same sign lengths that are NOT
        in the place list (is a place list special, or does any LB list match as often?), 1000 runs.
Output: data/la2_c1.json
"""
import json, random, os
from collections import Counter, defaultdict
from la2_common import *

random.seed(21); NR = 1000

def keys(s):
    ks = set()
    for i in range(len(s)):
        ks.add(('S', s[:i] + ('?',) + s[i + 1:]))
        ks.add(('D', s[:i] + s[i + 1:]))
    ks.add(('D', s))
    return ks

def match(la_types, targets):
    """returns {la_type: [(target, kind)]}"""
    tset = set(targets); idx = defaultdict(set)
    for t in targets:
        if len(t) >= 3:
            for k in keys(t): idx[k].add(t)
    out = {}
    for w in la_types:
        hits = []
        if w in tset: hits.append((w, 'exact'))
        if len(w) >= 3:
            cand = set()
            for k in keys(w): cand |= idx.get(k, set())
            for t in cand:
                if t != w and edit1(w, t): hits.append((t, 'var'))
        if hits: out[w] = hits
    return out

def counts(m):
    ex = sum(1 for w, h in m.items() if any(k == 'exact' for _, k in h))
    va = sum(1 for w, h in m.items() if any(k == 'var' for _, k in h))
    ex3 = sum(1 for w, h in m.items() if len(w) >= 3 and any(k == 'exact' for _, k in h))
    return ex, ex3, va

def main():
    toks = [t for t in la_tokens() if t['clean'] and len(t['s']) >= 2]
    ftype = Counter(t['s'] for t in toks)
    types = sorted(ftype)
    v, _ = lb_kn_vocab()
    P = lb_place_list(v)
    G = [tuple(g.split('-')) for g in GAZ]
    B = [tuple(p.split('-')) for p in P]
    res = {'n_la_types': len(types), 'n_la_tokens': len(toks), 'n_G': len(G), 'n_B': len(B)}
    obs = {}
    for name, tg in (('G', G), ('B', B)):
        m = match(types, tg); obs[name] = m
        res[name] = {'obs': counts(m), 'hits': {'-'.join(w): {'n_tok': ftype[w], 'targets': ['-'.join(t) + ':' + k for t, k in h]} for w, h in m.items()}}
    # null 1: shuffle LA signs across types
    pool = [s for w in types for s in w]
    for name, tg in (('G', G), ('B', B)):
        sims = []
        for _ in range(NR):
            random.shuffle(pool); i = 0; sh = []
            for w in types: sh.append(tuple(pool[i:i + len(w)])); i += len(w)
            sims.append(counts(match(sh, tg)))
        o = res[name]['obs']
        res[name]['null1'] = {lab: {'mean': sum(x[j] for x in sims) / NR, 'p': (1 + sum(x[j] >= o[j] for x in sims)) / (NR + 1)}
                              for j, lab in enumerate(['exact', 'exact3', 'var'])}
    # null 2: random non-place KN words of the same lengths
    nonp = defaultdict(list)
    for w in v:
        if '*' in w or w in P: continue
        nonp[len(w.split('-'))].append(tuple(w.split('-')))
    for name, tg in (('G', G), ('B', B)):
        sims = []
        for _ in range(NR):
            rt = [random.choice(nonp[len(t)]) for t in tg]
            sims.append(counts(match(types, rt)))
        o = res[name]['obs']
        res[name]['null2'] = {lab: {'mean': sum(x[j] for x in sims) / NR, 'p': (1 + sum(x[j] >= o[j] for x in sims)) / (NR + 1)}
                              for j, lab in enumerate(['exact', 'exact3', 'var'])}
    json.dump(res, open(os.path.join(D, 'la2_c1.json'), 'w'), indent=1)
    for name in ('G', 'B'):
        r = res[name]; print(name, 'obs exact/exact3/var', r['obs'])
        print('  null1', {k: (round(x['mean'], 2), round(x['p'], 3)) for k, x in r['null1'].items()})
        print('  null2', {k: (round(x['mean'], 2), round(x['p'], 3)) for k, x in r['null2'].items()})
        for w, h in sorted(r['hits'].items(), key=lambda x: -x[1]['n_tok']): print('   ', w, h)

if __name__ == '__main__': main()
