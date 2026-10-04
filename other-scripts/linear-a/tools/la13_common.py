#!/usr/bin/env python3
"""LA-13 shared code: 'Linear A contains its own dialects'.

Each find-site group is treated as a separate language. We search for regular sign-for-sign correspondences:
word pairs (w1 at site A, w2 at site B) of equal length that differ in exactly one sign (X at A, Y at B), or
by one inserted sign, where the same substitution recurs across many independent pairs.
A pair is CLEAN when w1 is not attested at B and w2 is not attested at A (complementary distribution).
No sound values are used in the search; transliteration labels are opaque sign names.
"""
import os, sys, json, random, collections, itertools, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from la5_common import la_docs, lb_docs
OUT = os.path.join(HERE, '..', 'data', 'la13')
os.makedirs(OUT, exist_ok=True)

GROUPS = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}
def la_group(site): return GROUPS.get(site, 'OTH')

def la_units(minlen=2):
    """-> list of (doc_id, group, support, [word tuples]) for every LA document with >=1 word of length>=minlen."""
    out = []
    for d in la_docs(admin_only=False):
        ws = [t[1] for L in d['lines'] for t in L if t[0] == 'W' and len(t[1]) >= minlen]
        if ws: out.append((d['id'], la_group(d['site']), d['support'], ws))
    return out

def lb_units(minlen=2, sites=None):
    out = []
    for d in lb_docs():
        if sites and d['site'] not in sites: continue
        ws = [t[1] for L in d['lines'] for t in L if t[0] == 'W' and len(t[1]) >= minlen]
        if ws: out.append((d['id'], d['site'], d.get('series', ''), ws))
    return out

def types_by_group(units, labels=None):
    T = collections.defaultdict(set)
    for i, u in enumerate(units):
        g = labels[i] if labels is not None else u[1]
        for w in u[3]: T[g].add(w)
    return T

def pclass(i, L):
    return 'I' if i == 0 else ('F' if i == L - 1 else 'M')

def pairs(T, indel=True, clean=True):
    """Enumerate cross-group one-sign-different pairs.
    -> list of (A, B, X, Y, pos, w1, w2) with A<B; X at A, Y at B. For indel: X or Y = '_' (absent)."""
    gs = sorted(T)
    # mask index: (len, i, masked tuple) -> {group: {sign: [words]}}
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in gs:
        for w in T[g]:
            L = len(w)
            for i in range(L):
                idx[(L, i, w[:i] + w[i + 1:])][g].append(w)
    res = []
    for (L, i, m), dg in idx.items():
        if len(dg) < 2 and not indel: continue
        gl = sorted(dg)
        for a, b in itertools.combinations(gl, 2):
            for w1 in dg[a]:
                for w2 in dg[b]:
                    if w1 == w2: continue
                    if clean and (w1 in T[b] or w2 in T[a]): continue
                    res.append((a, b, w1[i], w2[i], pclass(i, L), w1, w2))
    if indel:
        # w2 (len L+1) = w1 with a sign inserted at position i: key the deletion of w2 at i equal to w1
        full = {}
        for g in gs:
            for w in T[g]: full.setdefault(w, set()).add(g)
        for (L, i, m), dg in idx.items():
            if m not in full: continue
            for gshort in full[m]:
                for glong, ws in dg.items():
                    if glong == gshort: continue
                    for wl in ws:
                        if clean and (m in T[glong] or wl in T[gshort]): continue
                        a, b = sorted((gshort, glong))
                        if a == gshort: res.append((a, b, '_', wl[i], pclass(i, L), m, wl))
                        else: res.append((a, b, wl[i], '_', pclass(i, L), wl, m))
    # dedupe (an indel can be generated at several i for doubled signs)
    return list(set(res))

def rule_support(P, posmode=True):
    """-> dict rule -> (k_ind, n_pairs, list of pairs). rule = (A,B,X,Y,pos) if posmode else (A,B,X,Y,'*').
    k_ind = min(#distinct w1, #distinct w2) (independent-pair count)."""
    R = collections.defaultdict(list)
    for (a, b, x, y, p, w1, w2) in P:
        R[(a, b, x, y, p if posmode else '*')].append((w1, w2))
    out = {}
    for r, lst in R.items():
        k = min(len({u for u, _ in lst}), len({v for _, v in lst}))
        out[r] = (k, len(lst), lst)
    return out

def summary_stats(T, kmin=(2, 3, 4)):
    P = pairs(T)
    S = {}
    for pm in (True, False):
        R = rule_support(P, pm)
        ks = [v[0] for v in R.values()]
        tag = 'pos' if pm else 'any'
        S[tag + '_max'] = max(ks) if ks else 0
        for k in kmin: S[f'{tag}_n{k}'] = sum(1 for x in ks if x >= k)
    S['npairs'] = len(P)
    return S, P

def shuffle_labels(units, rnd, strat=False):
    labs = [u[1] for u in units]
    if not strat:
        rnd.shuffle(labs); return labs
    by = collections.defaultdict(list)
    for i, u in enumerate(units): by[u[2]].append(i)
    out = labs[:]
    for s, ii in by.items():
        l = [labs[i] for i in ii]; rnd.shuffle(l)
        for i, v in zip(ii, l): out[i] = v
    return out

def markov_types(T, rnd):
    """Per-group bigram (with start/end) resampled types; same number of types and length histogram per group."""
    out = {}
    for g, ws in T.items():
        tr = collections.defaultdict(collections.Counter)
        for w in ws:
            s = ('^',) + w
            for a, b in zip(s, s[1:]): tr[a][b] += 1
        lens = [len(w) for w in ws]
        S = set()
        tries = 0
        while len(S) < len(ws) and tries < 200 * len(ws):
            tries += 1
            L = rnd.choice(lens); w = []; prev = '^'
            for _ in range(L):
                c = tr[prev]
                if not c: break
                ks, vs = zip(*c.items()); prev = rnd.choices(ks, vs)[0]; w.append(prev)
            if len(w) == L: S.add(tuple(w))
        out[g] = S
    return out
