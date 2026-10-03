#!/usr/bin/env python3
"""Cycle 2: (a) a wider but fixed matching rule ('skeleton + affix'); (b) site concentration and
distance decay for every anchor, against frequency-matched Linear A words.

(a) Rule SK: a Linear A word contains a run of k signs (k = length of the LB place, k >= 3) whose
    consonant skeleton equals the LB place's skeleton (vowels free), with at most 1 sign before
    and at most 2 signs after the run.  Applied to all 15 gazetteer places.  Raw (uncleaned)
    Linear A words are used, so damaged/compound words count.  Nulls as in cycle 1:
    LA signs shuffled across types (1000), and random non-place KN words of same length (1000).
    NOTE: rule written after seeing a substring listing (post hoc); cycle 3 checks it on held-out data.
(b) For each anchor word: tokens' find sites -> mean great-circle distance to the named place
    (gazetteer coordinates fixed before any Linear A lookup).  Null: Linear A word types with the
    same token count (and, version 2, same document class mix), drawn 5000 times; p = P(null <= obs).
    Combined: product of per-anchor p's vs the same product under joint draws.
    Also 'near share': tokens within 15 km of the place vs corpus token share within 15 km.
Output: data/la2_c2.json
"""
import json, random, os, math
from collections import Counter, defaultdict
from la2_common import *

random.seed(33); NR = 1000; ND = 5000

def cons(sg):
    sg = sg.lower()
    if sg in ('a', 'e', 'i', 'o', 'u'): return '0'
    m = re.fullmatch(r'([a-z]+?)([aeiou])([0-9]?)', sg)
    if not m: return sg
    c, n = m.group(1), m.group(3)
    return c + ({'2': 'j', '3': 'jj'}.get(n, '') if c != 'pu' else '')

def sk_match(word, place):
    w = [cons(s) for s in word]; p = [cons(s) for s in place]; k = len(p)
    for st in range(0, min(2, len(w) - k + 1)):
        if w[st:st + k] == p and len(w) - (st + k) <= 2: return True
    return False

def raw_words():
    out = []
    for r in la_records():
        for t in r['tokens']:
            if t['t'] == 'word':
                s = tuple(norm(x) for x in t['s'])
                out.append((s, r['id'], r['site'], dclass(r['support'])))
    return out

def main():
    rw = raw_words()
    types = sorted(set(s for s, *_ in rw if len(s) >= 3))
    G = {g: tuple(g.split('-')) for g in GAZ}
    res = {}
    # (a) skeleton rule
    def sk_all(tl, targets):
        hits = defaultdict(list)
        for w in tl:
            for name, p in targets.items():
                if sk_match(w, p): hits[name].append(w)
        return hits
    obs = sk_all(types, G)
    n_obs = sum(len(x) for x in obs.values()); np_obs = len(obs)
    res['sk_hits'] = {k: ['-'.join(w) for w in v] for k, v in obs.items()}
    pool = [s for w in types for s in w]; sims = []
    for _ in range(NR):
        random.shuffle(pool); i = 0; sh = []
        for w in types: sh.append(tuple(pool[i:i + len(w)])); i += len(w)
        h = sk_all(sh, G); sims.append((sum(len(x) for x in h.values()), len(h)))
    res['sk_null1'] = {'types_mean': sum(a for a, b in sims) / NR, 'p_types': (1 + sum(a >= n_obs for a, b in sims)) / (NR + 1),
                       'places_mean': sum(b for a, b in sims) / NR, 'p_places': (1 + sum(b >= np_obs for a, b in sims)) / (NR + 1)}
    v, _ = lb_kn_vocab(); P = lb_place_list(v)
    nonp = defaultdict(list)
    for w in v:
        if '*' in w or w in P: continue
        nonp[len(w.split('-'))].append(tuple(w.split('-')))
    sims = []
    for _ in range(NR):
        rt = {str(i): random.choice(nonp[len(p)]) for i, p in enumerate(G.values())}
        h = sk_all(types, rt); sims.append((sum(len(x) for x in h.values()), len(h)))
    res['sk_null2'] = {'types_mean': sum(a for a, b in sims) / NR, 'p_types': (1 + sum(a >= n_obs for a, b in sims)) / (NR + 1),
                       'places_mean': sum(b for a, b in sims) / NR, 'p_places': (1 + sum(b >= np_obs for a, b in sims)) / (NR + 1)}
    res['sk_obs'] = {'types': n_obs, 'places': np_obs}
    # (b) geography
    tok_by_type = defaultdict(list)
    for s, rid, site, dc in rw:
        if site in SITES: tok_by_type[s].append((site, dc, rid))
    by_n = defaultdict(list); by_sig = defaultdict(list)
    for w, tl in tok_by_type.items():
        by_n[len(tl)].append(w)
        by_sig[tuple(sorted(Counter(dc for _, dc, _ in tl).items()))].append(w)
    allsites = [site for tl in tok_by_type.values() for site, _, _ in tl]
    def mdist(tl, place): return sum(km(SITES[s], place) for s, _, _ in tl) / len(tl)
    anchors = {}
    for g, p in G.items():
        for w in obs.get(g, []):
            if w in tok_by_type: anchors[(g, w)] = tok_by_type[w]
    ex_words = {('pa-i-to', ('pa', 'i', 'to')), ('su-ki-ri-ta', ('su', 'ki', 'ri', 'ta')), ('se-to-i-ja', ('se', 'to', 'i', 'ja'))}
    rows = []; draws1 = []; draws2 = []
    for (g, w), tl in sorted(anchors.items()):
        place = GAZ[g][1:3]; o = mdist(tl, place); n = len(tl)
        sig = tuple(sorted(Counter(dc for _, dc, _ in tl).items()))
        n1 = [mdist(tok_by_type[random.choice(by_n[n])], place) for _ in range(ND)]
        cand2 = by_sig.get(sig, []) if len(by_sig.get(sig, [])) >= 20 else by_n[n]
        n2 = [mdist(tok_by_type[random.choice(cand2)], place) for _ in range(ND)]
        p1 = (1 + sum(x <= o for x in n1)) / (ND + 1); p2 = (1 + sum(x <= o for x in n2)) / (ND + 1)
        near = sum(km(SITES[s], place) <= 15 for s, _, _ in tl)
        base = sum(km(SITES[s], place) <= 15 for s in allsites) / len(allsites)
        rows.append({'place': g, 'classical': GAZ[g][0], 'grade_loc': GAZ[g][3], 'la_word': '-'.join(w),
                     'exact': (g, w) in ex_words, 'tokens': [f'{rid}@{s}' for s, _, rid in tl],
                     'mean_km': round(o, 1), 'null_mean_km_freq': round(sum(n1) / ND, 1), 'p_freq': round(p1, 3),
                     'null_mean_km_class': round(sum(n2) / ND, 1), 'p_class': round(p2, 3),
                     'near15': f'{near}/{n}', 'base_share_near15': round(base, 3),
                     'p_near_binom': round(sum(math.comb(n, k) * base ** k * (1 - base) ** (n - k) for k in range(near, n + 1)), 3) if near else 1.0})
    res['geo'] = rows
    # combined (product of p's) for exact set and all-SK set
    def comb(sel, key):
        ps = [r[key] for r in rows if sel(r)]
        if not ps: return None
        stat = sum(math.log(p) for p in ps); k = len(ps)
        # under H0 each p ~ U(0,1) (approximately; discrete) -> Fisher chi2 with 2k df
        x = -2 * stat
        # survival of chi2 with even df
        s = sum((x / 2) ** i / math.factorial(i) for i in range(k)) * math.exp(-x / 2)
        return {'k': k, 'fisher_p': round(s, 4)}
    res['combined'] = {'exact_freq': comb(lambda r: r['exact'], 'p_freq'), 'exact_class': comb(lambda r: r['exact'], 'p_class'),
                       'all_freq': comb(lambda r: True, 'p_freq'), 'all_class': comb(lambda r: True, 'p_class'),
                       'all_but_exact_freq': comb(lambda r: not r['exact'], 'p_freq')}
    json.dump(res, open(os.path.join(D, 'la2_c2.json'), 'w'), indent=1)
    print('SK obs', res['sk_obs'], 'null1', res['sk_null1'], 'null2', res['sk_null2'])
    print(res['sk_hits'])
    for r in rows: print(r)
    print(res['combined'])

if __name__ == '__main__': main()
