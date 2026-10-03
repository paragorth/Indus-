#!/usr/bin/env python3
"""LA-3 cycle 2: object-level geography, and the alternating pairs tested on the
administrative corpus.

A. Geography at object level (one row per formula object, not per attestation):
   deviant = any formula word on the object differs from its family reference.
   Region x deviant chi2, permutation of the deviant labels (5,000).
   Support-type confound: stone vessel vs other.

B. Admin test.  Admin corpus = tablets, nodules, roundels, sealings, lames, bars,
   label; words of >= 2 signs (tokens).  For a sign pair X~Y:
   1. positional profile (initial / medial / final share) -> L1 distance
   2. neighbour profile (left and right neighbour sign counts) -> cosine
   3. minimal pairs: admin types with X at position i whose Y-swapped form is
      also an admin type, / min(#types with X, #types with Y)
   Null: 2,000 random pairs U~V with freq(U) within x2 of freq(X) and freq(V)
   within x2 of freq(Y).  Per pair, P = share of null pairs at least as similar.
   Group test: the mean percentile of all formula pairs whose signs both occur
   >= 5 times in admin, against the same statistic for random frequency-matched
   pair sets (2,000).
Inputs: data/corpus.json, data/la3_formula.json.  Output: data/la3_equiv_report.txt
"""
import json, os, random
from collections import Counter, defaultdict
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
random.seed(5)
NR = 2000
spec = importlib.util.spec_from_file_location('f', os.path.join(HERE, 'la3_formula.py'))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)

def main():
    recs = json.load(open(os.path.join(D, 'corpus.json')))
    R = json.load(open(os.path.join(D, 'la3_formula.json')))
    rep = ['LA-3 cycle 2: object-level geography and admin test of alternating pairs', '']
    # ---------------- A. geography per object
    formula = set(R['formula']); core = set(R['core'])
    sup = {r['id']: r['support'] for r in recs}
    objs = {}
    for h in R['hits']:
        if h['seed'] not in formula: continue
        o = objs.setdefault(h['obj'], {'site': h['site'], 'dev': False, 'n': 0})
        o['n'] += 1
        if any(a != b for a, b in h['cols']): o['dev'] = True
    rows = [(o, v['site'], v['dev']) for o, v in objs.items()]
    def chi(rows_, keyf):
        tab = defaultdict(lambda: [0, 0])
        for o, s, d in rows_: tab[keyf(o, s)][int(d)] += 1
        n = len(rows_); dt = sum(r[2] for r in rows_); x = 0
        for k, (a0, a1) in tab.items():
            t = a0 + a1
            for ob, ex in ((a1, t * dt / n), (a0, t * (n - dt) / n)):
                if ex > 0: x += (ob - ex) ** 2 / ex
        return x, tab
    for name, keyf in (('region', lambda o, s: F.region(s)), ('site', lambda o, s: s),
                       ('support', lambda o, s: 'stone vessel' if sup[o] == 'Stone vessel' else 'other')):
        x0, tab = chi(rows, keyf); ds = [r[2] for r in rows]; nl = []
        for it in range(5000):
            random.shuffle(ds); nl.append(chi([(o, s, d) for (o, s, _), d in zip(rows, ds)], keyf)[0])
        rep.append('A. objects deviating by %s: %s; chi2 %.2f, permutation P = %.4f'
                   % (name, ', '.join('%s %d/%d' % (k, v[1], v[0] + v[1]) for k, v in sorted(tab.items())), x0,
                      sum(1 for x in nl if x >= x0) / 5000))
    # stratified: region effect among stone vessels only
    sv = [r for r in rows if sup[r[0]] == 'Stone vessel']
    x0, tab = chi(sv, lambda o, s: F.region(s)); ds = [r[2] for r in sv]; nl = []
    for it in range(5000):
        random.shuffle(ds); nl.append(chi([(o, s, d) for (o, s, _), d in zip(sv, ds)], lambda o, s: F.region(s))[0])
    rep.append('A. stone vessels only, by region: %s; chi2 %.2f, P = %.4f'
               % (', '.join('%s %d/%d' % (k, v[1], v[0] + v[1]) for k, v in sorted(tab.items())), x0, sum(1 for x in nl if x >= x0) / 5000))
    # leave-Palaikastro-out
    npk = [r for r in rows if r[1] != 'Palaikastro']
    x0, tab = chi(npk, lambda o, s: F.region(s)); ds = [r[2] for r in npk]; nl = []
    for it in range(5000):
        random.shuffle(ds); nl.append(chi([(o, s, d) for (o, s, _), d in zip(npk, ds)], lambda o, s: F.region(s))[0])
    rep.append('A. without Palaikastro, by region: %s; chi2 %.2f, P = %.4f'
               % (', '.join('%s %d/%d' % (k, v[1], v[0] + v[1]) for k, v in sorted(tab.items())), x0, sum(1 for x in nl if x >= x0) / 5000))
    # ---------------- B. admin test
    words = []
    for r in recs:
        if r['support'] not in F.ADM: continue
        for t in r['tokens']:
            if t['t'] == 'word' and len(t['s']) >= 2: words.append(tuple(t['s']))
    types = set(words)
    freq = Counter(x for w in words for x in w)
    pos = defaultdict(lambda: [0, 0, 0]); left = defaultdict(Counter); right = defaultdict(Counter)
    for w in words:
        for i, x in enumerate(w):
            pos[x][0 if i == 0 else 2 if i == len(w) - 1 else 1] += 1
            left[x][w[i-1] if i else '#'] += 1
            right[x][w[i+1] if i < len(w) - 1 else '#'] += 1
    by_sign_types = defaultdict(set)
    for w in types:
        for x in set(w): by_sign_types[x].add(w)
    def posd(x, y):
        a = pos[x]; b = pos[y]; sa = sum(a); sb = sum(b)
        return -sum(abs(a[i] / sa - b[i] / sb) for i in range(3))     # higher = more similar
    def cos(c1, c2):
        num = sum(c1[k] * c2[k] for k in c1); n1 = sum(v * v for v in c1.values()) ** .5; n2 = sum(v * v for v in c2.values()) ** .5
        return num / (n1 * n2) if n1 and n2 else 0
    def ctx(x, y):
        return (cos(left[x], left[y]) + cos(right[x], right[y])) / 2
    def minp(x, y):
        hit = 0
        for w in by_sign_types[x]:
            for i, s in enumerate(w):
                if s == x and (w[:i] + (y,) + w[i+1:]) in types: hit += 1
        return hit / max(1, min(len(by_sign_types[x]), len(by_sign_types[y])))
    signs = [s for s in freq if freq[s] >= 5]
    def matched(x):
        return [s for s in signs if freq[x] / 2 <= freq[s] <= freq[x] * 2]
    pairs = [tuple(p.split('~')) for p in R['pairs_formula']]
    nev = {}
    for e in R['events']:
        if e['fam'] in formula and e['kind'] == 'sub':
            k = tuple(sorted((e['a'], e['b']))); nev[k] = nev.get(k, 0) + 1
    rep.append('')
    rep.append('B. admin corpus: %d word tokens (>=2 signs), %d types, %d signs with freq >= 5' % (len(words), len(types), len(signs)))
    rep.append('   pair (admin freqs) [formula events]: positional L1 / P, context cosine / P, minimal-pair rate / P   (P = share of freq-matched random pairs at least as similar)')
    pct = {}
    for x, y in sorted(pairs, key=lambda p: -nev.get(tuple(sorted(p)), 0)):
        if freq[x] < 5 or freq[y] < 5:
            rep.append('   %s~%s (%d,%d) [%d]: too rare in admin' % (x, y, freq[x], freq[y], nev.get(tuple(sorted((x, y))), 0))); continue
        mx, my = matched(x), matched(y)
        null = []
        while len(null) < NR:
            u, v = random.choice(mx), random.choice(my)
            if u != v: null.append((u, v))
        o = (posd(x, y), ctx(x, y), minp(x, y))
        ps = []
        for j, fn in enumerate((posd, ctx, minp)):
            ps.append(sum(1 for u, v in null if fn(u, v) >= o[j]) / NR)
        pct[(x, y)] = ps
        rep.append('   %s~%s (%d,%d) [%d]: %.2f / %.3f, %.2f / %.3f, %.3f / %.3f'
                   % (x, y, freq[x], freq[y], nev.get(tuple(sorted((x, y))), 0), -o[0], ps[0], o[1], ps[1], o[2], ps[2]))
    # group test
    tested = list(pct)
    if tested:
        for j, nm in enumerate(('positional', 'context', 'minimal pairs')):
            obs = sum(pct[p][j] for p in tested) / len(tested)
            # null: replace each tested pair by a random freq-matched pair and compute its P the same way
            # (its P vs its own matched null is ~uniform by construction, so mean ~0.5); a direct
            # draw: uniform mean of len(tested) values
            # empirical null: replace each tested pair by a random freq-matched pair (u,v) and
            # compute its P against its own matched null (300 pairs); 300 replicate sets
            fn = (posd, ctx, minp)[j]
            nl = []
            for it in range(300):
                tot = 0
                for x, y in tested:
                    mx, my = matched(x), matched(y)
                    while True:
                        u, v = random.choice(mx), random.choice(my)
                        if u != v: break
                    ov = fn(u, v); c = 0
                    for _ in range(300):
                        while True:
                            a_, b_ = random.choice(mx), random.choice(my)
                            if a_ != b_: break
                        c += fn(a_, b_) >= ov
                    tot += c / 300
                nl.append(tot / len(tested))
            pv = sum(1 for v in nl if v <= obs) / len(nl)
            rep.append('   group (%d pairs): mean P %s = %.3f; empirical null mean %.3f, P = %.4f' % (len(tested), nm, obs, sum(nl) / len(nl), pv))
        # leave-one-out on context: is the group effect carried by a few pairs?
        loo = sorted(tested, key=lambda p: pct[p][1])
        rep.append('   context P sorted: ' + ', '.join('%s~%s %.3f' % (p[0], p[1], pct[p][1]) for p in loo))
    # A vs JA detail: word-initial share and initial alternation in admin
    for x in ('A', 'JA', 'I', 'U', 'MA', 'MI', 'NA', 'TA', 'TI', 'SI', 'TE', 'E'):
        a = pos[x]; rep.append('   %-3s admin freq %4d  initial %.2f medial %.2f final %.2f' % (x, freq[x], a[0] / sum(a), a[1] / sum(a), a[2] / sum(a)))
    # initial A~JA minimal pairs in admin vs other initial vowel/CV pairs
    ini = Counter(w[0] for w in types)
    def ini_min(x, y):
        return sum(1 for w in types if w[0] == x and ((y,) + w[1:]) in types)
    o = ini_min('A', 'JA')
    nl = []
    candA = [s for s in signs if ini[s] and ini['A'] / 2 <= ini[s] <= ini['A'] * 2]
    candJ = [s for s in signs if ini[s] and ini['JA'] / 2 <= ini[s] <= ini['JA'] * 2]
    for it in range(NR):
        u, v = random.choice(candA), random.choice(candJ)
        if u != v: nl.append(ini_min(u, v))
    rep.append('   initial-swap A-/JA- admin minimal pairs: %d (A- types %d, JA- types %d); freq-matched random initial pairs mean %.2f, P = %.4f'
               % (o, ini['A'], ini['JA'], sum(nl) / len(nl), sum(1 for v in nl if v >= o) / len(nl)))
    rep.append('     examples: ' + ', '.join('-'.join(w) for w in types if w[0] == 'A' and (('JA',) + w[1:]) in types))
    open(os.path.join(D, 'la3_equiv_report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))

if __name__ == '__main__':
    main()
