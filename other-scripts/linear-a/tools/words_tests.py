#!/usr/bin/env python3
"""Tests 2c (commodity association), 2d (name-like repetition), 2f (affix alternations)."""
import json, os, random, math
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, '..', 'data', 'corpus.json')))
random.seed(2)

def base(l):
    l = l.lstrip('*') if l.startswith('*OLIV') else l
    b = l.split('+')[0]
    return b

# ---------------------------------------------------------------- 2c
def test_2c():
    print('2c. Commodity association')
    tabs = [i for i in C if i['support'] in ('Tablet', 'Lames (short thin tablet)', '3-sided bar', '4-sided bar')]
    lc = Counter(base(l) for i in tabs for l in i['logograms'])
    MAIN = [l for l, n in lc.most_common() if n >= 8 and l not in ('VIR',)]
    MAIN = [l for l in MAIN if not l.startswith('[')]
    MAIN.append('VIR')
    print('  commodity logograms used (tablets, >=8 tokens):', [(l, lc[l]) for l in MAIN])
    # (i) adjacency: word immediately followed (dividers skipped) by a logogram
    adj = defaultdict(Counter)
    for i in tabs:
        T = i['tokens']
        for k, t in enumerate(T):
            if t['t'] != 'word' or len(t['s']) < 2: continue
            j = k + 1
            while j < len(T) and T[j]['t'] == 'div': j += 1
            if j < len(T) and T[j]['t'] == 'logo' and base(T[j]['v']) in MAIN:
                adj['-'.join(t['s'])][base(T[j]['v'])] += 1
    multi = {w: c for w, c in adj.items() if sum(c.values()) >= 2}
    print('  words directly before a commodity sign, >=2 times:')
    for w, c in sorted(multi.items(), key=lambda x: -sum(x[1].values())):
        print('    ', w, dict(c))
    # (ii) tablet co-occurrence: word -> set of commodities on the same tablet
    wt = defaultdict(list)
    tab_comm = []
    for i in tabs:
        comm = frozenset(base(l) for l in i['logograms'] if base(l) in MAIN)
        ws = set(w for w in i['words'] if '-' in w)
        tab_comm.append((ws, comm))
    def profile(pairs):
        prof = defaultdict(Counter); ntab = Counter()
        for ws, comm in pairs:
            if not comm: continue
            for w in ws:
                ntab[w] += 1
                for c in comm: prof[w][c] += 1
        return prof, ntab
    prof, ntab = profile(tab_comm)
    words = [w for w in ntab if ntab[w] >= 3]
    def stats(prof, ntab, words):
        # a word is "tied" if one commodity is present on every tablet it occurs on
        tied = [w for w in words if max(prof[w].values()) == ntab[w]]
        spread = [w for w in words if len(prof[w]) >= 4]
        return tied, spread
    tied, spread = stats(prof, ntab, words)
    print(f'  words on >=3 tablets that carry commodity signs: {len(words)}')
    print(f'  tied (one commodity on every tablet with the word): {len(tied)} ->',
          {w: prof[w].most_common(1)[0][0] for w in tied})
    print(f'  spread (>=4 different commodities): {len(spread)} ->', spread)
    # control: shuffle commodity sets across tablets
    nt = []; ns = []
    comms = [c for _, c in tab_comm]
    for _ in range(2000):
        p = comms[:]; random.shuffle(p)
        pr, nn = profile([(ws, c) for (ws, _), c in zip(tab_comm, p)])
        wds = [w for w in nn if nn[w] >= 3]
        t, s = stats(pr, nn, wds)
        nt.append(len(t) / max(1, len(wds))); ns.append(len(s) / max(1, len(wds)))
    ot = len(tied) / len(words); os_ = len(spread) / len(words)
    print(f'  control (commodity sets shuffled across tablets): tied share obs {ot:.2f} vs null mean '
          f'{sum(nt)/len(nt):.2f} (P>=obs {sum(x >= ot for x in nt)/len(nt):.3f}); spread share obs {os_:.2f} '
          f'vs null {sum(ns)/len(ns):.2f} (P>=obs {sum(x >= os_ for x in ns)/len(ns):.3f})')
    # words that appear right before numbers 1 (persons?) vs before big numbers
    small = Counter(); big = Counter()
    for i in tabs:
        T = i['tokens']
        for k, t in enumerate(T[:-1]):
            if t['t'] == 'word' and len(t['s']) >= 2 and T[k + 1]['t'] == 'num':
                (small if T[k + 1]['v'] == 1 and not T[k + 1]['frac'] else big)['-'.join(t['s'])] += 1
    both = set(small) & set(big)
    print(f'  words followed by exactly "1": {len(small)} types; by other numbers: {len(big)} types; both: {len(both)} {sorted(both)[:15]}')

# ---------------------------------------------------------------- 2d
def bigram_unique_share(tokens, n_sims=200):
    # train sign-bigram model with start/end
    tr = defaultdict(Counter)
    for w in tokens:
        s = ['^'] + w.split('-') + ['$']
        for a, b in zip(s, s[1:]): tr[a][b] += 1
    tbl = {a: (list(c), list(c.values())) for a, c in tr.items()}
    shares = []
    for _ in range(n_sims):
        gen = []
        for _w in tokens:
            cur, out = '^', []
            while True:
                ks, ws = tbl[cur]
                cur = random.choices(ks, ws)[0]
                if cur == '$' or len(out) > 12: break
                out.append(cur)
            gen.append('-'.join(out))
        gen = [g for g in gen if '-' in g]
        cnt = Counter(gen)
        shares.append(sum(1 for g in gen if cnt[g] == 1) / len(gen))
    return sum(shares) / len(shares)

def test_2d():
    print('\n2d. Name-like repetition (share of word tokens that are unique types)')
    for label, sel in [('tablets', lambda i: i['support'] == 'Tablet'),
                       ('Hagia Triada tablets', lambda i: i['support'] == 'Tablet' and i['site'] == 'Haghia Triada'),
                       ('all objects', lambda i: True),
                       ('non-tablet (vessels, stone, metal)', lambda i: i['support'] not in ('Tablet', 'Nodule', 'Roundel'))]:
        toks = [w for i in C if sel(i) for w in i['words'] if '-' in w]
        cnt = Counter(toks)
        obs = sum(1 for w in toks if cnt[w] == 1) / len(toks)
        mod = bigram_unique_share(toks)
        print(f'  {label:36s} tokens {len(toks):5d} types {len(cnt):4d} unique-share obs {obs:.3f}  bigram model {mod:.3f}  ratio {obs/mod:.2f}')
    # "1"-list words: entries with quantity exactly 1 (person-list format)
    toks = []
    for i in C:
        T = i['tokens']
        for k, t in enumerate(T[:-1]):
            if t['t'] == 'word' and len(t['s']) >= 2 and T[k + 1]['t'] == 'num' and T[k + 1]['v'] == 1 and not T[k + 1]['frac']:
                toks.append('-'.join(t['s']))
    cnt = Counter(toks)
    obs = sum(1 for w in toks if cnt[w] == 1) / len(toks)
    print(f'  {"words with quantity 1 (list entries)":36s} tokens {len(toks):5d} types {len(cnt):4d} unique-share obs {obs:.3f}  bigram model {bigram_unique_share(toks):.3f}')

# ---------------------------------------------------------------- 2f
def test_2f():
    print('\n2f. Affix alternations (word types, all objects, >=2 signs)')
    types = sorted({w for i in C for w in i['words'] if '-' in w and '*' not in w})
    S = [w.split('-') for w in types]
    print('  word types used:', len(S))
    def count(S):
        init = Counter(); fin = Counter(); pre = Counter(); suf = Counter()
        st = set(tuple(s) for s in S)
        by_tail = defaultdict(set); by_head = defaultdict(set)
        for s in S:
            if len(s) >= 3: by_tail[tuple(s[1:])].add(s[0]); by_head[tuple(s[:-1])].add(s[-1])
        for g in by_tail.values():
            for a in g:
                for b in g:
                    if a < b: init[(a, b)] += 1
        for g in by_head.values():
            for a in g:
                for b in g:
                    if a < b: fin[(a, b)] += 1
        for s in S:
            if len(s) >= 3:
                if tuple(s[1:]) in st: pre[s[0]] += 1     # X-W and W both exist
                if tuple(s[:-1]) in st: suf[s[-1]] += 1   # W-X and W both exist
        return init, fin, pre, suf
    obs = count(S)
    # control: rebuild words by shuffling signs among positions *within the same length*,
    # keeping overall sign frequencies per position class (first / middle / last).
    def shuffled():
        firsts = [s[0] for s in S]; lasts = [s[-1] for s in S]; mids = [x for s in S for x in s[1:-1]]
        random.shuffle(firsts); random.shuffle(lasts); random.shuffle(mids)
        out = []; mi = 0
        for k, s in enumerate(S):
            m = mids[mi: mi + len(s) - 2]; mi += len(s) - 2
            out.append([firsts[k]] + m + [lasts[k]])
        return out
    sims = [count(shuffled()) for _ in range(300)]
    names = ['initial alternation (same rest, different first sign)', 'final alternation (same stem, different last sign)',
             'prefix added (W and X-W both exist)', 'suffix added (W and W-X both exist)']
    for k, nm in enumerate(names):
        o = obs[k]
        tot_o = sum(o.values()); tot_n = [sum(s[k].values()) for s in sims]
        mu = sum(tot_n) / len(tot_n); sd = (sum((x - mu) ** 2 for x in tot_n) / len(tot_n)) ** .5
        print(f'  {nm}: total obs {tot_o}, control {mu:.1f} +/- {sd:.1f}')
        rows = []
        for key, n in o.most_common(25):
            nn = [s[k].get(key, 0) for s in sims]
            m = sum(nn) / len(nn); p = sum(x >= n for x in nn) / len(nn)
            rows.append((key, n, m, p))
        rows = [r for r in rows if r[1] >= 3]
        for key, n, m, p in rows[:12]:
            tag = 'REAL' if p < 0.01 and n >= 2 * m + 2 else ''
            print(f'      {key}: obs {n}, control {m:.2f}, P {p:.3f} {tag}')

if __name__ == '__main__':
    test_2c(); test_2d(); test_2f()
