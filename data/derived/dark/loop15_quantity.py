"""Loop 15 cycle 3: quantity gaps (minimum-lot rule?).  For every sign that is immediately preceded by a stroke
numeral >= NMIN times: the distribution of numeral values; does it lack 1 and 2 (or any value) more than a null
in which numeral tokens are permuted among the items they count, stratified by series (short/tall) x region
(home = Mohenjo-daro+Harappa / held-out) x object class?  Also: numbered persons by site (S98/S98b), and trees.
Usage: python3 loop15_quantity.py [--seqkey seq_raw] [--nperm 5000]
"""
import json, sys, os, random, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
ARGS = sys.argv[1:]
def arg(n, d): return ARGS[ARGS.index(n) + 1] if n in ARGS else d
SEQKEY = arg('--seqkey', 'seq_raw'); NPERM = int(arg('--nperm', 5000)); NMIN = int(arg('--nmin', 10))
SHORT = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10,
         25: 5, 26: 6, 27: 7, 28: 8, 29: 9}
TALL = {31: 1, 32: 2, 33: 3, 34: 4, 35: 5, 36: 6, 37: 7}
VAL = {**SHORT, **TALL}; NUMS = set(VAL) | {55, 56}
OPEN = {817, 861, 820, 920, 692}
PERSON = {90, 91, 176, 140, 142, 100, 156}; TREE = {390, 405, 406, 407}
HOMES = {'Mohenjo-daro', 'Harappa'}
C = json.load(open(os.path.join(ROOT, 'data/derived/merged-corpus-canonical.json')))
def cls_of(ty): return 'SEAL' if ty.startswith('SEAL') else 'TAB' if ty.startswith('TAB') else 'OTHER'
rng = random.Random(15)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
P(f'LOOP15 cycle3 quantity gaps seqkey={SEQKEY} nperm={NPERM}')

# tokens: (item, numeral sign, value, series, region, cls, site, after_opener)
tok = []
for t in C:
    s = t[SEQKEY]
    if not s: continue
    reg = 'home' if t['site'] in HOMES else 'held'
    for i in range(len(s) - 1):
        n, z = s[i], s[i + 1]
        if n in VAL and z not in NUMS:
            tok.append(dict(item=z, num=n, val=VAL[n], ser='S' if n in SHORT else 'T', reg=reg, cls=cls_of(t['type']),
                            site=t['site'], after_open=(i >= 1 and s[i - 1] in OPEN) or i == 0 and False))
P('numeral+item adjacent tokens:', len(tok))

def table(toks, label):
    by = collections.defaultdict(collections.Counter)
    for x in toks: by[x['item']][x['val']] += 1
    items = [z for z, c in by.items() if sum(c.values()) >= NMIN]
    P(f'\n== {label}: items counted >= {NMIN} times ==')
    P('item   n   values(1..12)                         low(<=2)  share')
    for z in sorted(items, key=lambda z: -sum(by[z].values())):
        c = by[z]; n = sum(c.values()); low = c[1] + c[2]
        P(f'W{z:<5} {n:4d} ' + ' '.join(f'{v}:{c[v]}' for v in sorted(c)) + f'   low={low} ({low/n:.2f})')
    return by, items

by_all, items_all = table(tok, 'all tokens, all sites')
tok_nomark = [x for x in tok if x['num'] not in (1, 2)]
P('\n(W1 is a marker, S234; W2 is the opener marker, GRAMMAR caution) -> tokens excluding W1/W2 numerals:', len(tok_nomark))
by_nm, items_nm = table(tok_nomark, 'excluding W1 and W2 numeral signs (1 and 2 then come only from 31/32 tall and 12 two-tier)')
tok_nomark2 = [x for x in tok if not (x['num'] == 2 and x['after_open'])]
by2, items2 = table(tok_nomark2, 'W2 dropped only when it follows an opener (frame marker); W1 kept')

# permutation null: numeral tokens permuted among items within strata ser x reg x cls
def perm_test(toks, items, label):
    P(f'\n-- permutation null ({label}): values permuted among counted items within series x region x class, {NPERM}x --')
    strata = collections.defaultdict(list)
    for i, x in enumerate(toks): strata[(x['ser'], x['reg'], x['cls'])].append(i)
    obs_low = collections.Counter(); obs_one = collections.Counter(); obs_two = collections.Counter(); n_item = collections.Counter()
    for x in toks:
        if x['item'] in items:
            n_item[x['item']] += 1
            if x['val'] <= 2: obs_low[x['item']] += 1
            if x['val'] == 1: obs_one[x['item']] += 1
            if x['val'] == 2: obs_two[x['item']] += 1
    le_low = collections.Counter(); le_one = collections.Counter(); le_two = collections.Counter()
    zero_low_null = []; zero_one_null = []
    vals = [x['val'] for x in toks]
    for _ in range(NPERM):
        pv = vals[:]
        for idx in strata.values():
            vv = [pv[i] for i in idx]; rng.shuffle(vv)
            for i, v in zip(idx, vv): pv[i] = v
        low = collections.Counter(); one = collections.Counter(); two = collections.Counter()
        for x, v in zip(toks, pv):
            if x['item'] in items:
                if v <= 2: low[x['item']] += 1
                if v == 1: one[x['item']] += 1
                if v == 2: two[x['item']] += 1
        for z in items:
            if low[z] <= obs_low[z]: le_low[z] += 1
            if one[z] <= obs_one[z]: le_one[z] += 1
            if two[z] <= obs_two[z]: le_two[z] += 1
        zero_low_null.append(sum(1 for z in items if low[z] == 0)); zero_one_null.append(sum(1 for z in items if one[z] == 0))
    P('item   n  obs low(<=2)  P(low<=obs)  obs 1  P(1<=obs)  obs 2  P(2<=obs)   [Bonferroni x%d]' % len(items))
    for z in sorted(items, key=lambda z: le_low[z]):
        P(f'W{z:<5} {n_item[z]:4d}   {obs_low[z]:3d}   {le_low[z]/NPERM:8.4f}   {obs_one[z]:3d}  {le_one[z]/NPERM:8.4f}   {obs_two[z]:3d}  {le_two[z]/NPERM:8.4f}   {min(1, le_low[z]/NPERM*len(items)):.3f}')
    zl = sum(1 for z in items if obs_low[z] == 0); zo = sum(1 for z in items if obs_one[z] == 0)
    zero_low_null.sort(); zero_one_null.sort()
    P(f'items with NO value 1 or 2: observed {zl} of {len(items)}; null mean {sum(zero_low_null)/NPERM:.1f}, p95 {zero_low_null[int(0.95*NPERM)]}, max {zero_low_null[-1]}, P(null>=obs) {sum(1 for v in zero_low_null if v >= zl)/NPERM:.4f}')
    P(f'items with NO value 1:       observed {zo} of {len(items)}; null mean {sum(zero_one_null)/NPERM:.1f}, p95 {zero_one_null[int(0.95*NPERM)]}, max {zero_one_null[-1]}, P(null>=obs) {sum(1 for v in zero_one_null if v >= zo)/NPERM:.4f}')

perm_test(tok, items_all, 'all tokens')
perm_test(tok_nomark, items_nm, 'excluding W1/W2 signs')

# trees in detail
P('\n== TREES (W390/405/406/407) numbered, by value and region ==')
tr = collections.defaultdict(collections.Counter)
for x in tok:
    if x['item'] in TREE: tr[x['reg']][(x['val'], x['ser'])] += 1
for reg in ('home', 'held'):
    c = tr[reg]; n = sum(c.values())
    P(f'  {reg}: n={n} ' + ' '.join(f'{v}{s}:{c[(v, s)]}' for v, s in sorted(c)))
# trees total tokens and bare
tree_tot = collections.Counter(); tree_num = collections.Counter()
for t in C:
    s = t[SEQKEY]
    for i, z in enumerate(s):
        if z in TREE:
            reg = 'home' if t['site'] in HOMES else 'held'; tree_tot[reg] += 1
            if i >= 1 and s[i - 1] in VAL: tree_num[reg] += 1
P('  tree tokens:', dict(tree_tot), 'numbered:', dict(tree_num))
P('  texts with 1 or 2 + tree (any series):')
for t in C:
    s = t[SEQKEY]
    for i in range(len(s) - 1):
        if s[i] in VAL and VAL[s[i]] <= 2 and s[i + 1] in TREE:
            P('   ', t['cisi'], t['site'], t['type'], s)

# persons numbered, by site
P('\n== PERSON signs (W90/91/176/140/142/100/156) immediately after a numeral (W2 after opener excluded), by site ==')
pers = collections.defaultdict(collections.Counter); pers_tot = collections.Counter()
for t in C:
    s = t[SEQKEY]
    for i, z in enumerate(s):
        if z in PERSON:
            pers_tot[t['site']] += 1
            if i >= 1 and s[i - 1] in VAL and not (s[i - 1] == 2 and i >= 2 and s[i - 2] in OPEN):
                pers[t['site']][(z, VAL[s[i - 1]])] += 1
for site in sorted(pers_tot, key=lambda s: -pers_tot[s]):
    n = sum(pers[site].values())
    if pers_tot[site] >= 5 or n:
        P(f'  {site:<16} persons {pers_tot[site]:4d}  numbered {n:3d}  ' + ' '.join(f'W{z}={v}:{c}' for (z, v), c in pers[site].most_common(8)))
home_n = sum(sum(pers[s].values()) for s in HOMES); home_t = sum(pers_tot[s] for s in HOMES)
held_n = sum(sum(pers[s].values()) for s in pers if s not in HOMES); held_t = sum(pers_tot[s] for s in pers_tot if s not in HOMES)
P(f'  home: {home_n}/{home_t} numbered; held-out sites: {held_n}/{held_t}')
# which person is counted at home and by what
P('  home numbered persons (W156 burden-carrier takes fixed 3, S98):')
hc = collections.Counter()
for s in HOMES: hc.update(pers[s])
P('   ', ' '.join(f'W{z}+{v}:{c}' for (z, v), c in hc.most_common()))

open(os.path.join(HERE, f'loop15_c3_{SEQKEY}.txt'), 'w').write('\n'.join(out) + '\n')
