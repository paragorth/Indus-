"""S-DARK-24 cycle 3: (c) fragility audit: recompute the main results on the objects where both transcribers
read the whole text identically (cost-0 pairs), and on all matched objects, vs the full Wells corpus;
(d) does disagreement depend on object condition, preservation, material, site, object type?"""
import json, collections, random, math, sys
import numpy as np
R = '/home/user/Indus-/'; OUT = R + 'data/derived/dark/'
rng = random.Random(3)
corpus = json.load(open(R + 'data/derived/merged-corpus-canonical.json'))
P = [r for r in json.load(open(OUT + 'loop24_pairs.json')) if r['accepted']]
corr = {int(k): set(v) for k, v in json.load(open(OUT + 'loop24_corr.json')).items()}
L = []
def say(s): print(s); L.append(s)
agreed_idx = {r['widx'] for r in P if r['cost'] == 0}
matched_idx = {r['widx'] for r in P}
FULL = corpus
AGREED = [corpus[i] for i in sorted(agreed_idx)]
MATCHED = [corpus[i] for i in sorted(matched_idx)]
# objects matched but with a difference somewhere
DIFF = [corpus[i] for i in sorted(matched_idx - agreed_idx)]
say(f'(c) FRAGILITY AUDIT. Subsets: FULL Wells corpus {len(FULL)} objects; MATCHED (read by both) {len(MATCHED)}; AGREED (identical readings) {len(AGREED)}; DIFF (matched, some difference) {len(DIFF)}')
say('   seals in each: ' + ', '.join(f'{n} {sum(1 for x in S if x["type"].startswith("SEAL"))}' for n, S in (('FULL', FULL), ('MATCHED', MATCHED), ('AGREED', AGREED), ('DIFF', DIFF))))

# ---------- T1: S289 closer paradigm ----------
SUF = {400, 90}; JAR = 740
S289 = [154, 151, 158, 527, 520, 156, 226, 617, 236, 700]
def closer_stats(seqs, minn=10):
    seqs = [list(s) for s in seqs if len(s) >= 2]
    core = []
    for s in seqs:
        t = s[:-1] if s[-1] in SUF and len(s) > 1 else s
        core.append(t)
    cnt = collections.Counter(x for t in core for x in t)
    final = collections.Counter(t[-1] for t in core)
    jar_rate_by_len = collections.defaultdict(lambda: [0, 0])
    for t in core:
        jar_rate_by_len[len(t)][1] += 1
        if JAR in t: jar_rate_by_len[len(t)][0] += 1
    out = {}
    for s, n in cnt.items():
        if n < minn or s == JAR: continue
        texts = [t for t in core if s in t]
        obs = sum(1 for t in texts if JAR in t)
        exp = sum(jar_rate_by_len[len(t)][0] / jar_rate_by_len[len(t)][1] for t in texts)
        out[s] = (final[s] / n, obs / exp if exp else float('nan'), n)
    return out
def passing(st): return sorted(s for s, (f, j, n) in st.items() if f >= 0.4 and j <= 0.5)
def shuffled(seqs):
    out = []
    for s in seqs:
        t = list(s); rng.shuffle(t); out.append(t)
    return out
say('')
say('T1  S289 closer paradigm (final rate >= 0.4 before an optional W400/W90 suffix, jar co-occurrence <= 0.5x expected; signs with >= 10 tokens). Null: signs shuffled within texts, 200x')
for name, S in (('FULL', FULL), ('MATCHED', MATCHED), ('AGREED', AGREED), ('DIFF', DIFF)):
    seqs = [x['seq_raw'] for x in S]
    st = closer_stats(seqs); ps = passing(st)
    nulls = [len(passing(closer_stats(shuffled(seqs)))) for _ in range(200)]
    s289 = {s: st[s] for s in S289 if s in st}
    say(f'  {name}: {len(ps)} signs pass {ps}; null median {sorted(nulls)[100]}, 95% {sorted(nulls)[190]}, max {max(nulls)}; '
        f'S289 signs present with >= 10 tokens: ' + ', '.join(f'W{s} final {f:.2f} jar {j:.2f} n={n}' + (' PASS' if f >= 0.4 and j <= 0.5 else ' fail') for s, (f, j, n) in s289.items()))

# ---------- T2: S296 fish family ----------
FISH = [220, 240, 235, 233, 231, 226]
def ctx_vectors(seqs):
    left = collections.defaultdict(collections.Counter); right = collections.defaultdict(collections.Counter)
    for q in seqs:
        for i, s in enumerate(q):
            if i > 0: left[s][('L', q[i-1])] += 1
            if i < len(q) - 1: right[s][('R', q[i+1])] += 1
    return {s: left[s] + right[s] for s in set(left) | set(right)}
def cos(a, b):
    na = math.sqrt(sum(v*v for v in a.values())); nb = math.sqrt(sum(v*v for v in b.values()))
    return sum(a[k] * b[k] for k in set(a) & set(b)) / (na * nb) if na and nb else float('nan')
def mean_pair_cos(V, signs):
    vals = [cos(V[a], V[b]) for i, a in enumerate(signs) for b in signs[i+1:] if a in V and b in V]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float('nan')
say('')
say('T2  S296 fish family: mean pairwise context cosine of W220/240/235/233/231/226 vs 2000 frequency-matched random 6-sets')
for name, S in (('FULL', FULL), ('MATCHED', MATCHED), ('AGREED', AGREED), ('DIFF', DIFF)):
    seqs = [x['seq_raw'] for x in S]
    V = ctx_vectors(seqs); cnt = collections.Counter(s for q in seqs for s in q)
    fish = [f for f in FISH if cnt[f] >= 3]
    o = mean_pair_cos(V, fish)
    nulls = []
    for _ in range(2000):
        rs = []
        for f in fish:
            c = [s for s in cnt if 0.5 * cnt[f] <= cnt[s] <= 2 * cnt[f] and s not in rs] or [s for s in cnt if s not in rs]
            rs.append(rng.choice(c))
        nulls.append(mean_pair_cos(V, rs))
    nulls = [v for v in nulls if not math.isnan(v)]
    p = (1 + sum(1 for v in nulls if v >= o)) / (len(nulls) + 1)
    say(f'  {name}: fish present {fish} (tokens {[cnt[f] for f in fish]}); mean cosine {o:.3f} vs null median {sorted(nulls)[len(nulls)//2]:.3f}, 95% {sorted(nulls)[int(0.95*len(nulls))]:.3f} (P={p:.4f})')

# ---------- T3: S-DARK-15.3 minimum-lot rule ----------
ITEMS = [390, 405, 407, 520, 900, 585, 923, 845, 550, 220, 233, 235, 240]
say('')
say('T3  S-DARK-15.3 minimum-lot: numeral immediately before a counted item; short strokes W1..W9 (value = W number), tall W31..W39 (value = W-30). Rule: short 1 and 2 (grammar markers) do not count goods; counts start at 3')
for name, S in (('FULL', FULL), ('MATCHED', MATCHED), ('AGREED', AGREED), ('DIFF', DIFF)):
    short = collections.Counter(); tall = collections.Counter(); after_opener = 0
    for x in S:
        q = x['seq_raw']
        for i in range(1, len(q)):
            if q[i] in ITEMS:
                n = q[i-1]
                if 1 <= n <= 9:
                    short['1-2' if n <= 2 else '3+'] += 1
                    if n <= 2 and i >= 2 and q[i-2] in (817, 861, 820, 920, 692): after_opener += 1
                elif 31 <= n <= 39: tall['1-2' if n <= 32 else '3+'] += 1
    say(f'  {name}: short-stroke counts before items: value 1-2 {short["1-2"]} (of which directly after an opener = the W2 marker: {after_opener}), value 3+ {short["3+"]}; tall: 1-2 {tall["1-2"]}, 3+ {tall["3+"]}')

# ---------- T4: S321 / S-DARK-18 name ratio ----------
OP = {817, 861, 820}; MARK = {2, 60}
CLOSE = {740, 520, 151, 156, 527, 226, 617, 154, 158, 700, 236}
def middles(S):
    mids = []
    for x in S:
        if not x['type'].startswith('SEAL'): continue
        s = list(x['seq_raw'])
        if s and s[-1] in SUF: s = s[:-1]
        if s and s[-1] in CLOSE: s = s[:-1]
        if s and s[0] in OP: s = s[1:]
        if s and s[0] in MARK: s = s[1:]
        if len(s) >= 2: mids.append(tuple(s))
    return mids
def uniq(ms):
    c = collections.Counter(ms); return sum(1 for m in ms if c[m] == 1) / len(ms)
def bigram_ratio(mids, draws=100):
    big = collections.defaultdict(collections.Counter)
    for m in mids:
        p = 'S'
        for c in m: big[p][c] += 1; p = c
    def gen(n):
        out = []; p = 'S'
        for _ in range(n):
            src = big[p] if big[p] else big['S']
            ks, ws = zip(*src.items()); c = rng.choices(ks, ws)[0]; out.append(c); p = c
        return tuple(out)
    Ls = [len(m) for m in mids]
    nu = [uniq([gen(n) for n in Ls]) for _ in range(draws)]
    return uniq(mids), float(np.median(nu))
say('')
say('T4  S321/S-DARK-18 name test: seal middles (frame stripped), unique share / bigram-null unique share (ratio ~1 = behaves like random strings; Ur III names 0.43-0.62)')
full_m = middles(FULL)
for name, S in (('FULL', FULL), ('MATCHED', MATCHED), ('AGREED', AGREED), ('DIFF', DIFF)):
    mids = middles(S)
    if len(mids) < 30: say(f'  {name}: only {len(mids)} middles'); continue
    o, nm = bigram_ratio(mids)
    # size-matched FULL reference
    refs = []
    for _ in range(20):
        sub = rng.sample(full_m, len(mids)); a, b = bigram_ratio(sub, draws=30); refs.append(a / b)
    say(f'  {name}: middles {len(mids)}; unique {o:.3f} vs bigram {nm:.3f}, ratio {o/nm:.3f}; FULL corpus subsampled to the same n: ratio {np.mean(refs):.3f} (sd {np.std(refs):.3f})')

# ---------- T5: S29 opener / suffix exclusion ----------
say('')
say('T5  S29 opener vs suffix: suffix = final W400/W90; opener = initial W817/861/820. P(suffix | opener) vs P(suffix | no opener), texts >= 3 signs; and suffix rate seals vs tablets/sealings. Null: suffix labels permuted across texts (2000x)')
for name, S in (('FULL', FULL), ('MATCHED', MATCHED), ('AGREED', AGREED), ('DIFF', DIFF)):
    rows = [(x['seq_raw'][0] in OP, x['seq_raw'][-1] in SUF, x['type'].split(':')[0]) for x in S if len(x['seq_raw']) >= 3]
    op = [r for r in rows if r[0]]; nop = [r for r in rows if not r[0]]
    a = sum(r[1] for r in op); b = sum(r[1] for r in nop)
    obs = a / max(1, len(op)) - b / max(1, len(nop))
    labs = [r[1] for r in rows]; nulls = []
    for _ in range(2000):
        rng.shuffle(labs); nulls.append(sum(labs[:len(op)]) / max(1, len(op)) - sum(labs[len(op):]) / max(1, len(nop)))
    p = (1 + sum(1 for v in nulls if v <= obs)) / 2001
    seals = [r for r in rows if r[2] == 'SEAL']; tabs = [r for r in rows if r[2] in ('TAB', 'TAG')]
    say(f'  {name}: suffix after opener {a}/{len(op)} ({100*a/max(1,len(op)):.1f}%) vs without opener {b}/{len(nop)} ({100*b/max(1,len(nop)):.1f}%), one-sided permutation P={p:.4f}; '
        f'suffix on seals {sum(r[1] for r in seals)}/{len(seals)} ({100*sum(r[1] for r in seals)/max(1,len(seals)):.1f}%) vs tablets+sealings {sum(r[1] for r in tabs)}/{len(tabs)} ({100*sum(r[1] for r in tabs)/max(1,len(tabs)):.1f}%)')

# ---------- (d) condition dependence ----------
say('')
say('(d) DISAGREEMENT vs OBJECT CONDITION (Wells metadata) on all accepted pairs; flags per pair: any substitution / insertion, any lost-vs-read position, any difference at all; Mahadevan doubtful mark on the object (independent of Wells condition)')
def flags(r):
    sub = lost = ins = 0
    for o in r['ops']:
        if o[0] == 'S':
            w, m = o[1], o[2]
            if w and m and w in corr and m not in corr[w]: sub += 1
            if (w == 0) != (m == 0): lost += 1
        elif o[0] in ('W', 'M'):
            if (o[1] or o[2]): ins += 1
    return dict(sub=sub > 0, lost=lost > 0, ins=ins > 0, any=r['cost'] > 0, subins=(sub + ins) > 0, doubt=any(r['im']['doubt']))
F = [(r, flags(r)) for r in P]
def rate_table(key, getter, flag, minn=20, strat=lambda r: r['site_code'] + r['wells']['type'].split(':')[0]):
    groups = collections.defaultdict(list)
    for r, f in F:
        g = getter(r)
        if g: groups[g].append((r, f[flag]))
    groups = {g: v for g, v in groups.items() if len(v) >= minn}
    if len(groups) < 2: return
    rates = {g: sum(x for _, x in v) / len(v) for g, v in groups.items()}
    obs = max(rates.values()) - min(rates.values())
    # permutation within site x object class strata
    strata = collections.defaultdict(list)
    for g, v in groups.items():
        for r, x in v: strata[strat(r)].append(x)
    nulls = []
    for _ in range(1000):
        for s in strata.values(): rng.shuffle(s)
        ptr = collections.Counter(); rr = collections.defaultdict(list)
        for g, v in groups.items():
            for r, _ in v:
                k = strat(r); rr[g].append(strata[k][ptr[k]]); ptr[k] += 1
        nr = {g: sum(v) / len(v) for g, v in rr.items()}
        nulls.append(max(nr.values()) - min(nr.values()))
    p = (1 + sum(1 for v in nulls if v >= obs)) / 1001
    say(f'  {key} / flag {flag}: ' + ', '.join(f'{g} {100*rates[g]:.1f}% (n={len(groups[g])})' for g in sorted(groups, key=lambda g: -rates[g])) + f'; spread {100*obs:.1f} pts vs stratified null median {100*np.median(nulls):.1f} (P={p:.3f})')
for flag in ('any', 'lost', 'subins', 'sub', 'doubt'):
    rate_table('condition', lambda r: r['wells']['condition'] if r['wells']['condition'] in ('Poor', 'Fair', 'Good', 'Fine') else None, flag)
    rate_table('preservation', lambda r: r['wells']['preservation'] if r['wells']['preservation'] not in ('?', '-') else None, flag)
    rate_table('material', lambda r: r['wells']['material'] if r['wells']['material'] not in ('-',) else None, flag)
    rate_table('site', lambda r: r['site_code'], flag, strat=lambda r: r['wells']['type'].split(':')[0])
    rate_table('object type', lambda r: r['wells']['type'].split(':')[0], flag, strat=lambda r: r['site_code'])
    rate_table('IM77 source volume', lambda r: r['im']['source'] or None, flag, strat=lambda r: r['wells']['type'].split(':')[0])
# text length
say('  disagreement by text length: ' + ', '.join(f'L{lo}-{hi} any {100*np.mean([f["any"] for r, f in F if lo <= r["L"] <= hi]):.1f}% subins {100*np.mean([f["subins"] for r, f in F if lo <= r["L"] <= hi]):.1f}% (n={sum(1 for r, f in F if lo <= r["L"] <= hi)})' for lo, hi in ((3, 4), (5, 6), (7, 8), (9, 30))))
open(OUT + 'loop24_cycle3_log.txt', 'w').write('\n'.join(L) + '\n')
