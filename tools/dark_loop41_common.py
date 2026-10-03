"""Loop 41 (red team) shared code: corpus rebuild from the current inscriptions.csv, dedup regimes, and the
headline statistics of the top positive claims, each with its own null. No interpretation; numbers only."""
import csv, json, re, random, collections, statistics as st, math, sys

ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
BIG = ('Mohenjo-daro', 'Harappa')

OP = {817, 861, 820}; MARK = {2, 60}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
CLOSE18 = {740, 520, 151, 156, 527, 526, 226, 617, 154, 158, 700, 236, 705, 706, 704}; OP18 = {817, 861, 820, 825}; SUF18 = {400, 90, 93}
FRAGILE_DROP = {617, 388, 700}
FRAGILE_MERGE = {741: 740, 742: 740}

# ------------------------------------------------------------------ rebuild
def parse_text(t):
    t = t.strip()
    toks = re.split(r'[-/]', t.strip('+[]'))
    s = [int(x) for x in toks if x.strip().isdigit()]
    s = [x for x in s if x != 0]
    return s[::-1]           # canonical convention: every text reversed to reading order

def load_levels():
    d = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))
    strong = {m['form']: m['into'] for m in d['merges'] if m['level'] == 'strong'}
    allm = {m['form']: m['into'] for m in d['merges'] if m['level'] in ('strong', 'probable')}
    def close(mp):
        out = {}
        for k in mp:
            v = k
            seen = set()
            while v in mp and v not in seen: seen.add(v); v = mp[v]
            out[k] = v
        return out
    return close(strong), close(allm)

def rebuild(out=DARK + 'loop41_corpus_rebuilt.json'):
    strong, allm = load_levels()
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    C = []
    for r in rows:
        s = parse_text(r['text'])
        if not s: continue
        C.append({'cisi': r['cisi'], 'id': r['id'], 'site': r['site'], 'type': r['type'], 'dir.': r['dir.'], 'time': r['time'],
                  'period': r['period'], 'phase': r['phase'], 'area-section': r['area-section'], 'block-house': r['block-house'],
                  'material': r['material'], 'complete': r['complete'], 'condition': r['condition'], 'preservation': r['preservation'],
                  'symbol': r['symbol'], 'sides': r['sides'], 'ref': r['sanskrit'] if (r['sanskrit'] or '').startswith('ref:') else '',
                  'seq_raw': s, 'seq_strong': [strong.get(x, x) for x in s], 'seq_all': [allm.get(x, x) for x in s]})
    for r in C: r['seq'] = r['seq_all']
    json.dump(C, open(out, 'w'))
    return C

def load(which='canonical'):
    if which == 'canonical':
        return json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    return json.load(open(DARK + 'loop41_corpus_rebuilt.json'))

# ------------------------------------------------------------------ dedup regimes
def otype(t):
    return t.split(':')[0]

def dedup(C, level, mode):
    """mode: rows | site_text | site_type_text | die. Returns list of (row, tuple(seq)).
    die = one copy per (site, object class, text) for moulded/impressed classes (TAB, TAG, POT:T:s) and one per
    (cisi, text) for everything else (a seal is its own die)."""
    out = []; seen = set()
    for r in C:
        s = tuple(r[level])
        if not s: continue
        if mode == 'rows': k = ('row', id(r))
        elif mode == 'site_text': k = (r['site'], s)
        elif mode == 'site_type_text': k = (r['site'], otype(r['type']), s)
        elif mode == 'die':
            if otype(r['type']) in ('TAB', 'TAG') or r['type'].startswith('POT:T:s'): k = (r['site'], otype(r['type']), s)
            else: k = (r['cisi'] if r['cisi'] not in ('-', '') else id(r), s)
        else: raise ValueError(mode)
        if k in seen: continue
        seen.add(k); out.append((r, s))
    return out

def defragile(T):
    out = []
    for r, s in T:
        s2 = tuple(FRAGILE_MERGE.get(x, x) for x in s if x not in FRAGILE_DROP)
        if s2: out.append((r, s2))
    return out

# ------------------------------------------------------------------ statistics
def shuffled(T, rnd):
    return [(r, tuple(rnd.sample(s, len(s)))) for r, s in T]

def closer_paradigm(T, rnd, nnull=100, minn=15, exclude_jar_from_count=True):
    """S289: signs with >= minn tokens, final rate (before optional suffix) >= 0.4 and jar co-occurrence O/E <= 0.5.
    Returns (n_pass, pass_list, null_median, null_max)."""
    def score(T):
        texts = [s for _, s in T if len(s) >= 2]
        tok = collections.Counter(x for s in texts for x in s)
        fin = collections.Counter(); jar_with = collections.Counter(); jar_exp = collections.defaultdict(float)
        bylen = collections.defaultdict(list)
        for s in texts: bylen[len(s)].append(740 in s)
        jar_rate = {L: sum(v) / len(v) for L, v in bylen.items()}
        for s in texts:
            t = list(s)
            while len(t) > 1 and t[-1] in SUF: t.pop()
            fin[t[-1]] += 1
            for x in set(s):
                if 740 in s: jar_with[x] += 1
                jar_exp[x] += jar_rate[len(s)]
        passed = []
        for x, n in tok.items():
            if n < minn or x == 740 or x in SUF: continue
            if fin[x] / n >= 0.4 and (jar_with[x] / jar_exp[x] if jar_exp[x] else 9) <= 0.5: passed.append(x)
        return passed
    obs = score(T)
    null = [len(score(shuffled(T, rnd))) for _ in range(nnull)]
    null.sort()
    return len(obs), sorted(obs), null[len(null) // 2], null[-1]

def nest_stat(texts):
    S = set(texts); longer = [t for t in S if len(t) >= 4]; subs = set()
    for t in longer:
        for n in (3, 4, 5):
            for i in range(len(t) - n + 1):
                if n < len(t): subs.add(t[i:i + n])
    small = [t for t in S if 3 <= len(t) <= 5]
    return (sum(t in subs for t in small) / len(small)) if small else float('nan'), len(small)

def nesting(T, rnd, nnull=50, hosts=None):
    """S347 nesting: share of distinct 3-5-sign texts occurring whole inside a longer text. Null: within-text shuffle."""
    texts = [s for _, s in T]
    o, n = nest_stat(texts)
    null = []
    for _ in range(nnull):
        null.append(nest_stat([tuple(rnd.sample(s, len(s))) for s in texts])[0])
    null.sort()
    return o, null[len(null) // 2], max(null), n

def heldout_split(T):
    im = set(r['site'] for r in csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    imn = {s.lower().replace('-', '').replace(' ', '') for s in im}
    def inim(site): return site.lower().replace('-', '').replace(' ', '') in imn or site in ('Mohenjo-daro', 'Harappa', 'Chanhu-daro', 'Lothal', 'Kalibangan')
    A = [(r, s) for r, s in T if inim(r['site'])]; H = [(r, s) for r, s in T if not inim(r['site'])]
    return A, H

def runs_of(S):
    out = set()
    for s in S:
        for n in (3, 4, 5):
            for i in range(len(s) - n + 1): out.add(s[i:i + n])
    return out

def p1p2(T, rnd, nnull=50):
    """S349: held-out P1 nesting (held-out 3-5 texts inside any longer text) and P2 (held-out text contains a >= 3 run
    known from MD+H). Null = within-text shuffle of everything."""
    A, H = heldout_split([(r, s) for r, s in T if len(s) >= 3])
    big = {s for r, s in A if r['site'] in BIG}
    ht = [s for _, s in H]; at = [s for _, s in A] + ht
    def p2(texts, R): return sum(any(s[i:i + 3] in R for i in range(len(s) - 2)) for s in texts) / len(texts)
    def p1(texts, allt):
        sub = set()
        for t in allt:
            if len(t) < 4: continue
            for n in (3, 4, 5):
                for i in range(len(t) - n + 1):
                    if n < len(t): sub.add(t[i:i + n])
        sm = [t for t in texts if 3 <= len(t) <= 5]
        return sum(t in sub for t in sm) / max(1, len(sm))
    R = runs_of(big); o1 = p1(ht, at); o2 = p2(ht, R); n1 = []; n2 = []
    for _ in range(nnull):
        sh = [tuple(rnd.sample(s, len(s))) for s in ht]; sa = [tuple(rnd.sample(s, len(s))) for s in at]
        sb = {tuple(rnd.sample(s, len(s))) for s in big}
        n1.append(p1(sh, sa)); n2.append(p2(sh, runs_of(sb)))
    n1.sort(); n2.sort()
    return {'n_heldout': len(ht), 'P1': o1, 'P1_null': n1[len(n1) // 2], 'P1_max': n1[-1], 'P2': o2, 'P2_null': n2[len(n2) // 2], 'P2_max': n2[-1]}

# Markov nulls of matched length (same site x type x length skeleton)
def markov_fit(texts, order):
    m = collections.defaultdict(collections.Counter)
    for s in texts:
        h = ('S',) * order
        for x in list(s) + ['E']:
            m[h][x] += 1; h = (h + (x,))[1:]
    return m
def markov_gen(m, L, rnd, order, uni):
    out = []; h = ('S',) * order
    for i in range(L):
        src = m.get(h);
        if src: src = {k: v for k, v in src.items() if k != 'E'}
        if not src: src = uni
        ks, ws = zip(*src.items()); x = rnd.choices(ks, ws)[0]; out.append(x); h = (h + (x,))[1:]
    return tuple(out)
def markov_corpus(T, rnd, order, by_type=True):
    groups = collections.defaultdict(list)
    for r, s in T: groups[otype(r['type']) if by_type else 'all'].append(s)
    models = {g: markov_fit(v, order) for g, v in groups.items()}
    unis = {g: collections.Counter(x for s in v for x in s) for g, v in groups.items()}
    return [(r, markov_gen(models[otype(r['type']) if by_type else 'all'], len(s), rnd, order, unis[otype(r['type']) if by_type else 'all'])) for r, s in T]

def name_middle(seq):
    s = list(seq)
    if s and s[-1] in SUF18: s = s[:-1]
    if s and s[-1] in CLOSE18: s = s[:-1]
    if s and s[0] in OP18: s = s[1:]
    if s and s[0] in MARK: s = s[1:]
    return tuple(s) if len(s) >= 2 else None

def uniq(ms):
    c = collections.Counter(ms); return sum(1 for m in ms if c[m] == 1) / len(ms)
def bigram(ms):
    b = collections.defaultdict(collections.Counter)
    for m in ms:
        p = 'S'
        for c in m: b[p][c] += 1; p = c
    return b
def gen(b, n, rnd):
    out = []; p = 'S'
    for _ in range(n):
        src = b[p] if b[p] else b['S']; ks, ws = zip(*src.items()); c = rnd.choices(ks, ws)[0]; out.append(c); p = c
    return tuple(out)
def uniq_ratio(ms, n, rnd, draws=30):
    rs = []; ns = []
    for _ in range(draws):
        sub = rnd.sample(ms, min(n, len(ms)))
        rs.append(uniq(sub)); b = bigram(sub)
        ns.append(uniq([gen(b, len(m), rnd) for m in sub]))
    return st.mean(rs), st.mean(ns), st.mean(rs) / st.mean(ns)

def name_ratio(T, rnd, middle=name_middle, seals_only=True, n=None):
    ms = [middle(s) for r, s in T if (not seals_only or r['type'].startswith('SEAL'))]
    ms = [m for m in ms if m]
    if len(ms) < 30: return float('nan'), float('nan'), float('nan'), len(ms)
    u, b, ratio = uniq_ratio(ms, n or len(ms), rnd)
    return u, b, ratio, len(ms)

def anagram(T):
    """S-DARK-19.1: among multisets (>= 3 signs) on >= 2 objects, share of copy pairs in a different order."""
    g = collections.defaultdict(list)
    for r, s in T:
        if len(s) >= 3: g[tuple(sorted(s))].append(s)
    pairs = diff = groups = 0; agroups = 0
    for k, v in g.items():
        if len(v) < 2: continue
        groups += 1
        if len(set(v)) > 1: agroups += 1
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs += 1; diff += v[i] != v[j]
    return groups, agroups, pairs, diff

def fixed_pairs(T, minco=5, alpha=0.05):
    """S-DARK-19.2: unordered sign pairs co-occurring in >= minco distinct texts; A-before-B vs B-before-A (any distance);
    fixed = one-sided binomial with BH. Returns (tested, fixed, free)."""
    from scipy.stats import binomtest
    ab = collections.Counter()
    for _, s in T:
        seen = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if s[i] != s[j]: seen.add((s[i], s[j]))
        for p in seen: ab[p] += 1
    done = set(); tests = []
    for (a, b), n in ab.items():
        if (b, a) in done: continue
        done.add((a, b)); m = ab[(b, a)]
        if n + m < minco: continue
        k = max(n, m); tot = n + m
        p = binomtest(k, tot, 0.5, alternative='greater').pvalue
        tests.append((p, a, b, n, m))
    tests.sort()
    M = len(tests); fixed = 0
    for i, (p, *_) in enumerate(tests):
        if p <= alpha * (i + 1) / M: fixed = i + 1
    free = sum(1 for p, a, b, n, m in tests if min(n, m) / (n + m) >= 0.3)
    return M, fixed, free

def ghosts_dead(T):
    seals = [s for r, s in T if r['type'].startswith('SEAL') and len(s) >= 2]
    tags = {s for r, s in T if r['type'].startswith('TAG') and len(s) >= 2}
    used = {s for r, s in T if (r['type'].startswith('TAG') or r['type'].startswith('TAB')) and len(s) >= 2}
    sealset = set(seals)
    ghost = sum(1 for s in tags if s not in sealset)
    dead = sum(1 for s in seals if s not in used)
    return len(tags), ghost, len(seals), dead

def middle_closer_change(T, rnd, nnull=100):
    """S366: among middles seen in >= 2 texts (with a closer), share of middle-pairs whose closer differs; null = closers permuted."""
    recs = []
    for _, s in T:
        t = list(s)
        while len(t) > 1 and t[-1] in SUF: t.pop()
        if len(t) >= 3 and t[-1] in CL:
            m = tuple(t[:-1]);
            if m and m[0] in OP: m = m[1:]
            if m and m[0] in MARK: m = m[1:]
            if len(m) >= 1: recs.append((m, t[-1]))
    def share(recs):
        g = collections.defaultdict(list)
        for m, c in recs: g[m].append(c)
        pairs = chg = 0
        for v in g.values():
            for i in range(len(v)):
                for j in range(i + 1, len(v)): pairs += 1; chg += v[i] != v[j]
        return chg / pairs if pairs else float('nan'), pairs
    o, pairs = share(recs)
    cl = [c for _, c in recs]; null = []
    for _ in range(nnull):
        rnd.shuffle(cl); null.append(share([(m, c) for (m, _), c in zip(recs, cl)])[0])
    return o, st.mean(null), pairs

def w2_rules(T, rnd, nnull=100):
    """S-DARK-15: texts with W2 twice and texts with W2 and a marked jar 741/742; null = tokens permuted across texts
    within site x object class (lengths kept)."""
    def count(T):
        tw = sum(1 for _, s in T if s.count(2) >= 2)
        mj = sum(1 for _, s in T if 2 in s and (741 in s or 742 in s))
        return tw, mj
    o = count(T)
    groups = collections.defaultdict(list)
    for i, (r, s) in enumerate(T): groups[(r['site'], otype(r['type']))].append(i)
    tw = []; mj = []
    for _ in range(nnull):
        TT = list(T)
        for g, idx in groups.items():
            pool = [x for i in idx for x in T[i][1]]; rnd.shuffle(pool); k = 0
            for i in idx:
                L = len(T[i][1]); TT[i] = (T[i][0], tuple(pool[k:k + L])); k += L
        a, b = count(TT); tw.append(a); mj.append(b)
    return o, (st.mean(tw), st.mean(mj))

def frame_rates(T):
    """GRAMMAR frame: opener first, connective never initial, closer last (texts >= 2)."""
    tx = [s for _, s in T if len(s) >= 2]
    op = sum(s[0] in OP for s in tx) / len(tx)
    conn_tok = sum(s.count(2) + s.count(60) for s in tx); conn_init = sum(s[0] in MARK for s in tx)
    def last(s):
        t = list(s)
        while len(t) > 1 and t[-1] in SUF: t.pop()
        return t[-1]
    cl = sum(last(s) in CL for s in tx) / len(tx)
    return {'n': len(tx), 'opener_first': op, 'conn_initial': conn_init / max(1, conn_tok), 'closer_last': cl}

def fmt(x):
    if isinstance(x, float): return f'{x:.3f}'
    return str(x)

# ------------------------------------------------------------------ added for cycles 2-4
FISH = [235, 240, 233, 231]   # S296 chain hat -> whiskers -> bar -> stroke
def fish_order(T):
    """S296/S297: ordered pairs of two different modified fish adjacent or not; predicted vs reverse."""
    ok = rev = 0
    for _, s in T:
        pos = {x: i for i, x in enumerate(s) if x in FISH}
        for a in range(4):
            for b in range(a + 1, 4):
                if FISH[a] in pos and FISH[b] in pos:
                    if pos[FISH[a]] < pos[FISH[b]]: ok += 1
                    else: rev += 1
    return ok, rev

def qualifier_overlap(T, rnd, nnull=200, mint=15, closers=None):
    """S303: mean pairwise histogram intersection of left-partner distributions of the closers; null = closer labels
    permuted over texts keeping left partners."""
    closers = closers or CL
    recs = []
    for _, s in T:
        t = list(s)
        while len(t) > 1 and t[-1] in SUF: t.pop()
        if len(t) >= 2 and t[-1] in closers: recs.append((t[-2], t[-1]))
    cnt = collections.Counter(c for _, c in recs)
    keep = [c for c in cnt if cnt[c] >= mint]
    recs = [(l, c) for l, c in recs if c in keep]
    def ov(recs):
        d = collections.defaultdict(collections.Counter)
        for l, c in recs: d[c][l] += 1
        cs = list(d); vals = []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = d[cs[i]], d[cs[j]]; na, nb = sum(a.values()), sum(b.values())
                vals.append(sum(min(a[k] / na, b[k] / nb) for k in set(a) | set(b)))
        return st.mean(vals) if vals else float('nan')
    o = ov(recs); labs = [c for _, c in recs]; null = []
    for _ in range(nnull):
        rnd.shuffle(labs); null.append(ov([(l, c) for (l, _), c in zip(recs, labs)]))
    null = [x for x in null if x == x]
    return o, (st.mean(null) if null else float('nan')), (sum(1 for x in null if x <= o) + 1) / (len(null) + 1), len(keep), len(recs)

def determinative_proxy(T, minn=20, minhost=10):
    """S-DARK-25 proxy: a determinative is never a whole text on its own and is obligatory for some host (>= 50% of
    that host's texts carry it adjacent). Count signs with n >= minn, 0 stand-alone, and max host loyalty >= 0.5 over
    hosts with >= minhost texts, and >= 3 distinct hosts."""
    tok = collections.Counter(x for _, s in T for x in s)
    alone = collections.Counter(s[0] for _, s in T if len(s) == 1)
    texts_with = collections.Counter(x for _, s in T for x in set(s))
    adj = collections.defaultdict(collections.Counter)
    for _, s in T:
        for i in range(len(s) - 1):
            adj[s[i]][s[i + 1]] += 1; adj[s[i + 1]][s[i]] += 1
    hits = []; cands = 0
    for x, n in tok.items():
        if n < minn or alone[x] > 0: continue
        hosts = [y for y in adj[x] if texts_with[y] >= minhost and y != x]
        if len(hosts) < 3: continue
        cands += 1
        loy = max(adj[x][y] / texts_with[y] for y in hosts)
        if loy >= 0.5: hits.append((x, round(loy, 2)))
    return cands, hits

def nest_markov(T, rnd, nnull=30, order=1):
    """S347 nesting against an order-k Markov null fitted per object class, lengths kept (stricter than within-text shuffle)."""
    texts = [s for _, s in T]
    o, n = nest_stat(texts)
    null = []
    for _ in range(nnull):
        G = markov_corpus(T, rnd, order)
        null.append(nest_stat([s for _, s in G])[0])
    null.sort()
    return o, null[len(null) // 2], max(null), n

def p1p2_fixedR(T, rnd, nnull=50):
    """S349 as originally run: big-city run set R kept fixed; only held-out texts shuffled."""
    A, H = heldout_split([(r, s) for r, s in T if len(s) >= 3])
    big = {s for r, s in A if r['site'] in BIG}
    ht = [s for _, s in H]; R = runs_of(big)
    def p2(texts): return sum(any(s[i:i + 3] in R for i in range(len(s) - 2)) for s in texts) / max(1, len(texts))
    o = p2(ht); null = sorted(p2([tuple(rnd.sample(s, len(s))) for s in ht]) for _ in range(nnull))
    return o, null[len(null) // 2], null[-1], len(ht)

def p2_markov(T, rnd, nnull=30, order=1):
    """S349 P2 against a Markov null: held-out texts regenerated from an order-k chain fitted on the HELD-OUT texts
    themselves (per object class), lengths kept; R fixed. Measures how much reuse a local chain already predicts."""
    A, H = heldout_split([(r, s) for r, s in T if len(s) >= 3])
    big = {s for r, s in A if r['site'] in BIG}; R = runs_of(big)
    def p2(texts): return sum(any(s[i:i + 3] in R for i in range(len(s) - 2)) for s in texts) / max(1, len(texts))
    o = p2([s for _, s in H]); null = []
    for _ in range(nnull):
        G = markov_corpus(H, rnd, order); null.append(p2([s for _, s in G]))
    null.sort()
    return o, null[len(null) // 2], null[-1]

def all_stats(T, rnd, nnull=40, label=''):
    res = {'label': label, 'n': len(T)}
    npass, lst, nmed, nmax = closer_paradigm(T, rnd, nnull=nnull); res.update(closers=npass, closers_null_max=nmax, closer_list=lst)
    o, nm, nx, nsmall = nesting(T, rnd, nnull=min(30, nnull)); res.update(nest=o, nest_null=nm, nest_ratio=o / nm if nm else float('nan'), nest_n=nsmall)
    u, b, ratio, nm_ = name_ratio(T, rnd); res.update(name_ratio=ratio, name_u=u, name_b=b, name_n=nm_)
    g, ag, pairs, dif = anagram(T); res.update(anagram_groups=g, anagram_pairs=pairs, anagram_diff=(dif / pairs) if pairs else float('nan'))
    M, fx, fr = fixed_pairs(T); res.update(pairs_tested=M, pairs_fixed=fx, fixed_share=fx / M if M else float('nan'), pairs_free=fr)
    nt, gh, ns, dd = ghosts_dead(T); res.update(ghost=gh / nt if nt else float('nan'), dead=dd / ns if ns else float('nan'), n_tags=nt, n_seals=ns)
    o, nl, pr = middle_closer_change(T, rnd, nnull=nnull); res.update(mid_chg=o, mid_chg_null=nl, mid_pairs=pr)
    (tw, mj), (etw, emj) = w2_rules(T, rnd, nnull=nnull); res.update(w2_twice=tw, w2_twice_E=etw, w2_741=mj, w2_741_E=emj)
    res.update(frame_rates(T))
    ok, rev = fish_order(T); res.update(fish_ok=ok, fish_rev=rev)
    o, nl, p, k, nr = qualifier_overlap(T, rnd, nnull=nnull); res.update(qual_ov=o, qual_ov_null=nl, qual_p=p, qual_closers=k, qual_n=nr)
    c, hits = determinative_proxy(T); res.update(det_cands=c, det_hits=len(hits), det_list=hits)
    return res

def line(res):
    r = res
    return (f"{r['label']:34s} n={r['n']:5d} | closers {r['closers']} (null max {r['closers_null_max']}) | nest {r['nest']:.3f}/{r['nest_null']:.3f}={r['nest_ratio']:.2f}x (n {r['nest_n']}) | "
            f"name {r['name_ratio']:.3f} (n {r['name_n']}) | anagram {r['anagram_diff']:.3f} ({r['anagram_pairs']}) fixed {r['pairs_fixed']}/{r['pairs_tested']}={r['fixed_share']:.2f} free {r['pairs_free']} | "
            f"ghost {r['ghost']:.2f} ({r['n_tags']}) dead {r['dead']:.2f} | mid-chg {r['mid_chg']:.2f} vs {r['mid_chg_null']:.2f} ({r['mid_pairs']}) | W2x2 {r['w2_twice']} (E {r['w2_twice_E']:.1f}) W2+741 {r['w2_741']} (E {r['w2_741_E']:.1f}) | "
            f"opener {r['opener_first']:.3f} conn-init {r['conn_initial']:.3f} closer-last {r['closer_last']:.3f} | fish {r['fish_ok']}:{r['fish_rev']} | qual-ov {r['qual_ov']:.3f} vs {r['qual_ov_null']:.3f} P={r['qual_p']:.3f} ({r['qual_closers']} closers) | det {r['det_hits']}/{r['det_cands']} {r['det_list']}")
