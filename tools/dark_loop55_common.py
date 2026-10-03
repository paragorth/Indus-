"""Loop 55 (power audit) shared code.

Question: does the red-team battery (S366 statistics, S-DARK-41 Markov-1/2 nulls, S-DARK-25 detectors, frame /
closer-paradigm / once-rule / partial-order tests) detect 'beyond-chain' structure in KNOWN short formulaic writing when
that writing is cut to the Indus shape (n ~ 3,000, lengths 2-12, median 4, Indus-like duplication)? If not, the Indus
negatives are underpowered.

Everything here is corpus-generic: no Indus sign numbers. The Indus-specific slots (jar, W2, openers) are replaced by
their data-defined analogues (most frequent final sign, top-10 signs, first/last position), so that the identical code
runs on Ur III, Linear B, Latin, proto-cuneiform, designed codes and Indus.

Battery: 50 statistics (STATS). Nulls: within-text shuffle (SH), Markov-1 (M1) and Markov-2 (M2) chains fitted on the
sample itself per type stratum with lengths kept (tools/dark_loop41_common.markov_fit/markov_gen, copied), and M1E / M2E
(END-state chains, lengths free) as an extra. 'Beyond chain' = |z| >= 3 against the chain's null distribution AND outside
its min-max range (S366 convention)."""
import os, sys, json, math, random, collections, statistics as st
from math import comb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
DARK = os.path.join(ROOT, 'data/derived/dark')
LIB32 = os.path.join(DARK, 'loop32_corpora'); LIB48 = os.path.join(DARK, 'loop48_corpora')
OUT = os.path.join(DARK, 'loop55_samples')

# ------------------------------------------------------------------ corpora
# name -> (class, loader). class: L language, D designed code, A accounting, G grammar-without-language, I Indus
TYPE = {'ur3_syll': 'L', 'ur3_words': 'L', 'ur3_names_syll': 'L', 'linb_syll': 'L', 'linb_words': 'L', 'latin_edh': 'L',
        'latin_edh_chars': 'L', 'runes_words': 'L',
        'icd10': 'D', 'hts': 'D', 'aircraft_reg': 'D', 'unicode_names': 'D', 'heraldry': 'D',
        'chess_eco': 'G', 'chords': 'G',
        'proto_cuneiform': 'A', 'proto_elamite': 'A', 'khipu': 'A',
        'indus_seq_raw': 'I', 'indus_seq_strong': 'I', 'indus_seq_all': 'I'}


def load_ref(name):
    """-> list of (site, type, tuple seq). loop48 versions (site-tagged, one per impression) preferred."""
    p48 = os.path.join(LIB48, name + '.jsonl'); p32 = os.path.join(LIB32, name + '.jsonl')
    out = []
    if name == 'latin_edh_chars':
        for l in open(os.path.join(LIB48, 'latin_edh.jsonl')):
            d = json.loads(l); s = tuple(c for w in d['seq'] for c in w)
            out.append((d.get('site', 'x'), d.get('type', 'x'), s))
        return out
    path = p48 if os.path.exists(p48) else p32
    for l in open(path):
        d = json.loads(l)
        out.append((d.get('site', 'x'), d.get('type', 'x'), tuple(d['seq'])))
    return out


def load_indus(level='seq_all', regime='die'):
    import dark_loop41_common as L41
    C = L41.load()
    T = L41.dedup(C, level, regime)
    return [(r['site'], L41.otype(r['type']), tuple('W%d' % x for x in s)) for r, s in T if r['site'] != 'Unknown']


def indus_shape(level='seq_all'):
    """Target length histogram (2-12) and copy-count distribution of the Indus die regime."""
    T = [t for t in load_indus(level) if 2 <= len(t[2]) <= 12]
    hist = collections.Counter(len(s) for _, _, s in T)
    cnt = collections.Counter(s for _, _, s in T)
    copies = collections.Counter(cnt.values())
    return hist, copies, len(T)


# ------------------------------------------------------------------ sampler
def sample_indus_shaped(src, rnd, hist, copies=None, n=3000, dup='natural', minlen=2, maxlen=12):
    """Subsample src (list of (site,type,seq)) to n texts with the Indus length histogram.
    dup = 'natural': draw rows as they come (source duplication kept);
          'dedup'  : distinct texts only;
          'indus'  : distinct texts, each replicated with a copy count drawn from the Indus copy-count distribution.
    Draws without replacement per length; if a length pool is short, the pool is exhausted and the shortfall is
    redistributed over the other lengths (reported in meta)."""
    by = collections.defaultdict(list)
    if dup == 'natural':
        for t in src:
            if minlen <= len(t[2]) <= maxlen: by[len(t[2])].append(t)
    else:
        seen = set()
        for t in src:
            if minlen <= len(t[2]) <= maxlen and t[2] not in seen:
                seen.add(t[2]); by[len(t[2])].append(t)
    for L in by: rnd.shuffle(by[L])
    tot = sum(v for L, v in hist.items() if minlen <= L <= maxlen)
    want = {L: n * v / tot for L, v in hist.items() if minlen <= L <= maxlen}
    if dup == 'indus':
        # expected copies per distinct text
        ec = sum(k * v for k, v in copies.items()) / sum(copies.values())
        want = {L: w / ec for L, w in want.items()}
    out = []; short = 0
    take = {}
    for L, w in want.items():
        k = int(round(w)); pool = by.get(L, [])
        take[L] = min(k, len(pool)); short += k - take[L]
    short0 = short
    # redistribute shortfall proportionally over lengths with spare supply
    for _ in range(3):
        if short <= 0: break
        spare = {L: len(by.get(L, [])) - take[L] for L in want if len(by.get(L, [])) > take[L]}
        if not spare: break
        tots = sum(spare.values())
        for L, sp in spare.items():
            add = min(sp, int(math.ceil(short * sp / tots)))
            take[L] += add; short -= add
            if short <= 0: break
    ck, cw = (zip(*sorted(copies.items())) if copies else ((1,), (1,)))
    for L, k in take.items():
        for t in by[L][:k]:
            if dup == 'indus':
                c = rnd.choices(ck, cw)[0]
                out += [t] * c
            else:
                out.append(t)
    rnd.shuffle(out)
    if len(out) > n * 1.15: out = out[:int(n * 1.15)]
    ach = collections.Counter(len(s) for _, _, s in out)
    # shape mismatch: total variation distance between achieved and target length histograms (0 = exact Indus shape)
    na = max(1, sum(ach.values())); nt = max(1, sum(want.values()))
    tvd = 0.5 * sum(abs(ach.get(L, 0) / na - want.get(L, 0) / nt) for L in set(ach) | set(want))
    meta = {'n': len(out), 'shortfall0': short0 / max(1, sum(int(round(w)) for w in want.values())), 'len_tvd': tvd,
            'distinct_share': len(set(s for _, _, s in out)) / max(1, len(out)),
            'median_len': st.median([len(s) for _, _, s in out]) if out else 0}
    return out, meta


# ------------------------------------------------------------------ nulls
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
        src = m.get(h)
        if src: src = {k: v for k, v in src.items() if k != 'E'}
        if not src: src = uni
        ks, ws = zip(*src.items()); x = rnd.choices(ks, ws)[0]; out.append(x); h = (h + (x,))[1:]
    return tuple(out)


def markov_gen_end(m, rnd, order, uni, maxlen=14):
    out = []; h = ('S',) * order
    while len(out) < maxlen:
        src = m.get(h) or uni
        ks, ws = zip(*src.items()); x = rnd.choices(ks, ws)[0]
        if x == 'E':
            if out: break
            continue
        out.append(x); h = (h + (x,))[1:]
    return tuple(out)


def markov_corpus(T, rnd, order, by_type=True, end=False):
    groups = collections.defaultdict(list)
    for site, ty, s in T: groups[ty if by_type else 'all'].append(s)
    models = {g: markov_fit(v, order) for g, v in groups.items()}
    unis = {g: collections.Counter(x for s in v for x in s) for g, v in groups.items()}
    out = []
    for site, ty, s in T:
        g = ty if by_type else 'all'
        if end: out.append((site, ty, markov_gen_end(models[g], rnd, order, unis[g])))
        else: out.append((site, ty, markov_gen(models[g], len(s), rnd, order, unis[g])))
    return out


def shuffled(T, rnd):
    return [(site, ty, tuple(rnd.sample(s, len(s)))) for site, ty, s in T]


# ------------------------------------------------------------------ battery helpers
def H(counter):
    n = sum(counter.values())
    return -sum(v / n * math.log2(v / n) for v in counter.values() if v) if n else 0.0


def n80(counter):
    n = sum(counter.values()); acc = 0
    for i, (_, v) in enumerate(counter.most_common(), 1):
        acc += v
        if acc >= 0.8 * n: return i
    return len(counter)


def mi_pairs(pairs):
    if not pairs: return float('nan')
    ja = collections.Counter(pairs); a = collections.Counter(x for x, _ in pairs); b = collections.Counter(y for _, y in pairs)
    n = len(pairs)
    return sum(c / n * math.log2(c * n / (a[x] * b[y])) for (x, y), c in ja.items())


def zipf_slope(counter, top=50):
    f = sorted(counter.values(), reverse=True)[:top]
    if len(f) < 5: return float('nan')
    xs = [math.log(r + 1) for r in range(len(f))]; ys = [math.log(v) for v in f]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def binom_one_sided(k, n):
    if n <= 80: return sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    z = (k + 0.5 - n / 2) / math.sqrt(n / 4); return 0.5 * math.erfc(-z / math.sqrt(2))


def pair_counts(texts, mind=1):
    """ordered co-occurrence counts of sign pairs (each sign once per text), at distance >= mind."""
    cnt = collections.Counter()
    for s in texts:
        c = collections.Counter(s)
        u = [(i, a) for i, a in enumerate(s) if c[a] == 1]
        for x in range(len(u)):
            for y in range(x + 1, len(u)):
                (i, a), (j, b) = u[x], u[y]
                if j - i < mind: continue
                if a < b: cnt[(a, b, 0)] += 1
                else: cnt[(b, a, 1)] += 1
    out = {}
    for (a, b, d), n in cnt.items():
        x = out.setdefault((a, b), [0, 0]); x[d] += n
    return out


def order_feats(texts, mind=1, nmin=5, alpha=0.05):
    pc = pair_counts(texts, mind)
    tested = [(k, v) for k, v in pc.items() if sum(v) >= nmin]
    ps = []; free = 0
    for (a, b), (nab, nba) in tested:
        n = nab + nba; mn = min(nab, nba); ps.append(binom_one_sided(mn, n))
        if mn / n >= 0.3 and n >= 8: free += 1
    ps.sort(); m = max(1, len(tested)); k = 0
    for i, p in enumerate(ps, 1):
        if p <= alpha * i / m: k = i
    return k / m, free / m, len(tested)


def anagram_rate(texts):
    groups = collections.defaultdict(list)
    for s in texts:
        if len(s) >= 3: groups[tuple(sorted(s))].append(tuple(s))
    g2 = [v for v in groups.values() if len(v) >= 2]
    if len(g2) < 5: return float('nan')
    return sum(1 for v in g2 if len(set(v)) > 1) / len(g2)


def k80(cnt, cov=0.8):
    tot = sum(cnt.values()); acc = 0; core = []
    for a, n in cnt.most_common():
        core.append(a); acc += n
        if acc / tot >= cov: break
    return len(core), core


def comp_det_feats(texts, minn=30):
    """S-DARK-25 compact port (dark_loop32.py): complement hubs, hub free share, hub mobility, determinative loyalty."""
    toks = collections.Counter(a for s in texts for a in s)
    signs = [a for a, n in toks.items() if n >= minn]; sset = set(signs)
    R_ = collections.defaultdict(collections.Counter); L_ = collections.defaultdict(collections.Counter)
    alone = collections.Counter(s[0] for s in texts if len(s) == 1)
    for s in texts:
        for k, a in enumerate(s):
            if a in sset:
                if k + 1 < len(s): R_[a][s[k + 1]] += 1
                if k > 0: L_[a][s[k - 1]] += 1
    small = {}
    for a in signs:
        for side, T in (('R', R_), ('L', L_)):
            cnt = T[a]
            if sum(cnt.values()) < 20: continue
            k, core = k80(cnt)
            if k <= 4: small[(a, side)] = core
    hostsof = collections.defaultdict(set)
    for (a, side), core in small.items():
        for c in core: hostsof[(c, side)].add(a)
    hubs = {k: v for k, v in hostsof.items() if len(v) >= 3}
    frees = []; mobs = []
    for (c, side), hosts in hubs.items():
        att = free = 0; pos_att = collections.Counter(); st_att = collections.Counter(); unif = collections.Counter(); ustart = collections.Counter()
        for s in texts:
            L = len(s)
            for k, a in enumerate(s):
                if a != c: continue
                hostpos = k - 1 if side == 'R' else k + 1
                if 0 <= hostpos < L and s[hostpos] in hosts:
                    att += 1; pos_att[L - 1 - k] += 1; st_att[k] += 1
                    for j in range(L): unif[L - 1 - j] += 1; ustart[j] += 1
                else: free += 1
        frees.append(free / max(1, att + free))
        hu = min(H(unif), H(ustart))
        if hu: mobs.append(min(H(pos_att), H(st_att)) / hu)
    final = collections.Counter(); initial = collections.Counter()
    for s in texts:
        L = len(s)
        for k, a in enumerate(s):
            if a not in sset: continue
            if k == L - 1: final[a] += 1
            if k == 0: initial[a] += 1
    best = 0.0; nloyal = 0
    for a in signs:
        if alone[a] >= 3: continue
        for side, T, edge in (('R', R_, final), ('L', L_, initial)):
            att = sum(T[a].values()); fix = att / (att + edge[a]) if att + edge[a] else 0
            hosts = [h for h, c in T[a].items() if c >= 2]
            if fix < 0.9 or len(hosts) < 8 or H(T[a]) < 3.0: continue
            hl = [T[a][h] / toks[h] for h in hosts]; loy50 = sum(1 for x in hl if x >= 0.5) / len(hl)
            loy = sum(T[a][h] for h in hosts) / sum(toks[h] for h in hosts)
            best = max(best, loy50)
            if loy >= 0.5: nloyal += 1
    return {'hubs_per_sign': len(hubs) / max(1, len(signs)), 'hub_free_median': (sorted(frees)[len(frees) // 2] if frees else float('nan')),
            'hub_mobility': (sum(mobs) / len(mobs) if mobs else float('nan')), 'det_loy50_max': best, 'loyal_per_sign': nloyal / max(1, len(signs))}


def model_pref(texts, rnd, kfold=5):
    """held-out bits/token: bigram model vs position-slot model (dark_loop32.py). Positive = slot model better."""
    multi = [t for t in texts if len(t) >= 2]; rnd.shuffle(multi)
    folds = [multi[i::kfold] for i in range(kfold)]
    tot_b = tot_s = 0.0; n = 0
    for f in range(kfold):
        test = folds[f]; train = [t for g in range(kfold) if g != f for t in folds[g]]
        uni = collections.Counter(a for t in train for a in t); V = len(uni) + 1; Nt = sum(uni.values())
        big = collections.defaultdict(collections.Counter); slot = collections.defaultdict(collections.Counter)
        for t in train:
            p = '<s>'; L = len(t)
            for k, a in enumerate(t):
                big[p][a] += 1; p = a
                slot[(min(k, 3), min(L - 1 - k, 2))][a] += 1
        def pu(a): return (uni.get(a, 0) + 0.5) / (Nt + 0.5 * V)
        for t in test:
            p = '<s>'; L = len(t)
            for k, a in enumerate(t):
                c = big[p]; nb = sum(c.values())
                tot_b += -math.log2((c.get(a, 0) + 1.0 * pu(a) * V) / (nb + 1.0 * V)) if nb else -math.log2(pu(a))
                c = slot[(min(k, 3), min(L - 1 - k, 2))]; ns = sum(c.values())
                tot_s += -math.log2((c.get(a, 0) + 1.0 * pu(a) * V) / (ns + 1.0 * V)) if ns else -math.log2(pu(a))
                p = a; n += 1
    return (tot_s - tot_b) / max(1, n)


def bigram_gen(texts, rnd):
    b = collections.defaultdict(collections.Counter)
    for m in texts:
        p = 'S'
        for c in m: b[p][c] += 1; p = c
    out = []
    for m in texts:
        s = []; p = 'S'
        for _ in range(len(m)):
            src = b[p] if b[p] else b['S']; ks, ws = zip(*src.items()); c = rnd.choices(ks, ws)[0]; s.append(c); p = c
        out.append(tuple(s))
    return out


# ------------------------------------------------------------------ the battery (50 statistics, corpus-generic)
def battery(data, rnd):
    """data: list of (site, type, seq). Returns an OrderedDict of statistics (S366 battery made corpus-generic, plus the
    S-DARK-19 order census, S-DARK-25 detectors, S-DARK-15 once-rule analogue, S289 closer-paradigm analogue, S321 ratio)."""
    S = collections.OrderedDict()
    texts = [s for _, _, s in data]; n = len(texts)
    lens = [len(s) for s in texts]
    distinct = collections.Counter(texts)
    S['unique_text_share'] = len(distinct) / n
    S['texts_rep3'] = sum(1 for t, c in distinct.items() if len(t) >= 3 and c >= 3)
    D = set(distinct); subs = set()
    for t in D:
        if len(t) >= 4:
            for m in (3, 4, 5):
                if m < len(t):
                    for i in range(len(t) - m + 1): subs.add(t[i:i + m])
    small = [t for t in D if 3 <= len(t) <= 5]
    S['nesting_rate'] = sum(t in subs for t in small) / max(1, len(small))
    # cross-site sharing (sites as given; for corpora without sites this is nan)
    sites = set(x[0] for x in data)
    if len(sites) >= 2:
        run_sites = collections.defaultdict(set); text_sites = collections.defaultdict(set)
        for site, _, s in data:
            if len(s) >= 3: text_sites[s].add(site)
            for i in range(len(s) - 2): run_sites[s[i:i + 3]].add(site)
        S['run3_multisite_share'] = sum(len(v) >= 2 for v in run_sites.values()) / max(1, len(run_sites))
        S['text3_multisite_share'] = sum(len(v) >= 2 for v in text_sites.values()) / max(1, len(text_sites))
    else:
        S['run3_multisite_share'] = S['text3_multisite_share'] = float('nan')
    for d in (1, 2, 3, 4):
        S[f'mi_d{d}'] = mi_pairs([(s[i], s[i + d]) for s in texts for i in range(len(s) - d)])
    S['repeat_rate'] = sum(len(set(s)) < len(s) for s in texts if len(s) >= 3) / max(1, sum(l >= 3 for l in lens))
    S['adj_double_rate'] = sum(any(s[i] == s[i + 1] for i in range(len(s) - 1)) for s in texts) / n
    S['nonadj_repeat_rate'] = sum(any(s[i] == s[j] for i in range(len(s)) for j in range(i + 2, len(s))) for s in texts) / n
    uni = collections.Counter(x for s in texts for x in s)
    top10 = [a for a, _ in uni.most_common(10)]
    S['top10_twice_texts'] = sum(sum(1 for s in texts if s.count(a) >= 2) for a in top10) / n
    # long-range co-occurrence (distance >= 2) vs independence
    has = collections.Counter(); both = collections.Counter()
    for s in texts:
        for x in set(s): has[x] += 1
        seen = set()
        for i in range(len(s)):
            for j in range(i + 2, len(s)):
                if s[i] != s[j]:
                    p = (s[i], s[j]) if s[i] < s[j] else (s[j], s[i]); seen.add(p)
        for p in seen: both[p] += 1
    attract = avoid = 0
    for (x, y), c in both.items():
        e = has[x] * has[y] / n
        if c >= 5 and c / e >= 3: attract += 1
    common = [x for x, c in has.items() if c >= 40]
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]; p = (x, y) if x < y else (y, x)
            e = has[x] * has[y] / n
            if e >= 5 and both.get(p, 0) <= e / 3: avoid += 1
    S['longrange_attract_pairs'] = attract
    S['longrange_avoid_pairs'] = avoid
    # positional entropies (texts >= 3)
    T3 = [s for s in texts if len(s) >= 3]
    for p in (1, 2, 3):
        S[f'H_start{p}'] = H(collections.Counter(s[p - 1] for s in T3))
        S[f'H_end{p}'] = H(collections.Counter(s[-p] for s in T3))
    multi = [t for t in texts if len(t) >= 2]
    last = collections.Counter(t[-1] for t in multi); first = collections.Counter(t[0] for t in multi)
    S['top10_last_share'] = sum(v for _, v in last.most_common(10)) / max(1, len(multi))
    S['top10_first_share'] = sum(v for _, v in first.most_common(10)) / max(1, len(multi))
    S['n80_last/n80_all'] = n80(last) / n80(uni)
    S['n80_first/n80_all'] = n80(first) / n80(uni)
    pos = collections.defaultdict(collections.Counter)
    for t in multi:
        L = len(t)
        for k, a in enumerate(t): pos[a]['first' if k == 0 else 'last' if k == L - 1 else 'mid'] += 1
    excl = [max(c.values()) / sum(c.values()) for a, c in pos.items() if sum(c.values()) >= 10]
    S['slot_exclusivity'] = (sum(1 for x in excl if x >= 0.9) / len(excl)) if excl else float('nan')
    long_ = [t for t in texts if len(t) >= 4]
    if len(long_) >= 50:
        mid = collections.Counter(a for t in long_ for a in t[1:-1])
        S['H_second/H_mid'] = H(collections.Counter(t[1] for t in long_)) / max(1e-9, H(mid))
        S['H_penult/H_mid'] = H(collections.Counter(t[-2] for t in long_)) / max(1e-9, H(mid))
    else:
        S['H_second/H_mid'] = S['H_penult/H_mid'] = float('nan')
    # closer paradigm analogue (S289): J = most frequent final sign; count signs >= 15 tokens, final rate >= 0.4, O/E with J <= 0.5
    J = last.most_common(1)[0][0] if last else None
    tok = collections.Counter(x for s in multi for x in s)
    fin = collections.Counter(s[-1] for s in multi)
    bylen = collections.defaultdict(list)
    for s in multi: bylen[len(s)].append(J in s)
    jrate = {L: sum(v) / len(v) for L, v in bylen.items()}
    jw = collections.Counter(); je = collections.defaultdict(float)
    for s in multi:
        for x in set(s):
            if J in s: jw[x] += 1
            je[x] += jrate[len(s)]
    npar = 0
    for x, c in tok.items():
        if c < 15 or x == J: continue
        if fin[x] / c >= 0.4 and (jw[x] / je[x] if je[x] else 9) <= 0.5: npar += 1
    S['closer_paradigm_n'] = npar
    S['J_final_share'] = fin[J] / max(1, tok[J]) if J else float('nan')      # how final the top closer is
    S['J_twice_per_1000'] = 1000 * sum(1 for s in multi if s.count(J) >= 2) / max(1, sum(1 for s in multi if J in s))
    # order census (S-DARK-19.2): any distance and non-adjacent only
    fx, fr, nt = order_feats(texts, 1); S['fixed_pair_share'] = fx; S['free_pair_share'] = fr
    fx2, fr2, nt2 = order_feats(texts, 2); S['fixed_pair_share_d2'] = fx2
    S['anagram_rate'] = anagram_rate(texts)
    S['mi_first_last'] = mi_pairs([(s[0], s[-1]) for s in T3])
    S['mi_first_len'] = mi_pairs([(s[0], len(s)) for s in multi])
    # same prefix, different last sign (middle -> closer change analogue), and mirrored
    pre = collections.defaultdict(set); suf = collections.defaultdict(set); cpre = collections.Counter(); csuf = collections.Counter()
    for s in T3:
        pre[s[:-1]].add(s[-1]); cpre[s[:-1]] += 1; suf[s[1:]].add(s[0]); csuf[s[1:]] += 1
    rp = [m for m, c in cpre.items() if c >= 2]; rs = [m for m, c in csuf.items() if c >= 2]
    S['same_prefix_diff_last'] = sum(len(pre[m]) >= 2 for m in rp) / len(rp) if len(rp) >= 5 else float('nan')
    S['same_suffix_diff_first'] = sum(len(suf[m]) >= 2 for m in rs) / len(rs) if len(rs) >= 5 else float('nan')
    # Zipf and inventory
    S['zipf_all'] = zipf_slope(uni)
    S['zipf_initial'] = zipf_slope(first)
    S['zipf_final'] = zipf_slope(last)
    S['n_sign_types'] = len(uni)
    S['hapax_sign_share'] = sum(c == 1 for c in uni.values()) / len(uni)
    bg = collections.Counter(s[i:i + 2] for s in texts for i in range(len(s) - 1))
    tg = collections.Counter(s[i:i + 3] for s in texts for i in range(len(s) - 2))
    S['bigram_hapax_share'] = sum(c == 1 for c in bg.values()) / max(1, len(bg))
    S['trigram_hapax_share'] = sum(c == 1 for c in tg.values()) / max(1, len(tg))
    r4 = collections.Counter(s[i:i + 4] for s in texts for i in range(len(s) - 3))
    S['run4_rep3'] = sum(c >= 3 for c in r4.values())
    ctx = collections.defaultdict(collections.Counter)
    for s in texts:
        ss = list(s) + ['E']
        for i in range(len(ss) - 1): ctx[ss[i]][ss[i + 1]] += 1
    tot = sum(sum(v.values()) for v in ctx.values())
    S['H_cond_bigram'] = sum(sum(v.values()) / tot * H(v) for v in ctx.values())
    # detectors and models
    S.update(comp_det_feats(texts))
    S['slot_minus_bigram_bits'] = model_pref(list(texts), rnd)
    g = bigram_gen(multi, rnd)
    ug = len(set(g)) / len(g)
    S['uniq_vs_bigram'] = (len(set(multi)) / len(multi)) / ug if ug else float('nan')
    return S


STATS = ['unique_text_share', 'texts_rep3', 'nesting_rate', 'run3_multisite_share', 'text3_multisite_share', 'mi_d1', 'mi_d2', 'mi_d3', 'mi_d4',
         'repeat_rate', 'adj_double_rate', 'nonadj_repeat_rate', 'top10_twice_texts', 'longrange_attract_pairs', 'longrange_avoid_pairs',
         'H_start1', 'H_start2', 'H_start3', 'H_end1', 'H_end2', 'H_end3', 'top10_last_share', 'top10_first_share', 'n80_last/n80_all',
         'n80_first/n80_all', 'slot_exclusivity', 'H_second/H_mid', 'H_penult/H_mid', 'closer_paradigm_n', 'J_final_share', 'J_twice_per_1000',
         'fixed_pair_share', 'free_pair_share', 'fixed_pair_share_d2', 'anagram_rate', 'mi_first_last', 'mi_first_len', 'same_prefix_diff_last',
         'same_suffix_diff_first', 'zipf_all', 'zipf_initial', 'zipf_final', 'n_sign_types', 'hapax_sign_share', 'bigram_hapax_share',
         'trigram_hapax_share', 'run4_rep3', 'H_cond_bigram', 'hubs_per_sign', 'hub_free_median', 'hub_mobility', 'det_loy50_max',
         'loyal_per_sign', 'slot_minus_bigram_bits', 'uniq_vs_bigram']
# family labels for the report
FAMILY = {'reuse': ['unique_text_share', 'texts_rep3', 'nesting_rate', 'run4_rep3', 'uniq_vs_bigram', 'run3_multisite_share', 'text3_multisite_share'],
          'adjacency': ['mi_d1', 'H_cond_bigram', 'bigram_hapax_share', 'trigram_hapax_share', 'adj_double_rate'],
          'long-range': ['mi_d2', 'mi_d3', 'mi_d4', 'longrange_attract_pairs', 'longrange_avoid_pairs', 'mi_first_last', 'nonadj_repeat_rate', 'top10_twice_texts', 'repeat_rate', 'J_twice_per_1000'],
          'order': ['fixed_pair_share', 'free_pair_share', 'fixed_pair_share_d2', 'anagram_rate'],
          'frame/position': ['H_start1', 'H_start2', 'H_start3', 'H_end1', 'H_end2', 'H_end3', 'top10_last_share', 'top10_first_share', 'n80_last/n80_all', 'n80_first/n80_all',
                             'slot_exclusivity', 'H_second/H_mid', 'H_penult/H_mid', 'closer_paradigm_n', 'J_final_share', 'mi_first_len', 'same_prefix_diff_last', 'same_suffix_diff_first', 'slot_minus_bigram_bits'],
          'inventory': ['zipf_all', 'zipf_initial', 'zipf_final', 'n_sign_types', 'hapax_sign_share'],
          'detectors': ['hubs_per_sign', 'hub_free_median', 'hub_mobility', 'det_loy50_max', 'loyal_per_sign']}


def compare(real, syn):
    rows = {}
    for k in STATS:
        rv = real.get(k, float('nan'))
        vals = [s[k] for s in syn if k in s and not (isinstance(s[k], float) and math.isnan(s[k]))]
        if not vals or (isinstance(rv, float) and math.isnan(rv)):
            rows[k] = dict(real=rv, mean=float('nan'), sd=float('nan'), z=float('nan'), lo=float('nan'), hi=float('nan'), beyond=False, outside=False); continue
        m = st.mean(vals); sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        z = (rv - m) / sd if sd > 1e-12 else (0.0 if abs(rv - m) < 1e-9 else math.copysign(99.0, rv - m))
        outside = bool(rv < min(vals) or rv > max(vals))
        rows[k] = dict(real=rv, mean=m, sd=sd, z=z, lo=min(vals), hi=max(vals), outside=outside, beyond=bool(abs(z) >= 3 and outside))
    return rows


def run_one(data, rnd, nnull=30, nulls=('SH', 'M1', 'M2'), by_type=True):
    """observed battery + null batteries; returns {null: compare rows}, observed."""
    obs = battery(data, rnd)
    res = {}
    for nm in nulls:
        syn = []
        for _ in range(nnull):
            if nm == 'SH': G = shuffled(data, rnd)
            elif nm == 'M1': G = markov_corpus(data, rnd, 1, by_type)
            elif nm == 'M2': G = markov_corpus(data, rnd, 2, by_type)
            elif nm == 'M1E': G = markov_corpus(data, rnd, 1, by_type, end=True)
            elif nm == 'M2E': G = markov_corpus(data, rnd, 2, by_type, end=True)
            syn.append(battery(G, rnd))
        res[nm] = compare(obs, syn)
    return obs, res


def fmt(x):
    if isinstance(x, float):
        if math.isnan(x): return 'nan'
        return f'{x:.3f}' if abs(x) < 100 else f'{x:.0f}'
    return str(x)


# ------------------------------------------------------------------ site-structured sampling (cycle 4)
def indus_site_shares(level='seq_all'):
    T = [t for t in load_indus(level) if 2 <= len(t[2]) <= 12]
    c = collections.Counter(site for site, _, _ in T); n = sum(c.values())
    return [(s, v / n) for s, v in c.most_common()]


def sample_site_structured(src, rnd, hist, copies, n=3000, k_sites=6):
    """Map the Indus site shares (largest first) onto the source corpus's largest sites, then draw each site's quota with
    the Indus length histogram and natural duplication; the remaining Indus share goes to the pooled small sites."""
    shares = indus_site_shares()
    by_site = collections.defaultdict(list)
    for t in src:
        if 2 <= len(t[2]) <= 12: by_site[t[0]].append(t)
    big = [s for s, _ in collections.Counter({s: len(v) for s, v in by_site.items()}).most_common(k_sites)]
    out = []; mapping = {}
    for i, (isite, sh) in enumerate(shares[:k_sites]):
        pool = by_site[big[i]]
        smp, _ = sample_indus_shaped(pool, rnd, hist, copies, n=int(round(n * sh)), dup='natural')
        out += [(isite, ty, s) for _, ty, s in smp]; mapping[isite] = big[i]
    rest = sum(sh for _, sh in shares[k_sites:])
    pool = [t for s, v in by_site.items() if s not in big for t in v]
    smp, _ = sample_indus_shaped(pool, rnd, hist, copies, n=int(round(n * rest)), dup='natural')
    out += [('small:' + site, ty, s) for site, ty, s in smp]
    rnd.shuffle(out)
    meta = {'n': len(out), 'shortfall0': 0.0, 'len_tvd': 0.0, 'mapping': mapping,
            'distinct_share': len(set(s for _, _, s in out)) / max(1, len(out)), 'median_len': st.median([len(s) for _, _, s in out])}
    return out, meta
