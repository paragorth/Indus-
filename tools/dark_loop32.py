"""S-DARK-32: a TYPOLOGY CLASSIFIER. Where does Indus fall among short-text systems of KNOWN type when the repository's
metrics are used jointly?

Library: data/derived/dark/loop32_corpora/*.jsonl (tools/dark_loop32_prep.py; SOURCES.txt has provenance and type).
Battery (same code paths as the Indus work): tools/strat_codelibrary.battery (Zipf slope, positional entropies, closed-class
ends, bigram MI / conditional entropy, uniqueness vs bigram generator = S321 statistic, repeat ratio, nesting, frames, vocabulary
growth); order rigidity = pair census + BH of tools/dark_loop19.py (S-DARK-19.2, functions copied verbatim); complement hubs /
mobility and determinative loyalty of tools/dark_loop25.py (S-DARK-25.1-2, compact port); plus slot exclusivity, closer paradigm
(top-10 last share), numeral behaviour, and a bigram-vs-slot held-out model preference (the S360 strat_adequacy question made
corpus-generic: position-slot model vs bigram model, held-out bits per token).
Every corpus is subsampled to the Indus length histogram (2-13 signs, cap 3,000 texts) as in strat_codelibrary, 8 times with
different seeds; each subsample is one feature row. Classifier: L2 multinomial logistic regression (numpy) on z-scored
structural features (length features excluded), leave-one-CORPUS-out; also nearest centroid and 1-NN. Classes: L language
writing, D designed code (G grammar-without-language folded in), A accounting / tally. Controls: Indus slot-shuffled (S310 parse,
tokens permuted within slot labels), Indus Markov-2 synthetic, Indus bigram synthetic, and a label-permutation null for the CV.
Usage: python3 tools/dark_loop32.py features [nrows]   -> loop32_features.json
       python3 tools/dark_loop32.py classify           -> tables
"""
import os, sys, json, math, random, collections, re, statistics as st
from math import comb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
from strat_codelibrary import battery, match_lengths, bigram_gen, H, MAXLEN
LIB = os.path.join(ROOT, 'data/derived/dark/loop32_corpora')
OUTD = os.path.join(ROOT, 'data/derived/dark')
TYPE = {'ur3_words': 'L', 'ur3_syll': 'L', 'ur3_names_syll': 'L', 'linb_words': 'L', 'linb_syll': 'L', 'latin_edh': 'L', 'runes_words': 'L',
        'heraldry': 'D', 'icd10': 'D', 'hts': 'D', 'unicode_names': 'D', 'aircraft_reg': 'D', 'chess_eco': 'G', 'chords': 'G',
        'khipu': 'A', 'proto_elamite': 'A', 'proto_cuneiform': 'A',
        'indus_seq_raw': 'I', 'indus_seq_strong': 'I', 'indus_seq_all': 'I', 'indus_im77': 'I',
        'indus_slotshuf': 'X', 'indus_markov2': 'X', 'indus_bigram': 'X'}
TRAIN_EXCLUDE = {'runes_words'}          # below the 1,000-text bar: test only
TRAIN3 = {'L': 'L', 'D': 'D', 'G': 'D', 'A': 'A'}


def load(name):
    return [tuple(json.loads(l)['seq']) for l in open(os.path.join(LIB, name + '.jsonl'))]


# ---------------------------------------------------------------- numerals per corpus
INDUS_NUM_W = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
BR = json.load(open(os.path.join(ROOT, 'data/derived/bridge_extended.json')))
INDUS_NUM_M = {m for w in INDUS_NUM_W for m in (BR.get(str(w)) or [])}
LAT_NUM = re.compile(r'^(?=[ivxlcdm]{1,12}$)(?!^(i|vi|di|mi|vim|cum|civi|dici|dic|vivi|lici)$)m*(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$')
FR_NUM = {'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf', 'dix', 'onze', 'douze', 'treize', 'quinze', 'seize', 'vingt'}
EN_NUM = {'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ZERO', 'DIGIT'}


def numeral_test(name):
    if name.startswith('indus_im77'):
        return lambda t: t[1:].isdigit() and int(t[1:]) in INDUS_NUM_M
    if name.startswith('indus'):
        return lambda t: t[1:].isdigit() and int(t[1:]) in INDUS_NUM_W
    if name in ('linb_words', 'linb_syll'):
        return lambda t: t == 'NUM'
    if name in ('proto_elamite', 'proto_cuneiform'):
        return lambda t: re.fullmatch(r'N\d+[A-Z]?', t) is not None
    if name == 'latin_edh':
        return lambda t: LAT_NUM.match(t) is not None
    if name == 'heraldry':
        return lambda t: t in FR_NUM
    if name == 'unicode_names':
        return lambda t: t in EN_NUM
    if name in ('icd10', 'hts', 'aircraft_reg'):
        return lambda t: t.isdigit()
    if name == 'khipu':
        return lambda t: not t.endswith(':0')      # cord carries a knotted value
    return lambda t: False


# ---------------------------------------------------------------- order rigidity (dark_loop19.py, verbatim)
def pair_counts(texts):
    cnt = collections.Counter()
    for s in texts:
        c = collections.Counter(s)
        u = [a for a in s if c[a] == 1]
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                a, b = u[i], u[j]
                if a < b: cnt[(a, b, 0)] += 1
                else: cnt[(b, a, 1)] += 1
    out = {}
    for (a, b, d), n in cnt.items():
        x = out.setdefault((a, b), [0, 0]); x[d] += n
    return out


def binom_one_sided(k, n):
    if n <= 80: return sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    z = (k + 0.5 - n / 2) / math.sqrt(n / 4); return 0.5 * math.erfc(-z / math.sqrt(2))


def classify_pairs(pc, nmin=5, alpha=0.05):
    fixed = []; free = []; amb = []
    for (a, b), (nab, nba) in pc.items():
        n = nab + nba
        if n < nmin: continue
        mn = min(nab, nba); p = binom_one_sided(mn, n)
        if p <= alpha: fixed.append((a, b, nab, nba, p))
        elif mn / n >= 0.3 and n >= 8: free.append((a, b, nab, nba, p))
        else: amb.append((a, b, nab, nba, p))
    return fixed, free, amb


def bh(fixed, allpairs_n, alpha=0.05):
    ps = sorted(f[4] for f in fixed); m = allpairs_n; k = 0
    for i, p in enumerate(ps, 1):
        if p <= alpha * i / m: k = i
    thr = ps[k - 1] if k else -1
    return [f for f in fixed if f[4] <= thr], thr


def anagram_rate(texts):
    groups = collections.defaultdict(list)
    for s in texts:
        if len(s) >= 3: groups[tuple(sorted(s))].append(tuple(s))
    g2 = [v for v in groups.values() if len(v) >= 2]
    if not g2: return 0.0, 0
    return sum(1 for v in g2 if len(set(v)) > 1) / len(g2), len(g2)


def order_feats(texts):
    pc = pair_counts(texts)
    tested = sum(1 for v in pc.values() if sum(v) >= 5)
    fixed, free, amb = classify_pairs(pc); fb, _ = bh(fixed, max(1, tested))
    tot = sum(sum(v) for v in pc.values() if sum(v) >= 5)
    fx = sum(f[2] + f[3] for f in fb); fr = sum(f[2] + f[3] for f in free)
    ar, ng = anagram_rate(texts)
    return {'fixed_pair_share': len(fb) / max(1, tested), 'free_pair_share': len(free) / max(1, tested),
            'fixed_inst_share': fx / max(1, tot), 'free_inst_share': fr / max(1, tot), 'anagram_rate': ar}


# ---------------------------------------------------------------- complements / determinatives (dark_loop25.py, compact port)
def k80(cnt, cov=0.8):
    tot = sum(cnt.values()); acc = 0; core = []
    for a, n in cnt.most_common():
        core.append(a); acc += n
        if acc / tot >= cov: break
    return len(core), core


def comp_det_feats(texts, minn=30):
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
    # determinative loyalty (loop25 cycle 2, loyalty branch only)
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
    return {'hubs_per_sign': len(hubs) / max(1, len(signs)), 'hub_free_median': (sorted(frees)[len(frees) // 2] if frees else 0.0),
            'hub_mobility': (sum(mobs) / len(mobs) if mobs else 0.0), 'det_loy50_max': best, 'loyal_per_sign': nloyal / max(1, len(signs)),
            'smallset_share': len(small) / max(1, 2 * len(signs))}


# ---------------------------------------------------------------- slots, closers, numerals, model preference
def slot_feats(texts, isnum):
    multi = [t for t in texts if len(t) >= 2]
    pos = collections.defaultdict(collections.Counter)
    for t in multi:
        L = len(t)
        for k, a in enumerate(t):
            pos[a]['first' if k == 0 else 'last' if k == L - 1 else 'mid'] += 1
    excl = [max(c.values()) / sum(c.values()) for a, c in pos.items() if sum(c.values()) >= 10]
    r = {'slot_exclusivity': (sum(1 for x in excl if x >= 0.9) / len(excl) if excl else 0.0)}
    last = collections.Counter(t[-1] for t in multi); first = collections.Counter(t[0] for t in multi)
    r['top10_last_share'] = sum(v for _, v in last.most_common(10)) / max(1, len(multi))
    r['top10_first_share'] = sum(v for _, v in first.most_common(10)) / max(1, len(multi))
    # positional conditional entropy profile on texts >= 4: H(pos2)/H(mid) and H(last-1)/H(mid)
    long = [t for t in texts if len(t) >= 4]
    if len(long) >= 50:
        mid = collections.Counter(a for t in long for a in t[1:-1])
        r['H_second/H_mid'] = H(collections.Counter(t[1] for t in long)) / max(1e-9, H(mid))
        r['H_penult/H_mid'] = H(collections.Counter(t[-2] for t in long)) / max(1e-9, H(mid))
    else:
        r['H_second/H_mid'] = r['H_penult/H_mid'] = 1.0
    # numerals
    N = sum(len(t) for t in multi); nn = 0; nlast = 0; nfirst = 0; nadj = 0; nrun = 0
    for t in multi:
        f = [isnum(a) for a in t]
        nn += sum(f); nlast += f[-1]; nfirst += f[0]
        for k in range(len(t) - 1):
            if f[k] and f[k + 1]: nrun += 1
            if f[k] != f[k + 1]: nadj += 1
    r['num_share'] = nn / max(1, N)
    r['num_end_bias'] = (nlast - nfirst) / max(1, nn)   # +1 numerals at the end, -1 at the start, 0 none or balanced
    r['num_run_share'] = nrun / max(1, nn)
    return r


def model_pref(texts, rnd, kfold=5):
    """held-out bits/token: bigram model (unigram backoff, add-0.5) vs slot model P(sign | slot), slot = (min(k,3), min(L-1-k,2)).
    Positive = the slot model predicts the text better than the bigram model."""
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
    return {'bits_bigram': tot_b / max(1, n), 'bits_slot': tot_s / max(1, n), 'slot_minus_bigram_bits': (tot_s - tot_b) / max(1, n)}


def full_battery(texts, name, rnd):
    r = battery(list(texts), rnd)
    r.update(order_feats([list(t) for t in texts]))
    r.update(comp_det_feats([list(t) for t in texts]))
    r.update(slot_feats(texts, numeral_test(name)))
    r.update(model_pref([list(t) for t in texts], rnd))
    return r


# ---------------------------------------------------------------- Indus controls
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]; FISH = {235, 240, 233, 231, 220}; NUM = INDUS_NUM_W


def make_parser(seqs):
    left = collections.defaultdict(collections.Counter)
    for s in seqs:
        s = list(s)
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
    QUAL = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        QUAL[c] = q

    def parse(s):
        lab = ['NAME'] * len(s); i = 0; j = len(s)
        if s[0] in OPEN:
            lab[0] = 'OPENER'; i = 1
            if len(s) > 1 and s[1] in MARK:
                lab[1] = 'MARKER'; i = 2
                if s[0] == 920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
        while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CL or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
        if j - 1 >= i and s[j - 1] in CL:
            c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
            if c == 520:
                if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
                while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
            elif c == 740:
                if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
                if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
            elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
        for k in range(i, j - 1):
            if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
        for k in range(i, j):
            if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
        return lab
    return parse


def indus_controls(seed=32):
    """slot-shuffled Indus (tokens permuted ACROSS texts within slot label x length bin, so slot identity is kept and
    within-slot sequence order destroyed), Markov-2 synthetic Indus, bigram synthetic Indus; all on seq_all / seq_raw ints."""
    C = json.load(open(os.path.join(ROOT, 'data/derived/merged-corpus-canonical.json')))
    R = random.Random(seed)
    seqs = [list(r['seq_all']) for r in C if r.get('seq_all')]
    parse = make_parser(seqs)
    pools = collections.defaultdict(list); idx = collections.defaultdict(list)
    labs = [parse(s) for s in seqs]
    for i, (s, lab) in enumerate(zip(seqs, labs)):
        lb = min(len(s), 7)
        for p, (a, l) in enumerate(zip(s, lab)):
            pools[(l, lb)].append(a); idx[(l, lb)].append((i, p))
    new = [list(s) for s in seqs]
    for k, pl in pools.items():
        R.shuffle(pl)
        for (i, p), a in zip(idx[k], pl): new[i][p] = a
    slotshuf = [tuple(f'W{a}' for a in s) for s in new]
    raw = [tuple(f'W{a}' for a in r['seq_raw']) for r in C if r.get('seq_raw')]
    # Markov-2
    tri = collections.defaultdict(collections.Counter); big = collections.defaultdict(collections.Counter)
    for t in raw:
        p2, p1 = '<s>', '<s>'
        for a in t:
            tri[(p2, p1)][a] += 1; big[p1][a] += 1; p2, p1 = p1, a
    m2 = []
    for t in raw:
        p2, p1 = '<s>', '<s>'; g = []
        for _ in range(len(t)):
            c = tri.get((p2, p1)) or big.get(p1) or big['<s>']
            a = R.choices(list(c.keys()), list(c.values()))[0]; g.append(a); p2, p1 = p1, a
        m2.append(tuple(g))
    bg = [tuple(t) for t in bigram_gen([list(t) for t in raw], seed=seed)]
    return {'indus_slotshuf': slotshuf, 'indus_markov2': m2, 'indus_bigram': bg}


# ---------------------------------------------------------------- features stage
LEN_FEATS = {'mean_len', 'median_len', 'single_share'}


def features(nrows=8):
    names = [n for n in TYPE if n not in ('indus_slotshuf', 'indus_markov2', 'indus_bigram')]
    data = {n: load(n) for n in names}
    data.update(indus_controls())
    target = collections.Counter(min(len(t), MAXLEN) for t in data['indus_seq_raw'])
    rows = []
    for name, texts in data.items():
        for b in range(nrows):
            rnd = random.Random(1000 * b + 7)
            if TYPE[name] in ('I', 'X'):
                multi = [t for t in texts if 2 <= len(t) <= MAXLEN]
                sample = multi if b == 0 else rnd.sample(multi, int(0.85 * len(multi)))
            else:
                sample = match_lengths(texts, target, rnd)
            if len(sample) < 300:
                print(f'  {name}: only {len(sample)} matched texts', file=sys.stderr)
            r = full_battery(sample, name, rnd)
            r['single_share'] = sum(1 for t in texts if len(t) == 1) / len(texts)
            r['n_texts'] = len(sample)
            rows.append({'corpus': name, 'type': TYPE[name], 'row': b, 'feats': r})
            print(f'{name} row {b}: n={len(sample)} fixed {r["fixed_pair_share"]:.2f} uniq/bigram {r["uniq_vs_bigram"]:.2f} '
                  f'slot-bigram {r["slot_minus_bigram_bits"]:+.2f} hubs {r["hubs_per_sign"]:.2f} loy {r["det_loy50_max"]:.2f}', flush=True)
    json.dump(rows, open(os.path.join(OUTD, 'loop32_features.json'), 'w'), indent=0)


# ---------------------------------------------------------------- classifier stage
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z); return E / E.sum(axis=1, keepdims=True)


def fit_logreg(X, y, ncls, lam=1.0, iters=3000, lr=0.05):
    n, d = X.shape; W = np.zeros((d + 1, ncls)); Xb = np.hstack([X, np.ones((n, 1))])
    Y = np.eye(ncls)[y]
    # class weights: balance classes
    cw = n / (ncls * np.bincount(y, minlength=ncls).astype(float)); sw = cw[y][:, None]
    for _ in range(iters):
        P = softmax(Xb @ W)
        G = Xb.T @ ((P - Y) * sw) / n + lam * np.vstack([W[:-1], np.zeros((1, ncls))]) / n
        W -= lr * G
    return W


def predict(W, X):
    return softmax(np.hstack([X, np.ones((X.shape[0], 1))]) @ W)


def classify():
    rows = json.load(open(os.path.join(OUTD, 'loop32_features.json')))
    feats = sorted(k for k in rows[0]['feats'] if k not in LEN_FEATS and k not in ('n_texts', 'bits_bigram', 'bits_slot'))
    CLS = ['L', 'D', 'A']; ci = {c: i for i, c in enumerate(CLS)}
    corp = sorted({r['corpus'] for r in rows}, key=lambda c: (TYPE[c], c))
    byc = {c: [r for r in rows if r['corpus'] == c] for c in corp}
    X = {c: np.array([[r['feats'][f] for f in feats] for r in byc[c]]) for c in corp}
    out = []; P = out.append
    P('S-DARK-32 typology classifier. Features (structural, length features excluded): ' + ', '.join(feats))
    P(f'Corpora: {len(corp)}; rows per corpus: {len(byc[corp[0]])} (length-matched subsamples, cap 3,000 texts).')
    # ---- table of per-corpus means
    P('\n== Feature table (mean over rows) ==')
    P('corpus'.ljust(17) + 'T ' + ''.join(f[:11].rjust(12) for f in feats))
    for c in corp:
        P(c[:16].ljust(17) + TYPE[c] + ' ' + ''.join(f'{X[c][:, i].mean():12.3f}' for i in range(len(feats))))
    train_c = [c for c in corp if TYPE[c] in TRAIN3 and c not in TRAIN_EXCLUDE]

    def zfit(cs):
        A = np.vstack([X[c] for c in cs]); return A.mean(axis=0), A.std(axis=0) + 1e-9

    def run_cv(labels, lam=1.0):
        """leave-one-corpus-out; returns per-corpus mean prob vectors"""
        res = {}
        for c in train_c:
            tr = [d for d in train_c if d != c]
            mu, sd = zfit(tr)
            Xtr = np.vstack([(X[d] - mu) / sd for d in tr]); ytr = np.array([ci[labels[d]] for d in tr for _ in range(len(X[d]))])
            W = fit_logreg(Xtr, ytr, 3, lam=lam)
            res[c] = predict(W, (X[c] - mu) / sd).mean(axis=0)
        return res
    labels = {c: TRAIN3[TYPE[c]] for c in train_c}
    P('\n== Leave-one-corpus-out cross-validation (L2 multinomial logistic regression, class-balanced) ==')
    cv = run_cv(labels)
    acc = sum(1 for c in train_c if CLS[int(np.argmax(cv[c]))] == labels[c]) / len(train_c)
    rowacc = 0; nrow = 0
    for c in train_c:
        P(f'  {c:16s} true {labels[c]}  P(L,D,A) = {cv[c][0]:.2f} {cv[c][1]:.2f} {cv[c][2]:.2f}  -> {CLS[int(np.argmax(cv[c]))]}' + ('' if CLS[int(np.argmax(cv[c]))] == labels[c] else '  WRONG'))
    P(f'  corpus-level accuracy {acc:.2f} ({sum(1 for c in train_c if CLS[int(np.argmax(cv[c]))] == labels[c])}/{len(train_c)}); chance (majority class) {max(collections.Counter(labels.values()).values())/len(train_c):.2f}')
    # label permutation null
    R = random.Random(5); accs = []
    for _ in range(40):
        vals = list(labels.values()); R.shuffle(vals); lab2 = dict(zip(train_c, vals))
        cv2 = run_cv(lab2); accs.append(sum(1 for c in train_c if CLS[int(np.argmax(cv2[c]))] == lab2[c]) / len(train_c))
    P(f'  label-permutation null (40 relabellings): mean accuracy {st.mean(accs):.2f}, 95th pct {sorted(accs)[int(0.95*len(accs))]:.2f}, max {max(accs):.2f}; '
      f'P(null >= observed) = {(sum(1 for a in accs if a >= acc)+1)/(len(accs)+1):.3f}')
    # nearest centroid and 1-NN in z space (LOCO)
    mu_all, sd_all = zfit(train_c)
    Z = {c: ((X[c] - mu_all) / sd_all).mean(axis=0) for c in corp}
    nc_ok = nn_ok = 0
    for c in train_c:
        tr = [d for d in train_c if d != c]
        cent = {k: np.mean([Z[d] for d in tr if labels[d] == k], axis=0) for k in CLS}
        pc = min(CLS, key=lambda k: np.linalg.norm(Z[c] - cent[k])); nc_ok += pc == labels[c]
        nn = min(tr, key=lambda d: np.linalg.norm(Z[c] - Z[d])); nn_ok += labels[nn] == labels[c]
    P(f'  nearest-centroid LOCO accuracy {nc_ok/len(train_c):.2f}; 1-NN LOCO accuracy {nn_ok/len(train_c):.2f}')
    # ---- full model, feature weights
    Xtr = np.vstack([(X[d] - mu_all) / sd_all for d in train_c]); ytr = np.array([ci[labels[d]] for d in train_c for _ in range(len(X[d]))])
    W = fit_logreg(Xtr, ytr, 3)
    P('\n== Full-model standardised weights (L, D, A); |w| ranks the drivers ==')
    order = sorted(range(len(feats)), key=lambda i: -np.abs(W[i]).sum())
    for i in order:
        P(f'  {feats[i]:24s} {W[i][0]:+.2f} {W[i][1]:+.2f} {W[i][2]:+.2f}')
    # leave-one-feature-out CV accuracy drop
    P('\n== Leave-one-feature-out LOCO accuracy (drop = feature matters) ==')
    drops = []
    for i, f in enumerate(feats):
        keep = [j for j in range(len(feats)) if j != i]
        res = {}
        for c in train_c:
            tr = [d for d in train_c if d != c]
            A = np.vstack([X[d][:, keep] for d in tr]); mu, sd = A.mean(axis=0), A.std(axis=0) + 1e-9
            Wf = fit_logreg((A - mu) / sd, np.array([ci[labels[d]] for d in tr for _ in range(len(X[d]))]), 3, iters=1500)
            res[c] = predict(Wf, (X[c][:, keep] - mu) / sd).mean(axis=0)
        a = sum(1 for c in train_c if CLS[int(np.argmax(res[c]))] == labels[c]) / len(train_c)
        drops.append((acc - a, f))
    for d, f in sorted(drops, reverse=True)[:10]:
        P(f'  without {f:24s} accuracy {acc-d:.2f} (drop {d:+.2f})')
    # ---- classify Indus, IM77, controls, runes
    P('\n== Classification of the unknowns and controls (full model; P over rows: mean, min-max) ==')
    tests = [c for c in corp if TYPE[c] in ('I', 'X') or c in TRAIN_EXCLUDE]
    for c in tests:
        Pr = predict(W, (X[c] - mu_all) / sd_all)
        m = Pr.mean(axis=0); lo = Pr.min(axis=0); hi = Pr.max(axis=0)
        dists = sorted(((np.linalg.norm(Z[c] - Z[d]), d) for d in train_c))
        P(f'  {c:16s} P(L)={m[0]:.2f} [{lo[0]:.2f}-{hi[0]:.2f}] P(D)={m[1]:.2f} [{lo[1]:.2f}-{hi[1]:.2f}] P(A)={m[2]:.2f} [{lo[2]:.2f}-{hi[2]:.2f}] -> {CLS[int(np.argmax(m))]}; '
          f'nearest (z-dist): ' + ', '.join(f'{d} {v:.1f}' for v, d in dists[:4]))
    # ---- which metric most separates Indus from language writing
    P('\n== Univariate separation of Indus (seq_raw/strong/all/IM77 mean) from the LANGUAGE corpora (z = (Indus - mean_L) / sd_L over L rows) ==')
    Lrows = np.vstack([X[c] for c in train_c if labels[c] == 'L']); Drows = np.vstack([X[c] for c in train_c if labels[c] == 'D']); Arows = np.vstack([X[c] for c in train_c if labels[c] == 'A'])
    ind = np.vstack([X[c] for c in ('indus_seq_raw', 'indus_seq_strong', 'indus_seq_all', 'indus_im77')]).mean(axis=0)
    sep = []
    for i, f in enumerate(feats):
        zL = (ind[i] - Lrows[:, i].mean()) / (Lrows[:, i].std() + 1e-9); zD = (ind[i] - Drows[:, i].mean()) / (Drows[:, i].std() + 1e-9); zA = (ind[i] - Arows[:, i].mean()) / (Arows[:, i].std() + 1e-9)
        # share of L corpora whose mean lies on the far side of Indus (0 = Indus outside the whole L range)
        inrange = sum(1 for c in train_c if labels[c] == 'L' and min(X[c][:, i]) <= ind[i] <= max(X[c][:, i]))
        sep.append((abs(zL), f, zL, zD, zA, inrange, ind[i], Lrows[:, i].mean(), Drows[:, i].mean(), Arows[:, i].mean()))
    for a, f, zL, zD, zA, inr, v, mL, mD, mA in sorted(sep, reverse=True):
        P(f'  {f:24s} Indus {v:7.3f} | L mean {mL:7.3f} z {zL:+5.1f} (inside {inr} L corpora) | D mean {mD:7.3f} z {zD:+5.1f} | A mean {mA:7.3f} z {zA:+5.1f}')
    # ---- 4-class view: G separate
    P('\n== Four-class view (G = chess, chords kept apart): nearest reference corpora of each Indus version in z space, with type ==')
    for c in ('indus_seq_raw', 'indus_seq_strong', 'indus_seq_all', 'indus_im77', 'indus_slotshuf', 'indus_markov2', 'indus_bigram'):
        dists = sorted(((np.linalg.norm(Z[c] - Z[d]), d) for d in corp if TYPE[d] not in ('I', 'X')))
        P(f'  {c:16s} ' + ', '.join(f'{d}[{TYPE[d]}] {v:.1f}' for v, d in dists[:6]))
    txt = '\n'.join(out)
    open(os.path.join(OUTD, 'loop32_classify.txt'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    if sys.argv[1] == 'features': features(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else: classify()
