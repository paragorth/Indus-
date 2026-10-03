"""Loop 21: TRUE REPLICATION ON MATERIAL WELLS NEVER TRANSCRIBED.
Step 1. Identify IM77 texts (one text = all lines of one text_no+side, reading order) that have NO counterpart in the
        Wells/ICIT merged corpus. Matching follows tools/audit_overlap.py (bridge W -> M, exact sequence) but is extended:
        - site-aware (IM77 site group must equal the Wells site group; 'Other sites'/'West Asian' match any non-big-5 site);
        - wildcard positions: an unbridged W sign (no M in bridge_extended) or an unbridged M sign (no W maps to it) matches
          anything, so texts containing rare signs are not mis-labelled 'new' merely because the bridge is incomplete;
        - NEAR match = same site and edit distance <= 1 (one substitution, insertion or deletion), or exact reverse
          (direction disagreement). Near matches are almost surely the same object read differently.
        IM77-only(conservative) = complete texts (no sign 0) with neither an exact nor a near match.
        IM77-only(strict)       = no exact match (near matches kept).
Step 2. On the IM77-only set, in M numbers, re-test (1) slot frame, (2) fish order, (3) minimum-lot rule, (4) name
        calibration vs bigram and vs Ur III at matched n, (5) quantity-seal type, (6) nesting; each with a shuffle /
        permutation control computed on the IM77-only set alone. The same tests run on the OVERLAP set (IM77 texts that
        Wells also has) for comparison of effect sizes.
Usage: python3 loop21_engine.py <cycle 1|2|3> [--level seq_raw|seq_all] [--nodoubt]
"""
import sys, csv, json, random, collections, statistics as st, math, itertools, datetime
ROOT = '/home/user/Indus-/'; HERE = ROOT + 'data/derived/dark/'
ARGS = sys.argv[1:]
CYCLE = ARGS[0] if ARGS else '1'
LEVEL = ARGS[ARGS.index('--level') + 1] if '--level' in ARGS else 'seq_raw'
NODOUBT = '--nodoubt' in ARGS
TAG = f'loop21_cycle{CYCLE}' + ('' if LEVEL == 'seq_raw' else '_' + LEVEL) + ('_nodoubt' if NODOUBT else '')
OUT = open(HERE + TAG + '.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()

# ---------------------------------------------------------------- data
BR = {int(k): set(v) for k, v in json.load(open(ROOT + 'data/derived/bridge_extended.json')).items()}
MCOV = set(m for v in BR.values() for m in v)
WELLS = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
BIG5 = {'Mohenjo-daro': 'Mohenjodaro', 'Harappa': 'Harappa', 'Chanhu-daro': 'Chanhudaro', 'Lothal': 'Lothal', 'Kalibangan': 'Kalibangan'}
def wsite(s): return BIG5.get(s, 'OTHER')
def isite(s): return s if s in BIG5.values() else 'OTHER'

def wells_patterns(level):
    """(sitegroup, length) -> list of patterns; pattern = tuple of frozenset(M alternatives) or None (wildcard)."""
    idx = collections.defaultdict(list)
    for x in WELLS:
        s = x[level]
        if not s: continue
        pat = tuple(frozenset(BR[c]) if (c in BR and BR[c]) else None for c in s)
        idx[(wsite(x['site']), len(pat))].append((pat, x['type'].split(':')[0]))
    return idx

def im77_texts():
    """one text per (text_no, side); lines in order; returns list of dicts."""
    lines = collections.defaultdict(list)
    for r in csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')):
        if r['line'] == '9': continue
        lines[(r['text_no'], r['side'])].append(r)
    out = []
    for k, ls in lines.items():
        ls = sorted(ls, key=lambda r: int(r['line']))
        seq = []; doubt = False
        for l in ls:
            toks = l['signs_clean'].split(); seq += [int(t) for t in toks]
            if l['doubtful_positions'].strip(): doubt = True
        r0 = ls[0]
        out.append({'id': k, 'site': r0['site'], 'sg': isite(r0['site']), 'otype': r0['object_type'], 'seq': seq,
                    'damaged': 0 in seq, 'doubt': doubt, 'fs': r0['fs_category'], 'level': r0['level'], 'locus': r0['locus']})
    return out

MAXWILD = 2; MINCONC = 3
def concrete(m, cell): return cell is not None and m in MCOV
def pos_match(m, cell):
    return cell is None or m not in MCOV or m in cell

def exact(seq, pats, maxwild=None, minconc=None):
    """full-length match: every position agrees or is a wildcard; at most `maxwild` wildcard positions; at least `minconc`
    concrete agreeing positions."""
    mw = MAXWILD if maxwild is None else maxwild; mc = MINCONC if minconc is None else minconc
    L = len(seq); mc = min(mc, L); mw = 0 if L <= 3 else mw
    for pat, _ in pats:
        wild = conc = 0; ok = True
        for m, c in zip(seq, pat):
            if not concrete(m, c): wild += 1
            elif m in c: conc += 1
            else: ok = False; break
        if ok and wild <= mw and conc >= mc: return True
    return False

def exact_nowild(seq, pats): return exact(seq, pats, 0, 1)

def near(seq, idx, sg, maxwild=None, minconc=None):
    """edit distance 1 (one substitution, insertion or deletion) or exact reverse; same wildcard caps, with the mismatch
    position counted neither as wildcard nor as concrete."""
    mw = MAXWILD if maxwild is None else maxwild; mc = MINCONC if minconc is None else minconc
    L = len(seq); mc = min(mc, L - 1); mw = 0 if L <= 4 else mw
    def fits(s, pat, allow):
        wild = conc = mism = 0
        for m, c in zip(s, pat):
            if not concrete(m, c): wild += 1
            elif m in c: conc += 1
            else:
                mism += 1
                if mism > allow: return False
        return wild <= mw and conc >= mc + 1
    for pat, _ in idx.get((sg, L), []):
        if fits(seq, pat, 1): return True
        if fits(seq[::-1], pat, 0): return True
    for pat, _ in idx.get((sg, L + 1), []):
        for skip in range(L + 1):
            if fits(seq, pat[:skip] + pat[skip + 1:], 0): return True
    for pat, _ in idx.get((sg, L - 1), []):
        for skip in range(L):
            if fits(seq[:skip] + seq[skip + 1:], pat, 0): return True
    return False

def classify(level):
    idx = wells_patterns(level); T = im77_texts()
    for t in T:
        s = t['seq']; sg = t['sg']; g = sg
        t['unbr'] = sum(m not in MCOV for m in s)
        if t['damaged'] or not s:
            t['status'] = 'damaged'; t['nowild'] = False; continue
        t['nowild'] = exact_nowild(s, idx.get((g, len(s)), []))
        if exact(s, idx.get((g, len(s)), [])): t['status'] = 'exact'; continue
        if near(s, idx, g): t['status'] = 'near'; continue
        t['status'] = 'none' if t['unbr'] == 0 else 'indet'
    return T

def tune(T, level, rnd, nshuf=5):
    """false-positive table for rule variants: shuffled IM77 texts (>= 4 signs), share classified exact / near."""
    idx = wells_patterns(level)
    comp = [t for t in T if not t['damaged'] and t['seq'] and len(t['seq']) >= 4]
    P('  rule tuning (shuffled texts >= 4 signs, n=%d, %dx): maxwild minconc -> real exact%% / shuffled exact%% ; real near%% / shuffled near%%' % (len(comp), nshuf))
    for mw, mc in ((99, 1), (2, 3), (0, 3)):
        re_ = sum(exact(t['seq'], idx.get((t['sg'], len(t['seq'])), []), mw, mc) for t in comp)
        rn = sum((not exact(t['seq'], idx.get((t['sg'], len(t['seq'])), []), mw, mc)) and near(t['seq'], idx, t['sg'], mw, mc) for t in comp)
        se = []; sn = []
        for _ in range(nshuf):
            e = n = 0
            for t in comp:
                s = list(t['seq']); rnd.shuffle(s)
                if s == t['seq']: continue
                if exact(s, idx.get((t['sg'], len(s)), []), mw, mc): e += 1
                elif near(s, idx, t['sg'], mw, mc): n += 1
            se.append(e); sn.append(n)
        P(f'    maxwild={mw:2d} minconc={mc}: exact {re_ / len(comp):.1%} / {st.mean(se) / len(comp):.1%} ; near {rn / len(comp):.1%} / {st.mean(sn) / len(comp):.1%}')

def match_control(T, level, rnd, nshuf=20):
    """false-positive rate of the matcher: IM77 texts shuffled within themselves, same classification."""
    idx = wells_patterns(level)
    comp = [t for t in T if t['status'] != 'damaged' and len(t['seq']) >= 4]
    ex = []; nr = []
    for _ in range(nshuf):
        e = n = 0
        for t in comp:
            s = list(t['seq']); rnd.shuffle(s); g = t['sg']
            if s == t['seq']: continue
            if exact(s, idx.get((g, len(s)), [])): e += 1
            elif near(s, idx, g): n += 1
        ex.append(e); nr.append(n)
    return len(comp), st.mean(ex), max(ex), st.mean(nr), max(nr)

def cross_perm(T, rnd):
    """sign tokens permuted across texts (lengths kept) - null for co-occurrence/exclusivity."""
    pool = [c for t in T for c in t['seq']]; rnd.shuffle(pool); out = []; p = 0
    for t in T:
        L = len(t['seq']); out.append(dict(t, seq=pool[p:p + L])); p += L
    return out

def length_matched(src, ref, rnd, k=3):
    """sample from src k texts per ref text with the same length (without replacement where possible)."""
    by = collections.defaultdict(list)
    for t in src: by[len(t['seq'])].append(t)
    for v in by.values(): rnd.shuffle(v)
    out = []; used = collections.Counter()
    for t in ref:
        L = len(t['seq'])
        cand = by.get(L) or by.get(L - 1) or by.get(L + 1) or []
        for _ in range(k):
            if not cand: break
            out.append(cand[used[L] % len(cand)]); used[L] += 1
    return out

# ---------------------------------------------------------------- frame sets (M numbers; GRAMMAR.md / S18-S44 / S291 / S297)
OPEN = {267, 391, 293, 150}; MARK = {99, 100, 123}; CLOSE = {342, 211, 12, 15, 254, 60, 328}; SUF = {176, 1}
JAR = 342
FISH_ORDER = [65, 67, 72, 70]   # hat -> whiskers -> bar -> stroke (M numbers via bridge: W235, W240, W233, W231)
PLAIN_FISH = 59
SHORT = {97: 1, 98: 1, 99: 2, 100: 2, 102: 3, 103: 3, 104: 4, 105: 4, 106: 5, 107: 5, 108: 6, 109: 6, 110: 7, 111: 7, 112: 7, 114: 8}
TALL = {86: 1, 87: 2, 89: 3}
NUMERAL = set(SHORT) | set(TALL) | {113, 119, 121}
TREES = {161, 162, 167, 168, 169}
GOODS = TREES | {211, 287, 407, 296, 216}
PERSON = 1

def strip_suffix(s):
    s = list(s)
    while s and s[-1] in SUF: s = s[:-1]
    return s
def middle(s):
    s = strip_suffix(s)
    if s and s[-1] in CLOSE: s = s[:-1]
    if s and s[0] in OPEN: s = s[1:]
    if s and s[0] in MARK: s = s[1:]
    return tuple(s) if len(s) >= 2 else None

def shuffled(T, rnd):
    out = []
    for t in T:
        s = list(t['seq']); rnd.shuffle(s); out.append(dict(t, seq=s))
    return out

def binom_two(k, n):
    if n == 0: return float('nan')
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)

# ---------------------------------------------------------------- test 1: frame
def frame_stats(T):
    T2 = [t for t in T if len(t['seq']) >= 2]
    n = len(T2)
    op = sum(t['seq'][0] in OPEN for t in T2)
    mk_init = sum(t['seq'][0] in MARK for t in T2)
    mk_tot = sum(c in MARK for t in T2 for c in t['seq'])
    mk_second_after_open = sum(len(t['seq']) > 1 and t['seq'][0] in OPEN and t['seq'][1] in MARK for t in T2)
    cl = sum(bool(strip_suffix(t['seq'])) and strip_suffix(t['seq'])[-1] in CLOSE for t in T2)
    # order opener before closer, texts containing both
    both = 0; ok = 0
    for t in T2:
        s = t['seq']; o = [i for i, c in enumerate(s) if c in OPEN]; c_ = [i for i, c in enumerate(s) if c in CLOSE]
        if o and c_: both += 1; ok += (min(o) < max(c_))
    # exclusivity: texts holding 2+ distinct closer-class signs
    multi = sum(len(set(t['seq']) & CLOSE) >= 2 for t in T2)
    jar_with_other = sum(JAR in t['seq'] and len((set(t['seq']) & CLOSE) - {JAR}) > 0 for t in T2)
    return {'n': n, 'opener_first': op, 'marker_initial': mk_init, 'marker_tokens': mk_tot, 'open+marker': mk_second_after_open,
            'closer_last': cl, 'both': both, 'order_ok': ok, 'multi_closer': multi, 'jar_with_other_closer': jar_with_other}

def closer_table(T):
    rows = []
    for c in sorted(CLOSE):
        tok = sum(t['seq'].count(c) for t in T)
        fin = sum(bool(strip_suffix(t['seq'])) and strip_suffix(t['seq'])[-1] == c for t in T)
        withjar = sum(c in t['seq'] and JAR in t['seq'] for t in T) if c != JAR else None
        rows.append((c, tok, fin, withjar))
    return rows

def test_frame(T, label, rnd, nshuf=200):
    P(f'\n## (1) slot frame on {label} (n texts >= 2 signs)')
    o = frame_stats(T)
    nulls = collections.defaultdict(list)
    for _ in range(nshuf):
        s = frame_stats(shuffled(T, rnd))
        for k, v in s.items(): nulls[k].append(v)
    def line(k, lab):
        nm = st.mean(nulls[k]); hi = sorted(nulls[k])[int(0.975 * nshuf)]; lo = sorted(nulls[k])[int(0.025 * nshuf)]
        P(f'  {lab:46s} obs {o[k]:4d} / {o["n"]} = {o[k] / o["n"]:.3f}   shuffled mean {nm:.1f} [{lo}-{hi}]')
    line('opener_first', 'opener (M267/391/293/150) in first position')
    line('marker_initial', 'connective (M99/100/123) text-initial')
    line('open+marker', 'opener followed directly by connective')
    line('closer_last', 'closer (M342/211/12/15/254/60/328) last')
    xn = collections.defaultdict(list)
    for _ in range(nshuf):
        s = frame_stats(cross_perm(T, rnd))
        for k in ('multi_closer', 'jar_with_other_closer'): xn[k].append(s[k])
    for k, lab in (('multi_closer', 'texts with 2+ distinct closer signs (exclusivity)'), ('jar_with_other_closer', 'jar together with another closer sign')):
        v = sorted(xn[k]); P(f'  {lab:46s} obs {o[k]:4d} / {o["n"]} = {o[k] / o["n"]:.3f}   cross-text permutation mean {st.mean(v):.1f} [{v[int(0.025 * nshuf)]}-{v[int(0.975 * nshuf)]}]  P(<=obs) {(sum(x <= o[k] for x in v) + 1) / (nshuf + 1):.3f}')
    P(f'  connective tokens {o["marker_tokens"]}, of which text-initial {o["marker_initial"]} '
      f'(shuffled mean {st.mean(nulls["marker_initial"]):.1f})')
    P(f'  opener before closer in texts with both: {o["order_ok"]}/{o["both"]} (shuffled {st.mean(nulls["order_ok"]):.1f}/{st.mean(nulls["both"]):.1f})')
    P('  closer sign: tokens, final, texts co-occurring with the jar')
    for c, tok, fin, wj in closer_table(T):
        P(f'    M{c:<4d} tokens {tok:3d} final {fin:3d} ({fin / tok if tok else float("nan"):.2f}) with-jar {wj}')
    # seals vs other objects: opener and suffix (S29)
    seals = [t for t in T if t['otype'] == 'seal' and len(t['seq']) >= 2]; oth = [t for t in T if t['otype'] != 'seal' and len(t['seq']) >= 2]
    def rate(TT, f): return (sum(f(t) for t in TT), len(TT))
    a = rate(seals, lambda t: t['seq'][0] in OPEN); b = rate(oth, lambda t: t['seq'][0] in OPEN)
    c = rate(seals, lambda t: t['seq'][-1] in SUF); d = rate(oth, lambda t: t['seq'][-1] in SUF)
    P(f'  seals vs other objects: opener-first {a[0]}/{a[1]} vs {b[0]}/{b[1]}; suffix M176/M1 last {c[0]}/{c[1]} vs {d[0]}/{d[1]}')
    return o, nulls

# ---------------------------------------------------------------- test 2: fish order
def fish_pairs(T):
    rank = {c: i for i, c in enumerate(FISH_ORDER)}
    fwd = rev = 0; pairs = collections.Counter()
    for t in T:
        s = t['seq']
        for i in range(len(s) - 1):
            a, b = s[i], s[i + 1]
            if a in rank and b in rank and a != b:
                pairs[(a, b)] += 1
                if rank[a] < rank[b]: fwd += 1
                else: rev += 1
    # non-adjacent (any distance) as supplement
    fwd2 = rev2 = 0
    for t in T:
        s = [c for c in t['seq'] if c in rank]
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if s[i] != s[j]:
                    if rank[s[i]] < rank[s[j]]: fwd2 += 1
                    else: rev2 += 1
    return fwd, rev, pairs, fwd2, rev2

def test_fish(T, label):
    P(f'\n## (2) fish-qualifier order hat(M65) -> whiskers(M67) -> bar(M72) -> stroke(M70) on {label}')
    f, r, pairs, f2, r2 = fish_pairs(T)
    P(f'  adjacent modified-fish pairs: predicted order {f}, reverse {r}, two-sided binomial P = {binom_two(min(f, r), f + r):.4f}  pairs: {dict(pairs)}')
    P(f'  any-distance pairs within a text: predicted {f2}, reverse {r2}, P = {binom_two(min(f2, r2), f2 + r2):.4f}')
    # plain fish has no fixed place: plain before/after modified
    bef = aft = 0
    for t in T:
        s = t['seq']
        for i in range(len(s) - 1):
            if s[i] == PLAIN_FISH and s[i + 1] in FISH_ORDER: bef += 1
            if s[i] in FISH_ORDER and s[i + 1] == PLAIN_FISH: aft += 1
    P(f'  plain fish M59 before a modified fish {bef}, after {aft} (prediction: no fixed place)')
    fishtexts = sum(any(c in FISH_ORDER for c in t['seq']) for t in T)
    P(f'  texts with a modified fish: {fishtexts}; texts with 2+ distinct modified fish: {sum(len(set(t["seq"]) & set(FISH_ORDER)) >= 2 for t in T)}')
    return f, r

# ---------------------------------------------------------------- test 3: minimum lot
def lot_counts(T):
    """returns Counter of (series, value) for numerals directly before a counted good, and total numeral tokens."""
    c = collections.Counter(); goods = 0
    for t in T:
        s = t['seq']
        for i in range(len(s) - 1):
            if s[i + 1] in GOODS and s[i] in NUMERAL:
                goods += 1
                if s[i] in SHORT: c[('short', SHORT[s[i]])] += 1
                elif s[i] in TALL: c[('tall', TALL[s[i]])] += 1
                else: c[('other', s[i])] += 1
    return c, goods

def test_lot(T, label, rnd, nperm=2000):
    P(f'\n## (3) minimum-lot rule on {label}: numeral directly before a counted good (trees M161/162/167/168/169, arrow M211, bracket M287, M407, M296, M216)')
    c, goods = lot_counts(T)
    P(f'  numeral+good pairs: {goods}; distribution: {sorted(c.items(), key=lambda kv: (kv[0][0], kv[0][1]))}')
    bad = c[('short', 1)] + c[('short', 2)]
    tall1 = c[('tall', 1)]
    inrange = sum(v for (ser, val), v in c.items() if ser in ('short', 'tall') and 3 <= val <= 8)
    P(f'  short-1/short-2 before a good: {bad}; tall-1 before a good: {tall1}; counts 3-8: {inrange}; tall-2 (fixed term): {c[("tall", 2)]}')
    # null: permute numeral identities among all numeral slots (keeps the position of every numeral)
    slots = []; vals = []
    for ti, t in enumerate(T):
        for i, x in enumerate(t['seq']):
            if x in NUMERAL: slots.append((ti, i)); vals.append(x)
    nextgood = [T[ti]['seq'][i + 1] in GOODS if i + 1 < len(T[ti]['seq']) else False for ti, i in slots]
    nb = []
    for _ in range(nperm):
        v = vals[:]; rnd.shuffle(v)
        nb.append(sum(1 for x, g in zip(v, nextgood) if g and x in (97, 98, 99, 100)))
    P(f'  numeral tokens {len(vals)} (short-1/2 tokens {sum(1 for x in vals if x in (97, 98, 99, 100))}); '
      f'null (values permuted among numeral slots) short-1/2 before goods: mean {st.mean(nb):.1f}, min {min(nb)}, P(<= obs) = {(sum(x <= bad for x in nb) + 1) / (nperm + 1):.4f}')
    # plain person counted?
    pp = sum(1 for t in T for i in range(len(t['seq']) - 1) if t['seq'][i + 1] == PERSON and t['seq'][i] in NUMERAL)
    ppn = sum(t['seq'].count(PERSON) for t in T)
    P(f'  plain person M1 directly after a numeral: {pp} of {ppn} tokens')
    # which goods
    gc = collections.Counter(t['seq'][i + 1] for t in T for i in range(len(t['seq']) - 1) if t['seq'][i + 1] in GOODS and t['seq'][i] in NUMERAL)
    P(f'  goods counted: {dict(gc)}')
    return bad, goods, st.mean(nb)

# ---------------------------------------------------------------- test 4: name calibration
def uniq(ms):
    c = collections.Counter(ms); return sum(1 for m in ms if c[m] == 1) / len(ms) if ms else float('nan')
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
def uniq_ratio(ms, n, rnd, draws=50):
    rs = []; ns = []
    for _ in range(draws):
        sub = rnd.sample(ms, min(n, len(ms))); rs.append(uniq(sub)); b = bigram(sub)
        ns.append(uniq([gen(b, len(m), rnd) for m in sub]))
    return st.mean(rs), st.mean(ns), st.mean(rs) / st.mean(ns), st.pstdev([r / q for r, q in zip(rs, ns)])
def homonymy(ms):
    c = collections.Counter(ms)
    return sum(v for v in c.values() if v > 1) / len(ms), len(ms) / len(c), c.most_common(4)
STOP = {'dub', 'dumu', 'arad2', 'ir3', 'ir11', 'dam', 'szabra', 'ensi2', 'sukkal', 'kiszib3', 'kiszib', 'sanga', 'nu', 'gudu4',
        'sipa', 'nar', 'ugula', 'lugal', 'szagina', 'nin', 'gal'}
def ur3_legends():
    U = json.load(open(HERE + 'loop18_ur3_impressions.json'))
    seen = set(); leg = []; sobj = []; seenp = set()
    for r in U:
        n = []
        for w in r['line1']:
            if w[0] in STOP: break
            n.extend(w)
        if not (2 <= len(n) <= 6): continue
        k = tuple(tuple(w) for w in r['legend'])
        if k not in seen: seen.add(k); leg.append(tuple(n))
        if r['obj'].startswith('seal (') and r['pid'] not in seenp: seenp.add(r['pid']); sobj.append(tuple(n))
    return leg, sobj

def test_names(T, label, rnd):
    P(f'\n## (4) name calibration on {label}: middles (frame stripped), uniqueness vs bigram trained on the same middles; Ur III at matched n')
    seals = [middle(t['seq']) for t in T if t['otype'] == 'seal']; seals = [m for m in seals if m]
    allm = [middle(t['seq']) for t in T]; allm = [m for m in allm if m]
    leg, sobj = ur3_legends()
    for lab, ms in (('seals', seals), ('all objects', allm)):
        if len(ms) < 30: P(f'  {lab}: n={len(ms)} too small'); continue
        n = len(ms)
        ui = uniq_ratio(ms, n, rnd); ul = uniq_ratio(leg, n, rnd); us = uniq_ratio(sobj, n, rnd)
        h = homonymy(ms); hl = homonymy(rnd.sample(leg, n)); hs = homonymy(rnd.sample(sobj, n))
        P(f'  {lab}: n={n} distinct={len(set(ms))} mean len {st.mean(len(m) for m in ms):.2f}')
        P(f'    Indus  unique {ui[0]:.3f} vs bigram {ui[1]:.3f}  ratio {ui[2]:.3f} (sd {ui[3]:.3f}); share sharing a middle with another object {h[0]:.3f}; objects/middle {h[1]:.3f}; top {h[2]}')
        P(f'    Ur III legends   n={n}: unique {ul[0]:.3f} vs bigram {ul[1]:.3f}  ratio {ul[2]:.3f} (sd {ul[3]:.3f}); share shared {hl[0]:.3f}; seals/name {hl[1]:.3f}')
        P(f'    Ur III seal objs n={n}: unique {us[0]:.3f} vs bigram {us[1]:.3f}  ratio {us[2]:.3f} (sd {us[3]:.3f}); share shared {hs[0]:.3f}; seals/name {hs[1]:.3f}')
    return seals, allm

# ---------------------------------------------------------------- test 5: quantity seals
def qpattern(s):
    """complete text opener . connective . N(3-8) . tree ; also with plain fish"""
    if len(s) != 4: return None
    if s[0] in OPEN and s[1] in MARK and (s[2] in SHORT and 3 <= SHORT[s[2]] <= 8 or s[2] in TALL and TALL[s[2]] >= 2):
        if s[3] in TREES: return 'tree'
        if s[3] == PLAIN_FISH: return 'fish'
        if s[3] in GOODS: return 'other-good'
    return None
def qrun(s):
    for i in range(len(s) - 3):
        r = qpattern(s[i:i + 4])
        if r: return r
    return None

def test_quantity(T, label, rnd, nshuf=200):
    P(f'\n## (5) quantity-seal type opener . connective . N . tree on {label}')
    full = [t for t in T if qpattern(t['seq'])]
    runs = [t for t in T if len(t['seq']) > 4 and qrun(t['seq'])]
    P(f'  complete 4-sign texts of the type: {len(full)} ' + str(collections.Counter(qpattern(t["seq"]) for t in full)))
    for t in full: P(f'    {t["id"]} {t["site"]} {t["otype"]} {t["seq"]}')
    P(f'  longer texts containing the run: {len(runs)}')
    for t in runs[:12]: P(f'    {t["id"]} {t["site"]} {t["otype"]} {t["seq"]}')
    nf = []; nr = []
    for _ in range(nshuf):
        S = shuffled(T, rnd)
        nf.append(sum(1 for t in S if qpattern(t['seq']))); nr.append(sum(1 for t in S if len(t['seq']) > 4 and qrun(t['seq'])))
    P(f'  within-text shuffle null: complete type mean {st.mean(nf):.2f} max {max(nf)}; runs mean {st.mean(nr):.2f} max {max(nr)}')
    # sites and N values
    P(f'  sites: {collections.Counter(t["site"] for t in full + runs)}; N values: {collections.Counter(SHORT.get(t["seq"][2], ("tall", TALL.get(t["seq"][2]))) for t in full)}')
    return len(full), st.mean(nf)

# ---------------------------------------------------------------- test 6: nesting
def nest_rate(shorts, longs):
    L = [tuple(t) for t in longs]
    nested = 0
    for s in shorts:
        s = tuple(s); k = len(s); hit = False
        for l in L:
            if len(l) > k:
                for i in range(len(l) - k + 1):
                    if l[i:i + k] == s: hit = True; break
            if hit: break
        nested += hit
    return nested

def test_nesting(T_short_pool, T_long_pool, label, rnd, nshuf=50):
    P(f'\n## (6) nesting of 3-5-sign texts inside longer texts on {label}')
    shorts = list({tuple(t['seq']) for t in T_short_pool if 3 <= len(t['seq']) <= 5})
    longs = list({tuple(t['seq']) for t in T_long_pool if len(t['seq']) >= 4})
    obs = nest_rate(shorts, longs)
    nulls = []
    for _ in range(nshuf):
        sh = []
        for s in shorts: l = list(s); rnd.shuffle(l); sh.append(tuple(l))
        lg = []
        for s in longs: l = list(s); rnd.shuffle(l); lg.append(tuple(l))
        nulls.append(nest_rate(sh, lg))
    P(f'  distinct short texts {len(shorts)}, distinct long hosts {len(longs)}: nested {obs} = {obs / len(shorts):.3f}; '
      f'shuffled mean {st.mean(nulls):.1f} = {st.mean(nulls) / len(shorts):.3f} (max {max(nulls)}); ratio {obs / st.mean(nulls) if st.mean(nulls) else float("inf"):.2f}x')
    ex = []
    for s in shorts:
        for l in longs:
            if len(l) > len(s) and any(l[i:i + len(s)] == s for i in range(len(l) - len(s) + 1)): ex.append((s, l)); break
        if len(ex) >= 8: break
    for s, l in ex: P(f'    {list(s)} inside {list(l)}')
    return obs, len(shorts), st.mean(nulls)

# ---------------------------------------------------------------- P2: reuse of known runs
def runs_of(T, k=3):
    R = set()
    for t in T:
        s = t['seq']
        for i in range(len(s) - k + 1): R.add(tuple(s[i:i + k]))
    return R

def test_reuse(NEW, KNOWN, label, rnd, nshuf=100, k=3):
    P(f'\n## (P2) reuse: share of {label} texts containing a >= {k}-sign run attested in the OVERLAP (known) set')
    R = runs_of(KNOWN, k)
    def share(T):
        return sum(any(tuple(t['seq'][i:i + k]) in R for i in range(len(t['seq']) - k + 1)) for t in T)
    obs = share(NEW); nulls = [share(shuffled(NEW, rnd)) for _ in range(nshuf)]
    P(f'  obs {obs}/{len(NEW)} = {obs / len(NEW):.3f}; shuffled mean {st.mean(nulls):.1f} = {st.mean(nulls) / len(NEW):.3f} (max {max(nulls)}); ratio {obs / st.mean(nulls) if st.mean(nulls) else float("inf"):.2f}x')
    # which known runs are reused most
    cnt = collections.Counter()
    for t in NEW:
        seen = set()
        for i in range(len(t['seq']) - k + 1):
            r = tuple(t['seq'][i:i + k])
            if r in R and r not in seen: cnt[r] += 1; seen.add(r)
    P(f'  most reused runs: {cnt.most_common(10)}')
    return obs, len(NEW), st.mean(nulls)

# ---------------------------------------------------------------- driver
def describe(T, label):
    P(f'\n### set {label}: {len(T)} texts; sites {dict(collections.Counter(t["site"] for t in T))}')
    P(f'   object types {dict(collections.Counter(t["otype"] for t in T))}; length distribution {dict(sorted(collections.Counter(len(t["seq"]) for t in T).items()))}')
    P(f'   mean length {st.mean(len(t["seq"]) for t in T):.2f}; texts with a doubtful reading {sum(t["doubt"] for t in T)}')

def main():
    rnd = random.Random(21 + int(CYCLE))
    P(f'# LOOP 21 cycle {CYCLE}: true replication on IM77 texts absent from Wells/ICIT  (level={LEVEL}, nodoubt={NODOUBT}, {datetime.datetime.now().isoformat(timespec="minutes")})')
    T = classify(LEVEL)
    if NODOUBT: T = [t for t in T if not t['doubt']]
    P(f'IM77 texts (text_no+side): {len(T)}; status: {dict(collections.Counter(t["status"] for t in T))}')
    tune(T, LEVEL, rnd)
    comp = [t for t in T if t['status'] != 'damaged']
    P(f'  indeterminate (IM77 text has an unbridged M sign and no match): {sum(t["status"] == "indet" for t in comp)}')
    P(f'complete texts {len(comp)}: exact {sum(t["status"] == "exact" for t in comp)} ({sum(t["status"] == "exact" for t in comp) / len(comp):.1%}), '
      f'near {sum(t["status"] == "near" for t in comp)}, none {sum(t["status"] == "none" for t in comp)} ({sum(t["status"] == "none" for t in comp) / len(comp):.1%})')
    for L_ in (1, 2, 3, 4, 5):
        sub = [t for t in comp if len(t['seq']) == L_ or (L_ == 5 and len(t['seq']) >= 5)]
        P(f'  length {"5+" if L_ == 5 else L_}: n={len(sub)} exact {sum(t["status"] == "exact" for t in sub)} near {sum(t["status"] == "near" for t in sub)} none {sum(t["status"] == "none" for t in sub)} indet {sum(t["status"] == "indet" for t in sub)}')
    P('  by site (complete texts): ' + '; '.join(f'{s}: none {sum(t["status"] == "none" for t in comp if t["site"] == s)}/{sum(t["site"] == s for t in comp)}'
                                             for s in ['Mohenjodaro', 'Harappa', 'Lothal', 'Kalibangan', 'Chanhudaro', 'Other sites', 'West Asian finds']))
    P('  by object type (complete): ' + '; '.join(f'{o}: none {sum(t["status"] == "none" for t in comp if t["otype"] == o)}/{sum(t["otype"] == o for t in comp)}'
                                             for o in ['seal', 'sealing', 'miniature tablet', 'copper tablet', 'pottery graffito', 'ivory/bone rod']))
    # audit_overlap-style figure for comparison (len>=4, site-agnostic exact, fully bridgeable)
    idx = wells_patterns(LEVEL); allpats = collections.defaultdict(list)
    for (sg, L_), v in idx.items(): allpats[L_] += v
    fb = [t for t in comp if len(t['seq']) >= 4 and all(m in MCOV for m in t['seq'])]
    P(f'  audit_overlap-style (len>=4, fully bridgeable, any site, exact): {sum(exact(t["seq"], allpats[len(t["seq"])]) for t in fb)}/{len(fb)} = '
      f'{sum(exact(t["seq"], allpats[len(t["seq"])]) for t in fb) / len(fb):.1%}')
    ex_nw = sum(t['status'] == 'exact' and t['nowild'] for t in comp)
    P(f'  exact matches that use NO wildcard position: {ex_nw} of {sum(t["status"] == "exact" for t in comp)}; '
      f'IM77 texts containing an unbridged M sign: {sum(any(m not in MCOV for m in t["seq"]) for t in comp)}')
    n4, em, emax, nm, nmax = match_control(T, LEVEL, rnd)
    P(f'  matcher false-positive control (texts >= 4 signs shuffled within themselves, n={n4}, 20x): exact mean {em:.1f} (max {emax}) = {em / n4:.1%}; near mean {nm:.1f} (max {nmax}) = {nm / n4:.1%}')
    NEW = [t for t in comp if t['status'] == 'none']
    STRICT = [t for t in comp if t['status'] != 'exact']
    OVER = [t for t in comp if t['status'] == 'exact']
    json.dump({'new': [t['id'] for t in NEW], 'strict': [t['id'] for t in STRICT], 'overlap': [t['id'] for t in OVER]}, open(HERE + TAG + '_sets.json', 'w'))
    describe(NEW, 'IM77-only (conservative: no exact, no near match)'); describe(OVER, 'OVERLAP (exact match in Wells)')
    if CYCLE == '1':
        P('\n### the IM77-only texts (text_no:side, site, object, field symbol, sequence in M numbers; * = contains a doubtful reading)')
        for t in sorted(NEW, key=lambda t: t['id']): P(f'  {t["id"][0]}:{t["id"][1]} {t["site"]:12s} {t["otype"]:10s} {t["fs"]:14s} {"*" if t["doubt"] else " "} {t["seq"]}')
    OVERM = length_matched(OVER, NEW, rnd)
    sets = [('IM77-only', NEW), ('OVERLAP length-matched (3 per new text)', OVERM), ('OVERLAP', OVER)] if CYCLE != '3' else [('IM77-only', NEW), ('IM77-only STRICT (near matches kept)', STRICT)]
    if CYCLE == '1':
        for lab, S in sets:
            test_frame(S, lab, rnd); test_fish(S, lab); test_lot(S, lab, rnd)
    elif CYCLE == '2':
        for lab, S in sets:
            test_names(S, lab, rnd); test_quantity(S, lab, rnd)
            test_nesting(S, S, lab + ' (hosts = same set)', rnd)
        test_nesting(NEW, comp, 'IM77-only shorts, hosts = all complete IM77 texts', rnd)
        test_nesting(OVER, NEW, 'OVERLAP (known) shorts nested in IM77-only long texts as hosts', rnd)
        test_reuse(NEW, OVER, 'IM77-only', rnd)
        test_reuse(OVERM, [t for t in OVER if t not in OVERM], 'OVERLAP length-matched (known set minus itself)', rnd)
        test_reuse(STRICT, OVER, 'IM77-only STRICT', rnd)
    elif CYCLE == '3':
        # robustness: strict set; seals only; big-city vs other; per-site counts
        for lab, S in sets[1:]:
            test_frame(S, lab, rnd, 100); test_fish(S, lab); test_lot(S, lab, rnd, 1000); test_names(S, lab, rnd); test_quantity(S, lab, rnd, 100)
            test_nesting(S, S, lab, rnd, 30)
        for lab, S in (('IM77-only seals', [t for t in NEW if t['otype'] == 'seal']), ('IM77-only non-seals', [t for t in NEW if t['otype'] != 'seal']),
                       ('IM77-only Mohenjodaro', [t for t in NEW if t['site'] == 'Mohenjodaro']), ('IM77-only Harappa', [t for t in NEW if t['site'] == 'Harappa']),
                       ('IM77-only small sites', [t for t in NEW if t['site'] not in ('Mohenjodaro', 'Harappa')])):
            describe(S, lab); test_frame(S, lab, rnd, 100); test_lot(S, lab, rnd, 500); test_nesting(S, S, lab, rnd, 30)

if __name__ == '__main__':
    main()
