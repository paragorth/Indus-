"""S-DARK-49 cycle 1: is any Indus field a CALENDAR / CYCLIC slot rather than a name or title slot?
A month slot has ~12 elements of near-equal frequency, a day slot ~30, a year-name slot a few dozen; names and titles are Zipfian.
Fields tested: S310/S331 parser slots (OPENER, MARKER, NAME, COUNT, TITLE = qualifier before the closer, CLOSER head, SUFFIX),
positions counted from the closer (C-1..C-3) and from the start (P0..P2), the S-DARK-19 partial-order classes, and every state of the
S-DARK-33 ALERGIA automaton (outgoing symbol distribution = the field at that state).
Metrics per field: K types, N tokens, Zipf slope (OLS log f ~ log rank), Gini, Simpson effective number D=1/sum p^2 and evenness D/K,
Shannon evenness H/log K, top share, exclusivity (texts with >= 2 tokens of the field).
Null: slot labels permuted over all tokens of the corpus (NPERM x) = frequency-matched random partition of the same signs; P = share of null
fields at least as even (Simpson evenness).  Calendar band (fixed before running): 8 <= K <= 40, D/K >= 0.6, slope > -0.5.
Controls: Ur III seal legends (ur3_words: word positions 1..3, last), Ur III administrative month (iti) and year (mu) lines from the CDLI
ATF dump when present, Linear B (linb_words: word before me-no = month names; ideogram slot; first word), Proto-Elamite (first sign,
numeral-type signs), and a planted uniform 12-element month slot added to every Indus text.
Usage: python3 tools/dark_loop49_c1.py <seq_raw|seq_strong|seq_all> [nperm]
"""
import sys, json, csv, re, os, math, collections, random
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import OPEN, MARK, MJAR, SUF, CL, FISH, NUM, learn_qual, make_parser, pval
ROOT = '/home/user/Indus-/'
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rnd = random.Random(49)
OUT = open(ROOT + f'data/derived/dark/loop49_c1_{LV}.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()

# ------------------------------------------------------------- metrics
def metrics(cnt):
    """cnt: Counter element -> tokens"""
    f = sorted(cnt.values(), reverse=True); N = sum(f); K = len(f)
    if K == 0 or N == 0: return None
    p = [x / N for x in f]
    D = 1 / sum(x * x for x in p); H = -sum(x * math.log(x) for x in p if x > 0)
    # Zipf slope: OLS of log f on log rank
    if K >= 3:
        xs = [math.log(i + 1) for i in range(K)]; ys = [math.log(x) for x in f]
        mx = sum(xs) / K; my = sum(ys) / K
        sxx = sum((x - mx) ** 2 for x in xs); slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else float('nan')
    else: slope = float('nan')
    # Gini
    fs = sorted(f); cum = 0; g = 0
    for i, x in enumerate(fs): g += (2 * (i + 1) - K - 1) * x
    gini = g / (K * N) if K > 1 else 0.0
    return dict(K=K, N=N, D=D, ED=D / K, EH=(H / math.log(K)) if K > 1 else 1.0, slope=slope, gini=gini, top=p[0],
                single=sum(1 for x in f if x == 1) / K)
def band(m):
    if m is None: return '-'
    if 8 <= m['K'] <= 40 and m['ED'] >= 0.6 and (math.isnan(m['slope']) or m['slope'] > -0.5): return 'CALENDAR-band'
    if m['ED'] < 0.3 or (not math.isnan(m['slope']) and m['slope'] <= -0.7): return 'name-band'
    return 'between'
def fmt(m, P=None, nullED=None):
    if m is None: return 'empty'
    s = (f"K={m['K']:4d} N={m['N']:5d} D={m['D']:6.2f} D/K={m['ED']:.2f} H/logK={m['EH']:.2f} slope={m['slope']:+.2f} "
         f"gini={m['gini']:.2f} top={m['top']:.2f} singletons={m['single']:.2f} [{band(m)}]")
    if P is not None: s += f"  null D/K {nullED[0]:.2f} [{nullED[1]:.2f}-{nullED[2]:.2f}] P_even={P:.3f}"
    return s

# ------------------------------------------------------------- Indus texts
def otype(t):
    t0 = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t0, 'other')
C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
T = []
for r in C:
    s = r[LV]
    if not s or len(s) < 2 or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
    T.append(dict(id=r['cisi'], site=r['site'], ot=otype(r['type']), seq=list(s)))
parse = make_parser(learn_qual([t['seq'] for t in T]))
say(f'# S-DARK-49 cycle 1, level {LV}, nperm {NPERM}: calendar / cyclic slot hunt. Wells complete direction-recorded texts >= 2 signs: {len(T)}')

# S-DARK-19 partial-order classes (longest chain groups), as fixed sets of signs
PO = collections.OrderedDict([
    ('PO1 marked jar / leaf-tree first', {741, 742, 745, 803, 806}),
    ('PO2 tall numerals', {31, 32, 33, 34}),
    ('PO3 350-798-415', {350, 798, 415}),
    ('PO4 fish words', set(FISH)),
    ('PO5 705-33 / 255-435-690 titles', {705, 706, 255, 435, 690}),
    ('PO6 closer paradigm', set(CL)),
    ('PO0 openers', set(OPEN)),
    ('numeral values (all stroke signs)', set(NUM)),
])

def fields_from_texts(texts, labels):
    """returns dict fieldname -> Counter(element), and exclusivity dict fieldname -> share of texts with >= 2 tokens"""
    F = collections.defaultdict(collections.Counter); multi = collections.Counter(); ntext = collections.Counter()
    for s, lab in zip(texts, labels):
        per = collections.Counter()
        for i, (w, l) in enumerate(zip(s, lab)):
            F['slot ' + l][w] += 1; per['slot ' + l] += 1
            if l == 'OPENER': F['pos P0 opener sign'][w] += 1
        # positions from the closer
        cl = [i for i, l in enumerate(lab) if l == 'CLOSER']
        if cl:
            ci = cl[0]
            for k in (1, 2, 3):
                if ci - k >= 0 and lab[ci - k] in ('NAME', 'COUNT', 'TITLE'):
                    F[f'pos C-{k} (sign {k} before closer)'][s[ci - k]] += 1; per[f'pos C-{k} (sign {k} before closer)'] += 1
        # positions from the start, after opener+marker
        i0 = 0
        while i0 < len(s) and lab[i0] in ('OPENER', 'MARKER'): i0 += 1
        for k in (0, 1, 2):
            if i0 + k < len(s) and lab[i0 + k] in ('NAME', 'COUNT', 'TITLE'):
                F[f'pos P{k+1} (sign {k+1} of the middle)'][s[i0 + k]] += 1; per[f'pos P{k+1} (sign {k+1} of the middle)'] += 1
        for nm, S in PO.items():
            hits = [w for w in s if w in S]
            for w in hits: F['class ' + nm][w] += 1
            per['class ' + nm] += len(hits)
        for k, v in per.items():
            ntext[k] += 1
            if v >= 2: multi[k] += 1
    excl = {k: (multi[k], ntext[k]) for k in ntext}
    return F, excl

texts = [t['seq'] for t in T]; labels = [parse(s) for s in texts]
F, excl = fields_from_texts(texts, labels)

# null: permute slot labels over all tokens (frequency-matched random partition of the same signs)
alltok = [w for s in texts for w in s]; alllab = [l for lab in labels for l in lab]
nullED = collections.defaultdict(list)
for it in range(NPERM):
    rnd.shuffle(alllab)
    cnt = collections.defaultdict(collections.Counter)
    for w, l in zip(alltok, alllab): cnt[l][w] += 1
    for l, c in cnt.items():
        m = metrics(c); nullED['slot ' + l].append(m['ED'])
# null for class / position fields: random sign sets of the same K drawn with probability ~ frequency, evaluated on their token counts
tokfreq = collections.Counter(alltok); types = list(tokfreq); wts = [tokfreq[w] for w in types]
def null_class(K, N):
    """K sign types drawn with probability ~ token frequency; the field = their corpus frequencies"""
    out = []
    for it in range(NPERM):
        pick = set()
        while len(pick) < K: pick.update(rnd.choices(types, wts, k=K))
        pick = list(pick)[:K]
        c = collections.Counter({w: tokfreq[w] for w in pick}); out.append(metrics(c)['ED'])
    return out
def null_sample(N):
    """N tokens drawn at random from all tokens of the corpus (a frequency-matched random field of the same size)"""
    out = []
    for it in range(NPERM):
        c = collections.Counter(rnd.sample(alltok, N)); out.append(metrics(c)['ED'])
    return out

say('\n== Indus fields (Wells). P_even = share of null fields at least as even (Simpson D/K); null for slots = labels permuted over tokens, for classes = K signs drawn by frequency, for positions = N tokens sampled at random')
summary = {}
for name in sorted(F, key=lambda k: (k.split()[0], k)):
    m = metrics(F[name])
    if name in nullED and nullED[name]: nl = nullED[name]
    elif name.startswith('class'): nl = null_class(m['K'], m['N'])
    else: nl = null_sample(m['N'])
    nl_s = sorted(nl); P = (sum(1 for v in nl if v >= m['ED']) + 1) / (len(nl) + 1)
    say(f"{name:44s} {fmt(m, P, (sum(nl)/len(nl), nl_s[int(0.025*len(nl))], nl_s[int(0.975*len(nl))-1]))}  excl: >=2 tokens in {excl.get(name,(0,0))[0]}/{excl.get(name,(0,0))[1]} texts")
    summary[name] = dict(m, P=P, band=band(m))
    top = ', '.join(f'{w}:{n}' for w, n in F[name].most_common(14))
    say(f"{'':44s}   top: {top}")

# ------------------------------------------------------------- automaton states (S-DARK-33)
say('\n== S-DARK-33 automaton states as fields (outgoing symbol distribution at each state; END counted as an element; alphabet is CLASS-level: OTHER lumps rare signs)')
for kind in ('seals', 'all+type'):
    fn = ROOT + f'data/derived/dark/loop33_automaton_{kind}_{LV}.json'
    if not os.path.exists(fn): say('  missing', fn); continue
    A = json.load(open(fn))
    say(f'-- {kind}: {A["states"]} states')
    for st in range(A['states']):
        tr = A['trans'][st]; c = collections.Counter({sym: v[0] for sym, v in tr.items()})
        if A['final'][st]: c['END'] = A['final'][st]
        m = metrics(c)
        if m is None or m['N'] < 20: continue
        c2 = collections.Counter({k: v for k, v in c.items() if k not in ('END', 'OTHER')}); m2 = metrics(c2)
        say(f"  state {st:2d} n={A['n'][st]:5d}: {fmt(m)} | without END/OTHER: {fmt(m2) if m2 else 'empty'}")
        say(f"           top: {', '.join(f'{k}:{v}' for k, v in c.most_common(10))}")

# ------------------------------------------------------------- controls
say('\n== Controls (same metrics)')
def jl(fn): return [json.loads(l)['seq'] for l in open(ROOT + 'data/derived/dark/loop32_corpora/' + fn)]
ur3 = jl('ur3_words.jsonl')
for nm, c in [('Ur III legend word 1 (owner name)', collections.Counter(s[0] for s in ur3 if s)),
              ('Ur III legend word 2 (title / dumu)', collections.Counter(s[1] for s in ur3 if len(s) > 1)),
              ('Ur III legend word 3', collections.Counter(s[2] for s in ur3 if len(s) > 2)),
              ('Ur III legend last word', collections.Counter(s[-1] for s in ur3 if len(s) > 1)),
              ('Ur III legend all title-like words (dumu/dub-sar/arad/sanga/...)', collections.Counter(w for s in ur3 for w in s if re.fullmatch(r'_?(dumu|dub-sar|arad2?|sanga|nu-banda3|szabra|ensi2|lugal|sagi|gudu4|ugula|kuruszda|szagina|nar|simug|ma2-lah5|aszgab|nagar|azlag2|ensi2|lu2|nin|munus)_?', w)))]:
    say(f'{nm:60s} {fmt(metrics(c))}'); say(f"{'':60s}   top: {', '.join(f'{w}:{n}' for w, n in c.most_common(12))}")
# Ur III administrative calendar fields from the CDLI ATF dump (if downloaded): month lines 'iti X', year lines 'mu ...'
atf = os.environ.get('CDLI_ATF', ROOT + 'data/cache/cdli.atf')  # CDLI ATF dump (github cdli-gh/data, cdliatf_unblocked.atf via the LFS media URL, 87 MB; not committed)
if os.path.exists(atf) and os.path.getsize(atf) > 10_000_000:
    iti = collections.Counter(); mu = collections.Counter(); mu2 = collections.Counter(); ntexts = 0; cur_lang = ''
    for line in open(atf, errors='ignore'):
        if line.startswith('&P'): ntexts += 1
        s = line.strip()
        if '. ' not in s: continue
        body = s.split('. ', 1)[1].strip()
        if body.startswith('iti '):
            w = body.split()[1:3]
            iti[' '.join(w).strip('#?![]')] += 1
        elif body.startswith('mu ') and not body.startswith('mu-'):
            w = body.split()
            mu[' '.join(w[:3])] += 1; mu2[' '.join(w[:5])] += 1
    say(f'CDLI ATF: {ntexts} texts scanned')
    for nm, c in [('CDLI month lines (iti + 1-2 words), all periods', iti), ('CDLI year lines (mu + 2 words)', mu), ('CDLI year lines (mu + 4 words)', mu2)]:
        c2 = collections.Counter({k: v for k, v in c.items() if v >= 2})
        say(f'{nm:60s} {fmt(metrics(c))} | tokens>=2 only: {fmt(metrics(c2))}'); say(f"{'':60s}   top: {', '.join(f'{w}:{n}' for w, n in c.most_common(14))}")
        if nm.startswith('CDLI month'):
            top12 = collections.Counter(dict(c.most_common(12))); say(f"{'':60s}   top-12 month names only: {fmt(metrics(top12))}")
else:
    say('CDLI ATF dump not present yet (download running); Ur III month / year fields deferred to a later cycle')
# Linear B
lb = jl('linb_words.jsonl')
mon = collections.Counter(s[i - 1] for s in lb for i in range(1, len(s)) if s[i] == 'me-no')
ideo = collections.Counter(w for s in lb for w in s if re.fullmatch(r'[A-Z][A-Z0-9*+]*', w) and w not in ('NUM',))
first = collections.Counter(s[0] for s in lb if s and not re.fullmatch(r'[A-Z][A-Z0-9*+]*', s[0]))
for nm, c in [('Linear B month names (word before me-no)', mon), ('Linear B ideogram slot', ideo), ('Linear B first word (names / places)', first)]:
    say(f'{nm:60s} {fmt(metrics(c))}'); say(f"{'':60s}   top: {', '.join(f'{w}:{n}' for w, n in c.most_common(12))}")
# Proto-Elamite
pe = jl('proto_elamite.jsonl')
pe_first = collections.Counter(s[0] for s in pe if s and not s[0].startswith('N'))
pe_num = collections.Counter(w for s in pe for w in s if re.fullmatch(r'N\d+[A-Z]?', w))
for nm, c in [('Proto-Elamite entry first sign (owner / object)', pe_first), ('Proto-Elamite numeral signs', pe_num)]:
    say(f'{nm:60s} {fmt(metrics(c))}'); say(f"{'':60s}   top: {', '.join(f'{w}:{n}' for w, n in c.most_common(12))}")
# planted: a uniform 12-element month slot and a 30-element day slot added to every Indus text; and a 'year-name' slot with 60 elements, geometric 0.95
for nm, K, dist in [('PLANTED month slot (12, uniform)', 12, None), ('PLANTED day slot (30, uniform)', 30, None), ('PLANTED year-name slot (60, geometric 0.95)', 60, 0.95)]:
    if dist is None: draw = [rnd.randrange(K) for _ in texts]
    else:
        w = [dist ** i for i in range(K)]; draw = rnd.choices(range(K), w, k=len(texts))
    c = collections.Counter(draw); say(f'{nm:60s} {fmt(metrics(c))}')
# Indus whole-sign Zipf for reference, and the head inventory (S-DARK-45 style)
say(f"{'Indus all sign tokens (reference)':60s} {fmt(metrics(tokfreq))}")
heads = collections.Counter()
for s, lab in zip(texts, labels):
    cl = [i for i, l in enumerate(lab) if l == 'CLOSER']; heads['C%d' % s[cl[0]] if cl else 'none'] += 1
say(f"{'Indus head (closer) inventory incl. none':60s} {fmt(metrics(heads))}"); say(f"{'':60s}   {dict(heads.most_common())}")

json.dump(summary, open(ROOT + f'data/derived/dark/loop49_c1_{LV}.json', 'w'), indent=1)
cal = [k for k, v in summary.items() if v['band'] == 'CALENDAR-band']
say(f'\nFIELDS IN THE CALENDAR BAND (8<=K<=40, D/K>=0.6, slope>-0.5): {cal if cal else "none"}')
