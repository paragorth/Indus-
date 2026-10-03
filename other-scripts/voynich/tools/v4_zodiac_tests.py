#!/usr/bin/env python3
"""V4: are the zodiac nymph labels a numbered sequence (days / degrees)?

Same statistics on the Voynich zodiac labels and on controls with the SAME
month and ring sizes:
  numbers  - Latin ordinals, Roman-calendar day names (kalends/nones/ides),
             Italian cardinals, Latin weekday names (period 7);
  names    - saints' names, star names;
  random   - other Voynich label words, char-bigram pseudo-words.
Each control is shown plain, through a verbose cipher (1-2 symbols per letter),
and through a verbose homophonic cipher with 10% glyph noise.

Statistics (all against a within-month shuffle that keeps ring sizes):
  types    distinct/total; xrep = share of labels found verbatim in another month
  align_f  same index in two months -> same feature f (whole, pre2, end2, end1)
  rot_f    same, best cyclic shift / reflection per month pair (start unknown)
  lenprof  mean pairwise Spearman of label length vs index across months
  adj      label i and i+1 share pre2 (stem) within month
  lag L    end2 / pre2 match at lag L within month, L = 1..15 (period 7/10/12/15)
  ringMI   mutual information ring-rank (outer->inner) x end2 / pre2
Usage: python3 v4_zodiac_tests.py [cycle]   (1 = Voynich, 2 = controls, 3 = robustness)
"""
import json, math, os, random, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from vlib import glyphs
from v4_zodiac_extract import extract

ORDER = ['Pisces', 'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius']
NPERM = 300

# ---------------------------------------------------------------- data
def voynich_months(labs=None, unit='month', strict=False, outer_first=False):
    labs = labs or extract()
    key = (lambda l: l['month']) if unit == 'month' else (lambda l: l['page'])
    groups = defaultdict(list)
    for l in labs:
        groups[key(l)].append(l)
    out = []
    for k, ls in groups.items():
        # ring rank: by ring size within page, largest = 0 (outer)
        size = Counter((l['page'], l['ring']) for l in ls)
        pages = sorted({l['page'] for l in ls}, key=lambda p: [x['page'] for x in labs].index(p))
        rank = {}
        for p in pages:
            rs = sorted({r for (pp, r) in size if pp == p}, key=lambda r: -size[(p, r)])
            for i, r in enumerate(rs): rank[(p, r)] = i
        seq = []
        for p in pages:
            pl = [l for l in ls if l['page'] == p]
            if outer_first:
                pl.sort(key=lambda l: (rank[(p, l['ring'])], l['page_idx']))
            for l in pl:
                if strict and l['uncertain']: continue
                seq.append((tuple(glyphs(l['label'])), rank[(p, l['ring'])], l['angle']))
        out.append((k, seq))
    return out

def rings_template(vm):
    return [[r for (_, r, _) in seq] for _, seq in vm]

# ---------------------------------------------------------------- controls
LAT_ORD = ['primus', 'secundus', 'tertius', 'quartus', 'quintus', 'sextus', 'septimus', 'octavus', 'nonus',
           'decimus', 'undecimus', 'duodecimus', 'tertius decimus', 'quartus decimus', 'quintus decimus',
           'sextus decimus', 'septimus decimus', 'duodevicesimus', 'undevicesimus', 'vicesimus',
           'vicesimus primus', 'vicesimus secundus', 'vicesimus tertius', 'vicesimus quartus',
           'vicesimus quintus', 'vicesimus sextus', 'vicesimus septimus', 'vicesimus octavus',
           'vicesimus nonus', 'tricesimus', 'tricesimus primus']
LAT_ABL = ['', '', '', 'tertio', 'quarto', 'quinto', 'sexto', 'septimo', 'octavo', 'nono', 'decimo', 'undecimo',
           'duodecimo', 'tertio decimo', 'quarto decimo', 'quinto decimo', 'sexto decimo', 'septimo decimo',
           'duodevicesimo', 'undevicesimo']
ITA = ['uno', 'due', 'tre', 'quattro', 'cinque', 'sei', 'sette', 'otto', 'nove', 'dieci', 'undici', 'dodici',
       'tredici', 'quattordici', 'quindici', 'sedici', 'diciassette', 'diciotto', 'diciannove', 'venti',
       'ventuno', 'ventidue', 'ventitre', 'ventiquattro', 'venticinque', 'ventisei', 'ventisette', 'ventotto',
       'ventinove', 'trenta', 'trentuno']
FERIA = ['dominica', 'feria secunda', 'feria tertia', 'feria quarta', 'feria quinta', 'feria sexta', 'sabbato']
SAINTS = """stephanus iohannes thomas silvester hilarius felix marcellus antonius prisca fabianus sebastianus agnes
vincentius timotheus paulus polycarpus agatha dorothea scholastica valentinus iuliana petrus mathias
albinus perpetua felicitas gregorius patricius cuthbertus benedictus gabriel ambrosius leo tiburtius
georgius marcus vitalis philippus iacobus athanasius crux gordianus pancratius dunstanus urbanus
augustinus germanus beda bonifatius medardus barnabas basilius vitus albanus alban iohannes paulus
leo processus martinianus benedictus swithunus kenelmus arnulphus margareta praxedis magdalena
christina anna samson pantaleon martha abdon germanus oswaldus laurentius hippolytus eusebius
bartholomaeus rufus augustinus sabina felix aegidius cuthburga nativitas gorgonius protus hyacinthus
cornelius cyprianus lambertus matthaeus mauritius cosmas damianus michael hieronymus remigius
leodegarius fides franciscus dionysius gereon wilfridus calixtus lucas undecim romanus crispinus
simon iudas quintinus omnes leonardus quattuor martinus britius machutus edmundus hilda elisabeth
caecilia clemens catharina linus saturninus andreas eligius barbara nicolaus conceptio lucia
thomas ignatius innocentes eadmundus odilia otmarus gallus florianus erasmus kilianus ulricus
afra emmeramus wenceslaus vedastus amandus blasius valerius eulalia quiriacus""".split()
STARS = """aldebaran algol alcor alfard alferaz alioth alkaid almak alnath alphecca altair antares arcturus
bellatrix betelgeuze capella caphir deneb denebola dubhe enif fomalhaut hamal izar kochab markab
menkalinan menkar merak mirach mirfak mizar nunki pollux procyon rasalhague regulus rigel sadalmelik
scheat schedar shaula sirius spica suhail thuban unukalhai vega zubeneschamali zubenelgenubi alhena
achernar alnilam alnitak mintaka saiph sadr albireo alderamin algenib alpheratz ankaa ruchbah""".split()
MONTHLEN = {'Pisces': 28, 'Aries': 31, 'Taurus': 30, 'Gemini': 31, 'Cancer': 30, 'Leo': 31, 'Virgo': 31,
            'Libra': 30, 'Scorpio': 31, 'Sagittarius': 30}   # Feb..Nov
NONES7 = {'Aries', 'Gemini', 'Leo', 'Scorpio'}                   # Mar, May, Jul, Oct

def roman_day(month, d):
    L = MONTHLEN[month]; non = 7 if month in NONES7 else 5; ide = non + 8
    d = (d - 1) % L + 1
    if d == 1: return 'kalendis'
    if d < non: return 'pridie nonas' if non - d == 1 else LAT_ABL[non - d + 1] + ' nonas'
    if d == non: return 'nonis'
    if d < ide: return 'pridie idus' if ide - d == 1 else LAT_ABL[ide - d + 1] + ' idus'
    if d == ide: return 'idibus'
    n = L - d + 2
    return 'pridie kalendas' if n == 2 else LAT_ABL[n] + ' kalendas'

def gen_words(kind, months, rng, pool=None):
    out = []; phase = 0
    for name, n in months:
        if kind == 'latord': ws = [LAT_ORD[i % 31] for i in range(n)]
        elif kind == 'roman': ws = [roman_day(name, i + 1) for i in range(n)]
        elif kind == 'italian': ws = [ITA[i % 31] for i in range(n)]
        elif kind == 'feria':
            ws = [FERIA[(phase + i) % 7] for i in range(n)]; phase += MONTHLEN.get(name, 30)
        elif kind in ('saints', 'stars'):
            src = SAINTS if kind == 'saints' else STARS
            ws = rng.sample(src, n) if n <= len(src) else [rng.choice(src) for _ in range(n)]
        elif kind == 'vlabels': ws = [rng.choice(pool) for _ in range(n)]
        elif kind == 'markov': ws = [pool(rng) for _ in range(n)]
        out.append([w.replace(' ', '') if isinstance(w, str) else w for w in ws])
    return out

SYM = list('oaeydklrchstpinmfgqABCDEFGHIJ')
def make_cipher(rng, homo=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'; used = set(); book = {}
    for c in letters:
        codes = []
        while len(codes) < (2 if homo else 1):
            k = 2 if rng.random() < 0.5 else 1
            code = ''.join(rng.choice(SYM[:20]) for _ in range(k))
            if code not in used: used.add(code); codes.append(code)
        book[c] = codes
    def enc(w, r):
        s = ''.join(r.choice(book[c]) for c in w)
        if homo:
            s = ''.join(r.choice(SYM[:20]) if r.random() < 0.10 else ch for ch in s)
        return s
    return enc

def markov_model(words, rng0):
    big = defaultdict(Counter)
    for w in words:
        g = ['^'] + list(w) + ['$']
        for a, b in zip(g, g[1:]): big[a][b] += 1
    def draw(rng):
        s, a = [], '^'
        while len(s) < 15:
            ks, vs = zip(*big[a].items()); a = rng.choices(ks, vs)[0]
            if a == '$': break
            s.append(a)
        return tuple(s) if s else ('o',)
    return draw

def control_months(kind, vm, seed, cipher=None):
    rng = random.Random(seed)
    months = [(k, len(seq)) for k, seq in vm]
    pool = None
    if kind == 'vlabels':
        pool = [tuple(glyphs(w)) for w in other_label_words()]
    if kind == 'markov':
        pool = markov_model([lab for _, seq in vm for (lab, _, _) in seq], rng)
    words = gen_words(kind, months, rng, pool)
    enc = make_cipher(random.Random(seed + 7), homo=(cipher == 'homo')) if cipher else None
    er = random.Random(seed + 11)
    out = []
    for (k, seq), ws in zip(vm, words):
        s2 = []
        for (w, (_, r, a)) in zip(ws, seq):
            if isinstance(w, str):
                w = tuple(enc(w, er)) if enc else tuple(w)
            s2.append((w, r, a))
        out.append((k, s2))
    return out

def other_label_words():
    recs = json.load(open(os.path.join(ROOT, 'data', 'derived', 'ZL3b_lines.json')))
    return [w for r in recs if r['ltype'] == 'L' and r['illus'] != 'Z' for w in r['words'] if '?' not in w]

# ---------------------------------------------------------------- statistics
FEAT = {'whole': lambda w: w, 'pre2': lambda w: w[:2], 'end2': lambda w: w[-2:], 'end1': lambda w: w[-1:],
        'pre1': lambda w: w[:1]}

def mi(pairs):
    n = len(pairs); cx = Counter(a for a, _ in pairs); cy = Counter(b for _, b in pairs); cxy = Counter(pairs)
    return sum(c / n * math.log2(c * n / (cx[a] * cy[b])) for (a, b), c in cxy.items())

def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            for t in range(i, j + 1): r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rk(x), rk(y); n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((u - ma) ** 2 for u in a)); sb = math.sqrt(sum((u - mb) ** 2 for u in b))
    return 0 if sa == 0 or sb == 0 else sum((u - ma) * (v - mb) for u, v in zip(a, b)) / (sa * sb)

def stats(months, rot=True):
    seqs = [[w for (w, _, _) in s] for _, s in months]
    rings = [[r for (_, r, _) in s] for _, s in months]
    S = {}
    allw = [w for s in seqs for w in s]
    S['types'] = len(set(allw)) / len(allw)
    where = defaultdict(set)
    for i, s in enumerate(seqs):
        for w in s: where[w].add(i)
    S['xrep'] = sum(len(where[w]) > 1 for w in allw) / len(allw)
    for f in ('whole', 'pre2', 'end2', 'end1'):
        F = [[FEAT[f](w) for w in s] for s in seqs]; m = t = 0; rbest = []
        for a in range(len(F)):
            for b in range(a + 1, len(F)):
                L = min(len(F[a]), len(F[b]))
                m += sum(F[a][k] == F[b][k] for k in range(L)); t += L
                if rot:
                    best = 0
                    for B in (F[b], F[b][::-1]):
                        for sh in range(len(B)):
                            BB = B[sh:] + B[:sh]
                            best = max(best, sum(F[a][k] == BB[k] for k in range(L)) / L)
                    rbest.append(best)
        S['align_' + f] = m / t
        if rot: S['rot_' + f] = sum(rbest) / len(rbest)
    lens = [[len(w) for w in s] for s in seqs]; cs = []
    for a in range(len(lens)):
        for b in range(a + 1, len(lens)):
            L = min(len(lens[a]), len(lens[b])); cs.append(spearman(lens[a][:L], lens[b][:L]))
    S['lenprof'] = sum(cs) / len(cs)
    S['adj'] = sum(FEAT['pre2'](s[i]) == FEAT['pre2'](s[i + 1]) for s in seqs for i in range(len(s) - 1)) / \
        sum(len(s) - 1 for s in seqs)
    for f in ('end2', 'pre2'):
        for L in range(1, 16):
            num = sum(FEAT[f](s[i]) == FEAT[f](s[i + L]) for s in seqs for i in range(len(s) - L))
            den = sum(max(0, len(s) - L) for s in seqs)
            S['lag%s_%d' % (f, L)] = num / den
    for f in ('end2', 'pre2'):
        S['ringMI_' + f] = mi([(r, FEAT[f](w)) for s, rr in zip(seqs, rings) for w, r in zip(s, rr)])
    return S

def shuffle_months(months, rng):
    out = []
    for k, s in months:
        ws = [w for (w, _, _) in s]; rng.shuffle(ws)
        out.append((k, [(w, r, a) for w, (_, r, a) in zip(ws, s)]))
    return out

def evaluate(months, nperm=NPERM, seed=0, rot=True):
    obs = stats(months, rot); rng = random.Random(seed); null = defaultdict(list)
    for _ in range(nperm):
        for k, v in stats(shuffle_months(months, rng), rot).items(): null[k].append(v)
    res = {}
    for k, v in obs.items():
        nv = null[k]; mu = sum(nv) / len(nv); sd = math.sqrt(sum((x - mu) ** 2 for x in nv) / len(nv)) or 1e-9
        p = (1 + sum(x >= v for x in nv)) / (1 + len(nv))
        res[k] = (v, mu, (v - mu) / sd, p)
    return res

KEYS = ['types', 'xrep', 'align_whole', 'align_pre2', 'align_end2', 'rot_whole', 'rot_pre2', 'rot_end2',
        'lenprof', 'adj', 'ringMI_end2', 'ringMI_pre2']
def summary(name, res):
    def z(k): return res[k][2]
    lagz = {f: {L: z('lag%s_%d' % (f, L)) for L in range(2, 16)} for f in ('end2', 'pre2')}
    best = {f: max(lagz[f], key=lagz[f].get) for f in lagz}
    s = '%-26s types %.2f xrep %.2f | align whole z%+.1f pre2 z%+.1f end2 z%+.1f | rot whole z%+.1f pre2 z%+.1f end2 z%+.1f' % (
        name, res['types'][0], res['xrep'][0], z('align_whole'), z('align_pre2'), z('align_end2'),
        z('rot_whole'), z('rot_pre2'), z('rot_end2'))
    s += ' | lenprof %.2f (z%+.1f) | adj z%+.1f | lag z end2[7,10,12,15]=%s best L%d z%+.1f; pre2 best L%d z%+.1f | ringMI end2 z%+.1f pre2 z%+.1f' % (
        res['lenprof'][0], z('lenprof'), z('adj'),
        ','.join('%+.1f' % lagz['end2'][L] for L in (7, 10, 12, 15)), best['end2'], lagz['end2'][best['end2']],
        best['pre2'], lagz['pre2'][best['pre2']], z('ringMI_end2'), z('ringMI_pre2'))
    return s

if __name__ == '__main__':
    cyc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    vm = voynich_months()
    vm = sorted(vm, key=lambda x: ORDER.index(x[0]))
    out = {}
    if cyc == 1:
        r = evaluate(vm); out['voynich'] = r; print(summary('VOYNICH zodiac', r))
        for L in range(1, 16):
            print('lag', L, 'end2 %.3f null %.3f z%+.1f' % r['lagend2_%d' % L][:3], ' pre2 %.3f null %.3f z%+.1f' % r['lagpre2_%d' % L][:3])
        for k in KEYS: print(k, 'obs %.3f null %.3f z %+.2f p %.3f' % r[k])
    elif cyc == 2:
        for kind in ('latord', 'roman', 'italian', 'feria', 'saints', 'stars', 'vlabels', 'markov'):
            for ci in ((None, 'verbose', 'homo') if kind not in ('vlabels', 'markov') else (None,)):
                cm = control_months(kind, vm, seed=5, cipher=ci)
                r = evaluate(cm, nperm=150); out['%s/%s' % (kind, ci)] = r
                print(summary('%s/%s' % (kind, ci or 'plain'), r), flush=True)
    elif cyc == 3:
        variants = {
            'pages-as-units': sorted(voynich_months(unit='page'), key=lambda x: x[0]),
            'strict(no ? labels)': sorted(voynich_months(strict=True), key=lambda x: ORDER.index(x[0])),
            'outer-ring-first': sorted(voynich_months(outer_first=True), key=lambda x: ORDER.index(x[0])),
            'full months only': [m for m in vm if m[0] not in ('Aries', 'Taurus')],
        }
        for name, m in variants.items():
            r = evaluate(m, nperm=200); out[name] = r; print(summary(name, r), flush=True)
        # angle alignment: clock bin (12 bins) -> same end2/pre2 across months (outer ring only)
        rng = random.Random(3)
        def angstat(mm, f):
            bins = defaultdict(list)
            for k, s in mm:
                for w, r, a in s:
                    if a is not None: bins[(int(a // 30), r == 0)].append((k, FEAT[f](w)))
            m = t = 0
            for b, xs in bins.items():
                for i in range(len(xs)):
                    for j in range(i + 1, len(xs)):
                        if xs[i][0] != xs[j][0]: t += 1; m += xs[i][1] == xs[j][1]
            return m / t
        for f in ('end2', 'pre2', 'whole'):
            o = angstat(vm, f); nl = [angstat(shuffle_months(vm, rng), f) for _ in range(300)]
            mu = sum(nl) / len(nl); sd = math.sqrt(sum((x - mu) ** 2 for x in nl) / len(nl))
            line = 'clock-bin x ring alignment %s: obs %.3f null %.3f z %+.2f p %.3f' % (
                f, o, mu, (o - mu) / sd, (1 + sum(x >= o for x in nl)) / 301)
            print(line); out['angle_' + f] = line
    json.dump({k: v for k, v in out.items()}, open(os.path.join(ROOT, 'data', 'results', 'v4_zodiac_c%d.json' % cyc), 'w'), indent=1, default=str)
