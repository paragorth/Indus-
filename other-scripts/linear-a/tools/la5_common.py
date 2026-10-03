#!/usr/bin/env python3
"""LA-5 shared code: code-vs-language battery on administrative 'words'.

Arrow in the dark: assume Linear A administrative sign-groups are CODES (account heads, product codes, personal marks),
not words of speech. Run one battery, identically coded, on:
  LA      Linear A administrative words (corpus.json; tablet, nodule, roundel, sealing, lames, bars, label)
  LB      Linear B administrative words (DAMOS; same tablet structure: sign-group + ideogram + number) = positive language control
  LBpers  Linear B personnel names (S-DARK-56 corpus, one per distinct name) = names-of-a-language control
  PE      Proto-Elamite entry middles (loop56 corpus; numerals removed) = undeciphered accounting reference
  HTS, ICD  designed codes (tariff numbers in 2-digit groups; ICD-10-CM characters) = positive code controls
  planted codes built on the LA and LB vocabularies (random IDs and slot codes) = code controls with the same alphabet.
Methods are the Indus loop 41 / 56 / 73 statistics (shuffle and Markov nulls, closed edge sets, Heaps, uniqueness),
re-implemented here so that every population goes through the same lines of code.
"""
import json, os, re, math, random, collections, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
LAD = os.path.join(HERE, '..', 'data')
ROOT = '/home/user/Indus-/'
C56 = ROOT + 'data/derived/dark/loop56_corpora/'
C48 = ROOT + 'data/derived/dark/loop48_corpora/'
ADMIN = {'Tablet', 'Nodule', 'Roundel', 'Sealing', 'Lames (short thin tablet)', '3-sided bar', '4-sided bar', 'Label'}

def norm_la(s):
    return s.replace('₂', '2').replace('₃', '3').upper()

def strip_diac(w):
    return ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')

# ---------------------------------------------------------------- Linear A
def la_docs(admin_only=True):
    """-> list of docs: dict(id, site, support, lines=[[tok,...],...]); tok = ('W', signs) | ('F', label) | ('B',) break."""
    C = json.load(open(os.path.join(LAD, 'corpus.json')))
    out = []
    for c in C:
        if admin_only and c['support'] not in ADMIN: continue
        lines = [[]]
        for t in c['tokens']:
            if t['t'] == 'nl': lines.append([]); continue
            if t['t'] == 'word': lines[-1].append(('W', tuple(norm_la(x) for x in t['s'])))
            elif t['t'] == 'logo': lines[-1].append(('F', 'L:' + t['v'].split('+')[0]))
            elif t['t'] == 'num': lines[-1].append(('F', 'NUM'))
            elif t['t'] == 'div': lines[-1].append(('D',))
            else: lines[-1].append(('B',))
        out.append(dict(id=c['id'], site=c['site'], support=c['support'], lines=[l for l in lines if l]))
    return out

# ---------------------------------------------------------------- Linear B (DAMOS)
WORD = re.compile(r"^[a-z0-9*]+(-[a-z0-9*]+)*$")
EDIT = {'mut.', 'inf.', 'sup.', 'vac.', 'vest.', 'lat.', 'deest', 'v.', 'v.↓', 'v.→', 'α', 'β', 'γ', '⟦', '⟧', 'sup', 'inf', 'mut'}
def lb_docs():
    out = []
    for line in open(os.path.join(LAD, 'damos_items.jsonl')):
        d = json.loads(line); h = d.get('heading') or ''
        site = h[:2] if h[:2].isalpha() else '??'
        lines = []
        for raw in (d.get('content') or '').split('\n'):
            L = []
            for t in raw.split():
                t0 = strip_diac(t).replace("'", '').replace('"', '')
                if re.match(r'^\.[0-9a-zA-Z]+$', t0) or t0 in EDIT: continue
                if t0 in (',', '/', '//', '|'): L.append(('D',)); continue
                dam = '[' in t0 or ']' in t0 or '⟦' in t0 or '⟧' in t0
                core = t0.strip('[]⟦⟧')
                if not core or core.endswith('.') or core in EDIT: L.append(('B',)); continue
                if WORD.match(core) and any(ch.isalpha() for ch in core) and core[0] != '-' and core[-1] != '-':
                    if dam: L.append(('B',)); continue
                    sg = tuple(x.upper() for x in core.split('-'))
                    L.append(('W', sg))
                elif core[:1].isdigit():
                    L.append(('F', 'NUM'))
                elif core[:1].isupper() or core[:1] == '*':
                    L.append(('F', 'L:' + core.split(':')[0].split('+')[0].split(';')[0]))
                    if dam: L.append(('B',))
                else:
                    L.append(('B',))
            if L: lines.append(L)
        out.append(dict(id=h, site=site, series=h.split(' ')[1][:2] if ' ' in h else '', lines=lines))
    return out

def words_of(docs, minlen=2):
    """word tokens with doc index and site."""
    return [(i, d['site'], tok[1]) for i, d in enumerate(docs) for L in d['lines'] for tok in L if tok[0] == 'W' and len(tok[1]) >= minlen]

# ---------------------------------------------------------------- comparators (types)
def jl(path): return [tuple(json.loads(l)['seq']) for l in open(path)]
def comparators():
    out = {}
    out['LBpers'] = [tuple(x.upper() for x in s if x) for s in jl(C56 + 'linb_personnel_dedup.jsonl')]
    out['PE'] = jl(C56 + 'proto_elamite_mid.jsonl')
    out['HTS'] = jl(C56 + 'hts.jsonl')
    out['ICD'] = jl(C56 + 'icd10.jsonl')
    return {k: sorted(set(s for s in v if len(s) >= 2)) for k, v in out.items()}

def pe_tokens():
    """Proto-Elamite entry middles as tokens (M signs only, numerals removed), with site; loop48 corpus."""
    out = []
    for l in open(C48 + 'proto_elamite.jsonl'):
        d = json.loads(l); m = tuple(x for x in d['seq'] if not x.startswith('N'))
        if len(m) >= 2: out.append((d['site'], m))
    return out

# ---------------------------------------------------------------- planted codes
def random_id_code(types, rnd):
    """each type -> iid string from the population's sign unigram, same length (frequency-matched random ID)."""
    pool = [a for t in types for a in t]
    return {t: tuple(rnd.choice(pool) for _ in t) for t in types}

def slot_code(types, rnd):
    """each type -> string whose k-th sign (k counted from the start, last kept as its own slot) is drawn from that
    slot's sign distribution in the population: a designed slot code with the same positional profile, no adjacency."""
    slots = collections.defaultdict(list)
    for t in types:
        for i, a in enumerate(t): slots[pclass(i, len(t))].append(a)
    return {t: tuple(rnd.choice(slots[pclass(i, len(t))]) for i in range(len(t))) for t in types}

# ---------------------------------------------------------------- statistics
def H(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values() if v) if n else 0.0

def pclass(i, L):
    if i == 0: return 'I'
    if i == L - 1: return 'F'
    if i == 1: return 'S'
    if i == L - 2: return 'P'
    return 'M'

def shuffle_global(types, rnd):
    pool = [a for t in types for a in t]; rnd.shuffle(pool); out = []; k = 0
    for t in types: out.append(tuple(pool[k:k + len(t)])); k += len(t)
    return out

def shuffle_pos(types, rnd):
    """signs permuted across words within the same position class (I, S, M, P, F): keeps every slot profile."""
    cl = collections.defaultdict(list)
    for t in types:
        for i, a in enumerate(t): cl[pclass(i, len(t))].append(a)
    for v in cl.values(): rnd.shuffle(v)
    ptr = collections.Counter(); out = []
    for t in types:
        s = []
        for i in range(len(t)):
            c = pclass(i, len(t)); s.append(cl[c][ptr[c]]); ptr[c] += 1
        out.append(tuple(s))
    return out

def markov2_gen(types, rnd):
    """order-2 Markov chain fit on the types (start-padded, with END), regenerated at the SAME length (loop 41 rule)."""
    m = collections.defaultdict(collections.Counter); uni = collections.Counter()
    for t in types:
        h = ('^', '^')
        for a in t: m[h][a] += 1; uni[a] += 1; h = (h[1], a)
    out = []
    for t in types:
        h = ('^', '^'); s = []
        for _ in t:
            src = m.get(h) or m.get(('^', h[1])) or uni
            ks, ws = zip(*src.items()); a = rnd.choices(ks, ws)[0]; s.append(a); h = (h[1], a)
        out.append(tuple(s))
    return out

def adj_pairs(types):
    return [(t[i], t[i + 1]) for t in types for i in range(len(t) - 1)]

def mi_pairs(pp):
    a = collections.Counter(x for x, _ in pp); b = collections.Counter(y for _, y in pp); j = collections.Counter(pp); N = len(pp)
    return sum(v / N * math.log2(v * N / (a[x] * b[y])) for (x, y), v in j.items())

def gap_rate(types, emin=3.0):
    """phonotactic gaps: among sign pairs whose expected adjacency count (independence of left and right marginals) >= emin,
    the share never observed adjacent."""
    pp = adj_pairs(types); N = len(pp)
    a = collections.Counter(x for x, _ in pp); b = collections.Counter(y for _, y in pp); j = collections.Counter(pp)
    tot = gap = 0
    for x, cx in a.items():
        for y, cy in b.items():
            if cx * cy / N >= emin:
                tot += 1; gap += j.get((x, y), 0) == 0
    return gap / tot if tot else float('nan'), tot

def edge_cov(types):
    ini = collections.Counter(t[0] for t in types); fin = collections.Counter(t[-1] for t in types); N = len(types)
    return sum(v for _, v in ini.most_common(10)) / N, sum(v for _, v in fin.most_common(10)) / N

def pos_mean(types, mink=5):
    """mean |P(initial) - P(final)| over signs with >= mink tokens."""
    ini = collections.Counter(t[0] for t in types); fin = collections.Counter(t[-1] for t in types)
    tot = collections.Counter(a for t in types for a in t)
    v = [abs(ini[a] - fin[a]) / n for a, n in tot.items() if n >= mink]
    return sum(v) / len(v) if v else float('nan')

def ext_share(types):
    """share of types that equal another type of the sample plus ONE sign added at the start (pre) or the end (suf)."""
    S = set(types); pre = sum(1 for t in types if len(t) >= 3 and t[1:] in S); suf = sum(1 for t in types if len(t) >= 3 and t[:-1] in S)
    return pre / len(types), suf / len(types)

def gap_topk(types, K=15):
    """phonotactic gaps: share of ordered pairs among the K commonest signs never seen adjacent inside a word."""
    tot = collections.Counter(a for t in types for a in t); top = [a for a, _ in tot.most_common(K)]
    seen = set(adj_pairs(types))
    return sum(1 for a in top for b in top if (a, b) not in seen) / (len(top) ** 2)

def adj_gain(types, seed=0, kk=5.0):
    """held-out bits per sign saved by knowing the previous sign BEYOND the position class: fit on one half of the types,
    score the other half (both ways); baseline P(a | class) add-0.5, model (c(class,prev,a) + kk P(a|class)) / (n + kk)."""
    r = random.Random(seed); idx = list(range(len(types))); r.shuffle(idx)
    halves = [[types[i] for i in idx[::2]], [types[i] for i in idx[1::2]]]
    V = len(set(a for t in types for a in t)); g = []; n = 0
    for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        c0 = collections.defaultdict(collections.Counter); c1 = collections.defaultdict(collections.Counter)
        for t in fit:
            for i, a in enumerate(t):
                cl = pclass(i, len(t)); c0[cl][a] += 1
                if i: c1[(cl, t[i - 1])][a] += 1
        for t in test:
            for i in range(1, len(t)):
                cl = pclass(i, len(t)); a = t[i]; N0 = sum(c0[cl].values())
                p0 = (c0[cl][a] + 0.5) / (N0 + 0.5 * V)
                cc = c1.get((cl, t[i - 1])); n1 = sum(cc.values()) if cc else 0
                p1 = ((cc[a] if cc else 0) + kk * p0) / (n1 + kk)
                g.append(math.log2(p1 / p0))
    return sum(g) / len(g) if g else float('nan')

def type_stats(types):
    ic, fc = edge_cov(types); pre, suf = ext_share(types)
    return dict(gap=gap_topk(types), gain=adj_gain(types), init10=ic, fin10=fc, pos=pos_mean(types), pre=pre, suf=suf)

def battery(types, rnd, nnull=10):
    """observed statistics and their excess over three nulls: G (global sign shuffle), P (position-class shuffle),
    M2 (order-2 Markov regeneration). Edge coverage excess is over frequency-matched iid strings (= G)."""
    o = type_stats(types)
    res = dict(o)
    for nm, f in (('G', shuffle_global), ('P', shuffle_pos), ('M2', markov2_gen)):
        acc = collections.defaultdict(list)
        for _ in range(nnull):
            for k, v in type_stats(f(types, rnd)).items(): acc[k].append(v)
        for k in o:
            vv = [x for x in acc[k] if not math.isnan(x)]
            res[k + '_ex' + nm] = o[k] - (sum(vv) / len(vv) if vv else float('nan'))
    return res

def q(v, p):
    v = sorted(x for x in v if not (isinstance(x, float) and math.isnan(x)))
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')

def summ(v):
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return (sum(v) / len(v), q(v, 0.025), q(v, 0.975)) if v else (float('nan'),) * 3

def heaps(tokens, rnd, norders=10):
    n = len(tokens); grid = sorted(set(int(round(n * 0.05 * 1.25 ** i)) for i in range(40) if n * 0.05 * 1.25 ** i <= n) | {n})
    acc = collections.defaultdict(float)
    for _ in range(norders):
        idx = list(range(n)); rnd.shuffle(idx); seen = set(); gi = 0; m = 0
        for i in idx:
            seen.add(tokens[i]); m += 1
            if gi < len(grid) and m == grid[gi]: acc[m] += len(seen); gi += 1
    pts = [(math.log(m), math.log(acc[m] / norders)) for m in grid if m >= max(10, n * 0.05)]
    mx = sum(a for a, _ in pts) / len(pts); my = sum(b for _, b in pts) / len(pts)
    return sum((a - mx) * (b - my) for a, b in pts) / sum((a - mx) ** 2 for a, _ in pts)

VOW = re.compile(r'^(D|J|K|M|N|P|Q|R|S|T|W|Z|DW|NW|TW|PH)?([AEIOU])[23]?$')
def vowel(sign):
    m = VOW.match(sign); return m.group(2) if m else None
def is_pure_vowel(sign):
    return sign in ('A', 'E', 'I', 'O', 'U')
