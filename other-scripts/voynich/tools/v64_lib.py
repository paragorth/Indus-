"""v64: are Voynich words Lullian concept combinations?

Corpora (each a list of lines: {'words': [unit-strings], 'sec': str, 'fol': int}):
  voy      ZL3b paragraph text (glyph units, one char per glyph)
  voyit    IT2a
  voy_gshuf  glyphs shuffled inside each word (null)
  voy_mk2  glyph Markov-2 resynthesis (null)
  voy_sc   self-citation generator (null)
  ars      POSITIVE CONTROL: text read off a reconstructed Ars brevis tabula generalis
           (concepts B C D E F G H I K + T, 20-cell columns of each triple), opaque glyphs,
           Voynich-like padding, 15% free words, section-biased columns
  med      POSITIVE CONTROL: Lullist medical combinatoria (quality, degree, humour, organ;
           canonical order, degree turned like a wheel along each line), opaque glyphs, padding
  lat      NEGATIVE CONTROL: Isidore Latin, letters -> opaque glyph codes, same padding
  ars_mk2  Markov-2 resynthesis of ars (null for the positive control)
"""
import os, sys, random, re, json, subprocess, math
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib, gen

ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v64_ckpt')
LOOPS = os.path.join(ROOT, 'loops')
os.makedirs(CK, exist_ok=True)
SCORER = os.path.join(CK, 'v64_score')

OPAQUE = list('ABDGHIJMRUVWXZ0123456789')
PAD_I, PAD_M, PAD_F = ['q', 'o'], ['e'], ['y', 'n', 'l']


def build_scorer():
    src = os.path.join(HERE, 'v64_score.c')
    if not os.path.exists(SCORER) or os.path.getmtime(SCORER) < os.path.getmtime(src):
        subprocess.check_call(['gcc', '-O2', '-o', SCORER, src, '-lm'])


# ---------------- Voynich ----------------
def voynich(name='ZL3b', ltypes=('P',)):
    L = vlib.load_voynich(name, ltypes=ltypes)
    fols = {}
    out = []
    for r in L:
        f = fols.setdefault(r['folio'], len(fols))
        ws = [''.join(vlib.glyphs(w)) for w in r['words']]
        ws = [w for w in ws if w and '?' not in w]
        if ws:
            out.append({'words': ws, 'sec': r['illus'], 'fol': f, 'folio': r['folio'],
                        'lang': r['lang']})
    return out


# ---------------- encoding with padding ----------------
def make_codes(concepts, rng, maxlen=2):
    codes, used = {}, set()
    for c in concepts:
        while True:
            L = rng.choice([1, 2] if maxlen >= 2 else [1])
            s = ''.join(rng.choice(OPAQUE) for _ in range(L))
            # avoid exact duplicates and single-glyph codes that are prefixes of others
            if s not in used:
                break
        used.add(s); codes[c] = s
    return codes


def encode_word(seq, codes, rng, pi=0.4, pm=0.25, pf=0.6):
    w = ''
    if rng.random() < pi:
        w += rng.choice(PAD_I)
    for i, c in enumerate(seq):
        if i and rng.random() < pm:
            w += rng.choice(PAD_M)
        w += codes[c]
    if rng.random() < pf:
        w += rng.choice(PAD_F)
    return w


def layout(words_meta, rng, mean_len=8.5, lines_per_fol=30):
    """words_meta: list of (word, sec). Cut into lines of Voynich-like length."""
    out, i, nline = [], 0, 0
    while i < len(words_meta):
        k = max(2, int(rng.gauss(mean_len, 2.5)))
        chunk = words_meta[i:i + k]; i += k
        out.append({'words': [w for w, s in chunk], 'sec': chunk[0][1],
                    'fol': nline // lines_per_fol})
        nline += 1
    return out


# ---------------- positive control 1: Ars brevis tabula ----------------
ARS = list('BCDEFGHIK')


def tabula_column(tri):
    x = sorted(tri, key=ARS.index)
    cells = []
    from itertools import combinations
    for np_ in (3, 2, 1, 0):
        for P in combinations(x, np_):
            for Q in combinations(x, 3 - np_):
                if np_ == 3:
                    cells.append(list(P) + ['T'])
                elif np_ == 0:
                    cells.append(['T'] + list(Q))
                else:
                    # Llull: after T a single letter (np=2) or a pair (np=1)
                    cells.append(list(P) + ['T'] + list(Q))
    # keep 20 cells as in the tabula: np=2 -> 3 pairs x 3 singles, np=1 -> 3 x 3 pairs
    seen, out = set(), []
    for c in cells:
        t = tuple(c)
        if t not in seen:
            seen.add(t); out.append(c)
    return out[:20]


def ars_text(n_words=35000, seed=1):
    from itertools import combinations
    rng = random.Random(seed)
    concepts = ARS + ['T']
    codes = make_codes(concepts, rng)
    cols = list(combinations(ARS, 3))
    secs = 'HSBTPC'
    focus = {s: rng.sample(ARS, 3) for s in secs}
    free = [''.join(rng.choice(OPAQUE) for _ in range(rng.choice([2, 3, 4]))) for _ in range(12)]
    out, truth = [], []
    sec_sizes = [0.4, 0.25, 0.15, 0.08, 0.07, 0.05]
    for s, frac in zip(secs, sec_sizes):
        target = int(n_words * frac); n = 0
        wts = [1 + 6 * len(set(c) & set(focus[s])) ** 2 for c in cols]
        while n < target:
            col = rng.choices(cols, wts)[0]
            cells = tabula_column(col)
            st = rng.randrange(len(cells)); run = rng.randint(3, 8)
            for j in range(run):
                if rng.random() < 0.15:
                    out.append((rng.choice(free), s)); truth.append(None)
                else:
                    cell = cells[(st + j) % len(cells)]
                    out.append((encode_word(cell, codes, rng), s)); truth.append(cell)
                n += 1
    return layout(out, rng), codes, truth


# ---------------- positive control 2: Lullist medical combinatoria ----------------
MED_Q, MED_G, MED_U, MED_O = ['hot', 'cold', 'moist', 'dry'], ['g1', 'g2', 'g3', 'g4'], \
    ['sang', 'chol', 'phleg', 'mel'], ['cor', 'hep', 'cereb', 'stom']
MED = MED_Q + MED_G + MED_U + MED_O


def med_text(n_words=35000, seed=2):
    rng = random.Random(seed)
    codes = make_codes(MED, rng)
    free = [''.join(rng.choice(OPAQUE) for _ in range(rng.choice([2, 3, 4]))) for _ in range(12)]
    secs = 'HSBTPC'
    prof = {s: {g: [rng.random() ** 2 + 0.05 for _ in range(4)] for g in 'QGUO'} for s in secs}
    out = []
    sec_sizes = [0.4, 0.25, 0.15, 0.08, 0.07, 0.05]
    for s, frac in zip(secs, sec_sizes):
        target = int(n_words * frac); n = 0
        P = prof[s]
        while n < target:
            q = rng.choices(MED_Q, P['Q'])[0]
            d = rng.randrange(4)
            for j in range(rng.randint(4, 10)):
                if rng.random() < 0.15:
                    out.append((rng.choice(free), s)); n += 1; continue
                g = MED_G[(d + j) % 4]           # the degree wheel turns one step per word
                r = rng.random()
                if r < 0.4:
                    seq = [q, g]
                elif r < 0.7:
                    seq = [q, g, rng.choices(MED_U, P['U'])[0]]
                elif r < 0.85:
                    seq = [rng.choices(MED_U, P['U'])[0], rng.choices(MED_O, P['O'])[0]]
                else:
                    seq = [q, rng.choices(MED_U, P['U'])[0], rng.choices(MED_O, P['O'])[0]]
                out.append((encode_word(seq, codes, rng), s)); n += 1
    return layout(out, rng), codes


# ---------------- negative control: Latin ----------------
def latin_text(n_words=35000, seed=3):
    rng = random.Random(seed)
    txt = open(os.path.join(ROOT, 'data', 'plain', 'la.txt'), encoding='utf-8').read().lower()
    words = re.findall(r'[a-z]+', txt.replace('j', 'i').replace('v', 'u'))
    words = words[200:200 + n_words]
    letters = sorted(set(''.join(words)))
    codes = make_codes(letters, rng)
    secs = 'HSBTPC'; sizes = [0.4, 0.25, 0.15, 0.08, 0.07, 0.05]
    out, i = [], 0
    for s, fr in zip(secs, sizes):
        for w in words[i:i + int(n_words * fr)]:
            out.append((encode_word(list(w), codes, rng, pi=0.3, pm=0.1, pf=0.5), s))
        i += int(n_words * fr)
    return layout(out, rng), codes


# ---------------- nulls ----------------
def glyph_shuffle(lines, seed=1):
    rng = random.Random(seed); out = []
    for L in lines:
        ws = []
        for w in L['words']:
            c = list(w); rng.shuffle(c); ws.append(''.join(c))
        nl = dict(L); nl['words'] = ws; out.append(nl)
    return out


def all_corpora():
    p = os.path.join(CK, 'corpora.json')
    if os.path.exists(p):
        return json.load(open(p))
    C = {}
    C['voy'] = voynich('ZL3b')
    C['voyit'] = voynich('IT2a')
    C['voy_gshuf'] = glyph_shuffle(C['voy'])
    C['voy_mk2'] = gen.char_markov(C['voy'], 2, 1)
    C['voy_sc'] = gen.self_citation(C['voy'], 1)
    a, acodes, _ = ars_text(); C['ars'] = a
    m, mcodes = med_text(); C['med'] = m
    l, lcodes = latin_text(); C['lat'] = l
    C['ars_mk2'] = gen.char_markov(C['ars'], 2, 1)
    C['med_mk2'] = gen.char_markov(C['med'], 2, 1)
    json.dump({'corpora': C, 'codes': {'ars': acodes, 'med': mcodes, 'lat': lcodes}}, open(p, 'w'))
    return json.load(open(p))


SPLITS = {'all': ((0, 2), (1, 3)), 'sel': ((0,), (1,)), 'held': ((2,), (3,))}


def type_counts(lines, split='all'):
    a, b = SPLITS[split]
    tr, te = Counter(), Counter()
    for L in lines:
        q = L['fol'] % 4
        tgt = tr if q in a else te if q in b else None
        if tgt is None:
            continue
        for w in L['words']:
            tgt[w] += 1
    return tr, te


def write_corpus(lines, path, split='all'):
    tr, te = type_counts(lines, split)
    with open(path, 'w') as f:
        for w in set(tr) | set(te):
            if len(w) <= 60 and ' ' not in w:
                f.write('%d %d %s\n' % (tr[w], te[w], w))
    return tr, te


def ngram_pool(tr, top=200, nmax=4):
    c = Counter()
    for w, n in tr.items():
        for k in range(1, nmax + 1):
            for i in range(len(w) - k + 1):
                c[w[i:i + k]] += n
    return c.most_common(top)


def sample_alphabet(pool, rng, kmin=6, kmax=20):
    k = rng.randint(kmin, kmax)
    items = [g for g, _ in pool]; wts = [n ** 0.5 for _, n in pool]
    chosen = set()
    while len(chosen) < k:
        chosen.add(rng.choices(items, wts)[0])
    return sorted(chosen)


def score(corpus_path, alphabets, gapmode=1):
    build_scorer()
    inp = '\n'.join(' '.join(a) for a in alphabets) + '\n'
    r = subprocess.run([SCORER, corpus_path, str(gapmode)], input=inp, capture_output=True, text=True, check=True)
    out = []
    for line in r.stdout.strip().split('\n'):
        v = line.split()
        out.append(dict(zip(['cov', 'canon', 'bfree', 'bm1', 'bmk2', 'Gfree', 'Gm1', 'm'], map(float, v))))
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


# ---------------- python-side parse (for single alphabets) ----------------
def make_parser(alpha):
    al = sorted(alpha, key=len, reverse=True)
    rx = re.compile('|'.join(re.escape(a) for a in al) + '|(.)', re.S)
    idx = {a: i for i, a in enumerate(alpha)}
    cache = {}

    def parse(w):
        if w in cache:
            return cache[w]
        seq = []
        for m in rx.finditer(w):
            if m.group(1) is None:
                seq.append(idx[m.group(0)])
        cache[w] = tuple(seq)
        return cache[w]
    return parse


def canonical_rank(lines, parse, k):
    prec = [[0] * k for _ in range(k)]
    for L in lines:
        for w in L['words']:
            s = parse(w)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    if s[i] != s[j]:
                        prec[s[i]][s[j]] += 1
    sc = []
    for a in range(k):
        v = 0
        for b in range(k):
            if a != b:
                t = prec[a][b] + prec[b][a]
                v += prec[a][b] / t if t else 0.5
        sc.append(v)
    order = sorted(range(k), key=lambda a: -sc[a])
    return {a: r for r, a in enumerate(order)}


def section_info(lines, parse, k, unit='concept', min_sec=200):
    """Mutual information (bits) between token unit and section label,
    computed on concept tokens or on glyph tokens; with a Miller-Madow correction."""
    joint = Counter(); secs = Counter(); us = Counter(); n = 0
    for L in lines:
        for w in L['words']:
            items = parse(w) if unit == 'concept' else list(w)
            for u in items:
                joint[(u, L['sec'])] += 1; secs[L['sec']] += 1; us[u] += 1; n += 1
    if n == 0:
        return 0.0
    I = 0.0
    for (u, s), c in joint.items():
        I += c / n * math.log2(c * n / (us[u] * secs[s]))
    bias = (len(us) - 1) * (len(secs) - 1) / (2 * n * math.log(2))
    return I - bias
