"""v54 textual criticism without a language.

Corpus format: list of pages; page = dict(id, vars=dict(...), lines=[[word, ...], ...]); a word is a string of
single-character units (Voynich: glyph units from vlib.glyphs; controls: letters mapped to opaque symbols).

Pipeline
  families(C)      near-repeated word trigrams (C helper v54_pairs) = pairs of 'witnesses' of one passage
  collate(...)     edit operations between aligned witness words: substitution a~b, indel x, doubling x~xx
  reduction(...)   reduced alphabet from the top-K collation operations; random reductions of the same make-up
  evaluate(...)    held-out page-variable information (naive Bayes on 20-token chunks) and held-out word-bigram
                   gain, on pages never used to learn the reduction
"""
import os, sys, json, random, math, re, subprocess, collections, unicodedata, glob
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v54_ckpt')
os.makedirs(CK, exist_ok=True)
PAIRS_BIN = os.path.join(CK, 'v54_pairs')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'


# ------------------------------------------------------------------ corpora
def voynich(name='ZL3b'):
    import vlib
    L = vlib.load_voynich(name)
    pages = collections.OrderedDict()
    for r in L:
        ws = [''.join(vlib.glyphs(w)) for w in r['words'] if '?' not in w and w]
        if not ws: continue
        p = pages.setdefault(r['folio'], dict(id=r['folio'], vars=dict(sec=r['illus'], lang=r['lang'] or '-', hand=r['hand'],
                                                                      quire=r['quire']), lines=[]))
        p['lines'].append(ws)
    return [p for p in pages.values() if sum(len(l) for l in p['lines']) >= 20]


# planted medieval spelling variation (Brumati Italian herbal, 1844), per scribe preferences + free variation
PLANT_TRUTH = dict(sub=[('u', 'v'), ('i', 'j'), ('i', 'y'), ('j', 'y'), ('t', 'c'), ('n', 'm')], dele=['h'], dup='consonants')
SCRIBES = [dict(v=0.15, j=0.30, y=0.10, ci=0.6, m=0.15, h=0.10, dd=0.4, gg=0.05),
           dict(v=0.60, j=0.10, y=0.25, ci=0.2, m=0.40, h=0.25, dd=0.15, gg=0.15),
           dict(v=0.35, j=0.45, y=0.05, ci=0.4, m=0.25, h=0.05, dd=0.6, gg=0.02)]
VOW = set('aeiou')


def medieval(w, s, rng):
    out = []
    i = 0
    while i < len(w):
        c = w[i]; nxt = w[i + 1] if i + 1 < len(w) else ''
        prv = out[-1] if out else ''
        if c in 'bcdfglmnprstvz' and nxt == c:                       # geminate: drop one
            if rng.random() < s['dd']: i += 1; continue
        if c in 'uv':
            c = 'v' if rng.random() < s['v'] else 'u'
        elif c in 'ij':
            r = rng.random()
            c = 'j' if r < s['j'] else ('y' if r < s['j'] + s['y'] else 'i')
        elif c in 'tc' and nxt == 'i' and i + 2 < len(w) and w[i + 2] in VOW:
            c = 'c' if rng.random() < s['ci'] else 't'
        elif c in 'nm' and i == len(w) - 1:
            c = 'm' if rng.random() < s['m'] else 'n'
        out.append(c)
        if c in 'ctp' and nxt in VOW and rng.random() < s['h'] * 0.5: out.append('h')
        if c in 'lnrst' and prv in VOW and nxt in VOW and rng.random() < s['gg']: out.append(c)   # spurious gemination
        i += 1
    if out and out[0] in VOW and rng.random() < s['h'] * 0.4: out.insert(0, 'h')
    return ''.join(out)


def opaque_map(alpha, seed):
    rng = random.Random(seed)
    sy = [chr(c) for c in range(0x3B1, 0x3B1 + 25)] + [chr(c) for c in range(0x410, 0x410 + 32)]
    rng.shuffle(sy)
    return {a: sy[i] for i, a in enumerate(sorted(alpha))}


def brumati(seed=54, cap_tokens=40000, line_w=9, plant=True):
    import v49_lib
    rng = random.Random(seed)
    ents = v49_lib.brumati_entries()
    groups = {}
    for e in ents: groups.setdefault(e['cls'], len(groups))
    pages, cur, ntok = [], None, 0
    for e in ents:
        if ntok >= cap_tokens: break
        ws = []
        for pi, p in enumerate(e['paras']):
            if pi == 0: p = re.sub(r'^(\d+a?\.?|[IVXLC]+°?\.)\s', '', p)
            ws += v49_lib.brumati_words(p)
        if len(ws) < 15: continue
        sec = 'G%d' % min(5, groups[e['cls']] // 4)
        if cur is None or cur['_n'] >= 180 or cur['vars']['sec'] != sec:
            scribe = (len(pages) // 6) % 3
            cur = dict(id='br%03d' % len(pages), vars=dict(sec=sec, hand=str(scribe)), lines=[], _n=0); pages.append(cur)
        s = SCRIBES[int(cur['vars']['hand'])]
        ws2 = [medieval(w, s, rng) if plant else w for w in ws]
        cur['lines'] += [ws2[i:i + line_w] for i in range(0, len(ws2), line_w)]
        cur['_n'] += len(ws); ntok += len(ws)
    for p in pages: p.pop('_n')
    alpha = {c for p in pages for l in p['lines'] for w in l for c in w}
    M = opaque_map(alpha, seed)
    for p in pages: p['lines'] = [[''.join(M[c] for c in w) for w in l] for l in p['lines']]
    return pages, M


def german():
    C = json.load(open(os.path.join(ROOT, 'data', 'v30_ckpt', 'corpora.json')))['corpora']
    pages = []
    for k in ['G_Bav1', 'G_Bav2', 'G_Rip']:
        P = C[k]
        for i, ws in enumerate(P):
            ws = [w for w in ws if w]
            pages.append(dict(id='%s_%03d' % (k, i), vars=dict(sec=k, pos='%s%d' % (k, (4 * i) // len(P))),
                              lines=[ws[j:j + 9] for j in range(0, len(ws), 9)]))
    return pages


# ------------------------------------------------------------------ nulls
def _tokens_by(pages, key):
    by = collections.defaultdict(list)
    for p in pages:
        by[p['vars'][key]] += [w for l in p['lines'] for w in l]
    return by


def null_lineshuf(pages, seed=1, key='sec'):
    """Words shuffled across lines within each section: same words, same section frequencies, phrases destroyed."""
    rng = random.Random(seed)
    by = _tokens_by(pages, key)
    for v in by.values(): rng.shuffle(v)
    it = {k: iter(v) for k, v in by.items()}
    return [dict(p, lines=[[next(it[p['vars'][key]]) for _ in l] for l in p['lines']]) for p in pages]


def null_markov(pages, seed=1, key='sec'):
    """Word-internal unit-trigram resynthesis, trained per section (keeps section unit statistics, no phrases)."""
    rng = random.Random(seed)
    by = _tokens_by(pages, key)
    tabs = {}
    for k, ws in by.items():
        tri = collections.defaultdict(collections.Counter)
        for w in ws:
            s = '\x01\x01' + w + '\x02'
            for i in range(2, len(s)): tri[s[i - 2:i]][s[i]] += 1
        tabs[k] = {c: (list(v.keys()), list(v.values())) for c, v in tri.items()}
    out = []
    for p in pages:
        tab = tabs[p['vars'][key]]
        nl = []
        for l in p['lines']:
            q = []
            for _ in l:
                ctx, w = '\x01\x01', ''
                while True:
                    ks, vs = tab[ctx]; c = rng.choices(ks, vs)[0]
                    if c == '\x02' or len(w) > 20: break
                    w += c; ctx = ctx[1] + c
                q.append(w or ks[0])
            nl.append(q)
        out.append(dict(p, lines=nl))
    return out


def null_selfcit(pages, seed=1, key='sec', window=60, p_mod=0.5):
    """Self-citation (copy-and-modify) generator run separately inside each section."""
    rng = random.Random(seed)
    by = _tokens_by(pages, key)
    gen = {}
    for k, words in by.items():
        big = collections.defaultdict(collections.Counter)
        for w in words:
            for a, b in zip('\x01' + w, w): big[a][b] += 1
        bt = {a: (list(c.keys()), list(c.values())) for a, c in big.items()}
        lens = [len(w) for w in words]
        out = list(words[:window])
        while len(out) < len(words):
            w = list(rng.choice(out[-window:]))
            if rng.random() < p_mod:
                tl = rng.choice(lens)
                if len(w) < tl:
                    pos = rng.randrange(len(w) + 1); kk, vv = bt.get(w[pos - 1] if pos else '\x01', bt['\x01'])
                    w.insert(pos, rng.choices(kk, vv)[0])
                elif len(w) > tl and len(w) > 1: w.pop(rng.randrange(len(w)))
                elif w:
                    pos = rng.randrange(len(w)); kk, vv = bt.get(w[pos - 1] if pos else '\x01', bt['\x01'])
                    w[pos] = rng.choices(kk, vv)[0]
            out.append(''.join(w) or rng.choice(words))
        gen[k] = iter(out)
    return [dict(p, lines=[[next(gen[p['vars'][key]]) for _ in l] for l in p['lines']]) for p in pages]


# ------------------------------------------------------------------ families (C helper)
def families(pages, tag=None):
    toks = []   # (line, page, word)
    li = 0
    for pi, p in enumerate(pages):
        for l in p['lines']:
            for w in l: toks.append((li, pi, w))
            li += 1
    types = sorted({w for _, _, w in toks})
    units = sorted({c for w in types for c in w})
    um = {c: chr(33 + i) if i < 90 else chr(33 + 89) for i, c in enumerate(units)}
    tid = {w: i for i, w in enumerate(types)}
    inp = [str(len(types))] + [''.join(um[c] for c in w)[:200] or '!' for w in types] + [str(len(toks))] + \
          ['%d %d %d' % (a, b, tid[w]) for a, b, w in toks]
    r = subprocess.run([PAIRS_BIN], input='\n'.join(inp) + '\n', capture_output=True, text=True, check=True)
    pairs = [tuple(map(int, x.split())) for x in r.stdout.split('\n') if x]
    return toks, pairs, r.stderr.strip()


# ------------------------------------------------------------------ collation
def align_ops(a, b):
    """Levenshtein alignment of two unit strings; returns list of ops:
       ('S', x, y, posclass) substitution (x<y sorted), ('D', x, posclass) indel of x,
       ('G', x, posclass) indel of x next to an identical x (doubling)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1): d[i][0] = i
    for j in range(lb + 1): d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            d[i][j] = min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], la, lb
    L = max(la, lb)

    def pc(k, n):
        return 'I' if k == 0 else ('F' if k >= n - 1 else 'M')
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                x, y = sorted((a[i - 1], b[j - 1])); ops.append(('S', x, y, pc(i - 1, la)))
            i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            x = a[i - 1]; dbl = (i >= 2 and a[i - 2] == x) or (i < la and a[i] == x)
            ops.append(('G' if dbl else 'D', x, pc(i - 1, la))); i -= 1
        else:
            x = b[j - 1]; dbl = (j >= 2 and b[j - 2] == x) or (j < lb and b[j] == x)
            ops.append(('G' if dbl else 'D', x, pc(j - 1, lb))); j -= 1
    return ops


def collate(toks, pairs, pages=None, page_ok=None):
    """Collation tallies over witness pairs. page_ok: set of page indices allowed (both witnesses must be in it)."""
    occ = collections.Counter()      # unit occurrences in aligned witness words
    occpos = collections.Counter()   # (posclass) occurrences
    ops = collections.Counter()
    oppos = collections.Counter()
    npair = 0
    for (i, j, *_ds) in pairs:
        if page_ok is not None and (toks[i][1] not in page_ok or toks[j][1] not in page_ok): continue
        npair += 1
        for k in range(3):
            a, b = toks[i + k][2], toks[j + k][2]
            for w in (a, b):
                for q, c in enumerate(w):
                    occ[c] += 1; occpos['I' if q == 0 else ('F' if q == len(w) - 1 else 'M')] += 1
            if a != b:
                for o in align_ops(a, b):
                    key = o[:-1]; ops[key] += 1; oppos[o[-1]] += 1
    return dict(npair=npair, occ=occ, ops=ops, occpos=occpos, oppos=oppos)


def op_scores(col, min_occ=30):
    occ = col['occ']; out = []
    for key, n in col['ops'].items():
        if key[0] == 'S':
            if occ[key[1]] < min_occ or occ[key[2]] < min_occ: continue
            s = n / math.sqrt(occ[key[1]] * occ[key[2]])
        else:
            if occ[key[1]] < min_occ: continue
            s = n / occ[key[1]]
        out.append((s, n, key))
    out.sort(reverse=True)
    return out


# ------------------------------------------------------------------ reductions
def make_reduction(ops_list):
    """ops_list: list of keys ('S',x,y) / ('D',x) / ('G',x). Returns a function word -> reduced word."""
    par = {}

    def f(x):
        while par.get(x, x) != x: x = par[x]
        return x
    dele, dup = set(), set()
    for k in ops_list:
        if k[0] == 'S':
            a, b = f(k[1]), f(k[2])
            if a != b: par[max(a, b)] = min(a, b)
        elif k[0] == 'D': dele.add(k[1])
        else: dup.add(k[1])
    cmap = {}
    dupr = {f(x) for x in dup}

    def red(w):
        out = []
        for c in w:
            if c in dele: continue
            r = cmap.get(c)
            if r is None: r = cmap[c] = f(c)
            if out and out[-1] == r and (c in dup or r in dupr): continue
            out.append(r)
        return ''.join(out) or '_'
    return red


def random_ops(template, units_by_freq, rng):
    """Same make-up as template (number of S, D, G ops), units drawn from the frequent units."""
    out = []
    for k in template:
        if k[0] == 'S':
            x, y = rng.sample(units_by_freq, 2); out.append(('S',) + tuple(sorted((x, y))))
        else:
            out.append((k[0], rng.choice(units_by_freq)))
    return out


def apply_red(pages, red):
    cache = {}
    def r(w):
        v = cache.get(w)
        if v is None: v = cache[w] = red(w)
        return v
    return [dict(p, lines=[[r(w) for w in l] for l in p['lines']]) for p in pages]


# ------------------------------------------------------------------ held-out evaluation
def chunks(p, n=20):
    ws = [w for l in p['lines'] for w in l]
    return [ws[i:i + n] for i in range(0, len(ws) - n // 2, n)]


def nb_info(train, test, var, alpha=0.3):
    """Held-out accuracy of a multinomial naive Bayes (add-alpha) predicting page variable `var` from 20-token chunks."""
    cnt = collections.defaultdict(collections.Counter); tot = collections.Counter(); prior = collections.Counter()
    for p in train:
        y = p['vars'].get(var)
        for c in chunks(p): prior[y] += 1
        for l in p['lines']:
            for w in l: cnt[y][w] += 1; tot[y] += 1
    V = len({w for c in cnt.values() for w in c}) + 1
    labs = list(prior); pt = sum(prior.values())
    lt = {k: (math.log(prior[k] / pt), math.log(tot[k] + alpha * V)) for k in labs}
    hit, n = 0, 0
    for p in test:
        y = p['vars'].get(var)
        if y not in prior: continue
        for ch in chunks(p):
            best = max(labs, key=lambda k: lt[k][0] + sum(math.log(cnt[k][w] + alpha) for w in ch) - len(ch) * lt[k][1])
            hit += best == y; n += 1
    return hit / max(n, 1)


def bigram_gain(train, test, D=0.75):
    """Held-out bits/token gained by an interpolated absolute-discounting word bigram over the unigram."""
    uni = collections.Counter(); big = collections.defaultdict(collections.Counter)
    for p in train:
        for l in p['lines']:
            for a, b in zip(['<s>'] + l, l): uni[b] += 1; big[a][b] += 1
    N = sum(uni.values()); V = len(uni) + 1
    st = {a: (sum(c.values()), len(c)) for a, c in big.items()}
    g, n = 0.0, 0
    for p in test:
        for l in p['lines']:
            for a, b in zip(['<s>'] + l, l):
                u = (uni[b] + 0.5) / (N + 0.5 * V)
                c = big.get(a)
                if c:
                    ca, ta = st[a]; pb = max(c[b] - D, 0) / ca + D * ta / ca * u
                else: pb = u
                g += math.log2(pb / u); n += 1
    return g / max(n, 1)


def folds(pages, k=5, seed=0):
    idx = list(range(len(pages))); random.Random(seed).shuffle(idx)
    return [set(idx[i::k]) for i in range(k)]
