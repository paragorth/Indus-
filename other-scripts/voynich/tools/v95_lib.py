"""v95: shared helpers.  Corpora (v72 page format), planted real documents of a numeric / machine kind written through the
v72 merge code + Voynich-like surface, shuffled twins, and glyph-count feature arrays.

Corpus format: list of pages; page = dict(id, sec, lines=[dict(w=[words], ps=bool)]); one character per glyph unit.
"""
import os, sys, json, math, random, re, hashlib, pickle, zlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
os.environ.setdefault('VOY_MODE', 'glyph')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v95_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V


def psave(name, obj): pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    f = os.path.join(CK, name)
    return pickle.load(open(f, 'rb')) if os.path.exists(f) else None


def sha_obj(o): return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def half(pid):
    return V.leaf_half(pid)


# ------------------------------------------------------------------ real documents of a numeric kind
SIGNS = ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpius', 'sagittarius', 'capricornus',
         'aquarius', 'pisces']


def roman(n):
    """additive (non-subtractive) Roman numeral, as in most medieval tables; 0 -> 'nichil'."""
    n = int(n)
    if n <= 0: return 'nichil'
    s = ''
    for v, c in ((1000, 'm'), (500, 'd'), (100, 'c'), (50, 'l'), (10, 'x'), (5, 'v'), (1, 'i')):
        while n >= v: s += c; n -= v
    return s


def sun(n):
    """n = days from an epoch; returns ecliptic longitude (deg), declination (deg)."""
    L = (280.46 + 0.9856474 * n) % 360; g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = (L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360
    eps = math.radians(23.55)
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(math.radians(lam))))
    return lam, dec


def moon_lon(n): return (218.316 + 13.176396 * n + 6.289 * math.sin(math.radians(134.963 + 13.064993 * n))) % 360


def day_length(dec, lat):
    x = -math.tan(math.radians(lat)) * math.tan(math.radians(dec))
    return 2 * math.degrees(math.acos(max(-1, min(1, x)))) / 15.0


def altitude(dec, lat, hour_angle_h):
    h = math.radians(15 * hour_angle_h); p = math.radians(lat); d = math.radians(dec)
    return math.degrees(math.asin(math.sin(p) * math.sin(d) + math.cos(p) * math.cos(d) * math.cos(h)))


MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def doc_ephemeris(years=3, start=-210000):
    """a kalendar-ephemeris: one day per line: day, sun sign, sun degree, sun minute, day length h, m, moon sign, moon degree.
    paragraphs = months."""
    lines = []; n = start
    for y in range(years):
        for mi, md in enumerate(MDAYS):
            for d in range(1, md + 1):
                lam, dec = sun(n); dl = day_length(dec, 45.0); ml = moon_lon(n)
                w = [roman(d), SIGNS[int(lam // 30)], roman(int(lam % 30)), roman(int((lam % 1) * 60)),
                     roman(int(dl)), roman(int((dl % 1) * 60)), SIGNS[int(ml // 30)], roman(int(ml % 30))]
                lines.append((w, d == 1)); n += 1
    return lines


def doc_altitude(lats=(45.0, 50.0), start=-210000):
    """altitude table: one day per line, the sun's height at hour angles -4..+4 h (9 readings)."""
    lines = []
    for lat in lats:
        n = start
        for mi, md in enumerate(MDAYS):
            for d in range(1, md + 1):
                lam, dec = sun(n)
                w = [roman(round(altitude(dec, lat, h))) for h in range(-4, 5)]
                lines.append((w, d == 1)); n += 1
    return lines


def doc_mixed(years=3, start=-210000):
    """kalendar lines: 6 prose words (Circa instans, in order) with the sun's degree and the day-length minutes
    in fixed slots 1 and 4."""
    import v92_lib as L92
    prose = [w for _, ws in L92.entries('circa_instans_fr', cap=60000) for w in ws]
    lines = []; n = start; k = 0
    for y in range(years):
        for mi, md in enumerate(MDAYS):
            for d in range(1, md + 1):
                lam, dec = sun(n); dl = day_length(dec, 45.0)
                pw = prose[k:k + 6]; k += 6
                w = pw[:1] + [roman(int(lam % 30))] + pw[1:3] + [roman(int((dl % 1) * 60))] + pw[3:]
                lines.append((w, d == 1)); n += 1
    return lines


def pages_from_lines(lines, prefix, per_page=20):
    pages = []
    for i in range(0, len(lines), per_page):
        pages.append(dict(id='%s%03d' % (prefix, len(pages)), sec='x', lang='-', hand='-', quire='-',
                          lines=[dict(w=list(w), ps=ps) for w, ps in lines[i:i + per_page]]))
    return pages


def through_surface(pages, seed, sseed=None):
    words = [w for p in pages for l in p['lines'] for w in l['w']]
    code = V.payload_code(words, seed=seed, mode='merge')
    S = V.surface(V.encode_payload(pages, code), seed=seed + 1 if sseed is None else sseed)
    for p in S:
        for l in p['lines']: l.pop('orig', None)
    return S, code


DOCS = {'P_EPH': (doc_ephemeris, 'ep', 9501), 'P_ALT': (doc_altitude, 'al', 9502), 'P_MIX': (doc_mixed, 'mx', 9503)}


def _v92(name): return pickle.load(open(os.path.join(DATA, 'v92_ckpt', 'corp_%s.pkl' % name), 'rb'))


def corpus(name):
    c = pload('corp_%s.pkl' % name)
    if c is not None: return c
    if name in ('ZL3b', 'IT2a', 'GC2a'):
        c = V.voynich(name)
    elif name in DOCS:
        fn, pre, seed = DOCS[name]
        c, _ = through_surface(pages_from_lines(fn(), pre), seed)
    elif name[:-1] in DOCS and name.endswith('2'):     # same document, second surface (a 'second transcription')
        fn, pre, seed = DOCS[name[:-1]]
        c, _ = through_surface(pages_from_lines(fn(), pre), seed, sseed=seed + 50)
    elif name.startswith('PL_'):                       # plain (no code) version of a planted document
        fn, pre, seed = DOCS['P_' + name[3:]]
        c = pages_from_lines(fn(), pre)
    elif name.endswith('J') and name[:-1] in ('P_CIRCA', 'P_KONRAD'):
        c = justify(corpus(name[:-1]), seed=9531)
    elif name in ('G_SELF', 'G_LX', 'G_SEED', 'G_GM', 'P_CIRCA', 'P_KONRAD'):
        c = _v92(name)
    else:
        raise KeyError(name)
    psave('corp_%s.pkl' % name, c)
    return c


def twin(pages, seed):
    """order-destroying twin: words shuffled inside each line AND lines permuted inside their page; paragraph-start flags stay
    with the line slot. Keeps every line's content and every page's lines."""
    rng = random.Random(seed); out = []
    for p in pages:
        ls = [l['w'][:] for l in p['lines']]
        for w in ls: rng.shuffle(w)
        rng.shuffle(ls)
        out.append(dict(p, lines=[dict(l, w=w) for l, w in zip(p['lines'], ls)]))
    return out


# ------------------------------------------------------------------ features
ALPHA = list('oeyadiClkrnqtSspmTKfPgF') + ['?']
AIX = {c: i for i, c in enumerate(ALPHA)}
K = len(ALPHA); F = 3 * K           # glyph x position (0 first, 1 interior, 2 last)


def wvec(w):
    v = np.zeros(F)
    for j, ch in enumerate(w):
        pos = 0 if j == 0 else (2 if j == len(w) - 1 else 1)
        v[pos * K + AIX.get(ch, K - 1)] += 1
    return v


# ------------------------------------------------------------------ cycle 2: documents with a conserved quantity per line
def _siamese(n):
    M = [[0] * n for _ in range(n)]; i, j = 0, n // 2
    for k in range(1, n * n + 1):
        M[i][j] = k; ni, nj = (i - 1) % n, (j + 1) % n
        if M[ni][nj]: ni, nj = (i + 1) % n, j
        i, j = ni, nj
    return M


def _doubly_even(n):
    M = [[i * n + j + 1 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if (i % 4 in (0, 3)) == (j % 4 in (0, 3)): M[i][j] = n * n + 1 - M[i][j]
    return M


def _singly_even(n):                                    # Strachey (LUX-type) construction
    h = n // 2; A = _siamese(h); M = [[0] * n for _ in range(n)]; add = [0, 2, 3, 1]
    for i in range(h):
        for j in range(h):
            M[i][j] = A[i][j]; M[i + h][j + h] = A[i][j] + h * h
            M[i][j + h] = A[i][j] + 2 * h * h; M[i + h][j] = A[i][j] + 3 * h * h
    k = (n - 2) // 4
    for i in range(h):
        cols = list(range(k)) if i != h // 2 else list(range(1, k + 1))
        cols += list(range(n - k + 1, n))
        for j in cols: M[i][j], M[i + h][j] = M[i + h][j], M[i][j]
    return M


def magic(n):
    return _siamese(n) if n % 2 else (_doubly_even(n) if n % 4 == 0 else _singly_even(n))


def _syms(M):
    out = []; A = [r[:] for r in M]
    for _ in range(4):
        A = [list(r) for r in zip(*A[::-1])]; out.append(A); out.append([r[::-1] for r in A])
    return out


def doc_magic(reps=3):
    """the seven planetary squares (orders 3-9; constructed by the standard methods) in all 8 symmetries, rows as lines,
    numbers in additive Roman numerals; each square a paragraph headed by its planet word."""
    planets = {3: 'saturni', 4: 'iovis', 5: 'martis', 6: 'solis', 7: 'veneris', 8: 'mercurii', 9: 'lune'}
    lines = []
    for r in range(reps):
        for n in range(3, 10):
            for A in _syms(magic(n)):
                for i, row in enumerate(A):
                    lines.append(([roman(x) for x in row], i == 0))
    return lines


def _ring(n, pn, lead_end, bob, leads, rng):
    row = list(range(n)); out = [row[:]]
    def apply(row, places):
        r = row[:]; i = 0
        while i < n - 1:
            if i in places: i += 1; continue
            r[i], r[i + 1] = r[i + 1], r[i]; i += 2
        return r
    for _ in range(leads):
        for p in pn: row = apply(row, p); out.append(row[:])
        row = apply(row, bob if rng.random() < 0.3 else lead_end); out.append(row[:])
    return out


def doc_ringing(seed=95):
    """change-ringing rows (Plain Bob Minor and Major, plain and bobbed leads): each line names every bell once."""
    rng = random.Random(seed); names = ['prima', 'secunda', 'tertia', 'quarta', 'quinta', 'sexta', 'septima', 'octava']
    lines = []
    for n, leads in ((6, 70), (8, 25), (6, 40)):
        pn = []
        for k in range(2 * n - 1): pn.append(set() if k % 2 == 0 else {0, n - 1})
        rows = _ring(n, pn, {0, 1}, {0, 3}, leads, rng)
        for i, r in enumerate(rows): lines.append(([names[b] for b in r], i % (2 * n) == 0))
    return lines


DOCS.update({'P_MAG': (doc_magic, 'mg', 9511), 'P_RING': (doc_ringing, 'rg', 9512)})


def slot_swap(pages, seed):
    """the j-th word of every line exchanged among lines of the same section and word count (whole corpus): keeps each slot's
    word law, line lengths and edges; destroys any co-ordination inside a line, paragraph or page."""
    rng = random.Random(seed); grp = defaultdict(list)
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']): grp[(p['sec'], len(l['w']))].append((pi, li))
    new = [[l['w'][:] for l in p['lines']] for p in pages]
    for (sec, n), locs in grp.items():
        for j in range(n):
            ws = [pages[pi]['lines'][li]['w'][j] for pi, li in locs]; rng.shuffle(ws)
            for (pi, li), w in zip(locs, ws): new[pi][li][j] = w
    return [dict(p, lines=[dict(l, w=w) for l, w in zip(p['lines'], nw)]) for p, nw in zip(pages, new)]


# ------------------------------------------------------------------ cycle 3: computed (derived) tokens: geomantic charts
GEO = {  # figure bits (head, neck, body, feet), 1 = one dot (odd), 0 = two dots (even); the 16 Latin names
    (1, 1, 1, 1): 'via', (0, 0, 0, 0): 'populus', (1, 1, 0, 1): 'puer', (1, 0, 1, 1): 'puella',
    (0, 0, 1, 1): 'fortunamaior', (1, 1, 0, 0): 'fortunaminor', (0, 1, 0, 1): 'acquisitio', (1, 0, 1, 0): 'amissio',
    (1, 0, 0, 1): 'carcer', (0, 1, 1, 0): 'coniunctio', (1, 0, 0, 0): 'laetitia', (0, 0, 0, 1): 'tristitia',
    (0, 1, 1, 1): 'caputdraconis', (1, 1, 1, 0): 'caudadraconis', (0, 1, 0, 0): 'rubeus', (0, 0, 1, 0): 'albus'}


def geo_chart(rng):
    M = [tuple(rng.randint(0, 1) for _ in range(4)) for _ in range(4)]           # four mothers from random dots
    D = [tuple(M[j][i] for j in range(4)) for i in range(4)]                     # daughters: rows of the mothers
    x = lambda a, b: tuple(p ^ q for p, q in zip(a, b))
    N = [x(M[0], M[1]), x(M[2], M[3]), x(D[0], D[1]), x(D[2], D[3])]             # nieces
    W = [x(N[0], N[1]), x(N[2], N[3])]; J = x(W[0], W[1]); R = x(J, M[0])        # witnesses, judge, reconciler
    return M + D + N + W + [J, R]


def doc_geomancy(n=1100, seed=953):
    """geomantic charts as recorded in practice: 16 figure names; line 1 = mothers + daughters, line 2 = nieces, witnesses,
    judge, reconciler; each chart a paragraph."""
    rng = random.Random(seed); lines = []
    for _ in range(n):
        f = [GEO[t] for t in geo_chart(rng)]
        lines.append((f[:8], True)); lines.append((f[8:], False))
    return lines


DOCS.update({'P_GEO': (doc_geomancy, 'ge', 9521)})


def justify(pages, seed, width=(36, 46)):
    """negative control for layout: the same words in the same order re-wrapped greedily into lines of a fixed glyph width
    (one width per page, drawn from the Voynich range); paragraph starts kept at the first line of each old paragraph run."""
    rng = random.Random(seed); out = []
    for p in pages:
        W = rng.randint(*width); lines = []; cur = []; n = 0; ps = True
        for l in p['lines']:
            if l['ps'] and cur:
                lines.append(dict(w=cur, ps=ps)); cur = []; n = 0; ps = True
            elif l['ps']: ps = True
            for w in l['w']:
                if cur and n + len(w) > W:
                    lines.append(dict(w=cur, ps=ps)); cur = []; n = 0; ps = False
                cur.append(w); n += len(w) + 1
        if cur: lines.append(dict(w=cur, ps=ps))
        out.append(dict(p, lines=lines))
    return out


def slot_swap_len(pages, seed):
    """as slot_swap, but a word is exchanged only with a word of the same glyph length: every line keeps its exact glyph
    length (so any fixed ink width / justification is preserved in the null)."""
    rng = random.Random(seed); grp = defaultdict(list)
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']): grp[(p['sec'], len(l['w']))].append((pi, li))
    new = [[l['w'][:] for l in p['lines']] for p in pages]
    for (sec, n), locs in grp.items():
        for j in range(n):
            byl = defaultdict(list)
            for pi, li in locs: byl[len(pages[pi]['lines'][li]['w'][j])].append((pi, li))
            for ln, ll in byl.items():
                ws = [pages[pi]['lines'][li]['w'][j] for pi, li in ll]; rng.shuffle(ws)
                for (pi, li), w in zip(ll, ws): new[pi][li][j] = w
    return [dict(p, lines=[dict(l, w=w) for l, w in zip(p['lines'], nw)]) for p, nw in zip(pages, new)]
