"""v79 THE LINE-START CHAIN AS A SEPARATE CHANNEL: shared helpers.

Idea: the one-glyph-wide line-initial chain (v6/v10/v77) is not text. Medieval books carry non-linguistic channels down
the margin (line/verse numbers, entry numbers, calendar letters and golden numbers, class letters, cross-reference
keys). Test the chain against each kind by massive random hypothesis search, with controls that put a REAL channel of
each kind into an opaque Voynich-like surface (v72 machinery) and must give it back, and generators that must not.

Corpus format as v72/v77: list of pages; page = dict(id, sec, lang, hand, quire, lines=[dict(w=[words], ps=bool)]).
Channel corpora carry l['ch'] = the true channel symbol (or None) on each line.
"""
import os, sys, json, math, random, re, collections, pickle, unicodedata
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v79_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V72
import v77_lib as V77

leaf_half = V72.leaf_half
OPAQUE = ['d', 'y', 'o', 'q', 's', 't', 'S', 'C', 'l', 'p', 'k', 'f', 'r', 'a', 'T', 'K', 'n', 'e', 'i']


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def psave(name, obj):
    pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    p = os.path.join(CK, name)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


# ------------------------------------------------------------------ plain control texts with a real channel
def _norm(s):
    s = unicodedata.normalize('NFD', s.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return [w for w in re.findall(r'[a-z]+', s)]


def dante_plain():
    """Commedia: one verse = one line, canto = paragraph; channel = verse number in the canto (real line numbering).
    Pages of 24 verses (page break does not reset the verse count)."""
    t = open(os.path.join(ROOT, 'data', 'pg1000.txt'), encoding='utf-8', errors='ignore').read().replace('\r', '')
    a = t.find('Nel mezzo del cammin'); b = t.find('*** END')
    t = t[a - 200:b]
    lines, n = [], 0
    for raw in t.split('\n'):
        s = raw.strip()
        if not s: continue
        if re.match(r'^(Canto|Inferno|Purgatorio|Paradiso|INFERNO|PURGATORIO|PARADISO)\b', s):
            if s.startswith('Canto'): n = 0
            continue
        ws = _norm(s)
        if len(ws) < 3: continue
        n += 1
        lines.append(dict(w=ws, ps=(n == 1), ch=n))
    pages = []
    for i in range(0, len(lines), 24):
        pages.append(dict(id='da%03d' % (i // 24), sec='D', lang='-', hand='-', quire='-', lines=lines[i:i + 24]))
    return pages


def golden_numbers(ndays=365):
    """approximate Julian-calendar golden-number column: new moons of GN g march +8 mod 19 every ~1.554 days
    (Jan 1 = III, Jan 2 = XI, Jan 4 = XIX, ...); about 19 of every 29.5 days carry a number."""
    gn = [None] * ndays
    k = 0
    while True:
        d = int(k * 29.5306 / 19)
        if d >= ndays: break
        gn[d] = (3 + 8 * k - 1) % 19 + 1
        k += 1
    return gn


MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def calendar_plain(years=3):
    """a calendar: one line per day (Isidore text as the day's entry), month = paragraph, pages of ~16 days.
    channels: dom = dominical (ferial) letter a-g (day index mod 7), gold = golden number or None."""
    P = V72.isidore_plain()
    body = [l['w'] for p in P for l in p['lines'] if len(l['w']) >= 5]
    gn = golden_numbers()
    pages, bi, cur = [], 0, None
    for y in range(years):
        day = 0
        for m, nd in enumerate(MONTHS):
            for d in range(nd):
                if cur is None or len(cur['lines']) >= 16 or d == 0:
                    cur = dict(id='ca%03d' % len(pages), sec='M%d' % (m // 3), lang='-', hand='-', quire='-', lines=[])
                    pages.append(cur)
                cur['lines'].append(dict(w=body[bi % len(body)], ps=(d == 0), dom=(day + y) % 7, gold=gn[day]))
                bi += 1; day += 1
    return pages


def brumati_entries_plain():
    """Brumati Flora: entries (species/genus) as paragraphs; channels: cls = class of the entry (a real marginal class
    letter, constant down an entry), num = running entry number written on the entry's first line only."""
    import v49_lib
    ents = v49_lib.brumati_entries()
    order = {}
    for e in ents: order.setdefault(e['cls'], len(order))
    lines_all = []
    k = 0
    for e in ents:
        ws = []
        for pi, p in enumerate(e['paras']):
            if pi == 0: p = re.sub(r'^(\d+a?\.?|[IVXLC]+°?\.)\s', '', p)
            ws += v49_lib.brumati_words(p)
        if len(ws) < 15: continue
        k += 1
        for i in range(0, min(len(ws), 80), 8):
            lines_all.append(dict(w=ws[i:i + 8], ps=(i == 0), cls=order[e['cls']], num=k, ent=k))
    pages = []
    for i in range(0, len(lines_all), 20):
        pages.append(dict(id='bu%03d' % (i // 20), sec='B%d' % min(5, lines_all[i]['cls'] // 4), lang='-', hand='-',
                          quire='-', lines=lines_all[i:i + 20]))
    return pages


# ------------------------------------------------------------------ opaque surface with a channel at the line start
def surface_channel(pages, chan, p_write=0.7, seed=7, sym_seed=11, digits=None, tens_pos=None):
    """encode payload (v72 verbose code) + v72 surface; then on each line, with prob p_write, the channel value
    chan(line) (an int or None) replaces the surface's own line-start marker: written as an opaque glyph prefixed to
    the first word. digits: value -> glyph map (random from OPAQUE if None; values >= len map are reduced mod 10 =
    units digit, and if tens_pos == 'w2' the tens digit is prefixed to word 2)."""
    words = [w for p in pages for l in p['lines'] for w in l['w']]
    code = V72.payload_code(words, seed=72, mode='verbose')
    pay = V72.encode_payload(pages, code)
    rng = random.Random(seed)
    if digits is None:
        g = OPAQUE[:12]; r2 = random.Random(sym_seed); r2.shuffle(g); digits = g[:10]
    # surface with the planted random marker switched off on channel lines: run surface, then rebuild line starts
    surf = V72.surface(pay, seed=seed)
    out = []
    for p0, p1 in zip(pages, surf):
        nl = []
        for l0, l1 in zip(p0['lines'], p1['lines']):
            ws = list(l1['w'])
            v = chan(l0)
            if v is not None and ws and rng.random() < p_write:
                w0 = ws[0]
                if w0 and w0[0] in 'ysdt' and len(w0) > 1 and rng.random() < 0.9: w0 = w0[1:]   # drop random marker
                if w0 and w0[0] in 'pf' and l0['ps'] and len(w0) > 1: w0 = w0[1:]
                ws[0] = digits[v % len(digits)] + w0
                if tens_pos == 'w2' and len(ws) > 1:
                    ws[1] = digits[(v // 10) % len(digits)] + ws[1]
            nl.append(dict(l1, w=ws, ch=v))
        out.append(dict(p1, lines=nl))
    return out, digits


# ------------------------------------------------------------------ the chain
def chain_table(pages, alpha=None, k=10, half_fn=leaf_half):
    """per line: page idx, para idx, ps, first glyph code (top-k alphabet of NON-paragraph-first lines, else -1),
    half. Returns dict of arrays + alphabet."""
    rows = []
    para = -1
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']):
            if l['ps'] or para < 0: para += 1
            g = l['w'][0][0] if l['w'] and l['w'][0] else '?'
            rows.append((pi, para, int(l['ps']), g, half_fn(p['id']), li))
    if alpha is None:
        c = Counter(r[3] for r in rows if not r[2])
        alpha = [x for x, _ in c.most_common(k)]
    ix = {g: i for i, g in enumerate(alpha)}
    T = dict(page=np.array([r[0] for r in rows]), para=np.array([r[1] for r in rows]),
             ps=np.array([r[2] for r in rows]), g=np.array([ix.get(r[3], -1) for r in rows]),
             half=np.array([r[4] for r in rows]), li=np.array([r[5] for r in rows]), raw=[r[3] for r in rows])
    return T, alpha


def shuffle_within_para(T, rng):
    """permute the body-line glyphs within each paragraph (paragraph-first lines stay)."""
    g = T['g'].copy()
    for pa in np.unique(T['para']):
        idx = np.where((T['para'] == pa) & (T['ps'] == 0))[0]
        if len(idx) > 1: g[idx] = g[rng.permutation(idx)]
    return dict(T, g=g)


def markov_surrogate(T, rng, rowperm=False):
    """first-order surrogate of the body chain: P(g_{n+1} | g_n) fitted over the whole corpus (body lines, within
    paragraphs); the first body line of each paragraph keeps its real glyph. rowperm=True permutes each row's
    successor probabilities over the destinations (keeps each row's sharpness, destroys WHICH successor is favoured)."""
    A = int(T['g'].max()) + 2   # last index = 'other' (-1)
    g = T['g'].copy(); g[g < 0] = A - 1
    M = np.ones((A, A)) * 0.1
    n = len(g)
    for i in range(n - 1):
        if T['para'][i] == T['para'][i + 1] and T['ps'][i + 1] == 0 and T['ps'][i] == 0:
            M[g[i], g[i + 1]] += 1
    M /= M.sum(1, keepdims=True)
    if rowperm:
        for a in range(A): M[a] = M[a][rng.permutation(A)]
    out = g.copy()
    for i in range(1, n):
        if T['para'][i] == T['para'][i - 1] and T['ps'][i] == 0 and T['ps'][i - 1] == 0:
            out[i] = rng.choice(A, p=M[out[i - 1]])
    out[out == A - 1] = -1
    return dict(T, g=out)


def brumati_parts_plain():
    """Brumati entries with the real part of the entry each line belongs to (index of the source paragraph inside
    the entry: header/description, then further paragraphs: synonyms, habitat, uses); one entry = one paragraph."""
    import v49_lib
    ents = v49_lib.brumati_entries()
    lines_all = []
    for e in ents:
        k = 0
        first = True
        for pi, p in enumerate(e['paras']):
            if pi == 0: p = re.sub(r'^(\d+a?\.?|[IVXLC]+°?\.)\s', '', p)
            ws = v49_lib.brumati_words(p)
            for i in range(0, min(len(ws), 48), 8):
                lines_all.append(dict(w=ws[i:i + 8], ps=first, part=pi))
                first = False
    pages = []
    for i in range(0, len(lines_all), 20):
        pages.append(dict(id='bp%03d' % (i // 20), sec='B', lang='-', hand='-', quire='-', lines=lines_all[i:i + 20]))
    return pages
