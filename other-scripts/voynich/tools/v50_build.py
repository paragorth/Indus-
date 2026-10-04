"""v50 THE MESSAGE IS NOT IN THE LINES: build page-grid sets for the C path search (tools/v50_search.c).

Each set = list of pages; page = list of paragraph lines (ltype P) in page order; line = list of words with
para_start flag. Written as a flat integer text file data/v50_ckpt/sets/<name>.txt:
  npages; per page: nlines; per line: para_start nwords; per word: wid hid ng g1..gng
  wid = rank id among the 63 most frequent words of the REAL set (63 = other), hid = crc32(word) % 64,
  g = glyph code (EVA with ch sh cth ckh cph cfh as single glyphs; 23 codes, rare glyphs -> 23).

Sets:
  REAL / IT       ZL3b / IT2a paragraph text, pages with >= 4 lines
  MK              Markov resynthesis: same line lengths and paragraph flags; line-initial word from the
                  corpus line-initial distribution, later words from P(word | last glyph of previous word)
  <X>_LSk         lines shuffled within each page (k = 1..3)
  plants (positive controls, Latin = Isidore Etym., German = Kafka Verwandlung; letters -> glyphs by frequency
  rank at the target glyph slot; words -> Voynich word types by frequency rank):
    PA_<base>  acrostic: first glyph of the first word of every line carries the next Latin letter
    PC_<base>  column: the 4th word (j=3) of every line with >= 4 words is the next German code word
    PD_<base>  diagonal: last glyph of word (i mod len) on line i carries the next German letter
    PN_<base>  every 7th word in row-major page order: its first glyph carries the next Latin letter
  base = MK (filler) or REAL (hidden inside the real pages).
"""
import os, sys, json, random, re, zlib
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_lib import vglyphs

VOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(VOY, 'data', 'v50_ckpt'); SETS = os.path.join(CK, 'sets')
os.makedirs(SETS, exist_ok=True)
G = 23


def load_pages(src, min_lines=4):
    R = json.load(open(os.path.join(VOY, 'data', 'derived', f'{src}_lines.json')))
    pages = defaultdict(list); order = []
    for r in R:
        if r['ltype'] != 'P': continue
        if r['folio'] not in pages: order.append(r['folio'])
        pages[r['folio']].append((bool(r['para_start']), list(r['words'])))
    return [(f, pages[f]) for f in order if len(pages[f]) >= min_lines]


REALP = load_pages('ZL3b')
GC = Counter(g for _, p in REALP for _, ws in p for w in ws for g in vglyphs(w))
GCODE = {g: min(i, G) for i, (g, _) in enumerate(GC.most_common())}
WC = Counter(w for _, p in REALP for _, ws in p for w in ws)
WID = {w: i for i, (w, _) in enumerate(WC.most_common(63))}


def gcodes(w): return [GCODE.get(g, G) for g in vglyphs(w)]


def write_set(name, pages):
    out = [str(len(pages))]
    for _, p in pages:
        out.append(str(len(p)))
        for ps, ws in p:
            out.append(f'{int(ps)} {len(ws)}')
            for w in ws:
                gs = gcodes(w)
                out.append(f'{WID.get(w, 63)} {zlib.crc32(w.encode()) % 64} {len(gs)} ' + ' '.join(map(str, gs)))
    open(os.path.join(SETS, name + '.txt'), 'w').write('\n'.join(out) + '\n')
    json.dump([[f, [[ps, ws] for ps, ws in p]] for f, p in pages], open(os.path.join(SETS, name + '.json'), 'w'))


def line_shuffle(pages, seed):
    rng = random.Random(seed); out = []
    for f, p in pages:
        q = p[:]; rng.shuffle(q); out.append((f, q))
    return out


def markov(pages, seed):
    rng = random.Random(seed)
    init = Counter(); trans = defaultdict(Counter)
    for _, p in pages:
        for _, ws in p:
            init[ws[0]] += 1
            for a, b in zip(ws, ws[1:]): trans[vglyphs(a)[-1]][b] += 1
    def draw(c):
        ks = list(c); return rng.choices(ks, weights=[c[k] for k in ks])[0]
    # pre-flatten for speed
    def sampler(c):
        ks = list(c); ws = [c[k] for k in ks]; return lambda: rng.choices(ks, weights=ws)[0]
    si = sampler(init); st = {k: sampler(v) for k, v in trans.items()}
    out = []
    for f, p in pages:
        q = []
        for ps, ws in p:
            nw = [si()]
            for _ in range(len(ws) - 1):
                last = vglyphs(nw[-1])[-1]
                nw.append(st[last]() if last in st else si())
            q.append((ps, nw))
        out.append((f, q))
    return out


def letters(lang, n, skip=0):
    if lang == 'la':
        t = open(os.path.join(VOY, 'data', 'plain', 'la.txt'), encoding='utf-8').read()
    else:
        t = open(os.path.join(VOY, 'data', 'pg22367.txt'), encoding='utf-8').read()
        t = t[t.find('Als Gregor'):]
    t = t.lower().replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    s = re.sub(r'[^a-z]', '', t)[skip:skip + n]
    return s


def words_of(lang):
    t = open(os.path.join(VOY, 'data', 'pg22367.txt'), encoding='utf-8').read()
    t = t[t.find('Als Gregor'):].lower()
    return re.findall(r'[a-zäöüß]+', t)


def slot_glyph(w, slot):
    g = gcodes(w)
    return g[0] if slot == 'first' else g[-1]


def letter_map(text, slot):
    """Latin/German letters by frequency rank -> glyph codes by frequency rank at that slot (real corpus)."""
    sc = Counter(slot_glyph(w, slot) for _, p in REALP for _, ws in p for w in ws)
    gl = [g for g, c in sc.most_common() if c >= 50]
    lc = [l for l, _ in Counter(text).most_common()]
    return {l: gl[i % len(gl)] for i, l in enumerate(lc)}


def vocab_by_slot(slot):
    d = defaultdict(Counter)
    for w, c in WC.items(): d[slot_glyph(w, slot)][w] += c
    return {g: (list(c), [c[k] for k in c]) for g, c in d.items()}


def plant(base, kind, seed):
    rng = random.Random(seed)
    pages = [(f, [(ps, ws[:]) for ps, ws in p]) for f, p in base]
    if kind in ('PA', 'PN', 'PD'):
        slot = 'last' if kind == 'PD' else 'first'
        lang = 'de' if kind == 'PD' else 'la'
        txt = letters(lang, 60000, skip=5000)
        M = letter_map(txt, slot); V = vocab_by_slot(slot); k = 0
        def put(ws, j):
            nonlocal k
            g = M[txt[k % len(txt)]]; k += 1
            ks, wt = V[g]; ws[j] = rng.choices(ks, weights=wt)[0]
        for f, p in pages:
            if kind == 'PA':
                for ps, ws in p: put(ws, 0)
            elif kind == 'PD':
                for i, (ps, ws) in enumerate(p): put(ws, i % len(ws))
            else:
                flat = [(li, j) for li, (_, ws) in enumerate(p) for j in range(len(ws))]
                for li, j in flat[::7]: put(p[li][1], j)
    elif kind == 'PC':
        gw = words_of('de')[3000:]
        gr = [w for w, _ in Counter(gw).most_common()]
        vr = [w for w, _ in WC.most_common()]
        code = {w: vr[i % len(vr)] for i, w in enumerate(gr)}; k = 0
        for f, p in pages:
            for ps, ws in p:
                if len(ws) >= 4: ws[3] = code[gw[k % len(gw)]]; k += 1
    return pages


if __name__ == '__main__':
    IT = load_pages('IT2a')
    print('pages REAL', len(REALP), 'lines', sum(len(p) for _, p in REALP), 'IT', len(IT))
    MK = markov(REALP, 11)
    bases = {'REAL': REALP, 'IT': IT, 'MK': MK}
    for kind in ('PA', 'PC', 'PD', 'PN'):
        bases[f'{kind}_MK'] = plant(MK, kind, 100 + len(kind))
        bases[f'{kind}_REAL'] = plant(REALP, kind, 200 + len(kind))
    for name, pg in bases.items():
        write_set(name, pg)
        for k in (1, 2, 3, 4): write_set(f'{name}_LS{k}', line_shuffle(pg, 1000 * k + zlib.crc32(name.encode()) % 997))
        print('wrote', name, flush=True)
    json.dump({'GCODE': GCODE, 'WID': WID}, open(os.path.join(CK, 'codes.json'), 'w'))
