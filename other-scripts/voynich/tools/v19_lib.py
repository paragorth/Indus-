"""v19 'SOMEWHERE THE TEXT IS SORTED': corpora, stretches, planted controls, engine driver.

Engine: tools/v19_sortsearch.c (compiled to the scratch dir, or ./v19_sortsearch next to this file).
"""
import json, os, random, re, subprocess, math, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(DATA, 'results', 'v19')
os.makedirs(RES, exist_ok=True)
SCR = os.environ.get('V19_SCR', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v19')
os.makedirs(SCR, exist_ok=True)
ENGINE = os.path.join(SCR, 'sortsearch')
sys.path.insert(0, HERE)
from vlib import glyphs  # EVA -> glyph units (benched gallows, ch, sh merged)


def build_engine():
    src = os.path.join(HERE, 'v19_sortsearch.c')
    if not os.path.exists(ENGINE) or os.path.getmtime(ENGINE) < os.path.getmtime(src):
        subprocess.check_call(['gcc', '-O3', '-march=native', '-o', ENGINE, src, '-lm'])

# ---------------- corpora as lines ----------------
# line dict: words(list of glyph-unit lists), raw(list of str), page, para_start, section

def voynich_lines(name='ZL3b'):
    recs = json.load(open(os.path.join(DATA, 'derived', name + '_lines.json')))
    out = []
    for r in recs:
        if r['ltype'] != 'P':
            continue
        ws = [w for w in r['words'] if w and '?' not in w]
        if not ws:
            continue
        out.append({'raw': ws, 'words': [glyphs(w) for w in ws], 'page': r['folio'],
                    'para_start': r['para_start'], 'section': r['illus'], 'lang': r['lang']})
    return out


def voynich_labels(name='ZL3b'):
    """label stretches per folio, transcription order (ZL only file exists)."""
    recs = json.load(open(os.path.join(DATA, 'derived', 'v2_labels.json')))
    by = defaultdict(list)
    for r in recs:
        w = r['words'][0] if r['words'] else ''
        if not w or '?' in w:
            continue
        by[r['folio']].append(w)
    return by


ET = os.path.join(SCR, 'et.txt')


def _latin_norm(w):
    w = w.lower().replace('j', 'i').replace('v', 'u')
    w = w.replace('æ', 'ae').replace('œ', 'oe')
    w = ''.join(c for c in w if 'a' <= c <= 'z')
    return w


def isidore_book(roman_title):
    """Etymologiae (la.wikisource, Migne PL 82) -> paragraphs (lists of words) of the book."""
    t = open(ET, encoding='utf-8').read().split('\n')
    starts = [i for i, L in enumerate(t) if L.startswith('==LIBER')]
    i0 = [i for i in starts if roman_title in t[i]][0]
    i1 = min([i for i in starts if i > i0] + [len(t)])
    paras = []
    for L in t[i0 + 1:i1]:
        L = re.sub(r'\(\d{4}[A-D]?\)', ' ', L)
        L = re.sub(r'^\s*\d+\.\s*', '', L)
        L = re.sub(r'\s\d+\s', ' ', L)
        if L.startswith('CAPUT') or L.startswith('=') or len(L.strip()) <= 2:
            continue
        ws = [_latin_norm(w) for w in re.split(r'[\s\-—,;:.!?()\[\]"]+', L)]
        ws = [w for w in ws if w]
        if ws:
            paras.append(ws)
    return paras


def latin_lines(paras, per_line=9, lines_per_page=25, section='X'):
    out, pg, ln = [], 0, 0
    for ws in paras:
        for k in range(0, len(ws), per_line):
            out.append({'raw': ws[k:k + per_line], 'words': [list(w) for w in ws[k:k + per_line]],
                        'page': 'p%03d' % pg, 'para_start': k == 0, 'section': section})
            ln += 1
            if ln % lines_per_page == 0:
                pg += 1
    return out

# ---------------- stretches ----------------
# a stretch: (id, [entries]); entry = (cls, blk, glyph-unit list)

def pages_of(lines):
    by = defaultdict(list)
    order = []
    for i, L in enumerate(lines):
        if L['page'] not in by:
            order.append(L['page'])
        by[L['page']].append(i)
    return order, by


def make_stretches(lines, kind, matched=False, minn=8):
    order, by = pages_of(lines)
    st = []
    if kind == 'pagewords':
        for p in order:
            ent = []
            for bi, i in enumerate(by[p]):
                ws = lines[i]['words']
                for j, w in enumerate(ws):
                    c = 0 if not matched else (1 if j == 0 else (2 if j == len(ws) - 1 else 3))
                    ent.append((c, bi, w))
            if len(ent) >= minn:
                st.append((p, ent))
    elif kind == 'win5':
        for p in order:
            idx = by[p]
            for k in range(0, len(idx) - 4, 5):
                ent = []
                for bi, i in enumerate(idx[k:k + 5]):
                    ws = lines[i]['words']
                    for j, w in enumerate(ws):
                        c = 0 if not matched else (1 if j == 0 else (2 if j == len(ws) - 1 else 3))
                        ent.append((c, bi, w))
                if len(ent) >= minn:
                    st.append(('%s:%d' % (p, k), ent))
    elif kind == 'lineinit':
        for p in order:
            ent = [(0, bi, lines[i]['words'][0]) for bi, i in enumerate(by[p]) if not lines[i]['para_start']]
            if len(ent) >= minn:
                st.append((p, ent))
    elif kind == 'linefinal':
        for p in order:
            ent = [(0, bi, lines[i]['words'][-1]) for bi, i in enumerate(by[p])]
            if len(ent) >= minn:
                st.append((p, ent))
    elif kind == 'parainit':  # windows of 20 consecutive paragraph-initial words within a section
        cur, sec = [], None
        heads = [(L['section'], L['page'], L['words'][0]) for L in lines if L['para_start']]
        runs = defaultdict(list)
        seq = []
        for s, p, w in heads:
            if s != sec:
                seq.append([]); sec = s
            seq[-1].append((p, w))
        for r in seq:
            for k in range(0, len(r), 20):
                chunk = r[k:k + 20]
                if len(chunk) >= minn:
                    st.append(('%s..%s' % (chunk[0][0], chunk[-1][0]), [(0, bi, w) for bi, (p, w) in enumerate(chunk)]))
    elif kind == 'lines':
        for i, L in enumerate(lines):
            ws = L['words']
            if len(ws) >= 4:
                st.append(('%s#%d' % (L['page'], i), [(0, 0, w) for w in ws]))
    return st


def encode(stretches, symtab=None):
    if symtab is None:
        symtab = {'#': 0}
    for _, ent in stretches:
        for _, _, w in ent:
            for g in w:
                if g not in symtab:
                    symtab[g] = len(symtab)
    return symtab


def write_input(path, stretches, symtab):
    with open(path, 'w') as f:
        f.write('NSYM %d\n' % len(symtab))
        for sid, ent in stretches:
            f.write('STRETCH %s %d\n' % (sid.replace(' ', '_'), len(ent)))
            for c, b, w in ent:
                f.write('%d %d %d %s\n' % (c, b, len(w), ' '.join(str(symtab[g]) for g in w)))

KEYS = {'F': (1, 0), 'L2': (2, 0), 'L': (0, 0), 'R1': (1, 1), 'R': (0, 1)}


def run_engine(stretches, symtab, key, R=100, nullmode=0, pooled=0, restarts=6, ils=30, nrand=2000,
               seed=1, fixed=None, tag='x'):
    build_engine()
    inp = os.path.join(SCR, 'in_%s_%d.txt' % (tag, os.getpid()))
    write_input(inp, stretches, symtab)
    depth, rev = KEYS[key]
    env = dict(os.environ)
    if fixed is not None:
        env['V19_FIXED'] = ','.join(str(x) for x in fixed)
    else:
        env.pop('V19_FIXED', None)
    out = subprocess.run([ENGINE, str(depth), str(rev), str(R), str(nullmode), str(pooled), str(restarts),
                          str(ils), str(nrand), str(seed)], stdin=open(inp), capture_output=True, text=True, env=env)
    os.remove(inp)
    if out.returncode:
        raise RuntimeError(out.stderr)
    inv = {v: k for k, v in symtab.items()}
    rows = []
    for L in out.stdout.strip().split('\n'):
        if not L:
            continue
        f = L.split()
        d = dict(id=f[0], n=int(f[1]), T=float(f[2]), tau=float(f[3]), nmean=float(f[4]), nsd=float(f[5]),
                 nmax=float(f[6]), p=float(f[7]), randmax=float(f[8]),
                 order=[int(x) for x in f[9].split(',')] if len(f) > 9 else [])
        d['z'] = (d['tau'] - d['nmean']) / d['nsd'] if d['nsd'] > 0 else 0.0
        d['order_s'] = ' '.join(inv[x] for x in d['order'])
        rows.append(d)
    return rows


def order_agreement(learned, true_rank):
    """Kendall tau between a learned order (list of symbols) and true ranks, on shared symbols."""
    s = [g for g in learned if g in true_rank]
    c = d = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if true_rank[s[i]] < true_rank[s[j]]:
                c += 1
            else:
                d += 1
    return (c - d) / max(1, c + d)


def sortkey_fn(rank, depth=0, rev=False):
    def k(w):
        w = w[::-1] if rev else w
        if depth:
            w = w[:depth]
        return [rank.get(g, 99) + 1 for g in w] + [0]
    return k


def bh(ps, q=0.05):
    n = len(ps)
    o = sorted(range(n), key=lambda i: ps[i])
    k = 0
    for r, i in enumerate(o, 1):
        if ps[i] <= q * r / n:
            k = r
    return set(o[:k])
