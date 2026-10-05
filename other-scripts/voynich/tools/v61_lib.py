"""v61: undo the sandhi. Shared code.

A corpus is a list of lines; each line is a dict with
  words : list of surface words (strings of one-char glyph units)
  base  : list of true base forms (controls only) or None
  marked: list of bools, True where the source marks the word as altered (controls only)
  page, sec : page id and page-level variable
  para_start, para_end : bools
Words are strings of single characters (Voynich: vlib glyph units mapped to one char;
controls: letters remapped to opaque symbols).
"""
import json, math, os, random, re, sys, unicodedata, glob
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v61_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
import vlib

LOOPS = os.path.join(ROOT, 'loops')


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write('| %s | %s | %s | %s |\n' % (rid, method, result, verdict))


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'))


def jload(name):
    return json.load(open(os.path.join(CK, name)))

# ---------------------------------------------------------------- Voynich

GL = {'cth': 'T', 'ckh': 'K', 'cph': 'P', 'cfh': 'F', 'ch': 'C', 'sh': 'S'}


def vglyph(w):
    for a, b in GL.items():
        w = w.replace(a, b)
    return w


def unglyph(s):
    inv = {v: k for k, v in GL.items()}
    return ''.join(inv.get(c, c) for c in s)


def load_vms(name='ZL3b'):
    recs = vlib.load_voynich(name, ltypes=('P',))
    out = []
    for r in recs:
        ws = [vglyph(w) for w in r['words'] if re.fullmatch(r'[a-z]+', w)]
        if not ws:
            continue
        out.append({'words': ws, 'base': None, 'marked': None, 'page': r['folio'],
                    'sec': r['illus'], 'lang': r['lang'] or '?', 'hand': r['hand'] or '?',
                    'para_start': r['para_start'], 'para_end': r['para_end']})
    return out

# ---------------------------------------------------------------- controls

def opaque(lines, seed=61):
    """Remap every character (surface and base) to an opaque symbol; return lines, key."""
    chars = sorted({c for L in lines for w in L['words'] + (L['base'] or []) for c in w})
    rng = random.Random(seed)
    syms = [chr(0x4E00 + i) for i in range(len(chars))]
    rng.shuffle(syms)
    key = dict(zip(chars, syms))
    out = []
    for L in lines:
        M = dict(L)
        M['words'] = [''.join(key[c] for c in w) for w in L['words']]
        if L['base'] is not None:
            M['base'] = [''.join(key[c] for c in w) for w in L['base']]
        out.append(M)
    return out, key


def load_sanskrit(max_tokens=36000, seed=1):
    """Ramayana from the DCS: surface = '# text' tokens (sandhied), base = joined
    'Unsandhied' forms of the parts. Line = half-verse; page = chapter; sec = kanda."""
    files = sorted(glob.glob(os.path.join(CK, 'dcs/dcs/data/conllu/files/R*/*.conllu')))
    rng = random.Random(seed)
    rng.shuffle(files)
    lines, ntok = [], 0
    for fn in files:
        chap = None
        cur = None
        for raw in open(fn, encoding='utf-8'):
            raw = raw.rstrip('\n')
            if raw.startswith('## chapter:'):
                chap = raw.split(':', 1)[1].strip()
            elif raw.startswith('# text ='):
                cur = {'text': raw.split('=', 1)[1].strip(), 'toks': []}
            elif raw and raw[0].isdigit() and cur is not None:
                f = raw.split('\t')
                cur['toks'].append(f)
            elif not raw and cur is not None:
                L = _skt_line(cur, chap)
                if L:
                    lines.append(L); ntok += len(L['words'])
                cur = None
        if ntok >= max_tokens:
            break
    # paragraph = chapter: mark starts/ends
    for i, L in enumerate(lines):
        L['para_start'] = i == 0 or lines[i - 1]['page'] != L['page']
        L['para_end'] = i == len(lines) - 1 or lines[i + 1]['page'] != L['page']
    return lines


def _clean_skt(w):
    w = unicodedata.normalize('NFC', w.lower())
    return ''.join(c for c in w if c.isalpha() or unicodedata.category(c) == 'Mn')


def _edge_only(s, b, ke=3, ks=2):
    """True if s and b differ only within the last ke and first ks characters."""
    if s == b:
        return True
    i = 0
    while i < min(len(s), len(b)) and s[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(s), len(b)) - i and s[-1 - j] == b[-1 - j]:
        j += 1
    # differing core: s[i:len(s)-j] vs b[i:len(b)-j]
    return (i <= ks and len(s) - j <= ks + 1) or (len(s) - i <= ke and len(b) - i <= ke) or (i <= ks and len(b) - j <= ks + 1)


def _skt_line(cur, chap):
    surf = cur['text'].split()
    toks = cur['toks']
    # group tokens into surface units: range rows a-b cover parts a..b
    units, i = [], 0
    while i < len(toks):
        f = toks[i]
        if '-' in f[0]:
            a, b = map(int, f[0].split('-'))
            parts = []
            j = i + 1
            while j < len(toks) and '-' not in toks[j][0] and '.' not in toks[j][0] and int(toks[j][0]) <= b:
                parts.append(toks[j]); j += 1
            units.append((f[1], parts)); i = j
        elif '.' in f[0]:
            i += 1
        else:
            units.append((f[1], [f])); i += 1
    if len(units) != len(surf):
        return None
    words, base, marked = [], [], []
    for (form, parts), s in zip(units, surf):
        un = []
        for p in parts:
            m = re.search(r'Unsandhied=([^|]+)', p[9])
            un.append(m.group(1) if m else p[1])
        un = [re.sub(r'[sr]$', 'ḥ', u) for u in un]   # pausa form of final s/r
        sw, bw = _clean_skt(s), _clean_skt(''.join(un))
        if not sw or not bw:
            return None
        if not _edge_only(sw, bw):
            bw = sw   # lexical replacement (me/mama, tava/te...), not an edge process
        words.append(sw); base.append(bw); marked.append(sw != bw)
    kanda = chap.split(',')[1].strip() if chap and ',' in chap else '?'
    return {'words': words, 'base': base, 'marked': marked, 'page': chap, 'sec': kanda,
            'lang': '?', 'hand': '?'}


SOFT = {'p': 'b', 't': 'd', 'c': 'g', 'b': 'f', 'd': 'dd', 'g': '', 'll': 'l', 'm': 'f', 'rh': 'r'}
NASAL = {'p': 'mh', 't': 'nh', 'c': 'ngh', 'b': 'm', 'd': 'n', 'g': 'ng'}
ASP = {'p': 'ph', 't': 'th', 'c': 'ch'}


def _demutate(form, lemma):
    """Return (base, mutated?) for a Welsh form given its lemma."""
    f, l = form.lower(), lemma.lower()
    if not f or not l or f[0] == l[0] and not (l.startswith('ll') and not f.startswith('ll')) and not (l.startswith('rh') and not f.startswith('rh')):
        # h-prothesis before vowel-initial lemma
        if f.startswith('h') and l[:1] in 'aeiouwy' and f[1:3] == l[:2]:
            return f[1:], True
        return f, False
    for tab in (NASAL, ASP, SOFT):
        for o, m in sorted(tab.items(), key=lambda x: -len(x[1])):
            if l.startswith(o) and f.startswith(m):
                rest_f, rest_l = f[len(m):], l[len(o):]
                if rest_f[:2] == rest_l[:2] and (rest_f or not rest_l):
                    return o + rest_f, True
    return f, False


def load_welsh():
    lines = []
    for fn in ('cy_train.conllu', 'cy_test.conllu'):
        doc, cur = 'd0', []
        for raw in open(os.path.join(CK, fn), encoding='utf-8'):
            raw = raw.rstrip('\n')
            if raw.startswith('# newdoc id'):
                doc = raw.split('=', 1)[1].strip()
            elif raw and raw[0].isdigit():
                f = raw.split('\t')
                if '-' in f[0] or '.' in f[0]:
                    continue
                cur.append(f)
            elif not raw and cur:
                words, base, marked = [], [], []
                for f in cur:
                    w = f[1].lower()
                    if not re.fullmatch(r"[a-zâêîôûŵŷäëïöüẁẃẅỳýáéíóú]+", w):
                        continue
                    b, m = _demutate(w, f[2] if f[2] != '_' else w)
                    words.append(w); base.append(b); marked.append(m)
                if len(words) >= 2:
                    lines.append({'words': words, 'base': base, 'marked': marked, 'page': fn + doc,
                                  'sec': doc, 'lang': '?', 'hand': '?'})
                cur = []
    for i, L in enumerate(lines):
        L['para_start'] = i == 0 or lines[i - 1]['page'] != L['page']
        L['para_end'] = i == len(lines) - 1 or lines[i + 1]['page'] != L['page']
    return lines


IT_BASE = {'ch': 'che', 'com': 'come', 'd': 'di', 'ond': 'onde', 'quand': 'quando', 'diss': 'disse',
           'perch': 'perche', 'quell': 'quello', 'quest': 'questo', 'dov': 'dove', 's': 'si',
           'n': 'ne', 'm': 'mi', 't': 'ti', 'v': 'vi', 'c': 'ci', 'un': 'una', 'l': 'lo', 'gli': 'gli',
           'dell': 'dello', 'nell': 'nello', 'all': 'allo', 'sull': 'sullo', 'dall': 'dallo', 'coll': 'collo',
           'tutt': 'tutto', 'i': 'io', 'se': 'sei', 'de': 'dei', 'a': 'ai', 'e': 'ei', 'co': 'coi', 'ne': 'nei', 'que': 'quei', 'grand': 'grande', 'sant': 'santo', 'senz': 'senza', 'anch': 'anche', 'ov': 'ove'}


def load_italian(max_tokens=36000):
    """Dante, Commedia. Elided words (ending or starting with an apostrophe) are split off.
    Line = verse; paragraph = tercet block; page = canto; sec = cantica."""
    txt = open(os.path.join(DATA, 'pg1000.txt'), encoding='utf-8').read().replace('\r', '')
    m1 = re.search(r'\*\*\* ?START[^\n]*\n', txt); m2 = re.search(r'\*\*\* ?END', txt)
    txt = txt[m1.end():m2.start()]
    lines, cant, canto, prev_blank, ntok = [], '?', None, True, 0
    for raw in txt.split('\n'):
        s = raw.strip()
        if re.match(r'^(Inferno|Purgatorio|Paradiso)\b', s):
            cant = s.split()[0]; continue
        if re.match(r'^Canto [IVXLC]+', s) or re.match(r'^\s*(INFERNO|PURGATORIO|PARADISO)', s):
            canto = cant + ' ' + s; prev_blank = True; continue
        if not s:
            if lines:
                lines[-1]['para_end'] = True
            prev_blank = True; continue
        if canto is None:
            continue
        s = s.replace("'", '’')
        toks = re.findall(r"’?[A-Za-zàèéìíòóùÀÈÉÌÒÙ]+’?", s)
        words, base, marked = [], [], []
        for t in toks:
            core = t.strip('’').lower()
            core = unicodedata.normalize('NFD', core)
            core = ''.join(c for c in core if unicodedata.category(c) != 'Mn')
            if not core:
                continue
            mk = '’' in t
            if t.endswith('’'):
                b = IT_BASE.get(core, core + 'e')
            elif t.startswith('’'):
                b = {'l': 'il', 'n': 'in', 'mpero': 'impero'}.get(core, 'i' + core)
            else:
                b = core
            words.append(core); base.append(b); marked.append(mk)
        if not words:
            continue
        lines.append({'words': words, 'base': base, 'marked': marked, 'page': canto, 'sec': cant,
                      'lang': '?', 'hand': '?', 'para_start': prev_blank, 'para_end': False})
        prev_blank = False
    if lines:
        lines[-1]['para_end'] = True
    # keep every k-th canto so all three cantiche are present
    cantos = []
    for L in lines:
        if not cantos or cantos[-1] != L['page']:
            cantos.append(L['page'])
    ntot = sum(len(L['words']) for L in lines)
    k = max(1, round(ntot / max_tokens))
    keep = set(cantos[::k])
    return [L for L in lines if L['page'] in keep]

# ---------------------------------------------------------------- nulls / generators

def within_line_shuffle(lines, seed=1):
    rng = random.Random(seed)
    out = []
    for L in lines:
        M = dict(L); idx = list(range(len(L['words']))); rng.shuffle(idx)
        M['words'] = [L['words'][i] for i in idx]
        if L.get('base'):
            M['base'] = [L['base'][i] for i in idx]; M['marked'] = [L['marked'][i] for i in idx]
        out.append(M)
    return out


def pair_resample(lines, seed=1, by_page=True, line_blind_start=False):
    """Coupling planted by a generator, no sandhi: each word is a real token, drawn from
    tokens (same page if possible) whose real predecessor ended in the same glyph as the
    current predecessor. Line-blind (line ends are not known to the generator)."""
    rng = random.Random(seed)
    pool = defaultdict(list); poolg = defaultdict(list); first_pool = defaultdict(list)
    for L in lines:
        ws = L['words']
        first_pool[L['page']].append(ws[0])
        for a, b in zip(ws, ws[1:]):
            pool[(L['page'], a[-1])].append(b); poolg[a[-1]].append(b)
    out = []
    for L in lines:
        M = dict(L); ws = []
        for i in range(len(L['words'])):
            if i == 0:
                if line_blind_start and out and out[-1]['page'] == L['page']:
                    prevg = out[-1]['words'][-1][-1]
                    cand = pool.get((L['page'], prevg)) or poolg.get(prevg)
                    ws.append(rng.choice(cand)); continue
                fp = first_pool[L['page']]
                ws.append(rng.choice(fp)); continue
            k = (L['page'], ws[-1][-1])
            cand = pool.get(k) if by_page else None
            if not cand or len(cand) < 3:
                cand = poolg.get(ws[-1][-1]) or [w for w in L['words']]
            ws.append(rng.choice(cand))
        M['words'] = ws; M['base'] = None; M['marked'] = None
        out.append(M)
    return out


def self_citation(lines, seed=1, window=60, p_mod=0.5):
    sys.path.insert(0, HERE)
    import gen
    L2 = [dict(L) for L in lines]
    out = gen.self_citation(L2, seed=seed, window=window, p_mod=p_mod)
    for M in out:
        M['base'] = None; M['marked'] = None
    return out


def plant_sandhi(lines, rules, seed=1, prob=1.0):
    """Apply regressive rules (S_base_end, S_surf_end, ctx_set) to non-line-final words.
    base = original word, surface = rewritten."""
    rng = random.Random(seed)
    out = []
    for L in lines:
        ws = L['words']; nw, mk = [], []
        for i, w in enumerate(ws):
            new = w
            if i + 1 < len(ws):
                for b, s, ctx in rules:
                    if w.endswith(b) and len(w) > len(b) and ws[i + 1][0] in ctx and rng.random() < prob:
                        new = w[:len(w) - len(b)] + s; break
            nw.append(new); mk.append(new != w)
        M = dict(L); M['words'] = nw; M['base'] = list(ws); M['marked'] = mk
        out.append(M)
    return out


def reverse_text(lines):
    """Progressive edge processes become regressive on the reversed text."""
    out = []
    for L in reversed(lines):
        M = dict(L)
        M['words'] = [w[::-1] for w in reversed(L['words'])]
        if L.get('base'):
            M['base'] = [w[::-1] for w in reversed(L['base'])]
            M['marked'] = list(reversed(L['marked']))
        M['para_start'], M['para_end'] = L['para_end'], L['para_start']
        out.append(M)
    return out

# ---------------------------------------------------------------- tokens

def tokens(lines):
    """Flatten to token records with neighbour info.
    nxt: next word in line or None; fin: 'L' (line end inside paragraph), 'P' (paragraph end) or None;
    xnext: first word of next line in the same paragraph (for line-final tokens)."""
    T = []
    for li, L in enumerate(lines):
        ws = L['words']; n = len(ws)
        nxl = None
        if not L['para_end'] and li + 1 < len(lines) and lines[li + 1]['page'] == L['page'] and not lines[li + 1]['para_start']:
            nxl = lines[li + 1]['words'][0]
        for i, w in enumerate(ws):
            fin = None if i < n - 1 else ('P' if L['para_end'] else 'L')
            T.append({'w': w, 'nxt': ws[i + 1] if i < n - 1 else None, 'fin': fin,
                      'xnext': nxl if fin else None, 'page': L['page'], 'sec': L['sec'],
                      'line': li, 'pos': i, 'n': n,
                      'base': L['base'][i] if L.get('base') else None,
                      'marked': L['marked'][i] if L.get('marked') else None})
    return T


def page_split(lines, seed=0):
    pages = sorted({L['page'] for L in lines})
    rng = random.Random(seed); rng.shuffle(pages)
    return set(pages[::2])

# ---------------------------------------------------------------- the scan

def scan(lines, K=3, min_tok=15, n_random_classes=4000, seed=0, train_pages=None, keep=60,
         min_stems=2):
    """Exhaustive (S,B) ending pairs x context classes on the next word's first glyph.
    For each pair, the best class on train pages; reported with held-out (test pages) z,
    the line-end test and the cross-line test. Returns (records, n_hypotheses)."""
    rng = np.random.default_rng(seed)
    T = tokens(lines)
    if train_pages is None:
        train_pages = page_split(lines, seed)
    glyphs = sorted({c for t in T for c in t['w']})
    gi = {g: i for i, g in enumerate(glyphs)}; G = len(glyphs)
    types = sorted({t['w'] for t in T}); ti = {w: i for i, w in enumerate(types)}; V = len(types)
    # per-type count arrays: next first glyph (train/test), finals by kind, cross-line next glyph
    NX = np.zeros((2, V, G)); FL = np.zeros((V, 2)); XN = np.zeros((V, G))
    for t in T:
        v = ti[t['w']]
        if t['nxt'] is not None:
            NX[0 if t['page'] in train_pages else 1, v, gi[t['nxt'][0]]] += 1
        else:
            FL[v, 0 if t['fin'] == 'L' else 1] += 1
            if t['xnext']:
                XN[v, gi[t['xnext'][0]]] += 1
    # stems
    stem_end = defaultdict(dict)
    for w in types:
        for k in range(0, K + 1):
            if len(w) - k < 1:
                break
            stem_end[w[:len(w) - k]][w[len(w) - k:]] = ti[w]
    pairs = defaultdict(list)
    for st, ends in stem_end.items():
        if len(ends) < 2:
            continue
        es = list(ends)
        for a in es:
            for b in es:
                if a != b:
                    pairs[(a, b)].append((ends[a], ends[b]))
    # classes: singles, plus random subsets
    cls = [np.eye(G)[i] for i in range(G)]
    names = [glyphs[i] for i in range(G)]
    for _ in range(n_random_classes):
        k = rng.integers(2, max(3, G // 2 + 1))
        idx = rng.choice(G, size=k, replace=False)
        m = np.zeros(G); m[idx] = 1; cls.append(m); names.append(''.join(sorted(glyphs[i] for i in idx)))
    M = np.array(cls)  # C x G
    recs = []; nh = 0
    for (S, B), lst in pairs.items():
        if len(lst) < min_stems:
            continue
        sI = np.array([x[0] for x in lst]); bI = np.array([x[1] for x in lst])
        cS = NX[:, sI, :].sum(1); cB = NX[:, bI, :].sum(1)   # 2 x G
        if min(cS[0].sum(), cB[0].sum(), cS[1].sum(), cB[1].sum()) < min_tok:
            continue
        a = M @ cS[0]; b = cS[0].sum() - a; c = M @ cB[0]; d = cB[0].sum() - c
        lor = np.log((a + .5) * (d + .5) / ((b + .5) * (c + .5)))
        se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
        z = lor / se; nh += len(z)
        j = int(np.argmax(z))   # S enriched in class j  (the complement case is the (B,S) pair)
        m = M[j]
        at = m @ cS[1]; bt = cS[1].sum() - at; ct = m @ cB[1]; dt = cB[1].sum() - ct
        lt = math.log((at + .5) * (dt + .5) / ((bt + .5) * (ct + .5)))
        zt = lt / math.sqrt(1 / (at + .5) + 1 / (bt + .5) + 1 / (ct + .5) + 1 / (dt + .5))
        # all-data counts for the line-end test
        A_ = a[j] + at; B_ = b[j] + bt; C_ = c[j] + ct; D_ = d[j] + dt
        fS = FL[sI].sum(0); fB = FL[bI].sum(0)
        xs = XN[sI].sum(0); xb = XN[bI].sum(0)
        recs.append({'S': S, 'B': B, 'cls': names[j], 'nstem': len(lst), 'z_train': float(z[j]),
                     'lor_train': float(lor[j]), 'z_test': float(zt), 'lor_test': float(lt),
                     'abcd': [float(A_), float(B_), float(C_), float(D_)],
                     'finL': [float(fS[0]), float(fB[0])], 'finP': [float(fS[1]), float(fB[1])],
                     'xline': [float(m @ xs), float(xs.sum() - m @ xs), float(m @ xb), float(xb.sum() - m @ xb)]})
    recs.sort(key=lambda r: -r['z_train'])
    return recs, nh, {'G': G, 'V': V, 'pairs': len(pairs)}


def line_end_llr(r, kind='finL'):
    """Which context does the line end behave like? log-lik of final S/B counts under
    p = share of S in the trigger class (C), outside it (notC), or overall (blind)."""
    A, B, C, D = r['abcd']
    fS, fB = r[kind] if kind != 'both' else (r['finL'][0] + r['finP'][0], r['finL'][1] + r['finP'][1])
    def ll(p):
        p = min(max(p, 1e-3), 1 - 1e-3)
        return fS * math.log(p) + fB * math.log(1 - p)
    pC = (A + .5) / (A + C + 1); pN = (B + .5) / (B + D + 1); pA = (A + B + .5) / (A + B + C + D + 1)
    return {'n': fS + fB, 'llC': ll(pC) - ll(pA), 'llN': ll(pN) - ll(pA),
            'pS_C': pC, 'pS_N': pN, 'pS_all': pA, 'pS_fin': (fS + .5) / (fS + fB + 1)}


def xline_z(r):
    a, b, c, d = r['xline']
    lor = math.log((a + .5) * (d + .5) / ((b + .5) * (c + .5)))
    return lor / math.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))

# ---------------------------------------------------------------- evaluation helpers

def coupling_mi(lines):
    """MI (bits) between last glyph of a word and first glyph of the next, within lines."""
    c = Counter()
    for L in lines:
        for a, b in zip(L['words'], L['words'][1:]):
            c[(a[-1], b[0])] += 1
    n = sum(c.values()); ca = Counter(); cb = Counter()
    for (a, b), v in c.items():
        ca[a] += v; cb[b] += v
    return sum(v / n * math.log2(v * n / (ca[a] * cb[b])) for (a, b), v in c.items())


def apply_rules(lines, rules):
    """Undo: rules = list of (S, B, ctxset). A non-final word ending in S whose stem+B is a
    known type (in the lexicon of the input) and whose next word starts with a glyph in ctxset
    is rewritten to stem+B. Returns new lines and the number of rewrites."""
    lex = {w for L in lines for w in L['words']}
    out, n = [], 0
    for L in lines:
        ws = L['words']; nw = []
        for i, w in enumerate(ws):
            new = w
            if i + 1 < len(ws):
                for S, B, ctx in rules:
                    if (w.endswith(S) if S else True) and len(w) > len(S) and ws[i + 1][0] in ctx:
                        cand = w[:len(w) - len(S)] + B
                        if cand in lex:
                            new = cand; break
            n += new != w; nw.append(new)
        M = dict(L); M['words'] = nw; out.append(M)
    return out, n


def nb_page_score(lines, var='sec', seed=0, reps=4, alpha=0.5, min_count=1):
    """Held-out page classification: multinomial NB on word tokens, train on half the pages,
    mean log2-prob of the true class per held-out page and accuracy. Averaged over splits."""
    pages = defaultdict(list); lab = {}
    for L in lines:
        pages[L['page']].extend(L['words']); lab[L['page']] = L[var]
    P = sorted(pages)
    res = []
    for r in range(reps):
        rng = random.Random(seed * 100 + r); Q = P[:]; rng.shuffle(Q)
        tr, te = Q[::2], Q[1::2]
        cc = defaultdict(Counter); prior = Counter()
        for p in tr:
            cc[lab[p]].update(pages[p]); prior[lab[p]] += 1
        classes = sorted(cc)
        vocab = set(w for c in cc.values() for w in c)
        tot = {c: sum(cc[c].values()) for c in classes}
        lp_sum, acc, n = 0.0, 0, 0
        for p in te:
            if lab[p] not in cc:
                continue
            sc = {}
            for c in classes:
                s = math.log(prior[c] / len(tr))
                for w in pages[p]:
                    if w in vocab:
                        s += math.log((cc[c][w] + alpha) / (tot[c] + alpha * len(vocab)))
                sc[c] = s
            mx = max(sc.values()); Z = mx + math.log(sum(math.exp(v - mx) for v in sc.values()))
            lp_sum += (sc[lab[p]] - Z) / math.log(2); acc += max(sc, key=sc.get) == lab[p]; n += 1
        if n:
            res.append((lp_sum / n, acc / n))
    return float(np.mean([x[0] for x in res])), float(np.mean([x[1] for x in res]))


def vocab_stats(lines):
    ws = [w for L in lines for w in L['words']]
    c = Counter(ws)
    adj = sum(1 for L in lines for a, b in zip(L['words'], L['words'][1:]) if a == b)
    return {'tokens': len(ws), 'types': len(c), 'hapax': sum(1 for v in c.values() if v == 1),
            'zipf': vlib.zipf_slope(ws), 'adj_repeat_per_1k': 1000 * adj / max(1, len(ws)),
            'top10': [w for w, _ in c.most_common(10)]}
