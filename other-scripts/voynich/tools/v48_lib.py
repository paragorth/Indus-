"""v48 CLEAN THE SCRIPT, THEN TYPE THE LANGUAGE: shared code.

Corpus = dict(name, fam, docs=[dict(tag, lines=[[word, ...], ...])]); word = tuple of unit strings.
tag = 'A' / 'B' / None (Currier language for the Voynich; first / second half of the documents for controls).

(1) Normalisers
  explicit Voynich ladder (EVA string level, before the unit inventory is applied):
      E0 raw;  E1 gallows twins k=t, f=p (so ckh=cth, cfh=cph);  E2 + ch=sh;  E3 + benched=plain (cth=t, cph=p)
  blind normalisers, applied identically to the Voynich and to planted control corpora:
      merge_blind  greedy merge of unit pairs whose cross-half context-interchangeability index (v35/v39 index:
                   PPMI of L1 R1 L2 R2 contexts, alternating 400-word halves) is >= THR
      ab_blind     learns <= 8 ending and <= 4 beginning rewrites B -> A by stem-set overlap of tag-skewed
                   word edges, then rewrites the B documents
(2) Unit inventories for the Voynich: EVA characters, GLY (v21 glyph units), RUN (GLY + i-runs and e-runs as single
    units), STR (v25 stroke primitives, fixed order inside a glyph).
(3) Fingerprint: alphabet-free typological features per 8,000-token window (mean over 3 evenly spaced windows).
(4) Plants: positional twins (line-first / paragraph-first + decaying habit, like ch/sh and k/t), a neighbour-
    conditioned bench variant, and an A/B ending + beginning rewrite on the second half.  Nulls: global unit
    shuffle (word lengths kept) and a unit Markov-2 resynthesis.
"""
import os, sys, json, math, random, re, zlib, unicodedata, glob
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'v48_ckpt'); os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
RAW = os.path.join(SCRATCH, 'v48', 'raw')
GASK = os.path.join(SCRATCH, 'v31', 'gaskell', 'data', 'meaningful', 'texts')
WIN, NWIN, CAP = 8000, 3, 30000
LINE_W = 8


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ================================================================== Voynich
V_MULTI = [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]
EXPLICIT = {
    'E0': [],
    'E1': [('k', 't'), ('f', 'p')],
    'E2': [('k', 't'), ('f', 'p'), ('sh', 'ch')],
    'E3': [('k', 't'), ('f', 'p'), ('sh', 'ch'), ('cth', 't'), ('cph', 'p')],
}


def _strokes():
    sys.path.insert(0, HERE)
    import v25_shapes as S
    out = {}
    for g, d in S.VOYNICH.items():
        out[g] = tuple(p for p in S.V_PRIMS for _ in range(d.get(p, 0)))
    return out


_STR = None


def eva_units(w, inv):
    global _STR
    if inv == 'EVA':
        return tuple(w)
    g = w
    for a, b in V_MULTI:
        g = g.replace(a, b)
    g = list(g)
    if inv == 'GLY':
        return tuple(g)
    if inv == 'RUN':
        out, k = [], 0
        while k < len(g):
            c = g[k]
            if c in 'ie':
                j = k
                while j < len(g) and g[j] == c: j += 1
                out.append(c * (j - k)); k = j
            else:
                out.append(c); k += 1
        return tuple(out)
    if inv == 'STR':
        if _STR is None: _STR = _strokes()
        back = {b: a for a, b in V_MULTI}
        out = []
        for c in g:
            out.extend(_STR.get(back.get(c, c), ('X' + c,)))
        return tuple(out)
    raise ValueError(inv)


def voynich(src='ZL3b', norm='E0', inv='GLY'):
    recs = json.load(open(os.path.join(DATA, 'derived', f'{src}_lines.json')))
    pages, order = {}, []
    for r in recs:
        if r['ltype'] != 'P' or not r['words']: continue
        f = r['folio']
        if f not in pages:
            pages[f] = dict(tag=r.get('lang') if r.get('lang') in ('A', 'B') else None, lines=[], id=f,
                            hand=r.get('hand'), sect=r.get('illus')); order.append(f)
        ws = []
        for w, u in zip(r['words'], r['uncertain']):
            if u or not re.fullmatch('[a-z]+', w): continue
            for a, b in EXPLICIT[norm]:
                w = w.replace(a, b)
            ws.append(eva_units(w, inv))
        if ws: pages[f]['lines'].append(ws)
    docs = [pages[f] for f in order if pages[f]['lines']]
    return dict(name=f'VOY_{src}_{norm}_{inv}', fam='Voynich', docs=docs)


# ================================================================== natural-language corpora
STRIP_SCRIPTS = ('GREEK', 'HEBREW', 'ARABIC', 'COPTIC', 'SYRIAC', 'ARMENIAN', 'GEORGIAN', 'CYRILLIC')
SPECIAL = {'ſ': 's', 'ß': 'ss', 'æ': 'ae', 'œ': 'oe', 'ꝰ': 'us', 'ꝛ': 'r', 'ƿ': 'w', 'þ': 'th', 'ð': 'd'}


TOKRE = re.compile(r"[^\s\d_.,;:!?\"'()\[\]{}«»“”„‘’‹›\-–—/\\|*<>=+#%&@~`^$§·؟،؛۔।॥0-9]+")


def words_of(line, strip_all=False):
    s = line.lower()
    s = ''.join(SPECIAL.get(c, c) for c in s)
    s = unicodedata.normalize('NFC', s)
    out = []
    for w in TOKRE.findall(s):
        c0 = next((c for c in w if c.isalpha()), None)
        if c0 is None: continue
        nm = unicodedata.name(c0, '')
        if strip_all or any(n in nm for n in STRIP_SCRIPTS) or 'LATIN' in nm:
            w = unicodedata.normalize('NFD', w) if (strip_all or 'LATIN' not in nm) else w
            w = ''.join(c for c in w if c.isalpha())
            w = unicodedata.normalize('NFC', w)
        else:   # abugidas: keep letters and vowel signs / viramas as units
            w = ''.join(c for c in w if unicodedata.category(c)[0] in 'LM')
        if w: out.append(tuple(w))
    return out


def build(name, fam, raw_docs, strip_all=False, cap=CAP, wrap=LINE_W):
    """raw_docs: list of documents, each a list of text lines; re-wrapped into lines of `wrap` words."""
    allw = [[w for l in d for w in words_of(l, strip_all)] for d in raw_docs[:600]]
    uc = Counter(u for d in allw for w in d for u in w); tot = sum(uc.values())
    scr = unicodedata.name(uc.most_common(1)[0][0], 'X').split()[0]
    ok = {u for u, c in uc.items() if c >= 0.0005 * tot and unicodedata.name(u, 'X').split()[0] == scr}
    docs, n = [], 0
    for d in raw_docs:
        ws = [w for l in d for w in words_of(l, strip_all)]
        ws = [w for w in ws if all(u in ok for u in w)]
        if len(ws) < 40: continue
        lines = [ws[i:i + wrap] for i in range(0, len(ws), wrap)]
        docs.append(dict(tag=None, lines=lines)); n += len(ws)
        if n >= cap: break
    h = len(docs) // 2
    for i, d in enumerate(docs):
        d['tag'] = 'A' if i < h else 'B'
    return dict(name=name, fam=fam, docs=docs)


def _chunks(text, size=300):
    ws = text.split()
    return [[' '.join(ws[i:i + size])] for i in range(0, len(ws), size)]


EB = {'lat': 'Latin', 'ita': 'Romance', 'deu': 'Germanic', 'ces': 'Slavic', 'pol': 'Slavic', 'hun': 'Uralic',
      'grc': 'Greek', 'heb': 'Semitic', 'hbo': 'Semitic', 'arb': 'Semitic', 'tur': 'Turkic', 'rom': 'IndoAryan',
      'spa': 'Romance', 'por': 'Romance', 'fra': 'Romance', 'ron': 'Romance', 'cop': 'Egyptian', 'rus': 'Slavic',
      'srp': 'Slavic', 'slk': 'Slavic', 'nld': 'Germanic', 'eng': 'Germanic', 'dan': 'Germanic', 'fin': 'Uralic',
      'lit': 'Baltic', 'est': 'Uralic', 'bre': 'Celtic', 'pes': 'Iranian', 'ydd': 'Germanic', 'aii': 'Semitic',
      'hrv': 'Slavic', 'rmy': 'IndoAryan'}
WK = {'eu': 'Basque', 'ca': 'Romance', 'oc': 'Romance', 'ka': 'Kartvelian', 'sq': 'Albanian', 'mt': 'Semitic',
      'la': 'Latin'}
GK = {'pinyin': ('Modern - Chinese (Pinyin) - Literary - NT - Matthew.txt', 'Sinitic'),
      'quran': ('Historical - Arabic - Literary - Quran.txt', 'Semitic'),
      'avicenna': ('Historical - Arabic - Technical - Avicenna.txt', 'Semitic'),
      'sanskrit': ('Historical - Sanskrit - Literary - Mahabharata.txt', 'IndoAryan'),
      'nahuatl': ('Historical - Nahuatl - Technical - Florentine Codex.txt', 'Nahuan'),
      'deherb': ('Historical - German - Technical - German Herbarium.txt', 'Germanic'),
      'spamed': ('Historical - Spanish - Technical - De Materia Medica.txt', 'Romance'),
      'grcodor': ('Historical - Greek - Technical - De odoribus.txt', 'Greek'),
      'ang': ('Historical - Anglo-Saxon - Technical - Leechbook.txt', 'Germanic'),
      'plinyabbr': ('Historical - Latin (Abbreviated) - Technical - Pliny\'s Natural History.txt', 'Latin'),
      'flemherb': ('Historical - Flemish - Technical - Cruydeboeck.txt', 'Germanic')}
V30 = {'G_Bav1': 'Germanic', 'G_Alem': 'Germanic', 'G_Rip': 'Germanic', 'I_Ita': 'Romance', 'I_Lat': 'Latin',
       'C_Old': 'Slavic', 'I_Com1': 'Romance'}
GUT = {'pg218': ('Latin', 'caesar'), 'pg1000': ('Romance', 'dante'), 'pg45334': ('Romance', 'manzoni'),
       'pg22367': ('Germanic', 'kafka'), 'pg2000': ('Romance', 'cervantes')}


def all_languages():
    out = []
    for c, fam in EB.items():
        p = os.path.join(RAW, f'eb_{c}.json')
        if os.path.exists(p):
            out.append(build('eb_' + c, fam, json.load(open(p))))
    for c, fam in WK.items():
        p = os.path.join(RAW, f'wk_{c}.json')
        if os.path.exists(p):
            out.append(build('wk_' + c, fam, json.load(open(p))))
    p = os.path.join(RAW, 'mnc.json')
    if os.path.exists(p):
        out.append(build('mnc', 'Tungusic', json.load(open(p))))
    hy = glob.glob(os.path.join(SCRATCH, 'v28txt', '20231101.hy_*00000*.parquet'))
    if hy:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(hy[0]); docs = []
        for b in pf.iter_batches(columns=['text'], batch_size=200):
            for t in b.column('text').to_pylist():
                if len(t) > 800: docs.append([l for l in t.split('\n') if len(l.split()) >= 6])
            if len(docs) > 400: break
        out.append(build('wk_hy', 'Armenian', docs))
    for c, (fn, fam) in GK.items():
        p = os.path.join(GASK, fn)
        if os.path.exists(p):
            out.append(build('gk_' + c, fam, _chunks(open(p, encoding='utf-8', errors='replace').read()), strip_all=(c == 'pinyin')))
    p = os.path.join(SCRATCH, 'v15', 'oc_raw.txt')
    if os.path.exists(p):
        out.append(build('ws_oc', 'Romance', _chunks(open(p).read())))
    d = json.load(open(os.path.join(DATA, 'v30_ckpt', 'corpora.json')))['corpora']
    for c, fam in V30.items():
        docs = [[' '.join(x) for x in [doc]] for doc in d[c]]
        out.append(build('ms_' + c, fam, docs))
    for c, (fam, nm) in GUT.items():
        t = open(os.path.join(DATA, c + '.txt'), encoding='utf-8', errors='replace').read()
        t = t.split('*** START', 1)[-1].split('*** END', 1)[0]
        out.append(build('pg_' + nm, fam, _chunks(t)))
    return [C for C in out if len(tokens(C)) >= WIN]


# ================================================================== corpus helpers
def tokens(C):
    return [w for d in C['docs'] for l in d['lines'] for w in l]


def mapc(C, f, name=None):
    """apply f(word, doc, line_index, word_index) -> word to every word."""
    docs = []
    for d in C['docs']:
        nl = []
        for li, l in enumerate(d['lines']):
            nl.append([f(w, d, li, wi) for wi, w in enumerate(l)])
        nd = dict(d); nd['lines'] = [[w for w in l if w] for l in nl]
        docs.append(nd)
    return dict(C, name=name or C['name'], docs=docs)


def merge_units(C, mp, name=None):
    return mapc(C, lambda w, d, li, wi: tuple(mp.get(u, u) for u in w), name)


def windows(toks, win=WIN, nwin=NWIN):
    if len(toks) <= win: return [toks]
    k = min(nwin, len(toks) // win)
    starts = np.linspace(0, len(toks) - win, k).astype(int)
    return [toks[s:s + win] for s in starts]


# ================================================================== nulls
def null_shuffle(C, seed=0):
    rng = random.Random(seed)
    us = [u for w in tokens(C) for u in w]; rng.shuffle(us)
    it = iter(us)
    return mapc(C, lambda w, d, li, wi: tuple(next(it) for _ in w), C['name'] + '|shuf')


def null_markov2(C, seed=0):
    rng = random.Random(seed)
    T = defaultdict(Counter)
    for w in tokens(C):
        s = ('<', '<') + w + ('>',)
        for i in range(2, len(s)):
            T[s[i - 2:i]][s[i]] += 1
    tab = {k: (list(v), np.cumsum(list(v.values())) / sum(v.values())) for k, v in T.items()}

    def gen(w, d, li, wi):
        out, ctx = [], ('<', '<')
        while len(out) < 25:
            ks, cp = tab[ctx]
            u = ks[int(np.searchsorted(cp, rng.random()))]
            if u == '>': break
            out.append(u); ctx = (ctx[1], u)
        return tuple(out) or ('?',)
    return mapc(C, gen, C['name'] + '|mk2')


# ================================================================== plants
def plant(C, seed=0, pos=True, ab=True):
    """positional twins + bench variant + A/B rewrite.  Returns the planted corpus and the truth."""
    rng = random.Random(seed)
    cnt = Counter(u for w in tokens(C) for u in w)
    rank = [u for u, _ in cnt.most_common()]
    truth = {}
    if pos:
        a, b, c = rank[4], rank[7], rank[10]
        ta, tb, tc = '' + a, '' + b, '' + c
        truth['twins'] = {ta: a, tb: b, tc: c}
        ctx_set = set(rank[:6][::2])          # bench variant written after these units

        def f(w, d, li, wi, st={}):
            key = id(d), li
            if st.get('key') != key: st.clear(); st['key'] = key; st['ha'] = 0.; st['hb'] = 0.
            out, prev = [], '^'
            for u in w:
                if u == a:
                    p = (0.65 if (wi == 0 or (li == 0)) else 0.2) + 0.35 * st['ha']
                    m = rng.random() < min(p, 0.95); st['ha'] = 0.7 * st['ha'] + 0.3 * m
                    out.append(ta if m else a)
                elif u == b:
                    p = 0.3 + 0.1 * (wi >= len(d['lines'][li]) // 2) + 0.35 * st['hb']
                    m = rng.random() < min(p, 0.95); st['hb'] = 0.7 * st['hb'] + 0.3 * m
                    out.append(tb if m else b)
                elif u == c:
                    m = (prev in ctx_set) != (rng.random() < 0.1)
                    out.append(tc if m else c)
                else:
                    out.append(u)
                prev = u
            return tuple(out)
        C = mapc(C, f, C['name'] + '|P')
    if ab:
        toks = tokens(C)
        ends = Counter(w[-2:] for w in toks if len(w) >= 4)
        cand = [e for e, _ in ends.most_common(14)][2:]
        rng.shuffle(cand); E = cand[:4]
        bg = Counter((w[i], w[i + 1]) for w in toks for i in range(len(w) - 1)).most_common(6)
        ins = [bg[k][0] for k in (1, 3, 4, 5)]
        rules = {e: e[:1] + ins[i] + e[1:] for i, e in enumerate(E)}
        inits = [u for u, _ in Counter(w[0] for w in toks if len(w) >= 3).most_common(8)]
        i_from, i_to = inits[4], inits[1]
        truth['ab'] = dict(ends={''.join(k): ''.join(v) for k, v in rules.items()}, init=(i_from, i_to))

        def g(w, d, li, wi):
            if d['tag'] != 'B': return w
            if len(w) >= 4 and w[-2:] in rules and rng.random() < 0.85:
                w = w[:-2] + rules[w[-2:]]
            if len(w) >= 3 and w[0] == i_from and rng.random() < 0.8:
                w = (i_to,) + w[1:]
            return w
        C = mapc(C, g, C['name'] + 'R')
    return C, truth


# ================================================================== blind normalisers
def interchange(C, minc=50):
    sys.path.insert(0, HERE)
    import v25_lib as L
    words = tokens(C)
    c = Counter(g for w in words for g in w)
    A = sorted([g for g, n in c.items() if n >= minc], key=lambda g: -c[g])
    a, b = L.halves(words)
    Ps = []
    for h in (a, b):
        M, _, _ = L.contexts(h, A)
        X = L.ppmi(M)
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
        Ps.append(X / n)
    Xc = Ps[0] @ Ps[1].T
    dg = np.sqrt(np.maximum(np.outer(np.diag(Xc), np.diag(Xc)), 1e-12))
    I = 0.5 * (Xc + Xc.T) / dg
    pairs = [(float(I[i, j]), A[i], A[j]) for i in range(len(A)) for j in range(i + 1, len(A))]
    return sorted(pairs, reverse=True), c


def merge_blind(C, thr=0.85, maxm=6):
    pairs, c = interchange(C)
    mp, used = {}, set()
    for I, x, y in pairs:
        if I < thr or len(mp) >= maxm: break
        if x in used or y in used: continue
        hi, lo = (x, y) if c[x] >= c[y] else (y, x)
        mp[lo] = hi; used |= {x, y}
    return merge_units(C, mp, C['name'] + '|M'), mp, pairs[:8]


def ab_blind(C, max_end=8, max_beg=4, jmin=0.08, minp=0.003):
    toks = {'A': [], 'B': []}
    for d in C['docs']:
        if d['tag'] in toks:
            toks[d['tag']].extend(w for l in d['lines'] for w in l)
    if not toks['A'] or not toks['B']: return C, {}
    rules = {'end': {}, 'beg': {}}
    nA, nB = len(toks['A']), len(toks['B'])
    for side, L, mx in (('end', (1, 2, 3, 4), max_end), ('beg', (1, 2, 3), max_beg)):
        fr = {t: Counter() for t in 'AB'}
        st = {t: defaultdict(set) for t in 'AB'}
        for k in L:
            for t in 'AB':
                for w in toks[t]:
                    if len(w) > k:
                        e = w[-k:] if side == 'end' else w[:k]
                        fr[t][e] += 1; st[t][e].add(w[:-k] if side == 'end' else w[k:])
        Bh = [e for e in fr['B'] if fr['B'][e] / nB >= minp and fr['B'][e] / nB > 2 * fr['A'][e] / nA]
        Ah = [e for e in fr['A'] if fr['A'][e] / nA >= minp and fr['A'][e] / nA > 2 * fr['B'][e] / nB]
        cand = []
        for eb in Bh:
            for ea in Ah:
                sb, sa = st['B'][eb], st['A'][ea]
                j = len(sb & sa) / max(1, len(sb | sa))
                if j >= jmin: cand.append((j, eb, ea))
        cand.sort(reverse=True)
        done_b, done_a = set(), set()
        for j, eb, ea in cand:
            if len(rules[side]) >= mx: break
            # one rule per edge family: skip edges nested in one already used
            nest = lambda x, S: any((x[-len(y):] == y or y[-len(x):] == x) if side == 'end' else (x[:len(y)] == y or y[:len(x)] == x) for y in S)
            if nest(eb, done_b) or nest(ea, done_a): continue
            rules[side][eb] = ea; done_b.add(eb); done_a.add(ea)
    endr = sorted(rules['end'].items(), key=lambda x: -len(x[0]))
    begr = sorted(rules['beg'].items(), key=lambda x: -len(x[0]))

    def f(w, d, li, wi):
        if d['tag'] != 'B': return w
        for eb, ea in endr:
            if len(w) > len(eb) and w[-len(eb):] == eb:
                w = w[:-len(eb)] + ea; break
        for eb, ea in begr:
            if len(w) > len(eb) and w[:len(eb)] == eb:
                w = ea + w[len(eb):]; break
        return w
    out = mapc(C, f, C['name'] + '|AB')
    return out, {s: {''.join(k): ''.join(v) for k, v in r.items()} for s, r in rules.items()}


# ================================================================== fingerprint
def _H(c):
    v = np.array(list(c.values()), float)
    if v.sum() == 0: return 0.
    p = v / v.sum(); return float(-(p * np.log2(p)).sum())


def sukhotin(M):
    M = M.copy().astype(float); np.fill_diagonal(M, 0)
    r = M.sum(1); V = np.zeros(len(M), bool)
    while True:
        r2 = np.where(V, -np.inf, r); i = int(np.argmax(r2))
        if r2[i] <= 0: break
        V[i] = True; r -= 2 * M[:, i]
    return V


def adj(words, idx):
    n = len(idx); M = np.zeros((n, n))
    for w in words:
        for a, b in zip(w, w[1:]):
            M[idx[a], idx[b]] += 1
    return M + M.T


FEATS = ['h1', 'h2', 'h2r', 'vshare', 'alt', 'bip', 'agree', 'stab', 'initV', 'finV', 'clus', 'clus2', 'VV',
         'sylpw', 'skelH', 'wl', 'wlcv', 'wlsk', 'w1', 'wlong', 'morph', 'mpw', 'Hp1', 'Hp2', 'Hl1', 'Hl2',
         'sufconc', 'sufpre', 'burst', 'reuse', 'ttr', 'hapax', 'zipf', 'rep']


def fingerprint(words, seed=0):
    f = {}
    uc = Counter(u for w in words for u in w)
    units = [u for u, _ in uc.most_common()]
    idx = {u: i for i, u in enumerate(units)}
    tot = sum(uc.values())
    f['h1'] = _H(uc)
    big = Counter(); prev = Counter()
    for w in words:
        s = ('#',) + w
        for a, b in zip(s, s[1:] + ('#',)):
            big[(a, b)] += 1; prev[a] += 1
    hj = _H(big); f['h2'] = hj - _H(prev); f['h2r'] = f['h2'] / max(f['h1'], 1e-9)
    M = adj(words, idx)
    V = sukhotin(M)
    w_tok = np.array([uc[u] for u in units], float)
    f['vshare'] = float(w_tok[V].sum() / tot)
    isV = {u: bool(V[idx[u]]) for u in units}
    pairs = [(isV[a], isV[b]) for w in words for a, b in zip(w, w[1:])]
    f['alt'] = float(np.mean([a != b for a, b in pairs])) if pairs else 0.
    f['VV'] = float(np.mean([a and b for a, b in pairs])) if pairs else 0.
    A = M.copy(); np.fill_diagonal(A, 0)
    keep = A.sum(1) > 0
    A = A[np.ix_(keep, keep)]; d = A.sum(1)
    Nn = A / np.sqrt(np.outer(d, d))
    lam, vec = np.linalg.eigh(Nn)
    f['bip'] = float(-lam[0])
    lab = np.zeros(len(units), bool); lab[np.where(keep)[0]] = vec[:, 0] > 0
    ag = float(w_tok[lab == V].sum() / tot); f['agree'] = max(ag, 1 - ag)
    # stability of the split across alternating 200-word blocks
    h = [[], []]
    for i in range(0, len(words), 200):
        h[(i // 200) % 2].extend(words[i:i + 200])
    Vs = []
    for part in h:
        Vp = sukhotin(adj(part, idx)); Vs.append(Vp)
    f['stab'] = float(w_tok[Vs[0] == Vs[1]].sum() / tot)
    sk = Counter(); cl = []; nV = []
    for w in words:
        s = ''.join('V' if isV[u] else 'C' for u in w)
        sk[s] += 1
        runs = re.findall('C+', s); cl.extend(len(r) for r in runs); nV.append(len(re.findall('V+', s)))
    n = len(words)
    f['initV'] = sum(c for s, c in sk.items() if s[0] == 'V') / n
    f['finV'] = sum(c for s, c in sk.items() if s[-1] == 'V') / n
    f['clus'] = float(np.mean(cl)) if cl else 0.
    f['clus2'] = float(np.mean([x >= 2 for x in cl])) if cl else 0.
    f['sylpw'] = float(np.mean(nV)); f['skelH'] = _H(sk)
    L = np.array([len(w) for w in words], float)
    f['wl'] = float(L.mean()); f['wlcv'] = float(L.std() / L.mean())
    f['wlsk'] = float(((L - L.mean()) ** 3).mean() / L.std() ** 3)
    f['w1'] = float((L == 1).mean()); f['wlong'] = float((L >= 2 * L.mean()).mean())
    # Harris successor-entropy segmentation over types
    wc = Counter(words); types = list(wc)
    succ = defaultdict(Counter)
    for t in types:
        for i in range(len(t)):
            succ[t[:i]][t[i]] += 1
        succ[t][('>',)] += 1
    Hs = {p: _H(c) for p, c in succ.items()}
    ml, mp = [], []
    for t in types:
        if len(t) < 3: ml.append(len(t)); mp.append(1); continue
        H = [Hs[t[:i]] for i in range(len(t) + 1)]
        nb = sum(1 for i in range(2, len(t)) if H[i] > H[i - 1] and H[i] >= H[i + 1] and H[i] >= 1.0)
        mp.append(1 + nb); ml.append(len(t) / (1 + nb))
    f['morph'] = float(np.mean(ml)); f['mpw'] = float(np.mean(mp))
    for k, name in ((0, 'Hp1'), (1, 'Hp2')):
        f[name] = _H(Counter(w[k] for w in words if len(w) > k + 1)) / f['h1']
    for k, name in ((1, 'Hl1'), (2, 'Hl2')):
        f[name] = _H(Counter(w[-k] for w in words if len(w) > k)) / f['h1']
    long_t = [t for t in types if len(t) >= 4]
    ends = Counter(t[-2:] for t in long_t); begs = Counter(t[:2] for t in long_t)
    ce = sum(c for _, c in ends.most_common(10)) / max(1, len(long_t))
    cb = sum(c for _, c in begs.most_common(10)) / max(1, len(long_t))
    f['sufconc'] = ce; f['sufpre'] = math.log((ce + 1e-3) / (cb + 1e-3))
    # reuse: next recurrence within 50 tokens (burst) and 50-1000 tokens (long), over a token-shuffle baseline
    def rec(ws):
        last, a, b, m = {}, 0, 0, 0
        for i, w in enumerate(ws):
            if w in last:
                g = i - last[w]; m += 1
                a += g <= 50; b += 50 < g <= 1000
            last[w] = i
        return a / max(m, 1), b / max(m, 1)
    rng = random.Random(seed); sh = list(words); rng.shuffle(sh)
    r0, r1 = rec(words), rec(sh)
    f['burst'] = r0[0] / max(r1[0], 1e-9); f['reuse'] = r0[1] / max(r1[1], 1e-9)
    f['ttr'] = len(types) / n
    f['hapax'] = sum(1 for c in wc.values() if c == 1) / len(types)
    fr = np.array(sorted(wc.values(), reverse=True)[:300], float)
    rk = np.arange(1, len(fr) + 1)
    f['zipf'] = float(np.polyfit(np.log(rk), np.log(fr), 1)[0])
    f['rep'] = float(np.mean([a == b for a, b in zip(words, words[1:])]))
    return f


def fp_corpus(C, seed=0):
    ws = windows(tokens(C))
    fs = [fingerprint(w, seed) for w in ws]
    return {k: float(np.mean([x[k] for x in fs])) for k in FEATS}


# ================================================================== map
class Map:
    def __init__(self, fps, names, fams, feats=FEATS):
        self.feats = feats
        X = np.array([[fp[k] for k in feats] for fp in fps])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.Z = (X - self.mu) / self.sd
        self.names, self.fams = names, fams
        D = self.dist_matrix(self.Z, self.Z); np.fill_diagonal(D, np.inf)
        self.nn_lang = D.min(1)
        self.ref = float(np.median(self.nn_lang))

    def z(self, fp):
        return (np.array([fp[k] for k in self.feats]) - self.mu) / self.sd

    @staticmethod
    def dist_matrix(A, B):
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).mean(2))

    def place(self, fp, exclude=()):
        z = self.z(fp)
        d = np.sqrt(((self.Z - z) ** 2).mean(1))
        for i, n in enumerate(self.names):
            if n in exclude: d[i] = np.inf
        o = np.argsort(d)
        return dict(out=float(d[o[0]] / self.ref), nn=[(self.names[i], self.fams[i], float(d[i])) for i in o[:5]],
                    pct=float((self.nn_lang < d[o[0]]).mean()), d=d)

    def fam_vote(self, fp, k=5, exclude=()):
        p = self.place(fp, exclude)
        v = Counter()
        for n, fam, dd in p['nn'][:k]:
            v[fam] += 1 / (dd + 1e-6)
        return v.most_common()
