"""v56 THE BOOK IS HYPERTEXT: are some Voynich words pointers to other pages?

Corpus format: dict(pages=[...], toks=[...]) built by build(); a page = dict(id, sec, quire, folio, lines=[[w..]..],
lflags=[(para_start, ltype)..]); a word = string of single-character glyph units.

Core objects
  R[i, t]   residual context->page affinity for token i and target page t: how much the rare vocabulary
            (payload skeletons) of the lines around token i (token itself removed) is shared with page t,
            minus the mean over pages of the same section and the same distance band from i's own page,
            then standardised per token. A pointer rule maps token i to an address -> page; its score is
            z = sum_i R[i, target_i] / sqrt(n_selected) (invalid addresses count 0).
  Rf, Rq    the same aggregated to folios (leaves) and quires.

Rule = selector (which tokens are pointers) + feature (count or positional weight of each glyph in the full word
or in its skeleton) + value vector d (one integer per glyph) + address mode (absolute/relative, space, offset,
mod/clip). addr = C_feature @ d. Everything is linear, so random rules are evaluated in batches and a
coordinate ascent can retune one glyph value at a time over all its candidate values at once.
"""
import os, sys, json, random, math, re, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v56_ckpt')
os.makedirs(CK, exist_ok=True)

GALL = set('ktpfKTPF')
FRAME = set('oainylr')
SKEL_MAP = {'K': 'k', 'T': 't', 'P': 'p', 'F': 'f'}


def skel_voy(w):
    """payload skeleton (v52/v54): drop q, e, d, ch/sh, s, other padding; keep o a i n y l r + gallows."""
    return ''.join(SKEL_MAP.get(c, c) for c in w if c in FRAME or c in GALL)


# ------------------------------------------------------------------ corpora
def voynich(name='ZL3b', ltypes=('P', 'L', 'R', 'C')):
    import vlib
    L = vlib.load_voynich(name, ltypes=ltypes)
    pages = collections.OrderedDict()
    for r in L:
        ws = [''.join(vlib.glyphs(w)) for w in r['words'] if '?' not in w and '*' not in w and w]
        ws = [w for w in ws if re.fullmatch(r'[a-zA-Z]+', w)]
        if not ws: continue
        f = r['folio']
        p = pages.setdefault(f, dict(id=f, sec=r['illus'], quire=r['quire'], folio=int(re.match(r'f(\d+)', f).group(1)) if re.match(r'f\d', f) else 86,
                                     lang=r['lang'] or '-', hand=r['hand'], lines=[], lflags=[]))
        p['lines'].append(ws); p['lflags'].append((bool(r['para_start']), r['ltype']))
    return dict(pages=list(pages.values()), skel=skel_voy, name=name)


def markov_corpus(C, seed=1):
    """unit-trigram word resynthesis per section (generator; keeps page/line layout)."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for p in C['pages']:
        by[p['sec']] += [w for l in p['lines'] for w in l]
    tabs = {}
    for k, ws in by.items():
        tri = collections.defaultdict(collections.Counter)
        for w in ws:
            s = '\x01\x01' + w + '\x02'
            for i in range(2, len(s)): tri[s[i - 2:i]][s[i]] += 1
        tabs[k] = {c: (list(v.keys()), list(v.values())) for c, v in tri.items()}
    P = []
    for p in C['pages']:
        tab = tabs[p['sec']]; nl = []
        for l in p['lines']:
            q = []
            for _ in l:
                ctx, w = '\x01\x01', ''
                while True:
                    ks, vs = tab[ctx]; c = rng.choices(ks, vs)[0]
                    if c == '\x02' or len(w) > 15: break
                    w += c; ctx = ctx[1] + c
                q.append(w or 'o')
            nl.append(q)
        P.append(dict(p, lines=nl))
    return dict(C, pages=P, name=C['name'] + '_markov')


def selfcit_corpus(C, seed=1, window=60, p_mod=0.5):
    """copy-and-modify generator per section (the strongest generator in v54)."""
    import v54_lib
    pages = [dict(id=p['id'], vars=dict(sec=p['sec']), lines=p['lines']) for p in C['pages']]
    out = v54_lib.null_selfcit(pages, seed=seed, key='sec', window=window, p_mod=p_mod)
    return dict(C, pages=[dict(p, lines=o['lines']) for p, o in zip(C['pages'], out)], name=C['name'] + '_selfcit')


def relabel_corpus(C, seed=1):
    """pages relabelled: the binding order is randomly permuted (sections and quires travel with their pages)."""
    rng = random.Random(seed)
    P = list(C['pages']); rng.shuffle(P)
    return dict(C, pages=P, name=C['name'] + '_relab')


# ------------------------------------------------------------------ Culpeper control (real cross-references)
ROMAN_ADD = [(100, 'c'), (50, 'l'), (10, 'x'), (5, 'v'), (1, 'i')]


def roman_add(n):
    s = ''
    for v, c in ROMAN_ADD:
        while n >= v: s += c; n -= v
    return s


def culpeper_chapters():
    t = open(os.path.join(CK, 'culpeper.txt'), encoding='utf-8').read().replace('\r', '')
    i = t.find('    ADDER’S TONGUE'); j = t.find('THE DISPENSATORY', i) if 'THE DISPENSATORY' in t[i:] else len(t)
    t = t[i:j]
    lines = t.split('\n'); chaps = []; cur = None
    for ln in lines:
        m = re.match(r'^ {2,}([A-Z][A-Z’\'\-, ]{2,60})\.?\s*$', ln)
        if m and not ln.strip().startswith(('THE ', 'OF ')) or (m and len(chaps) and cur and len(cur['body']) > 40):
            name = m.group(1).strip().rstrip('.').strip()
            if cur: chaps.append(cur)
            cur = dict(head=name, body=[]); continue
        if cur is not None: cur['body'].append(ln)
    if cur: chaps.append(cur)
    out = []
    for c in chaps:
        txt = ' '.join(c['body'])
        txt = re.sub(r'_[A-Za-z. ]+\._\]', ' ', txt)
        words = re.findall(r"[a-z’']+", txt.lower().replace('’', "'"))
        if len(words) < 60: continue
        names = [n.strip().lower().replace('’', "'") for n in re.split(r'\bOR\b|,', c['head']) if n.strip()]
        out.append(dict(head=c['head'], names=names, words=words))
    return out


def culpeper(seed=56, max_words=180, line_w=9, pad=True, numerals=True):
    """Culpeper's herbal chapters (= pages, kept in their alphabetical order). Real cross-references: a mention in
    chapter i of the name of another chapter j. Each mention is replaced by an additive Roman numeral giving j's
    chapter number (as 'vide cap. xxxxii' minus the 'vide cap.'), the whole text is mapped to opaque glyphs and
    Voynich-like padding is added (optional prefix glyph, two-way alternation of one letter, doubled vowel glyph)."""
    rng = random.Random(seed)
    ch = culpeper_chapters()
    # name phrases; drop names that are common words (appear in > 8% of chapters as plain text)
    df = collections.Counter()
    for c in ch:
        txt = ' ' + ' '.join(c['words']) + ' '
        for n in {n for cc in ch for n in cc['names']}:
            pass
    allnames = {}
    for j, c in enumerate(ch):
        for n in c['names']:
            n2 = n.replace('-', ' ').strip()
            if len(n2) >= 4: allnames.setdefault(n2, j)
    texts = [' '.join(c['words']).replace('-', ' ') for c in ch]
    for n in allnames:
        pat = re.compile(r'\b' + re.escape(n) + r's?\b')
        df[n] = sum(1 for t in texts if pat.search(t))
    good = {n: j for n, j in allnames.items() if df[n] <= 0.08 * len(ch)}
    pats = sorted(good, key=len, reverse=True)
    rx = re.compile(r'\b(' + '|'.join(re.escape(n) for n in pats) + r')s?\b')
    pages = []; truth = []
    for i, c in enumerate(ch):
        t = texts[i]
        ws = []; pos = 0
        for m in rx.finditer(t):
            ws += t[pos:m.start()].split()
            j = good[m.group(1)]
            if j != i and numerals:
                ws.append('#%d' % j); truth.append((i, len(ws) - 1, j))
            else:
                ws += m.group(0).split()
            pos = m.end()
        ws += t[pos:].split()
        # keep a window of max_words that includes as many references as possible: take the first max_words
        ws = ws[:max_words]
        pages.append(dict(id='cu%03d' % i, sec='G%d' % (i * 6 // len(ch)), quire='Q%02d' % (i // 16), folio=i // 2,
                          words=ws))
    alpha = sorted({ch_ for p in pages for w in p['words'] if not w.startswith('#') for ch_ in w if ch_.isalpha()})
    syms = [chr(c) for c in range(0x3B1, 0x3B1 + 25)] + [chr(c) for c in range(0x410, 0x410 + 32)]
    r2 = random.Random(seed + 1); r2.shuffle(syms)
    M = {a: syms[k] for k, a in enumerate(alpha)}
    PADQ, ALT, DUP = syms[40], syms[41], syms[42]
    alt_src = M['e']
    def enc(w):
        if w.startswith('#'):
            w = roman_add(int(w[1:]) + 1)   # chapters numbered from 1
        g = [M[c] for c in w if c in M]
        if not pad: return ''.join(g)
        out = []
        if rng.random() < 0.25: out.append(PADQ)
        for c in g:
            if c == alt_src and rng.random() < 0.5: c = ALT
            out.append(c)
            if c in (M['o'], M['a']) and rng.random() < 0.2: out.append(DUP)
        return ''.join(out)
    P = []
    nref = 0
    for i, p in enumerate(pages):
        ws = [enc(w) for w in p['words']]
        lines = [ws[k:k + line_w] for k in range(0, len(ws), line_w)]
        P.append(dict(id=p['id'], sec=p['sec'], quire=p['quire'], folio=p['folio'], lang='-', hand='-', lines=lines,
                      lflags=[(k == 0, 'P') for k in range(len(lines))]))
    # truth restricted to kept words
    tr = [(i, k, j) for i, k, j in truth if k < max_words]
    pad_set = {PADQ, DUP}
    def skel(w):
        return ''.join(alt_src if c == ALT else c for c in w if c not in pad_set)
    roman_g = {c: M[c] for c in 'ivxlc'}
    return dict(pages=P, skel=skel, name='CULP' + ('' if numerals else '_names'), truth=tr, roman=roman_g, M=M)


# ------------------------------------------------------------------ planted pointer system inside the Voynich text
def plant_pointers(C, R_builder, rate_per_page=2, seed=7, base=7, ndig=3):
    """Replace ~rate_per_page random tokens per page by a pointer word that encodes (positional base 7 over the
    frame glyphs o a i n y l r, most significant first) the 1-based index of a page relevant to the context (drawn
    from the context's top-10 residual pages at distance >= 3), then pad it Voynich-style (q-, ch, e/ee, -dy)."""
    rng = random.Random(seed)
    E = R_builder(C)
    R, tokpage = E['R'], E['tok_page']
    digits = list('oainylr'); rng.shuffle(digits)
    dval = {g: k for k, g in enumerate(digits)}
    P = [dict(p, lines=[list(l) for l in p['lines']]) for p in C['pages']]
    truth = []
    N = len(P)
    for pi, p in enumerate(P):
        locs = [(li, k) for li, l in enumerate(p['lines']) for k in range(len(l))]
        if len(locs) < 10: continue
        for li, k in rng.sample(locs, min(rate_per_page, len(locs))):
            ti = E['tok_index'][(pi, li, k)]
            row = R[ti].copy()
            for t in range(N):
                if abs(t - pi) < 3: row[t] = -1e9
            cand = np.argsort(-row)[:10]
            t = int(rng.choice(list(cand)))
            a = t + 1
            ds = []
            for _ in range(ndig): ds.append(a % base); a //= base
            ds = ds[::-1]
            w = ''
            if rng.random() < 0.4: w += 'q'
            for x, dd in enumerate(ds):
                w += digits[dd]
                if rng.random() < 0.3: w += 'C' if rng.random() < 0.5 else 'S'
                if rng.random() < 0.2: w += 'e'
            w += rng.choice(['', 'd', 'ed', 'eed', 's'])
            p['lines'][li][k] = w
            truth.append((pi, li, k, t))
    return dict(C, pages=P, name=C['name'] + '_planted', truth=truth, digits=digits)


# ------------------------------------------------------------------ residual affinity
def build_R(C, use_skel=True, ctx_lines=2, df_max=0.17, bands=(1, 3, 6, 11, 21, 41), near=2, order=None):
    """tokens + residual affinity matrices. order: optional list of page indices giving an alternative binding
    order (address space); default = given order."""
    pages = C['pages']
    if order is not None: pages = [pages[k] for k in order]
    N = len(pages)
    sk = C['skel'] if use_skel else (lambda w: w)
    ptypes = [set(sk(w) for l in p['lines'] for w in l) - {''} for p in pages]
    df = collections.Counter(t for s in ptypes for t in s)
    dfm = df_max * N if df_max < 1 else df_max
    rare = {t for t, c in df.items() if 2 <= c <= dfm}
    vocab = {t: k for k, t in enumerate(sorted(rare))}
    V = len(vocab)
    idf = np.zeros(V, np.float32)
    for t, k in vocab.items(): idf[k] = math.log(N / df[t])
    PM = np.zeros((N, V), np.float32)
    for pi, s in enumerate(ptypes):
        for t in s & rare: PM[pi, vocab[t]] = 1.0
    pnorm = np.sqrt(PM.sum(1) + 5.0)
    secs = [p['sec'] for p in pages]
    # tokens
    toks = []; tok_index = {}
    for pi, p in enumerate(pages):
        nl = len(p['lines'])
        for li, l in enumerate(p['lines']):
            for k, w in enumerate(l):
                tok_index[(pi, li, k)] = len(toks)
                toks.append((pi, li, k, w))
    T = len(toks)
    R = np.zeros((T, N), np.float32)
    bandid = np.zeros((N, N), np.int32)
    for a in range(N):
        for b in range(N):
            d = abs(a - b); bandid[a, b] = sum(d >= x for x in bands)
    gkeys = [(p['sec'], p.get('lang', '-')) for p in pages]     # residual groups: section x Currier language
    secid = {s: k for k, s in enumerate(sorted(set(gkeys)))}
    sarr = np.array([secid[s] for s in gkeys])
    for pi, p in enumerate(pages):
        idxs = [tok_index[(pi, li, k)] for li, l in enumerate(p['lines']) for k in range(len(l))]
        if not idxs: continue
        rows = np.zeros((len(idxs), V), np.float32)
        r = 0
        for li, l in enumerate(p['lines']):
            ctx = collections.Counter()
            for lj in range(max(0, li - ctx_lines), min(len(p['lines']), li + ctx_lines + 1)):
                for w in p['lines'][lj]:
                    t = sk(w)
                    if t in vocab: ctx[t] += 1
            for k, w in enumerate(l):
                c2 = dict(ctx); t = sk(w)
                if t in vocab:
                    c2[t] -= 1
                for t2, n in c2.items():
                    if n > 0: rows[r, vocab[t2]] = idf[vocab[t2]]
                r += 1
        S = rows @ PM.T / pnorm[None, :]                      # (n, N)
        # residual vs same (band, section) group, excluding own page
        g = bandid[pi] * 100 + sarr
        g[max(0, pi - near):pi + near + 1] = -1        # own page and its +-near neighbours carry no score
        Rr = np.zeros_like(S)
        for gv in np.unique(g):
            if gv < 0: continue
            m = g == gv
            Rr[:, m] = S[:, m] - S[:, m].mean(1, keepdims=True)
        Rr[:, max(0, pi - near):pi + near + 1] = 0.0
        sd = Rr.std(1, keepdims=True) + 1e-6
        Rr = Rr / sd
        R[idxs] = Rr
    # column centring: a hub page that everything resembles must not reward rules that send many tokens to it
    tp_all = np.array([t[0] for t in toks])
    for t in range(N):
        m = np.abs(tp_all - t) > near
        R[m, t] -= R[m, t].mean()
        R[~m, t] = 0.0
    # aggregate to folios and quires (own unit excluded)
    fol = [p['folio'] for p in pages]; ufol = sorted(set(fol), key=lambda x: fol.index(x))
    qu = [p['quire'] for p in pages]; uqu = sorted(set(qu), key=lambda x: qu.index(x))
    def agg(keys, ukeys):
        A = np.zeros((N, len(ukeys)), np.float32)
        for pi, kk in enumerate(keys): A[pi, ukeys.index(kk)] = 1
        A = A / np.maximum(A.sum(0, keepdims=True), 1)
        out = R @ A
        tp = np.array([t[0] for t in toks])
        own = np.array([ukeys.index(keys[t]) for t in tp])
        out[np.arange(T), own] = 0.0
        return out
    Rf = agg(fol, ufol); Rq = agg(qu, uqu)
    return dict(R=R, Rf=Rf, Rq=Rq, toks=toks, tok_index=tok_index, tok_page=np.array([t[0] for t in toks]),
                N=N, Nf=len(ufol), Nq=len(uqu), pages=pages,
                page_folio=np.array([ufol.index(f) for f in fol]), page_quire=np.array([uqu.index(q) for q in qu]))


# ------------------------------------------------------------------ token features
def token_table(C, E):
    """per-token selector flags and glyph feature matrices."""
    pages = E['pages']; toks = E['toks']
    glyphs = sorted({c for (_, _, _, w) in toks for c in w})
    gi = {g: k for k, g in enumerate(glyphs)}
    sk = C['skel']
    T = len(toks); G = len(glyphs)
    cnt_full = np.zeros((T, G), np.int64)
    cnt_skel = np.zeros((T, G), np.int64)
    ln_full = np.zeros(T, np.int64); ln_skel = np.zeros(T, np.int64)
    first = np.zeros(T, bool); last = np.zeros(T, bool); pfirst = np.zeros(T, bool); pline = np.zeros(T, bool)
    label = np.zeros(T, bool)
    words = []
    for i, (pi, li, k, w) in enumerate(toks):
        s = sk(w)
        for c in w: cnt_full[i, gi[c]] += 1
        for c in s:
            if c in gi: cnt_skel[i, gi[c]] += 1
        ln_full[i] = len(w); ln_skel[i] = len(s)
        L = pages[pi]['lines'][li]
        first[i] = k == 0; last[i] = k == len(L) - 1
        ps, lt = pages[pi]['lflags'][li]
        pline[i] = ps; pfirst[i] = ps and k == 0; label[i] = lt != 'P'
        words.append((w, s))
    return dict(glyphs=glyphs, gi=gi, cnt_full=cnt_full, cnt_skel=cnt_skel, ln_full=ln_full, ln_skel=ln_skel,
                first=first, last=last, pfirst=pfirst, pline=pline, label=label, words=words)


def pos_weights(TT, use_skel, base, from_right=True, maxlen=10):
    """(T, G) coefficient matrix: sum over positions of glyph g of base**place."""
    G = len(TT['glyphs']); gi = TT['gi']
    out = np.zeros((len(TT['words']), G), np.int64)
    for i, (w, s) in enumerate(TT['words']):
        x = s if use_skel else w
        x = [c for c in x if c in gi][:maxlen]
        n = len(x)
        for k, c in enumerate(x):
            place = (n - 1 - k) if from_right else k
            out[i, gi[c]] += base ** place
    return out


# ------------------------------------------------------------------ selectors
def random_selector(TT, rng):
    G = TT['glyphs']; T = len(TT['words'])
    kinds = ['all', 'gall', 'first', 'last', 'pline', 'pfirst', 'label', 'glyph', 'skellen', 'subset', 'nogall']
    parts = []; m = np.ones(T, bool)
    for _ in range(rng.choice([1, 1, 2])):
        k = rng.choice(kinds)
        if k == 'all': mm = np.ones(T, bool); d = 'all'
        elif k == 'gall':
            idx = [TT['gi'][g] for g in G if g in GALL]
            mm = TT['cnt_full'][:, idx].sum(1) > 0 if idx else np.ones(T, bool); d = 'has-gallows'
        elif k == 'nogall':
            idx = [TT['gi'][g] for g in G if g in GALL]
            mm = TT['cnt_full'][:, idx].sum(1) == 0 if idx else np.ones(T, bool); d = 'no-gallows'
        elif k in ('first', 'last', 'pline', 'pfirst', 'label'): mm = TT[k].copy(); d = k
        elif k == 'glyph':
            g = rng.choice(G); mm = TT['cnt_full'][:, TT['gi'][g]] > 0; d = 'has-' + g
        elif k == 'skellen':
            a = rng.randint(1, 5); b = a + rng.randint(0, 3)
            mm = (TT['ln_skel'] >= a) & (TT['ln_skel'] <= b); d = 'skel%d-%d' % (a, b)
        else:
            S = set(rng.sample(G, rng.randint(3, max(3, min(9, len(G))))))
            idx = [TT['gi'][g] for g in G if g not in S]
            mm = TT['cnt_full'][:, idx].sum(1) == 0; d = 'only{' + ''.join(sorted(S)) + '}'
        m &= mm; parts.append(d)
    return m, '&'.join(parts)


# ------------------------------------------------------------------ evaluation
def targets(addr, tp, mode, E):
    """addr (n, V) int -> (target index, valid) in the chosen space. mode = (space, kind, offset)
    space: 'page'|'folio'|'quire'; kind: 'abs_mod'|'abs_clip'|'rel_fwd'|'rel_back'."""
    space, kind, off = mode
    Nsp = {'page': E['N'], 'folio': E['Nf'], 'quire': E['Nq']}[space]
    if kind == 'abs_mod':
        t = (addr + off) % Nsp; v = np.ones_like(t, bool)
    elif kind == 'abs_clip':
        t = addr + off; v = (t >= 0) & (t < Nsp)
    else:
        own = {'page': tp, 'folio': E['page_folio'][tp], 'quire': E['page_quire'][tp]}[space]
        s = 1 if kind == 'rel_fwd' else -1
        t = own[:, None] + s * (addr + off) if addr.ndim == 2 else own + s * (addr + off)
        v = (t >= 0) & (t < Nsp) & (addr + off != 0)
    t = np.where(v, t, 0)
    return t, v


def Rspace(E, space):
    return {'page': E['R'], 'folio': E['Rf'], 'quire': E['Rq']}[space]


def _perm_expect(RS, sel_idx, t, v, Nsp):
    """expected score when the same targets are shuffled among the same pointer tokens (kills class-hub rules:
    a token class whose contexts all resemble one group of pages)."""
    rbar = RS[sel_idx].sum(0)                                  # (Nsp,)
    n = len(sel_idx)
    if t.ndim == 1:
        h = np.bincount(t[v], minlength=Nsp)
        return float(rbar @ h) / n
    K = t.shape[1]
    out = np.zeros(K)
    for k in range(K):
        h = np.bincount(t[v[:, k], k], minlength=Nsp); out[k] = float(rbar @ h) / n
    return out


def _nsp(E, space): return {'page': E['N'], 'folio': E['Nf'], 'quire': E['Nq']}[space]


import ctypes
_K = None
KIND = {'abs_mod': 0, 'abs_clip': 1, 'rel_fwd': 2, 'rel_back': 3}


def _kern():
    global _K
    if _K is None:
        so = os.path.join(CK, 'bin', 'v56_kernel.so')
        if not os.path.exists(so):
            os.makedirs(os.path.dirname(so), exist_ok=True)
            os.system('cc -O3 -shared -fPIC -o %s %s -lm' % (so, os.path.join(HERE, 'v56_kernel.c')))
        _K = ctypes.CDLL(so)
        P = ctypes.c_void_p
        _K.cand_scores.argtypes = [ctypes.c_int, P, P, ctypes.c_int, P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   P, P, ctypes.c_int, P, P, P]
    return _K


def _prep(E, space, sel_idx):
    RS = Rspace(E, space)
    if not RS.flags['C_CONTIGUOUS'] or RS.dtype != np.float32:
        RS = np.ascontiguousarray(RS, np.float32)
        E[{'page': 'R', 'folio': 'Rf', 'quire': 'Rq'}[space]] = RS
    tp = E['tok_page'][sel_idx]
    own = {'page': tp, 'folio': E['page_folio'][tp], 'quire': E['page_quire'][tp]}[space].astype(np.int32)
    rbar = RS[sel_idx].sum(0).astype(np.float64)
    return RS, np.ascontiguousarray(own), np.ascontiguousarray(sel_idx.astype(np.int64)), rbar


def _cand(b0, cg, vals, mode, E, prep):
    RS, own, rows, rbar = prep
    b0 = np.ascontiguousarray(b0, np.int64); cg = np.ascontiguousarray(cg, np.int64)
    vals = np.ascontiguousarray(vals, np.int64)
    out = np.zeros(len(vals))
    _kern().cand_scores(len(b0), b0.ctypes.data, cg.ctypes.data, len(vals), vals.ctypes.data, KIND[mode[1]], int(mode[2]),
                        RS.shape[1], own.ctypes.data, RS.ctypes.data, RS.shape[1], rows.ctypes.data, rbar.ctypes.data,
                        out.ctypes.data)
    return out


def _nsp(E, space): return {'page': E['N'], 'folio': E['Nf'], 'quire': E['Nq']}[space]


def score_many(coef, sel_idx, D, mode, E):
    """coef (T,G) int, sel_idx selected token ids, D (G, K) int value vectors -> pairing-corrected z (K,)."""
    prep = _prep(E, mode[0], sel_idx)
    c = coef[sel_idx].astype(np.float64)
    addr = np.rint(c @ D.astype(np.float64)).astype(np.int64)
    zero = np.zeros(len(sel_idx), np.int64); v0 = np.zeros(1, np.int64)
    return np.array([_cand(addr[:, k], zero, v0, mode, E, prep)[0] for k in range(D.shape[1])])


def score_one(coef, sel_idx, d, mode, E):
    return float(score_many(coef, sel_idx, d[:, None], mode, E)[0])


def ascent(coef, sel_idx, d, mode, E, vmax, sweeps=3, rng=None, glyph_order=None, grid=None):
    """coordinate ascent on the pairing-corrected z: for each glyph try every candidate value at once (C kernel)."""
    prep = _prep(E, mode[0], sel_idx)
    c = coef[sel_idx]
    d = d.copy(); base = c @ d; best = None
    vals_range = np.arange(vmax, dtype=np.int64) if grid is None else np.asarray(grid, np.int64)
    used = np.where(c.sum(0) > 0)[0]
    for sw in range(sweeps):
        order = list(used)
        if rng: rng.shuffle(order)
        changed = False
        for g in order:
            b0 = base - c[:, g] * d[g]
            z = _cand(b0, c[:, g], vals_range, mode, E, prep)
            k = int(vals_range[int(np.argmax(z))])
            if k != d[g]: changed = True
            d[g] = k; base = b0 + c[:, g] * k; best = float(z.max())
        if not changed: break
    if best is None: best = score_one(coef, sel_idx, d, mode, E)
    return d, best


def score_robust(coef, sel_idx, d, mode, E, TT):
    """held-out score, cluster-robust by word type: per-token value minus its target-shuffle expectation, summed
    within skeleton type, z = sum_types X / sqrt(sum_types X^2). Tokens of one type share target and context, so the
    plain z is over-dispersed; this one is not."""
    if len(sel_idx) < 5: return 0.0
    RS = Rspace(E, mode[0]); Nsp = _nsp(E, mode[0])
    addr = coef[sel_idx] @ d
    t, v = targets(addr, E['tok_page'][sel_idx], mode, E)
    h = np.bincount(t[v], minlength=Nsp).astype(np.float64)
    n = len(sel_idx)
    x = RS[sel_idx, t] * v - (RS[sel_idx] @ h) / n
    keys = [TT['words'][i][1] for i in sel_idx]
    agg = collections.defaultdict(float)
    for k, xi in zip(keys, x): agg[k] += float(xi)
    X = np.array(list(agg.values()))
    return float(X.sum() / math.sqrt((X ** 2).sum() + 1e-12))


def ascent_offset(coef, sel_idx, d, mode, E):
    """choose the address offset (-2..2) and, for absolute addresses, mod vs clip."""
    best = (score_one(coef, sel_idx, d, mode, E), mode)
    kinds = [mode[1]] if mode[1].startswith('rel') else ['abs_mod', 'abs_clip']
    for kd in kinds:
        for off in (-2, -1, 0, 1, 2):
            m = (mode[0], kd, off)
            if m == mode: continue
            z = score_one(coef, sel_idx, d, m, E)
            if z > best[0]: best = (z, m)
    return best[1], best[0]


def split_types(TT, seed=0):
    """held-out by WORD TYPE (skeleton): a compositional address code generalises to word types never seen in
    training; a learned type->page table (topical words) does not."""
    import hashlib
    def h(s): return int(hashlib.md5(('%d|%s' % (seed, s)).encode()).hexdigest()[:8], 16) % 2 == 0
    cache = {}
    out = np.zeros(len(TT['words']), bool)
    for i, (w, s) in enumerate(TT['words']):
        if s not in cache: cache[s] = h(s)
        out[i] = cache[s]
    return out


def split_pages(E, seed=0):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for pi, p in enumerate(E['pages']): by[p['sec']].append(pi)
    train = set()
    for s, v in sorted(by.items()):
        v = list(v); rng.shuffle(v); train |= set(v[:len(v) // 2 + (len(v) % 2) * rng.randint(0, 1)])
    return np.array([pi in train for pi in range(E['N'])])


NUMERAL_GRID = [0] + list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100))
SPACES = ['page', 'page', 'page', 'folio', 'quire']
KINDS = ['abs_mod', 'abs_clip', 'abs_clip', 'rel_fwd', 'rel_back']


def random_config(TT, E, rng, poscache):
    sel, sdesc = random_selector(TT, rng)
    use_skel = rng.random() < 0.6
    if rng.random() < 0.5:
        numeral = rng.random() < 0.6          # alphabetic-numeral values (Greek/Hebrew/Roman-like) or free values
        feat = ('count', use_skel, 'numeral' if numeral else 'free'); coef = TT['cnt_skel'] if use_skel else TT['cnt_full']
        space = rng.choice(SPACES); vmax = {'page': E['N'], 'folio': E['Nf'], 'quire': E['Nq']}[space]
        if numeral: vmax = NUMERAL_GRID
    else:
        B = rng.randint(2, 12); fr = rng.random() < 0.7
        feat = ('pos', use_skel, B, fr)
        if feat not in poscache: poscache[feat] = pos_weights(TT, use_skel, B, fr)
        coef = poscache[feat]; space = rng.choice(SPACES); vmax = B
    kind = rng.choice(KINDS); off = rng.choice([-1, 0, 0, 1])
    if kind.startswith('rel'): off = rng.choice([0, 1])
    return sel, sdesc, feat, coef, (space, kind, off), vmax


def subset_mask(TT, S):
    G = TT['glyphs']
    notS = [TT['gi'][g] for g in G if g not in S]
    if 'pres' not in TT: TT['pres'] = TT['cnt_full'] > 0
    return ~TT['pres'][:, notS].any(1) if notS else np.ones(len(TT['words']), bool)


def search(C, n_cfg=300, n_rand=500, n_ascend=2, seed=0, log=None, time_budget=None, E=None, p_subset=0.4):
    import time
    t0 = time.time()
    if E is None: E = build_R(C)
    TT = token_table(C, E)
    tr_page = split_pages(E, seed)
    tok_tr = tr_page[E['tok_page']]
    rng = random.Random(seed + 11); nrng = np.random.default_rng(seed + 12)
    poscache = {}
    rows = []; n_eval = 0
    G_all = TT['glyphs']
    for ci in range(n_cfg):
        if time_budget and time.time() - t0 > time_budget: break
        sel, sdesc, feat, coef, mode0, vmax = random_config(TT, E, rng, poscache)
        mode = mode0
        S = None
        if rng.random() < p_subset:
            S = set(rng.sample(G_all, rng.randint(3, min(12, len(G_all)))))
            sel = subset_mask(TT, S); sdesc = 'subset'
        itr = np.where(sel & tok_tr)[0]; ite = np.where(sel & ~tok_tr)[0]
        if len(itr) < 30 or len(ite) < 30: continue
        G = coef.shape[1]
        grid = None
        if isinstance(vmax, list):
            grid = vmax; D = np.array(grid)[nrng.integers(0, len(grid), size=(G, n_rand))]
            D[nrng.random((G, n_rand)) < 0.5] = 0
        else:
            D = nrng.integers(0, vmax, size=(G, n_rand))
        z = score_many(coef, itr, D, mode, E); n_eval += n_rand
        for k in np.argsort(-z)[:n_ascend]:
            mode = mode0
            d, ztr = ascent(coef, itr, D[:, k].astype(np.int64), mode, E, vmax, sweeps=3, rng=rng, grid=grid)
            mode, ztr = ascent_offset(coef, itr, d, mode, E)
            d, ztr = ascent(coef, itr, d, mode, E, vmax, sweeps=1, rng=rng, grid=grid)
            n_eval += 4 * G * (len(grid) if grid else vmax) + 10
            if S is not None:
                for rnd in range(2):
                    for g in rng.sample(G_all, len(G_all)):
                        S2 = set(S) ^ {g}
                        if len(S2) < 2: continue
                        m2 = subset_mask(TT, S2); i2 = np.where(m2 & tok_tr)[0]
                        if len(i2) < 30: continue
                        z2 = score_one(coef, i2, d, mode, E); n_eval += 1
                        if z2 > ztr: S, ztr, itr = S2, z2, i2
                    d, ztr = ascent(coef, itr, d, mode, E, vmax, sweeps=2, rng=rng, grid=grid)
                    n_eval += 2 * G * (len(grid) if grid else vmax)
                sel = subset_mask(TT, S); ite = np.where(sel & ~tok_tr)[0]
                sdesc = 'only{' + ''.join(sorted(S)) + '}'
                if len(ite) < 10: continue
            zte = score_one(coef, ite, d, mode, E)
            rows.append(dict(sel=sdesc, feat=list(feat), mode=list(mode), vmax=(len(grid) if grid else vmax), ntr=len(itr), nte=len(ite),
                             z_rand=float(z[k]), z_tr=ztr, z_te=zte,
                             d={TT['glyphs'][g]: int(d[g]) for g in range(G) if d[g] and coef[itr, g].sum() > 0}))
        if log and ci % 50 == 0:
            print('%s cfg %d evals %.2e best_tr %.2f best_te %.2f elapsed %.0fs' % (C['name'], ci, n_eval,
                  max([r['z_tr'] for r in rows] or [0]), max([r['z_te'] for r in rows] or [0]), time.time() - t0),
                  file=log, flush=True)
    return rows, E, TT, n_eval


def summarise(rows, top=20):
    rs = sorted(rows, key=lambda r: -r['z_tr'])[:top]
    zte = np.array([r['z_te'] for r in rs]) if rs else np.zeros(1)
    allte = np.array([r['z_te'] for r in rows]) if rows else np.zeros(1)
    return dict(n=len(rows), top_tr_mean=float(np.mean([r['z_tr'] for r in rs])) if rs else 0,
                top_te_mean=float(zte.mean()), top_te_max=float(zte.max()), all_te_mean=float(allte.mean()),
                all_te_sd=float(allte.std()), n_te_gt3=int((allte > 3).sum()), best=rs[:3])


# ------------------------------------------------------------------ systematic grid search (cycle 1 main)
def fixed_selectors(TT):
    G = TT['glyphs']; T = len(TT['words'])
    gal = [TT['gi'][g] for g in G if g in GALL]
    out = [('all', np.ones(T, bool))]
    for k in range(1, 7): out.append(('skel=%d' % k, TT['ln_skel'] == k))
    for a in (2, 3, 4): out.append(('skel%d-%d' % (a, a + 1), (TT['ln_skel'] >= a) & (TT['ln_skel'] <= a + 1)))
    if gal:
        out.append(('gallows', TT['cnt_full'][:, gal].sum(1) > 0)); out.append(('no-gallows', TT['cnt_full'][:, gal].sum(1) == 0))
    for k in ('first', 'last', 'pfirst', 'label'): out.append((k, TT[k].copy()))
    for j in range(4): out.append(('subset%d' % j, None))
    return out


def grid_configs():
    feats = [('count', sk, nm) for sk in (True, False) for nm in ('numeral', 'free')]
    feats += [('pos', True, B, fr) for B in range(2, 13) for fr in (True, False)]
    feats += [('pos', False, B, True) for B in range(2, 13)]
    modes = [('page', 'abs_clip', 0), ('page', 'rel_fwd', 0), ('page', 'rel_back', 0), ('folio', 'abs_clip', 0),
             ('quire', 'abs_clip', 0)]
    return feats, modes


def search_grid(C, seed=0, starts=4, n_rand=300, log=None, time_budget=None, E=None, max_cfg=None, mode_filter=None):
    import time
    t0 = time.time()
    if E is None: E = build_R(C)
    TT = token_table(C, E)
    tok_tr = split_types(TT, seed)
    sels = fixed_selectors(TT)
    feats, modes = grid_configs()
    cfgs = [(f, s, m) for f in range(len(feats)) for s in range(len(sels)) for m in range(len(modes))
            if mode_filter is None or modes[m][1] in mode_filter]
    rng = random.Random(1234); rng.shuffle(cfgs)          # same visiting order for every corpus
    if max_cfg: cfgs = cfgs[:max_cfg]
    nrng = np.random.default_rng(seed + 12)
    poscache = {}; rows = []; n_eval = 0; done = 0
    G_all = TT['glyphs']
    for ci, (fi, si, mi) in enumerate(cfgs):
        if time_budget and time.time() - t0 > time_budget: break
        feat = feats[fi]; sname, sel = sels[si]; mode0 = modes[mi]
        if feat[0] == 'count':
            coef = TT['cnt_skel'] if feat[1] else TT['cnt_full']
            grid = NUMERAL_GRID if feat[2] == 'numeral' else None
            vmax = _nsp(E, mode0[0]) if grid is None else None
        else:
            key = ('pos', feat[1], feat[2], feat[3])
            if key not in poscache: poscache[key] = pos_weights(TT, feat[1], feat[2], feat[3])
            coef = poscache[key]; grid = None; vmax = feat[2]
        S = None
        if sel is None:
            S = set(rng.sample(G_all, rng.randint(3, min(12, len(G_all))))); sel = subset_mask(TT, S)
        itr = np.where(sel & tok_tr)[0]; ite = np.where(sel & ~tok_tr)[0]
        if len(itr) < 30 or len(ite) < 30: continue
        done += 1
        G = coef.shape[1]
        if grid is not None:
            D = np.array(grid)[nrng.integers(0, len(grid), size=(G, n_rand))]; D[nrng.random((G, n_rand)) < 0.5] = 0
        else:
            D = nrng.integers(0, vmax, size=(G, n_rand))
        z = score_many(coef, itr, D, mode0, E); n_eval += n_rand
        st = list(np.argsort(-z)[:2]) + list(nrng.integers(0, n_rand, size=max(0, starts - 2)))
        best = None
        for k in st:
            mode = mode0; itr_k = itr; S_k = S
            d, ztr = ascent(coef, itr_k, D[:, k].astype(np.int64), mode, E, vmax, sweeps=3, rng=rng, grid=grid)
            mode, ztr = ascent_offset(coef, itr_k, d, mode, E)
            d, ztr = ascent(coef, itr_k, d, mode, E, vmax, sweeps=2, rng=rng, grid=grid)
            nv = len(grid) if grid is not None else vmax
            n_eval += 5 * G * nv + 10
            if S_k is not None:
                for g in rng.sample(G_all, len(G_all)):
                    S2 = set(S_k) ^ {g}
                    if len(S2) < 2: continue
                    i2 = np.where(subset_mask(TT, S2) & tok_tr)[0]
                    if len(i2) < 30: continue
                    z2 = score_one(coef, i2, d, mode, E); n_eval += 1
                    if z2 > ztr: S_k, ztr, itr_k = S2, z2, i2
                d, ztr = ascent(coef, itr_k, d, mode, E, vmax, sweeps=2, rng=rng, grid=grid)
            if best is None or ztr > best[0]: best = (ztr, d, mode, itr_k, S_k)
        ztr, d, mode, itr_k, S_k = best
        ite_k = ite if S_k is None else np.where(subset_mask(TT, S_k) & ~tok_tr)[0]
        if len(ite_k) < 10: continue
        zte = score_robust(coef, ite_k, d, mode, E, TT)
        rows.append(dict(sel=sname if S_k is None else 'only{' + ''.join(sorted(S_k)) + '}', feat=list(feat),
                         mode=list(mode), ntr=len(itr_k), nte=len(ite_k), z_tr=ztr, z_te=zte,
                         z_te_plain=score_one(coef, ite_k, d, mode, E), z_tr_robust=score_robust(coef, itr_k, d, mode, E, TT),
                         d={G_all[g]: int(d[g]) for g in range(G) if d[g] and coef[itr_k, g].sum() > 0}))
        if log and done % 100 == 0:
            print('%s cfg %d/%d evals %.2e best_tr %.2f best_te %.2f elapsed %.0fs' % (C['name'], ci, len(cfgs), n_eval,
                  max(r['z_tr'] for r in rows), max(r['z_te'] for r in rows), time.time() - t0), file=log, flush=True)
    return rows, E, TT, n_eval, dict(visited=ci + 1, total=len(cfgs), done=done)
