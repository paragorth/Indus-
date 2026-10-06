"""v72 THE CRACK ATTEMPT: one explicit, machine-checkable model of the Voynich writing system.

Model (two layers)
  MESSAGE layer : a sequence of payload tokens p_1..p_n per page (what the payload carries).
  SURFACE layer : each payload token is written as a glyph word w = S(p, context) by
                  - padding slots       ch~sh (C), e~ee~d after gallows/ch and before final y, optional q-
                  - gallows twins       k~t, p~f chosen by run habit and line position
                  - benches             ch+gallows written as a benched gallows (cth, ckh, cph, cfh)
                  - line machinery      a line-initial marker glyph (y s d t) on some lines, p/f on paragraph
                                        starts, final -n written -m at line end; lines are sealed (no coupling
                                        across breaks)
                  - edge marking        q- after a -y word (junction pairing)
  The EXTRACTION rule E inverts the surface layer (deterministically, word by word, with the line flags).

Corpus format: list of pages; page = dict(id, sec, lang, lines=[dict(w=[words], ps=bool)]); a word is a string
of one character per glyph unit (vlib units: C=ch S=sh T=cth K=ckh P=cph F=cfh, lowercase EVA otherwise).
Payload tokens use o a i n y l r s ... plus G (k/t), H (p/f), C (ch/sh).
"""
import os, sys, json, math, random, re, hashlib, collections, zlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v72_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def jload(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ corpora
def voynich(name='ZL3b'):
    import vlib
    L = vlib.load_voynich(name, ltypes=('P',))
    pages = collections.OrderedDict()
    for r in L:
        ws = [''.join(vlib.glyphs(w)) for w in r['words'] if re.fullmatch(r'[a-z]+', w)]
        if not ws: continue
        p = pages.setdefault(r['folio'], dict(id=r['folio'], sec=r['illus'], lang=r['lang'] or '-', hand=r['hand'] or '-',
                                              quire=r['quire'], lines=[]))
        p['lines'].append(dict(w=ws, ps=bool(r['para_start'])))
    return [p for p in pages.values() if sum(len(l['w']) for l in p['lines']) >= 20]


def leaf_half(pid):
    """discovery (0) / holdout (1) by leaf number parity (both sides of a leaf stay together)."""
    m = re.match(r'f(\d+)', pid)
    return int(m.group(1)) % 2 if m else zlib.crc32(pid.encode()) % 2


def _pages_from_entries(entries, sec_of, line_w=8, page_tok=160, cap=36000, prefix='x'):
    """entries: list of (sec, [words]) ; each entry is a paragraph; pages filled to ~page_tok words, one section."""
    pages, cur, n = [], None, 0
    for sec, ws in entries:
        if n >= cap: break
        if cur is None or cur['_n'] >= page_tok or cur['sec'] != sec:
            cur = dict(id='%s%03d' % (prefix, len(pages)), sec=sec, lang='-', hand='-', quire='-', lines=[], _n=0)
            pages.append(cur)
        for i in range(0, len(ws), line_w):
            cur['lines'].append(dict(w=ws[i:i + line_w], ps=(i == 0)))
        cur['_n'] += len(ws); n += len(ws)
    for p in pages: p.pop('_n')
    return pages


def brumati_plain():
    import v49_lib
    ents = v49_lib.brumati_entries()
    order = {}
    for e in ents: order.setdefault(e['cls'], len(order))
    E = []
    for e in ents:
        ws = []
        for pi, p in enumerate(e['paras']):
            if pi == 0: p = re.sub(r'^(\d+a?\.?|[IVXLC]+°?\.)\s', '', p)
            ws += v49_lib.brumati_words(p)
        if len(ws) >= 15: E.append(('G%d' % min(5, order[e['cls']] // 4), ws))
    return _pages_from_entries(E, None, prefix='br')


def isidore_plain():
    import unicodedata
    t = open(os.path.join(ROOT, 'data', 'plain', 'la.txt'), encoding='utf-8').read()
    paras = [p.strip() for p in re.split(r'\n\s*\n', t) if p.strip()]
    E, cap = [], 0
    for p in paras:
        if p.startswith('CAPUT'): cap += 1; continue
        s = unicodedata.normalize('NFD', p.lower()); s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        s = re.sub(r'\(\d+[a-z]?\)', ' ', s)
        ws = [w.replace('j', 'i').replace('v', 'u') for w in re.findall(r'[a-z]+', s)]
        if len(ws) >= 12: E.append(('L%d' % min(5, cap // 12), ws))
    return _pages_from_entries(E, None, prefix='is')


def german_plain():
    C = json.load(open(os.path.join(ROOT, 'data', 'v30_ckpt', 'corpora.json')))['corpora']
    E = []
    docs = C['G_Alem']
    for i, ws in enumerate(docs):
        ws = [re.sub(r'[^a-z]', '', w.lower()) for w in ws]; ws = [w for w in ws if w]
        if len(ws) >= 12: E.append(('D%d' % (4 * i // len(docs)), ws))
    return _pages_from_entries(E, None, prefix='de')


# ------------------------------------------------------------------ planted surface machinery
PSYM = ['o', 'a', 'i', 'n', 'y', 'l', 'r', 's', 'G', 'H', 'C']


def payload_code(words, seed=72, mode='verbose'):
    """random prefix-free verbose code letters -> 1-2 payload symbols (frequent letters get single symbols);
    mode='merge': lossy many-to-one letter -> one payload symbol (frequency-balanced), Voynich-like word length."""
    rng = random.Random(seed)
    cnt = Counter(c for w in words for c in w)
    letters = [c for c, _ in cnt.most_common()]
    if mode == 'merge':
        sym = PSYM[:]; rng.shuffle(sym); load = {s: 0 for s in sym}; code = {}
        for c in letters:
            s = min(sym, key=lambda x: (load[x], sym.index(x))); code[c] = s; load[s] += cnt[c]
        return code
    sym = PSYM[:]; rng.shuffle(sym)
    singles, prefixes = sym[:6], sym[6:]
    twos = [a + b for a in prefixes for b in PSYM]; rng.shuffle(twos)
    code = {}
    for i, c in enumerate(letters):
        code[c] = singles[i] if i < len(singles) else twos[i - len(singles)]
    return code


def encode_payload(pages, code):
    return [dict(p, lines=[dict(l, w=[''.join(code[c] for c in w) for w in l['w']], orig=l['w']) for l in p['lines']])
            for p in pages]


def _pick(rng, d):
    r = rng.random(); s = 0
    for k, v in d:
        s += v
        if r < s: return k
    return d[-1][0]


PAD_MID = [('', 0.35), ('e', 0.30), ('ee', 0.15), ('d', 0.10), ('ed', 0.10)]
PAD_Y = [('', 0.40), ('e', 0.20), ('d', 0.25), ('ed', 0.15)]
MARK = ['y', 's', 'd', 't']


def surface(pages, seed=7, strength=1.0):
    """payload pages -> Voynich-like surface pages (planted machinery, see module doc)."""
    rng = random.Random(seed)
    out = []
    for p in pages:
        nl = []
        prev_mark = None
        for li, l in enumerate(p['lines']):
            hab = dict(G=rng.random() < 0.5, H=rng.random() < 0.5, C=rng.random() < 0.4)   # line-reset habits
            ws = []
            for wi, pw in enumerate(l['w']):
                s = ''
                syms = list(pw)
                for j, c in enumerate(syms):
                    if c in 'GHC':
                        if rng.random() < 0.25 * strength: hab[c] = not hab[c]
                        if c == 'C':
                            pS = 0.75 if (l['ps'] or wi == 0) else (0.2 if hab['C'] else 0.1)
                            g = 'S' if rng.random() < pS else 'C'
                        elif c == 'G':
                            g = 't' if (hab['G'] or (l['ps'] and rng.random() < 0.3)) else 'k'
                        else:
                            g = 'f' if hab['H'] else 'p'
                        if g in 'kt' and s.endswith('C') and rng.random() < 0.3 * strength:
                            s = s[:-1] + ('T' if g == 't' else 'K')
                        elif g in 'pf' and s.endswith('C') and rng.random() < 0.3 * strength:
                            s = s[:-1] + ('F' if g == 'f' else 'P')
                        else:
                            s += g
                        if j < len(syms) - 1 and rng.random() < strength: s += _pick(rng, PAD_MID)
                    elif c == 'y' and j == len(syms) - 1 and j > 0:
                        if rng.random() < strength: s += _pick(rng, PAD_Y)
                        s += 'y'
                    else:
                        s += c
                if wi > 0 and rng.random() < strength * (0.55 if ws[-1].endswith('y') else 0.08): s = 'q' + s
                if wi == len(l['w']) - 1 and s.endswith('n') and rng.random() < 0.3 * strength: s = s[:-1] + 'm'
                ws.append(s)
            if ws and rng.random() < 0.5 * strength:
                m = rng.choice([x for x in MARK if x != prev_mark]); prev_mark = m
                ws[0] = m + ws[0]
            if ws and l['ps'] and rng.random() < 0.5 * strength:
                ws[0] = rng.choice('pf') + ws[0]
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


# ------------------------------------------------------------------ extraction rules (the inverse surface map)
def _core(w, bench='split', drop_d=True, drop_q=True):
    if drop_q and len(w) > 1 and w[0] == 'q': w = w[1:]
    w = w.replace('S', 'C')
    if bench == 'split':
        w = w.replace('T', 'CG').replace('K', 'CG').replace('P', 'CH').replace('F', 'CH')
    elif bench == 'drop':
        w = w.replace('T', '').replace('K', '').replace('P', '').replace('F', '')
    w = w.replace('k', 'G').replace('t', 'G').replace('p', 'H').replace('f', 'H')
    w = w.replace('e', '')
    if drop_d: w = w.replace('d', '')
    return w


def extract_word(w, li0, para0, rule):
    """rule: dict with keys bench, drop_d, line (bool), lfin (l->y merge), beg (keep first k payload symbols or 0)."""
    if rule.get('identity'): return w
    if rule.get('line'):
        if para0 and len(w) > 1 and w[0] in 'pf': w = w[1:]
        if li0 and len(w) > 1 and w[0] in 'ysdt': w = w[1:]
        if len(w) > 1 and w[-1] in 'mg': w = w[:-1] + 'n'
    x = _core(w, rule.get('bench', 'split'), rule.get('drop_d', True))
    if rule.get('lfin') and len(x) > 1 and x[-1] == 'l': x = x[:-1] + 'y'
    if rule.get('beg'): x = x[:rule['beg']]
    return x or '_'


RULES = {
    'E0_identity': dict(identity=True),
    'E1_padding': dict(bench='split', drop_d=True, line=False),
    'E1b_benchdrop': dict(bench='drop', drop_d=True, line=False),
    'E1c_keepd': dict(bench='split', drop_d=False, line=False),
    'E2_line': dict(bench='split', drop_d=True, line=True),
    'E3_line_lfin': dict(bench='split', drop_d=True, line=True, lfin=True),
    'E4_begin3': dict(bench='split', drop_d=True, line=True, beg=3),
}


def rule_hash(name):
    import inspect
    src = inspect.getsource(_core) + inspect.getsource(extract_word) + json.dumps(RULES[name], sort_keys=True) + name
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def extract(pages, rule):
    out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            nl.append(dict(l, w=[extract_word(w, i == 0, l['ps'] and i == 0, rule) for i, w in enumerate(l['w'])]))
        out.append(dict(p, lines=nl))
    return out


# ------------------------------------------------------------------ generators (keep page, line, paragraph shape)
def _words_by_sec(pages):
    by = defaultdict(list)
    for p in pages:
        for l in p['lines']: by[p['sec']] += l['w']
    return by


def _cum(c):
    ks = list(c); v = np.cumsum([c[k] for k in ks]).astype(float); return ks, v / v[-1]


def _draw(kv, rng):
    ks, cp = kv
    return ks[min(int(np.searchsorted(cp, rng.random())), len(ks) - 1)]


def gen_wshuf(pages, seed=1):
    rng = random.Random(seed); by = _words_by_sec(pages)
    for v in by.values(): rng.shuffle(v)
    it = {k: iter(v) for k, v in by.items()}
    return [dict(p, lines=[dict(l, w=[next(it[p['sec']]) for _ in l['w']]) for l in p['lines']]) for p in pages]


def gen_mk2(pages, seed=1):
    """glyph-unit trigram per section (word-internal, word start/end symbols), with a separate model for
    line-initial words (keeps the line-initial glyph effect)."""
    rng = random.Random(seed)
    T = defaultdict(lambda: defaultdict(Counter))
    for p in pages:
        for l in p['lines']:
            for i, w in enumerate(l['w']):
                k = (p['sec'], i == 0)
                x = '^^' + w + '$'
                for j in range(2, len(x)): T[k][x[j - 2:j]][x[j]] += 1
    TT = {k: {c: _cum(v) for c, v in t.items()} for k, t in T.items()}
    def word(k):
        x = '^^'
        while len(x) < 20:
            c = _draw(TT[k][x[-2:]], rng)
            if c == '$': break
            x += c
        return x[2:] or 'o'
    return [dict(p, lines=[dict(l, w=[word((p['sec'], i == 0)) for i in range(len(l['w']))]) for l in p['lines']]) for p in pages]


def gen_selfcit(pages, seed=1, window=60, p_mod=0.5, p_copy=0.6):
    """self-citation: copy a word from the last `window` words of the page, change one glyph with prob p_mod,
    else draw from the section's word law (a copy-and-vary generator)."""
    rng = random.Random(seed); by = _words_by_sec(pages)
    law = {k: _cum(Counter(v)) for k, v in by.items()}
    gl = {k: _cum(Counter(c for w in v for c in w)) for k, v in by.items()}
    out = []
    for p in pages:
        hist = []; nl = []
        for l in p['lines']:
            ws = []
            for _ in l['w']:
                if len(hist) >= 3 and rng.random() < p_copy:
                    w = rng.choice(hist[-window:])
                    if rng.random() < p_mod:
                        j = rng.randrange(len(w)); w = w[:j] + _draw(gl[p['sec']], rng) + w[j + 1:]
                else:
                    w = _draw(law[p['sec']], rng)
                ws.append(w); hist.append(w)
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


def gen_junction(pages, seed=1):
    """v17 junction resynthesis: next word drawn from real words (same section) that follow a word with the
    same last glyph; line-initial words drawn from the section's line-initial words."""
    rng = random.Random(seed)
    F = defaultdict(list); I = defaultdict(list)
    for p in pages:
        for l in p['lines']:
            I[p['sec']].append(l['w'][0])
            for a, b in zip(l['w'], l['w'][1:]): F[(p['sec'], a[-1])].append(b)
    out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            ws = [rng.choice(I[p['sec']])]
            for _ in l['w'][1:]:
                c = F.get((p['sec'], ws[-1][-1])) or I[p['sec']]
                ws.append(rng.choice(c))
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


GENS = dict(WSHUF=gen_wshuf, MK2=gen_mk2, SELFCIT=gen_selfcit, JUNC=gen_junction,
            SC10=lambda pages, seed=1: gen_selfcit(pages, seed, p_copy=0.10))


# ------------------------------------------------------------------ the payload (message-layer) model
def tokens(pages):
    """flat token list with flags per page: list of (page_idx, line_idx, pos, token, para_first_line)."""
    T = []
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']):
            for wi, w in enumerate(l['w']):
                T.append((pi, li, wi, w, l['ps']))
    return T


class Spell:
    """char trigram spelling model over payload strings (for out-of-vocabulary tokens)."""
    def __init__(self, toks):
        self.c = defaultdict(Counter)
        for w in toks:
            x = '^^' + w + '$'
            for j in range(2, len(x)): self.c[x[j - 2:j]][x[j]] += 1
        self.A = len({ch for w in toks for ch in w}) + 1
        self.tot = {k: sum(v.values()) for k, v in self.c.items()}
    def logp(self, w):
        x = '^^' + w + '$'; s = 0.0
        for j in range(2, len(x)):
            k = x[j - 2:j]; n = self.tot.get(k, 0)
            s += math.log((self.c[k][x[j]] + 0.1) / (n + 0.1 * self.A)) if k in self.c else math.log(1.0 / self.A)
        return s


def _edits1(w, alpha):
    out = set()
    for i in range(len(w) + 1):
        if i < len(w): out.add(w[:i] + w[i + 1:])
        for a in alpha:
            out.add(w[:i] + a + w[i:])
            if i < len(w) and a != w[i]: out.add(w[:i] + a + w[i + 1:])
    out.discard(w)
    return out


COMPS_GEN = ['U', 'SEC', 'POS', 'CACHE', 'NEAR']
COMPS_MSG = ['PAGE', 'BIG']


class PayloadModel:
    """Component probabilities for every token of `test` pages using tables from `train` pages.
    U    corpus unigram (Witten-Bell escape to a spelling model)
    SEC  section unigram (smoothed to U)
    POS  line-position unigram: line-initial / paragraph-first line / other (smoothed to U)
    CACHE exact copy of one of the last W tokens of the page            (generator: self-citation)
    NEAR one-edit variant of one of the last W tokens                   (generator: copy-and-vary)
    PAGE exact recurrence of a token from earlier on the page, beyond W  (message: page topic)
    BIG  within-line payload bigram, absolute discounting to SEC        (message: sequence / syntax)
    """
    def __init__(self, train, W=20, beta=40.0, D=0.75):
        self.W, self.beta, self.D = W, beta, D
        T = tokens(train)
        toks = [t[3] for t in T]
        self.N = len(toks); self.cnt = Counter(toks); self.V = len(self.cnt)
        self.esc = self.V / (self.N + self.V)
        self.sp = Spell(toks)
        self.alpha = sorted({c for w in self.cnt for c in w})
        self.sec = defaultdict(Counter); self.pos = defaultdict(Counter); self.big = defaultdict(Counter)
        for (pi, li, wi, w, ps) in T:
            self.sec[train[pi]['sec']][w] += 1
            self.pos[self._pc(wi, ps)][w] += 1
        for p in train:
            for l in p['lines']:
                for a, b in zip(l['w'], l['w'][1:]): self.big[a][b] += 1
        self.secN = {k: sum(v.values()) for k, v in self.sec.items()}
        self.posN = {k: sum(v.values()) for k, v in self.pos.items()}
        self.bigN = {k: sum(v.values()) for k, v in self.big.items()}
        self.bigT = {k: len(v) for k, v in self.big.items()}
        self.jb = defaultdict(Counter)                      # junction class: last glyph of previous token
        for p in train:
            for l in p['lines']:
                for a, b in zip(l['w'], l['w'][1:]): self.jb[a[-1]][b] += 1
        self.jbN = {k: sum(v.values()) for k, v in self.jb.items()}
        self.jbT = {k: len(v) for k, v in self.jb.items()}
        self._e1 = {}
        self._pu = {}
        self._nv = {}

    def nvoc(self, q):
        r = self._nv.get(q)
        if r is None:
            r = sum(1 for x in self.e1(q) if x in self.cnt); self._nv[q] = r
        return r

    @staticmethod
    def _pc(wi, ps):
        return 'I' if wi == 0 else ('P' if ps else 'O')

    def pu(self, w):
        r = self._pu.get(w)
        if r is None:
            c = self.cnt.get(w, 0)
            r = (1 - self.esc) * c / self.N if c else self.esc * math.exp(self.sp.logp(w))
            r = max(r, 1e-12); self._pu[w] = r
        return r

    def e1(self, w):
        r = self._e1.get(w)
        if r is None:
            r = _edits1(w, self.alpha); self._e1[w] = r
        return r

    def comp_probs(self, test, comps):
        """returns matrix [ntok x ncomp] of probabilities and list of position classes."""
        rows, pcs = [], []
        for p in test:
            hist = []; seen = set(); seen_new = set()
            secC = self.sec.get(p['sec'], Counter()); secN = self.secN.get(p['sec'], 0)
            for l in p['lines']:
                prev = None
                for wi, w in enumerate(l['w']):
                    pu = self.pu(w)
                    pc = self._pc(wi, l['ps'])
                    psec = (secC.get(w, 0) + self.beta * pu) / (secN + self.beta)
                    r = []
                    for c in comps:
                        if c == 'U': r.append(pu)
                        elif c == 'SEC': r.append(psec)
                        elif c == 'POS':
                            r.append((self.pos[pc].get(w, 0) + self.beta * pu) / (self.posN.get(pc, 0) + self.beta))
                        elif c == 'CACHE':
                            win = hist[-self.W:]
                            r.append(win.count(w) / len(win) if win else pu)
                        elif c == 'NEAR':
                            win = hist[-self.W:]
                            if not win: r.append(pu); continue
                            okw = (w in self.cnt) or (w in seen)
                            s = 0.0; nz = 0
                            for q in win:
                                nb = self.e1(q)
                                k = self.nvoc(q) + sum(1 for x in seen_new if x in nb)
                                if k == 0: nz += 1
                                elif okw and w in nb: s += 1.0 / k
                            r.append((s + nz * pu) / len(win))
                        elif c == 'PAGE':
                            far = hist[:-self.W] if len(hist) > self.W else []
                            r.append(far.count(w) / len(far) if far else psec)
                        elif c == 'BIG':
                            if prev is None or prev not in self.big: r.append(psec)
                            else:
                                n = self.bigN[prev]; t = self.bigT[prev]
                                r.append((max(self.big[prev].get(w, 0) - self.D, 0) + self.D * t * psec) / n)
                        elif c == 'JBIG':
                            if prev is None or prev[-1] not in self.jb: r.append(psec)
                            else:
                                k = prev[-1]; n = self.jbN[k]; t = self.jbT[k]
                                r.append((max(self.jb[k].get(w, 0) - self.D, 0) + self.D * t * psec) / n)
                        elif c == 'BIGJ':
                            # word bigram backing off to the junction-class bigram (so it can only add what
                            # the identity of the previous token says beyond its last glyph)
                            if prev is None: r.append(psec); continue
                            k = prev[-1]
                            pj = ((max(self.jb[k].get(w, 0) - self.D, 0) + self.D * self.jbT[k] * psec) / self.jbN[k]
                                  if k in self.jb else psec)
                            if prev not in self.big: r.append(pj)
                            else:
                                n = self.bigN[prev]; t = self.bigT[prev]
                                r.append((max(self.big[prev].get(w, 0) - self.D, 0) + self.D * t * pj) / n)
                    rows.append(r); pcs.append(pc)
                    hist.append(w); seen.add(w); prev = w
                    if w not in self.cnt: seen_new.add(w)
        return np.maximum(np.array(rows, float), 1e-12), pcs


def em_weights(P, iters=60):
    k = P.shape[1]; lam = np.ones(k) / k
    for _ in range(iters):
        R = P * lam; R /= R.sum(1, keepdims=True); lam = R.mean(0)
    return lam


def all_probs(train, test, W=20, seed=0):
    """component probabilities (all of COMPS_ALL) for the test tokens, tables from train; BIGSH = the bigram
    table trained on section-shuffled training payloads ('message replaced by noise')."""
    M = PayloadModel(train, W=W)
    P, pc = M.comp_probs(test, COMPS_GEN + COMPS_MSG)
    sh = gen_wshuf(train, seed + 101)
    M.big = defaultdict(Counter)
    for p in sh:
        for l in p['lines']:
            for a, b in zip(l['w'], l['w'][1:]): M.big[a][b] += 1
    M.bigN = {k: sum(v.values()) for k, v in M.big.items()}; M.bigT = {k: len(v) for k, v in M.big.items()}
    Psh, _ = M.comp_probs(test, ['BIG'])
    return np.hstack([P, Psh]), np.array(pc)


COMPS_ALL = COMPS_GEN + COMPS_MSG + ['BIGSH']
MODELS = {'NOISE': ['U'], 'GEN': COMPS_GEN, 'MSG': COMPS_GEN + COMPS_MSG, 'GEN+PAGE': COMPS_GEN + ['PAGE'],
          'GEN+BIG': COMPS_GEN + ['BIG'], 'MSGSH': COMPS_GEN + ['PAGE', 'BIGSH']}


def probs_generic(train, test, comps, W=20):
    M = PayloadModel(train, W=W)
    P, pc = M.comp_probs(test, comps)
    return P, np.array(pc)


def ladder_generic(train, test, comps, models, W=20, seed=0, folds=4):
    """like ladder() for any component list and any dict of models (subsets of comps)."""
    rng = random.Random(seed)
    idx = list(range(len(train))); rng.shuffle(idx)
    Ps, PCs = [], []
    for f in range(folds):
        tes = set(idx[f::folds])
        P, pc = probs_generic([train[i] for i in idx if i not in tes], [train[i] for i in sorted(tes)], comps, W)
        Ps.append(P); PCs.append(pc)
    P = np.vstack(Ps); PCs = np.concatenate(PCs)
    Pt, pct = probs_generic(train, test, comps, W)
    col = {c: i for i, c in enumerate(comps)}
    res = {}
    for m, cs in models.items():
        ci = [col[c] for c in cs]
        lam = {g: em_weights(P[PCs == g][:, ci]) if (PCs == g).sum() > 20 else np.ones(len(ci)) / len(ci) for g in 'IPO'}
        mix = np.zeros(len(pct))
        for g in 'IPO':
            s = pct == g
            if s.any(): mix[s] = Pt[s][:, ci] @ lam[g]
        res[m] = -np.log2(mix)
    return res


def ladder(train, test, W=20, seed=0, folds=4):
    """held-out bits per payload token for NOISE (U), GEN (U SEC POS CACHE NEAR), MSG (GEN + PAGE BIG),
    MSGSH (bigram table trained on section-shuffled payloads), mixture weights cross-fitted on train
    (separately for line-initial / paragraph-first-line / other positions)."""
    rng = random.Random(seed)
    idx = list(range(len(train))); rng.shuffle(idx)
    Ps, PCs = [], []
    for f in range(folds):
        tes = set(idx[f::folds])
        P, pc = all_probs([train[i] for i in idx if i not in tes], [train[i] for i in sorted(tes)], W, seed + f)
        Ps.append(P); PCs.append(pc)
    P = np.vstack(Ps); PCs = np.concatenate(PCs)
    Pt, pct = all_probs(train, test, W, seed)
    col = {c: i for i, c in enumerate(COMPS_ALL)}
    res, pt = {}, {}
    for m, comps in MODELS.items():
        ci = [col[c] for c in comps]
        lam = {g: em_weights(P[PCs == g][:, ci]) if (PCs == g).sum() > 20 else np.ones(len(ci)) / len(ci) for g in 'IPO'}
        mix = np.zeros(len(pct))
        for g in 'IPO':
            s = pct == g
            if s.any(): mix[s] = Pt[s][:, ci] @ lam[g]
        pt[m] = -np.log2(mix)
        res[m] = dict(bits=float(pt[m].mean()), n=len(mix), lam={g: [float(x) for x in v] for g, v in lam.items()})
    d = pt['GEN'] - pt['MSG']
    res['gain_msg'] = float(d.mean()); res['gain_msg_z'] = float(d.mean() / (d.std() / math.sqrt(len(d)) + 1e-12))
    res['gain_gen'] = float((pt['NOISE'] - pt['GEN']).mean())
    res['gain_page'] = float((pt['GEN'] - pt['GEN+PAGE']).mean())
    res['gain_big'] = float((pt['GEN'] - pt['GEN+BIG']).mean())
    res['gain_msgsh'] = float((pt['GEN'] - pt['MSGSH']).mean())
    res['per_tok'] = pt
    return res


# ------------------------------------------------------------------ surface-layer cost
def surface_bits(train_s, train_p, test_s, test_p):
    """bits per word of the surface choice w given its payload p, with and without context
    (line-initial, paragraph-first line, previous word's last glyph). Witten-Bell backoff ctx -> p -> spelling."""
    def ctx(l, i):
        return ('I' if i == 0 else ('P' if l['ps'] else 'O'), l['w'][i - 1][-1] if i > 0 else '^')
    cp = defaultdict(Counter); cpc = defaultdict(Counter); allw = []
    for ps_, pp_ in zip(train_s, train_p):
        for ls, lp in zip(ps_['lines'], pp_['lines']):
            for i, (w, p) in enumerate(zip(ls['w'], lp['w'])):
                cp[p][w] += 1; cpc[(p,) + ctx(ls, i)][w] += 1; allw.append(w)
    sp = Spell(allw)
    def wb(C, w, back):
        n = sum(C.values()); t = len(C)
        if n == 0: return back
        return (C.get(w, 0) + t * back) / (n + t)
    b0 = b1 = 0.0; n = 0
    for ps_, pp_ in zip(test_s, test_p):
        for ls, lp in zip(ps_['lines'], pp_['lines']):
            for i, (w, p) in enumerate(zip(ls['w'], lp['w'])):
                spl = math.exp(sp.logp(w))
                q0 = wb(cp.get(p, Counter()), w, spl)
                q1 = wb(cpc.get((p,) + ctx(ls, i), Counter()), w, q0)
                b0 -= math.log2(max(q0, 1e-12)); b1 -= math.log2(max(q1, 1e-12)); n += 1
    return dict(surf_bits=b0 / n, surf_bits_ctx=b1 / n, n=n)


def recovery(pay_pages, plain_pages):
    """planted controls: does the extracted payload give back the real words? purity both ways."""
    pairs = Counter()
    for pp, op in zip(pay_pages, plain_pages):
        for lp, lo in zip(pp['lines'], op['lines']):
            for a, b in zip(lp['w'], lo['w']): pairs[(a, b)] += 1
    by_a = defaultdict(Counter); by_b = defaultdict(Counter)
    for (a, b), c in pairs.items(): by_a[a][b] += c; by_b[b][a] += c
    N = sum(pairs.values())
    pur = sum(max(v.values()) for v in by_a.values()) / N           # payload type -> one real word
    inv = sum(max(v.values()) for v in by_b.values()) / N           # real word -> one payload type
    return dict(purity=pur, inv_purity=inv, n_pay_types=len(by_a), n_word_types=len(by_b))
