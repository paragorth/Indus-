"""v21 FORGE THE VOYNICH, THEN CATCH THE FORGERY: shared library.

A corpus is a list of pages: {'id', 'sec', 'paras': [[line, ...], ...]} where a line is a list of word
strings in 'unit space' (Voynich: EVA glyph units collapsed to one character each, C=ch S=sh T=cth K=ckh
P=cph F=cfh; Latin / Italian: lower-case letters).

Forgers keep the page / paragraph / line / words-per-line skeleton and redraw every word.
Discriminators get page-level feature vectors of real and forged pages, cross-validated by page
(a page's real and forged versions always sit in the same fold).
"""
import os, sys, re, json, math, random, html, bisect
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'v21_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v21'
MAIN_SECS = ['HA', 'SB', 'BB', 'HB', 'PA', 'TB']


def U(w):
    return ''.join(vlib.glyphs(w))


# ------------------------------------------------------------------ corpora
def voynich_pages(name='ZL3b', minw=40):
    lines = vlib.load_voynich(name, drop_uncertain=True)
    pages = {}
    order = []
    cur = None
    for L in lines:
        f = L['folio']
        if f not in pages:
            sec = (L.get('illus') or 'x') + (L.get('lang') or 'x')
            if sec not in MAIN_SECS: sec = 'other'
            pages[f] = {'id': f, 'sec': sec, 'paras': [], 'hand': L.get('hand'), 'quire': L.get('quire')}
            order.append(f); cur = None
        pg = pages[f]
        ws = [U(w) for w in L['words']]
        ws = [w for w in ws if w and '?' not in w]
        if not ws: continue
        if L['para_start'] or cur is None:
            cur = []; pg['paras'].append(cur)
        cur.append(ws)
        if L.get('para_end'): cur = None
    out = [pages[f] for f in order]
    return [p for p in out if sum(len(l) for pa in p['paras'] for l in pa) >= minw]


def _clean_words(s):
    s = s.lower().replace('æ', 'ae').replace('œ', 'oe').replace('j', 'i').replace('v', 'u') if False else s.lower()
    s = re.sub(r'\[[^\]]*\]', ' ', s)
    s = re.sub(r'\([^)]*\)', ' ', s)
    return re.findall(r"[a-zàèéìíòóùúæœ]+", s)


def wrap(words, width):
    lines, cur, cw = [], [], 0
    for w in words:
        if cur and cw + len(w) + 1 > width:
            lines.append(cur); cur, cw = [], 0
        cur.append(w); cw += len(w) + 1
    if cur: lines.append(cur)
    return lines


def _build_pages(entries, width, minw, maxw, pfx):
    """entries: list of (sec, [paragraph word lists]) topical units -> pages of about minw..maxw words.
    A page never mixes two topical units; small units are dropped."""
    pages = []
    for sec, paras in entries:
        mp = []
        for p in paras:
            if not p: continue
            if mp and len(mp[-1]) < 25: mp[-1] = mp[-1] + p
            else: mp.append(list(p))
        chunk, n = [], 0
        for p in mp:
            if n >= minw and n + len(p) > maxw:
                pages.append((sec, chunk)); chunk, n = [], 0
            chunk.append(p); n += len(p)
        if chunk: pages.append((sec, chunk))
    out = []
    for sec, ch in pages:
        if sum(len(p) for p in ch) < minw: continue
        out.append({'id': f'{pfx}{len(out):03d}', 'sec': sec, 'paras': [wrap(p, width) for p in ch]})
    return out


def latin_herbal(width=48, minw=60, maxw=220):
    """Isidore, Etymologiae XVI (stones, metals) and XVII (plants, trees, herbs); chapter = topic."""
    entries = []
    for book in ('16', '17'):
        t = open(os.path.join(SCR, f'isid{book}.html'), encoding='latin-1').read()
        t = re.sub(r'<[^>]+>', ' ', t); t = html.unescape(t)
        parts = re.split(r'\n\s*([IVXL]+)\.\s+(DE [A-Z ]+)\.', t)
        for k in range(1, len(parts) - 2, 3):
            body = parts[k + 2]
            secs = re.split(r'\[\d+\]', body)
            paras = []
            for s in secs:
                s = re.sub(r'\(([^)]*)\)', ' ', s)
                ws = re.findall(r'[a-z]+', s.lower().replace('j', 'i').replace('v', 'u'))
                if ws: paras.append(ws)
            entries.append(('L' + book, paras))
    return _build_pages(entries, width, minw, maxw, 'la')


def italian_herbal(width=48, minw=60, maxw=220):
    """Brumati, Flora medico-economica (1844), it.wikisource; one genus entry = one topic."""
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(s):
            super().__init__(); s.out = []; s.inp = 0; s.cur = []; s.skip = 0
        def handle_starttag(s, tag, a):
            a = dict(a)
            if tag in ('script', 'style'): s.skip += 1
            if tag == 'p': s.inp = 1; s.cur = []
        def handle_endtag(s, tag):
            if tag in ('script', 'style'): s.skip -= 1
            if tag == 'p' and s.inp: s.out.append(''.join(s.cur)); s.inp = 0
        def handle_data(s, d):
            if s.inp and not s.skip: s.cur.append(d)

    entries = []
    classes = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV',
               'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV']
    for ci, c in enumerate(classes):
        p = P(); p.feed(open(os.path.join(SCR, 'flora', c + '.html'), encoding='utf-8').read())
        sec = 'I' + str(ci * 3 // len(classes))
        txt = '\n'.join(x for x in p.out if not x.startswith('<dc:'))
        txt = txt.replace('\xa0', ' ')
        txt = re.sub(r'\[p\.\s*\d+\s*modifica\]', '\n', txt)
        keep = []
        for ln in txt.split('\n'):
            st = ln.strip()
            if ('andria' in st or 'ginia' in st or 'Classe' in st) and len(st) < 70: continue
            keep.append(ln)
        txt = '\n'.join(keep)
        parts = re.split(r'(?:^|\s)[IVXLC]{1,8}°?\.\s+(?=[A-Z])', txt)
        for body in parts[1:]:
            paras = []
            for chunk in re.split(r'\n\s*\n|\s(?=\d{1,3}[a-z]?\.\s)', body):
                ws = _clean_words(re.sub(r'\d+', ' ', chunk))
                if ws: paras.append(ws)
            if paras: entries.append((sec, paras))
    return _build_pages(entries, width, minw, maxw, 'it')


def tokens(page):
    return [w for pa in page['paras'] for l in pa for w in l]


def ntok(C):
    return sum(len(tokens(p)) for p in C)


# ------------------------------------------------------------------ forgers
def _cum(c):
    ks = list(c.keys()); cs = np.cumsum(list(c.values())).tolist()
    return ks, cs


def _draw(rng, kc):
    ks, cs = kc
    return ks[bisect.bisect_right(cs, rng.random() * cs[-1])]


class Forger:
    """Ladder of word-redrawing forgers.

    scope: 'global' or 'sec' tables; pos: position-aware successor tables (p1, p2, mid, end);
    lam: probability of drawing from the page's own successor table (page topic);
    chain: line-first word conditioned on the previous line's first unit;
    redup: probability of repeating the previous word; cite: probability of copying a word from the
    previous line at a nearby position (self-citation); unigram: ignore junction (baseline).
    """

    def __init__(self, C, scope='sec', pos=False, lam=0.0, chain=False, redup=0.0, cite=0.0, unigram=False,
                 width=False, rich=False, cite_window=0, cite_edit=0.0, para_lam=False, end_room=8, name='forger'):
        self.cfg = dict(scope=scope, pos=pos, lam=lam, chain=chain, redup=redup, cite=cite, unigram=unigram)
        self.name = name
        self.scope, self.pos, self.lam, self.chain = scope, pos, lam, chain
        self.redup, self.cite, self.unigram, self.width, self.rich = redup, cite, unigram, width, rich
        self.cite_window, self.cite_edit, self.para_lam, self.end_room = cite_window, cite_edit, para_lam, end_room
        if cite_edit: self._learn_edits(C)
        T = defaultdict(Counter)
        for p in C:
            s = p['sec'] if scope == 'sec' else '*'
            pid = p['id']
            for qi, pa in enumerate(p['paras']):
                prevfirst = None
                Q = 'Q' + pid + '_' + str(qi)
                for li, ws in enumerate(pa):
                    T[('pf' if li == 0 else 'li', s)][ws[0]] += 1
                    T[('pf' if li == 0 else 'li', 'P' + pid)][ws[0]] += 1
                    if li > 0 and prevfirst is not None:
                        T[('ch', s, prevfirst)][ws[0]] += 1
                    prevfirst = ws[0][0]
                    for k in range(1, len(ws)):
                        key = ws[k - 1][-1]
                        pc = self._pc(k, len(ws))
                        T[('j', s, pc, key)][ws[k]] += 1
                        if rich:
                            T[('r', s, self._rk(ws, k))][ws[k]] += 1
                        T[('j', s, 'any', key)][ws[k]] += 1
                        T[('j', 'P' + pid, 'any', key)][ws[k]] += 1
                        if para_lam: T[('j', Q, 'any', key)][ws[k]] += 1
                        T[('u', s, pc)][ws[k]] += 1
                        T[('u', s)][ws[k]] += 1
        self.T = {k: _cum(v) for k, v in T.items()}
        self.Tn = {k: sum(v.values()) for k, v in T.items()}

    def _learn_edits(self, C, window=20):
        """Edit operations seen between real near-repeat pairs (edit distance 1, within `window` tokens)."""
        sub = defaultdict(Counter); ins = Counter(); dele = Counter(); kinds = Counter()
        for p in C:
            toks = tokens(p)
            for i in range(len(toks)):
                for j in range(max(0, i - window), i):
                    a, b = toks[j], toks[i]
                    if not near1(a, b): continue
                    if len(a) == len(b):
                        for x, y in zip(a, b):
                            if x != y: sub[x][y] += 1
                        kinds['sub'] += 1
                    elif len(b) > len(a):
                        for t in range(len(b)):
                            if b[:t] + b[t + 1:] == a: ins[b[t]] += 1; break
                        kinds['ins'] += 1
                    else:
                        for t in range(len(a)):
                            if a[:t] + a[t + 1:] == b: dele[a[t]] += 1; break
                        kinds['del'] += 1
                    break
        self.E_sub = {x: _cum(c) for x, c in sub.items()}
        self.E_ins = _cum(ins); self.E_del = dele; self.E_kind = _cum(kinds)

    def _edit(self, w, rng):
        for _ in range(5):
            kd = _draw(rng, self.E_kind)
            if kd == 'sub':
                pos = [i for i, c in enumerate(w) if c in self.E_sub]
                if not pos: continue
                i = rng.choice(pos); return w[:i] + _draw(rng, self.E_sub[w[i]]) + w[i + 1:]
            if kd == 'ins':
                i = rng.randint(0, len(w)); return w[:i] + _draw(rng, self.E_ins) + w[i:]
            if kd == 'del' and len(w) > 2:
                pos = [i for i, c in enumerate(w) if self.E_del.get(c)]
                if not pos: continue
                i = rng.choice(pos); return w[:i] + w[i + 1:]
        return w

    @staticmethod
    def _rk(line, k):
        """rich key: last unit of the previous word, its length bucket, first unit of the word two back."""
        a = line[k - 1]
        b = line[k - 2][0] if k >= 2 else '^'
        return (a[-1], min(len(a), 6), b)

    def _pc(self, k, n):
        if k == n - 1: return 'end'
        if not self.pos: return 'mid'
        return 'p1' if k == 1 else ('p2' if k == 2 else 'mid')

    def _get(self, *keys, minn=1):
        for k in keys:
            if k in self.T and self.Tn[k] >= minn: return self.T[k]
        return None

    def forge_page(self, p, rng):
        s = p['sec'] if self.scope == 'sec' else '*'
        P = 'P' + p['id']
        paras = []
        hist = []
        for qi, pa in enumerate(p['paras']):
            out = []; prevfirst = None; prevline = None
            Q = 'Q' + p['id'] + '_' + str(qi)
            if self.cite_window: hist = []
            for li, ws in enumerate(pa):
                n = len(ws)
                lt = 'pf' if li == 0 else 'li'
                src = None
                if self.lam and rng.random() < self.lam: src = self._get((lt, P), minn=1)
                if src is None and self.chain and li > 0: src = self._get(('ch', s, prevfirst), minn=5)
                if src is None: src = self._get((lt, s), (lt, '*'))
                w = _draw(rng, src); line = [w]
                target = sum(len(x) for x in ws) + len(ws) - 1   # units + spaces
                k = 0
                while True:
                    k += 1
                    if self.width:
                        room = target - (sum(len(x) for x in line) + len(line) - 1) - 1
                        if room < 2 or k > 3 * n + 5: break
                        pc = 'end' if room <= self.end_room else ('p1' if k == 1 else ('p2' if k == 2 else 'mid'))
                        if not self.pos and pc != 'end': pc = 'mid'
                    else:
                        if k >= n: break
                        pc = self._pc(k, n)
                    r = rng.random()
                    if self.redup and r < self.redup and pc != 'end':
                        nw = w
                    elif self.cite and self.cite_window and len(hist) + len(line) >= 3 and r < self.redup + self.cite:
                        pool = (hist + line)[-self.cite_window:-1]
                        nw = rng.choice(pool) if pool else w
                        if self.cite_edit and rng.random() < self.cite_edit: nw = self._edit(nw, rng)
                    elif self.cite and not self.cite_window and prevline and r < self.redup + self.cite:
                        j = min(len(prevline) - 1, max(0, k + rng.randint(-1, 1)))
                        nw = prevline[j]
                    else:
                        if self.unigram:
                            tab = self._get(('u', s, pc), ('u', s))
                        else:
                            key = w[-1]; tab = None
                            if self.lam and rng.random() < self.lam:
                                if self.para_lam: tab = self._get(('j', Q, 'any', key), minn=2)
                                if tab is None: tab = self._get(('j', P, 'any', key), minn=2)
                            if tab is None and self.rich and pc != 'end':
                                tab = self._get(('r', s, self._rk(line + ['x'], k)), minn=8)
                            if tab is None:
                                tab = self._get(('j', s, pc, key), ('j', s, 'any', key), ('u', s, pc), ('u', s))
                        nw = _draw(rng, tab)
                        if self.width and pc == 'end':
                            cands = [nw] + [_draw(rng, tab) for _ in range(11)]
                            nw = min(cands, key=lambda x: abs(len(x) - room))
                    line.append(nw); w = nw
                    if self.width and pc == 'end': break
                out.append(line); prevfirst = line[0][0]; prevline = line; hist.extend(line)
            paras.append(out)
        q = dict(p); q['paras'] = paras
        return q

    def forge(self, C, rng):
        return [self.forge_page(p, rng) for p in C]


class SlotForger:
    """Slot-template generator: each word built glyph by glyph from an in-word trigram (= the learned slot
    grammar), separate models for paragraph-first, line-first, line-final and medial words, per section.
    No word-to-word dependency."""

    def __init__(self, C, name='slot'):
        self.name = name
        T = defaultdict(Counter)
        for p in C:
            s = p['sec']
            for pa in p['paras']:
                for li, ws in enumerate(pa):
                    for k, w in enumerate(ws):
                        cls = 'pf' if (li == 0 and k == 0) else ('lf' if k == 0 else ('end' if k == len(ws) - 1 else 'mid'))
                        x = '^^' + w + '$'
                        for i in range(2, len(x)):
                            T[(s, cls, x[i - 2:i])][x[i]] += 1
                            T[('*', cls, x[i - 2:i])][x[i]] += 1
        self.T = {k: _cum(v) for k, v in T.items()}

    def word(self, rng, s, cls):
        x = '^^'
        while len(x) < 20:
            t = self.T.get((s, cls, x[-2:])) or self.T.get(('*', cls, x[-2:])) or self.T.get(('*', 'mid', x[-2:]))
            if t is None: break
            c = _draw(rng, t)
            if c == '$': break
            x += c
        return x[2:] or 'o'

    def forge(self, C, rng):
        out = []
        for p in C:
            paras = []
            for pa in p['paras']:
                o = []
                for li, ws in enumerate(pa):
                    n = len(ws)
                    o.append([self.word(rng, p['sec'], 'pf' if (li == 0 and k == 0) else ('lf' if k == 0 else ('end' if k == n - 1 else 'mid'))) for k in range(n)])
                paras.append(o)
            q = dict(p); q['paras'] = paras; out.append(q)
        return out


# ------------------------------------------------------------------ features
def near1(a, b):
    if a == b: return False
    if abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 3: return False
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b): a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]: i += 1
    return a[i:] == b[i + 1:]


def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])


class Featurizer:
    """Page-level features. Global frequency tables come from a reference corpus (the real one)."""

    def __init__(self, ref, labels=()):
        self.freq = Counter(w for p in ref for w in tokens(p))
        self.N = sum(self.freq.values())
        self.labels = set(labels)
        g = Counter(c for p in ref for w in tokens(p) for c in w)
        self.top_units = [c for c, _ in g.most_common(18)]
        # section-specificity
        self.secf = defaultdict(Counter)
        for p in ref:
            for w in tokens(p): self.secf[p['sec']][w] += 1
        self.secN = {s: sum(c.values()) for s, c in self.secf.items()}

    def page(self, p):
        F = {}
        lines = [l for pa in p['paras'] for l in pa]
        lpara = [pi for pi, pa in enumerate(p['paras']) for _ in pa]
        toks = [w for l in lines for w in l]
        n = len(toks)
        tl = [li for li, l in enumerate(lines) for _ in l]
        tp = [lpara[li] for li in tl]
        tk = [k for l in lines for k in range(len(l))]
        F['n_tok'] = n
        F['n_lines'] = len(lines)
        F['n_paras'] = len(p['paras'])
        cnt = Counter(toks)
        F['ttr'] = len(cnt) / n
        F['page_hapax'] = sum(1 for w, c in cnt.items() if c == 1) / n
        F['corpus_hapax'] = sum(1 for w in toks if self.freq[w] <= 1) / n
        F['rare5'] = sum(1 for w in toks if self.freq[w] <= 5) / n
        lf = np.array([math.log(self.freq[w] + 1) for w in toks])
        F['logfreq_mean'] = float(lf.mean()); F['logfreq_sd'] = float(lf.std())
        # section-specificity
        s = p['sec']
        if s in self.secf:
            v = [math.log((self.secf[s][w] + 0.5) / (self.secN[s] + 1)) - math.log((self.freq[w] + 0.5) / (self.N + 1)) for w in toks]
            F['sec_spec'] = float(np.mean(v))
        else:
            F['sec_spec'] = 0.0
        # repeats by distance (earliest previous occurrence)
        last = {}
        rep = Counter()
        for i, w in enumerate(toks):
            if w in last:
                j = last[w]
                if j == i - 1: rep['adj'] += 1
                elif tl[j] == tl[i]: rep['sameline'] += 1
                elif tl[i] - tl[j] == 1: rep['prevline'] += 1
                elif tl[i] - tl[j] <= 4: rep['line2_4'] += 1
                else: rep['line5p'] += 1
                if tp[j] != tp[i]: rep['otherpara'] += 1
            last[w] = i
        for k in ('adj', 'sameline', 'prevline', 'line2_4', 'line5p', 'otherpara'):
            F['rep_' + k] = rep[k] / n
        F['rep_any'] = sum(rep[k] for k in ('adj', 'sameline', 'prevline', 'line2_4', 'line5p')) / n
        # repeats of rare words (topic words): freq <= 20 in corpus, re-used on page
        F['rep_rare'] = sum(1 for w, c in cnt.items() if c >= 2 and self.freq[w] <= 20) / max(1, len(cnt))
        # near repeats (edit 1) by distance, sampled exhaustively for small pages
        nr = Counter()
        for i in range(n):
            for j in range(max(0, i - 40), i):
                if near1(toks[i], toks[j]):
                    d = tl[i] - tl[j]
                    nr['same' if d == 0 else ('prev' if d == 1 else 'far')] += 1
                    break
        for k in ('same', 'prev', 'far'):
            F['near_' + k] = nr[k] / n
        # vertical: same position in previous line
        vert = vert_last = vert_first = 0; vn = 0
        for li in range(1, len(lines)):
            a, b = lines[li - 1], lines[li]
            for k in range(min(len(a), len(b))):
                vn += 1
                vert += a[k] == b[k]; vert_last += a[k][-1] == b[k][-1]; vert_first += a[k][0] == b[k][0]
        F['vert_same'] = vert / max(1, vn); F['vert_last'] = vert_last / max(1, vn); F['vert_first'] = vert_first / max(1, vn)
        # drift: halves
        h = n // 2
        A, B = set(toks[:h]), set(toks[h:])
        F['jacc_halves'] = len(A & B) / max(1, len(A | B))
        ga = Counter(c for w in toks[:h] for c in w); gb = Counter(c for w in toks[h:] for c in w)
        keys = set(ga) | set(gb)
        va = np.array([ga[k] for k in keys], float); vb = np.array([gb[k] for k in keys], float)
        F['glyph_cos_halves'] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        if len(p['paras']) >= 2:
            sets = [set(w for l in pa for w in l) for pa in p['paras']]
            js = [len(sets[i] & sets[i + 1]) / max(1, len(sets[i] | sets[i + 1])) for i in range(len(sets) - 1)]
            F['jacc_paras'] = float(np.mean(js))
        else:
            F['jacc_paras'] = F['jacc_halves']
        # line shapes
        glen = [sum(len(w) for w in l) for l in lines]
        wl = [len(l) for l in lines]
        mwl = [np.mean([len(w) for w in l]) for l in lines]
        F['line_glen_sd'] = float(np.std(glen) / (np.mean(glen) + 1e-9))
        F['line_glen_ac1'] = _corr(glen[:-1], glen[1:])
        F['line_n_vs_mwl'] = _corr(wl, mwl)
        F['mwl_sd'] = float(np.std(mwl))
        F['wlen_mean'] = float(np.mean([len(w) for w in toks]))
        F['wlen_sd'] = float(np.std([len(w) for w in toks]))
        # word length autocorrelation within lines, first-unit agreement
        a1 = [(len(l[k]), len(l[k + 1])) for l in lines for k in range(len(l) - 1)]
        a2 = [(len(l[k]), len(l[k + 2])) for l in lines for k in range(len(l) - 2)]
        F['wlen_ac1'] = _corr(*zip(*a1)) if len(a1) > 3 else 0.0
        F['wlen_ac2'] = _corr(*zip(*a2)) if len(a2) > 3 else 0.0
        F['first_eq_lag1'] = float(np.mean([l[k][0] == l[k + 1][0] for l in lines for k in range(len(l) - 1)] or [0]))
        F['first_eq_lag2'] = float(np.mean([l[k][0] == l[k + 2][0] for l in lines for k in range(len(l) - 2)] or [0]))
        F['last_eq_lag1'] = float(np.mean([l[k][-1] == l[k + 1][-1] for l in lines for k in range(len(l) - 1)] or [0]))
        F['last_eq_lag2'] = float(np.mean([l[k][-1] == l[k + 2][-1] for l in lines for k in range(len(l) - 2)] or [0]))
        F['junc_same'] = float(np.mean([l[k][-1] == l[k + 1][0] for l in lines for k in range(len(l) - 1)] or [0]))
        # line-initial chain
        ch = [];
        for pa in p['paras']:
            for li in range(1, len(pa)):
                if li >= 2: ch.append(pa[li - 1][0][0] == pa[li][0][0])
        F['chain_same'] = float(np.mean(ch)) if ch else 0.0
        # ch / sh
        isC = ['C' in w for w in toks]; isS = ['S' in w for w in toks]
        F['C_rate'] = float(np.mean(isC)); F['S_rate'] = float(np.mean(isS))
        cs1 = cs2 = 0; n1 = n2 = 0
        for l in lines:
            for k in range(len(l) - 1):
                n1 += 1; cs1 += ('C' in l[k] and 'S' in l[k + 1]) or ('S' in l[k] and 'C' in l[k + 1])
            for k in range(len(l) - 2):
                n2 += 1; cs2 += ('C' in l[k] and 'S' in l[k + 2]) or ('S' in l[k] and 'C' in l[k + 2])
        F['CS_adj'] = cs1 / max(1, n1); F['CS_gap1'] = cs2 / max(1, n2)
        seq = ['C' if 'C' in w else 'S' for w in toks if ('C' in w) != ('S' in w)]
        F['CS_switch'] = float(np.mean([a != b for a, b in zip(seq, seq[1:])])) if len(seq) > 2 else 0.5
        # glyph variants / gallows
        units = Counter(c for w in toks for c in w); tu = sum(units.values())
        for c in 'TKPF':
            F['var_' + c] = units[c] / tu
        F['k_vs_t'] = (units['k'] + 1) / (units['t'] + units['k'] + 2)
        F['f_vs_p'] = (units['f'] + 1) / (units['f'] + units['p'] + 2)
        for c in self.top_units:
            F['u_' + c] = units[c] / tu
        # line make-up clumping (q-type vs a-type) and ch-e trade-off
        if len(lines) >= 3:
            q = [sum(w[0] == 'q' for w in l) / len(l) for l in lines]
            a = [sum(('a' in w) for w in l) / len(l) for l in lines]
            F['q_line_sd'] = float(np.std(q)); F['a_line_sd'] = float(np.std(a))
            F['q_a_corr'] = _corr(q, a)
            c = [sum(w.count('C') for w in l) / len(l) for l in lines]
            e = [sum(w.count('e') for w in l) / len(l) for l in lines]
            F['C_e_corr'] = _corr(c, e)
            mg = [sum(w.count('m') + w.count('g') for w in l) for l in lines]
            F['mg_line_var'] = float(np.var(mg) / (np.mean(mg) + 1e-3))
        else:
            for k in ('q_line_sd', 'a_line_sd', 'q_a_corr', 'C_e_corr', 'mg_line_var'): F[k] = 0.0
        F['label_rate'] = sum(1 for w in toks if w in self.labels) / n
        # paragraph structure: first line vs rest vocabulary, last line shortness
        fl = [w for pa in p['paras'] for w in pa[0]]
        rest = set(w for pa in p['paras'] for l in pa[1:] for w in l)
        F['para_first_in_rest'] = float(np.mean([w in rest for w in fl])) if fl and rest else 0.0
        F['para_first_wlen'] = float(np.mean([len(w) for w in fl])) if fl else 0.0
        F['gallows_first'] = float(np.mean([pa[0][0][0] in 'tkpf' for pa in p['paras']]))
        return F

    def matrix(self, C, keys=None):
        rows = [self.page(p) for p in C]
        if keys is None: keys = sorted(rows[0].keys())
        return np.array([[r.get(k, 0.0) for k in keys] for r in rows], float), keys


# ------------------------------------------------------------------ discriminators
def _auc(sr, sf):
    """AUC of real scores sr against forged scores sf (Mann-Whitney)."""
    x = np.r_[sr, sf]; r = np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort') + 1.0
    # average ranks for ties
    from scipy.stats import rankdata
    r = rankdata(x)
    n1, n0 = len(sr), len(sf)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def folds(n, nfold, seed):
    return np.random.RandomState(seed).permutation(n) % nfold


def cv_auc(Xr, Xf, model='ridge', nfold=5, seed=0, cols=None, C=1.0):
    """Paired design: Xr[i] and Xf[i] are page i real and forged; fold by page. Returns AUC.
    'ridge': closed-form regularised linear discriminant (LDA-like) on standardised features.
    'gbm': gradient-boosted trees."""
    n = len(Xr)
    if cols is not None: Xr = Xr[:, cols]; Xf = Xf[:, cols]
    fold = folds(n, nfold, seed)
    sr = np.zeros(n); sf = np.zeros(n)
    for f in range(nfold):
        tr = fold != f; te = fold == f
        X = np.vstack([Xr[tr], Xf[tr]])
        mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
        if model == 'ridge':
            Z = (X - mu) / sd
            y = np.r_[np.ones(tr.sum()), -np.ones(tr.sum())]
            A = Z.T @ Z + C * len(Z) * np.eye(Z.shape[1])
            w = np.linalg.solve(A, Z.T @ y)
            sr[te] = ((Xr[te] - mu) / sd) @ w; sf[te] = ((Xf[te] - mu) / sd) @ w
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            y = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
            m = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_leaf_nodes=8,
                                               random_state=seed).fit(X, y)
            sr[te] = m.predict_proba(Xr[te])[:, 1]; sf[te] = m.predict_proba(Xf[te])[:, 1]
    return _auc(sr, sf)


def paired_z(Xr, Xf):
    d = Xr - Xf
    sd = d.std(0, ddof=1); sd[sd == 0] = np.inf
    return d.mean(0) / (sd / math.sqrt(len(d)))


def save(name, obj):
    with open(os.path.join(CK, name), 'w') as f: json.dump(obj, f, indent=1, default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')
