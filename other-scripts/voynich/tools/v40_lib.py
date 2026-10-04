"""v40 'the page chooses the twin': shared code.

Corpus model: pages -> lines -> glyph units (word space = ' ').  For every twin slot (one of two look-alike
glyphs) we record the choice y and four families of predictors:
  SCRIBE  hand / section / Currier language (page-level)
  CTX     glyph before and after the slot, slot position in the word
  FRAME   the whole word with the slot masked (word identity beyond local context)
  LAYOUT  glyph(s) directly above in the previous line (same glyph offset), x position, distance to the line
          end, word index, line index in paragraph and page, paragraph-first/last line, line length (short
          lines ~ drawing intrusions), recto/verso
  PEN     the previous choice of the same twin pair earlier in the line / previous line (alternation, pen state)
"""
import json, os, re, math, random
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(DATA, 'v40_ckpt')
os.makedirs(CKPT, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v40'
LOG2 = math.log(2)

V_MULTI = [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]
V_TALL = set('ktpfKTPF')


def vglyphs(w):
    for a, b in V_MULTI:
        w = w.replace(a, b)
    return list(w)


# ---------------------------------------------------------------- corpora
def load_voynich(name='ZL3b'):
    recs = json.load(open(os.path.join(DATA, 'derived', name + '_lines.json')))
    pages = defaultdict(list)
    order = []
    for r in recs:
        if r['ltype'] != 'P' or not r['words']:
            continue
        if r['folio'] not in pages:
            order.append(r['folio'])
        pages[r['folio']].append(r)
    out = []
    for f in order:
        lines = []
        for r in pages[f]:
            g = []
            for i, w in enumerate(r['words']):
                if i:
                    g.append(' ')
                g.extend(vglyphs(w))
            lines.append(dict(g=g, ps=r['para_start'], pe=r['para_end']))
        r0 = pages[f][0]
        m = re.match(r'f(\d+)([rv])', f)
        out.append(dict(page=f, lines=lines, hand=r0['hand'], sect=r0['illus'], lang=r0['lang'],
                        quire=r0['quire'], recto=int(m.group(2) == 'r') if m else 0))
    return out


def load_dta(path, lines_min_words=3):
    """DTA diplomatic txt: pages '[n/m]'; lines kept as printed; blank line = paragraph break."""
    txt = open(path, encoding='utf-8').read()
    chunks = re.split(r'\n\[\d+/?\d*\]\n', txt)
    out = []
    for pi, ch in enumerate(chunks):
        raw = ch.split('\n')
        lines, prev_blank = [], True
        for L in raw:
            L = L.strip()
            if not L:
                if lines:
                    lines[-1]['pe'] = True
                prev_blank = True
                continue
            if len(L.split()) < lines_min_words:
                prev_blank = True
                continue
            lines.append(dict(g=list(L), ps=prev_blank, pe=False))
            prev_blank = False
        if len(lines) >= 5:
            out.append(dict(page='p%d' % pi, lines=lines, hand='1', sect='x', lang='x', quire='x', recto=pi % 2))
    return out


def load_gutenberg(fname, page_lines=36, lower=True):
    txt = open(os.path.join(DATA, fname), encoding='utf-8', errors='replace').read()
    m1 = re.search(r'\*\*\* ?START[^\n]*\n', txt); m2 = re.search(r'\*\*\* ?END', txt)
    if m1 and m2:
        txt = txt[m1.end():m2.start()]
    lines, prev_blank = [], True
    for L in txt.split('\n'):
        L = L.strip()
        if lower:
            L = L.lower()
        if not L:
            if lines:
                lines[-1]['pe'] = True
            prev_blank = True
            continue
        if len(L.split()) < 3:
            prev_blank = True
            continue
        lines.append(dict(g=list(L), ps=prev_blank, pe=False))
        prev_blank = False
    out = []
    for i in range(0, len(lines) - page_lines + 1, page_lines):
        out.append(dict(page='p%d' % (i // page_lines), lines=lines[i:i + page_lines], hand='1', sect='x',
                        lang='x', quire='x', recto=(i // page_lines) % 2))
    return out


# ---------------------------------------------------------------- twin slots
def word_spans(g):
    spans, s = [], None
    for i, c in enumerate(g):
        if c == ' ':
            if s is not None:
                spans.append((s, i)); s = None
        elif s is None:
            s = i
    if s is not None:
        spans.append((s, len(g)))
    return spans


def extract(pages, pairs, tall):
    """pairs: dict name -> (setA, setB, maskfn) ; y = 1 if glyph in setB.
    Returns list of token dicts (python) with raw layout info for later feature construction."""
    toks = []
    for pi, P in enumerate(pages):
        L = P['lines']
        # paragraph index of lines
        para_line, k = [], 0
        for li, ln in enumerate(L):
            if ln['ps'] or li == 0:
                k = 0
            para_line.append(k); k += 1
        lens = [len(ln['g']) for ln in L]
        for li, ln in enumerate(L):
            g = ln['g']
            spans = word_spans(g)
            # median length of the paragraph block (approx: page median)
            for wi, (s, e) in enumerate(spans):
                w = g[s:e]
                for j in range(s, e):
                    c = g[j]
                    for pname, (A, B, maskc) in pairs.items():
                        if c in A or c in B:
                            fr = ''.join(w[:j - s]) + maskc(c) + ''.join(w[j - s + 1:])
                            toks.append(dict(
                                pair=pname, y=int(c in B), page=pi, line=li, off=j, wi=wi, nw=len(spans),
                                jw=j - s, wl=e - s,
                                prev=(g[j - 1] if j > s else '#'), nxt=(g[j + 1] if j + 1 < e else '#'),
                                frame=pname + ':' + fr, word=''.join(w),
                                hand=P['hand'], sect=P['sect'], lang=P['lang'], recto=P['recto'],
                                linelen=len(g), xrel=(j + 0.5) / len(g), toend=len(g) - j,
                                lip=para_line[li], ps=int(ln['ps'] or li == 0), pe=int(ln['pe'] or li == len(L) - 1),
                                lpage=li / max(1, len(L) - 1), nlines=len(L)))
    return toks


def above_any(pages, toks, glyphset, dlo, dhi, dl=-1):
    """1 if any glyph of glyphset sits in line (line+dl) at offsets off+dlo .. off+dhi; -1 -> no such line (0)."""
    out = np.zeros(len(toks), dtype=np.int8)
    for i, t in enumerate(toks):
        li = t['line'] + dl
        L = pages[t['page']]['lines']
        if li < 0 or li >= len(L):
            continue
        g = L[li]['g']
        a, b = max(0, t['off'] + dlo), min(len(g), t['off'] + dhi + 1)
        for c in g[a:b]:
            if c in glyphset:
                out[i] = 1; break
    return out


def pen_feats(toks, y):
    """previous choice of the same pair in the same line (+1 B, -1 A, 0 none) and in the previous line."""
    last_line, last_prevline = {}, {}
    pl = np.zeros(len(toks)); pp = np.zeros(len(toks))
    byline = defaultdict(list)
    for i, t in enumerate(toks):
        byline[(t['pair'], t['page'], t['line'])].append(i)
    for i, t in enumerate(toks):
        pass
    for (pair, pg, li), idx in byline.items():
        idx = sorted(idx, key=lambda i: toks[i]['off'])
        prevl = byline.get((pair, pg, li - 1), [])
        lastprev = (2 * y[max(prevl, key=lambda i: toks[i]['off'])] - 1) if prevl else 0
        cur = 0
        for i in idx:
            pl[i] = cur; pp[i] = lastprev
            cur = 2 * y[i] - 1
    return pl, pp


# ---------------------------------------------------------------- design matrices
def onehot(cols, min_count=3):
    """cols: list of lists of string categories (one per token). -> scipy sparse matrix."""
    from scipy.sparse import csr_matrix
    n = len(cols[0])
    rows, cs, vocab = [], [], {}
    for k, col in enumerate(cols):
        cnt = Counter(col)
        for i, v in enumerate(col):
            if cnt[v] < min_count:
                continue
            key = (k, v)
            if key not in vocab:
                vocab[key] = len(vocab)
            rows.append(i); cs.append(vocab[key])
    return csr_matrix((np.ones(len(rows)), (rows, cs)), shape=(n, max(1, len(vocab))))


def bins(x, edges):
    return [str(int(np.searchsorted(edges, v))) for v in x]


def family_cols(toks, fam, extra=None):
    if fam == 'SCRIBE':
        return [[t['hand'] for t in toks], [t['sect'] for t in toks], [t['lang'] for t in toks],
                [t['hand'] + t['sect'] for t in toks]]
    if fam == 'CTX':
        return [[t['prev'] for t in toks], [t['nxt'] for t in toks], [t['prev'] + t['nxt'] for t in toks],
                [str(min(t['jw'], 3)) + '_' + str(min(t['wl'] - t['jw'] - 1, 3)) for t in toks]]
    if fam == 'FRAME':
        return [[t['frame'] for t in toks]]
    if fam == 'LAYOUT':
        e = extra
        return [[str(v) for v in e['tall_above']], [str(v) for v in e['tall_above2']],
                [str(v) for v in e['same_above']],
                bins([t['xrel'] for t in toks], [0.15, 0.35, 0.55, 0.75, 0.9]),
                bins([t['off'] for t in toks], [3, 8, 15, 25, 35, 45]),
                bins([t['toend'] for t in toks], [3, 6, 10, 20]),
                [str(min(t['wi'], 4)) for t in toks], [str(min(t['nw'] - t['wi'] - 1, 3)) for t in toks],
                [str(min(t['lip'], 4)) for t in toks], [str(t['ps']) for t in toks], [str(t['pe']) for t in toks],
                bins([t['lpage'] for t in toks], [0.2, 0.4, 0.6, 0.8]),
                bins([t['linelen'] for t in toks], [20, 30, 40, 50, 60]), [str(t['recto']) for t in toks]]
    if fam == 'PEN':
        return [[str(int(v)) for v in extra['pen_line']], [str(int(v)) for v in extra['pen_prev']]]
    raise ValueError(fam)


def oof_logloss(X, y, groups, nfold=5, C=1.0, seed=0):
    """out-of-fold predicted probabilities with page-grouped folds; returns p (len n)."""
    from sklearn.linear_model import LogisticRegression
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ug)
    fold_of = {g: i % nfold for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    p = np.zeros(len(y))
    for k in range(nfold):
        tr, te = f != k, f == k
        if te.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean(); continue
        m = LogisticRegression(C=C, max_iter=400, solver='liblinear')
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return np.clip(p, 1e-4, 1 - 1e-4), f


def bits(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() / LOG2


def offset_gain(y, p0, x, f):
    """fit logit(p) = logit(p0) + a + b x on train folds, evaluate held-out; gain (bits/token) of b over a-only."""
    z0 = np.log(p0 / (1 - p0))
    gain = np.zeros(len(y))
    for k in np.unique(f):
        tr, te = f != k, f == k
        ga = _fit_offset(y[tr], z0[tr], None)
        gb = _fit_offset(y[tr], z0[tr], x[tr])
        la = _ll(y[te], z0[te] + ga[0])
        lb = _ll(y[te], z0[te] + gb[0] + gb[1] * x[te])
        gain[te] = la - lb
    return gain.mean() / LOG2


def _ll(y, z):
    return np.logaddexp(0, z) - y * z


def _fit_offset(y, z0, x, it=12, l2=1.0):
    a = b = 0.0
    for _ in range(it):
        z = z0 + a + (b * x if x is not None else 0)
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p)
        ga = (y - p).sum()
        if x is None:
            a += ga / max(w.sum(), 1e-9)
            continue
        gb = ((y - p) * x).sum() - l2 * b
        haa, hab, hbb = w.sum(), (w * x).sum(), (w * x * x).sum() + l2
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            break
        a += (hbb * ga - hab * gb) / det
        b += (-hab * ga + haa * gb) / det
    return a, b
