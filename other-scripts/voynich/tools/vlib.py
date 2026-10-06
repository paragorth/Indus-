"""Shared helpers for the Voynich data-only tests.

Texts are represented as a list of "lines"; each line is a dict with
  words (list[str]), para_start (bool), para_end (bool), plus metadata.
"""
import json, math, os, random, re, unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(DATA, 'results')
os.makedirs(RES, exist_ok=True)

# ---------- Voynich ----------
GLYPH_MULTI = [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]

def glyphs(word):
    """EVA word -> list of glyph units (benched gallows, ch, sh merged)."""
    w = word
    for a, b in GLYPH_MULTI:
        w = w.replace(a, b)
    return list(w)

# Default transcription reading (v83, 6 Oct 2026): the uncertainty-aware parser tools/v83_parse.py.
# VOY_MODE=glyph (default): words with glyph-level doubt are dropped ([a:b] alternatives, ?, rare @nnn; glyphs,
#   {..} ligatures, ' marks, comments questioning the reading, damage comments, damaged pages/blocks); words next to an
#   uncertain space ',' are kept (the space still counts as a word break) and flagged; '<~>' is a word space.
# VOY_MODE=legacy reproduces every loop before v83 (tools/parse_ivtff.py -> data/derived/<name>_lines.json).
# Other modes: all, clean (no flag of any kind), agree (ZL3b = IT2a = GC2a on the same locus), agreeclean.
VOY_MODE = os.environ.get('VOY_MODE', 'glyph')


def load_voynich(name='ZL3b', ltypes=('P',), drop_uncertain=False, mode=None):
    mode = mode or VOY_MODE
    if mode == 'legacy':
        recs = json.load(open(os.path.join(DATA, 'derived', name + '_lines.json')))
    else:
        import sys as _sys
        _sys.path.insert(0, HERE)
        import v83_parse
        recs = v83_parse.load(name, mode)
    out = []
    for r in recs:
        if ltypes and r['ltype'] not in ltypes:
            continue
        ws = r['words']
        if drop_uncertain:
            ws = [w for w, u in zip(ws, r['uncertain']) if not u]
        if not ws:
            continue
        r = dict(r); r['words'] = ws
        out.append(r)
    return out

# ---------- reference texts ----------
REFS = {
    'Latin-Caesar': 'pg218.txt',
    'Latin-Descartes': 'pg23306.txt',
    'Italian-Manzoni': 'pg45334.txt',
    'Italian-Dante': 'pg1000.txt',
    'German-Kafka': 'pg22367.txt',
    'Spanish-Cervantes': 'pg2000.txt',
}

def _norm_word(w):
    w = w.lower()
    w = ''.join(ch for ch in w if ch.isalpha())
    return w

def load_ref(key, max_words=None, skip_frac=0.0):
    """Gutenberg text -> lines (typeset lines; paragraphs = blank-line blocks)."""
    txt = open(os.path.join(DATA, REFS[key]), encoding='utf-8', errors='replace').read()
    m1 = re.search(r'\*\*\* ?START[^\n]*\n', txt); m2 = re.search(r'\*\*\* ?END', txt)
    if m1 and m2:
        txt = txt[m1.end():m2.start()]
    txt = txt.replace('’', ' ').replace("'", ' ')
    raw_lines = txt.split('\n')
    start = int(len(raw_lines) * skip_frac)
    lines, prev_blank, total = [], True, 0
    for L in raw_lines[start:]:
        ws = [_norm_word(w) for w in re.split(r'[\s\-—]+', L)]
        ws = [w for w in ws if w]
        if not ws:
            if lines:
                lines[-1]['para_end'] = True
            prev_blank = True
            continue
        lines.append({'words': ws, 'para_start': prev_blank, 'para_end': False})
        prev_blank = False
        total += len(ws)
        if max_words and total >= max_words:
            break
    return lines

def chars_of(word, glyph_mode=False):
    return glyphs(word) if glyph_mode else list(word)

# ---------- statistics ----------
def words_of(lines):
    return [w for L in lines for w in L['words']]

def truncate(lines, n):
    out, t = [], 0
    for L in lines:
        out.append(L); t += len(L['words'])
        if t >= n:
            break
    return out

def zipf_slope(words, rmin=1, rmax=1000):
    c = sorted(Counter(words).values(), reverse=True)
    rmax = min(rmax, len(c))
    xs = [math.log(r) for r in range(rmin, rmax + 1)]
    ys = [math.log(c[r - 1]) for r in range(rmin, rmax + 1)]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)); sxx = sum((x - mx) ** 2 for x in xs)
    return sxy / sxx

def entropies(words, glyph_mode=False):
    """h1 and conditional h2 over the character stream with '_' as word separator."""
    seq = []
    for w in words:
        seq.extend(chars_of(w, glyph_mode)); seq.append('_')
    c1 = Counter(seq); n = len(seq)
    h1 = -sum(v / n * math.log2(v / n) for v in c1.values())
    c2 = Counter(zip(seq, seq[1:])); n2 = n - 1
    h12 = -sum(v / n2 * math.log2(v / n2) for v in c2.values())
    first = Counter(a for a, b in zip(seq, seq[1:]))
    hfirst = -sum(v / n2 * math.log2(v / n2) for v in first.values())
    c3 = Counter(zip(seq, seq[1:], seq[2:])); n3 = n - 2
    h123 = -sum(v / n3 * math.log2(v / n3) for v in c3.values())
    return {'alphabet': len(c1), 'h0': math.log2(len(c1)), 'h1': h1, 'h2': h12 - hfirst, 'h3': h123 - h12}

def word_bigram_mi(words):
    c1 = Counter(words); c2 = Counter(zip(words, words[1:])); n = len(words) - 1
    mi = 0.0
    for (a, b), v in c2.items():
        mi += v / n * math.log2((v / n) / ((c1[a] / len(words)) * (c1[b] / len(words))))
    return mi

def basic_stats(words, glyph_mode=False):
    n = len(words); c = Counter(words)
    lens = [len(chars_of(w, glyph_mode)) for w in words]
    ml = sum(lens) / n; sd = (sum((l - ml) ** 2 for l in lens) / n) ** 0.5
    d = {'tokens': n, 'types': len(c), 'ttr': len(c) / n,
         'hapax_frac_types': sum(1 for v in c.values() if v == 1) / len(c),
         'zipf_slope': zipf_slope(words), 'wlen_mean': ml, 'wlen_sd': sd,
         'wlen_cv': sd / ml, 'top10_share': sum(sorted(c.values(), reverse=True)[:10]) / n}
    d.update(entropies(words, glyph_mode))
    d['word_bigram_mi'] = word_bigram_mi(words)
    return d

def wlen_dist(words, glyph_mode=False, maxl=15):
    c = Counter(min(len(chars_of(w, glyph_mode)), maxl) for w in words)
    n = len(words)
    return [c.get(i, 0) / n for i in range(1, maxl + 1)]

def bh_fdr(pvals, q=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    thr = -1
    for k, i in enumerate(order, 1):
        if pvals[i] <= q * k / m:
            thr = k
    sig = set(order[:thr]) if thr > 0 else set()
    return [i in sig for i in range(m)]

def log_binom_sf(k, n, p):
    """P(X>=k), X~Bin(n,p), computed in log space then exponentiated."""
    if k <= 0:
        return 1.0
    lp = math.log(p) if p > 0 else -1e300; lq = math.log1p(-p) if p < 1 else -1e300
    terms = []
    for i in range(k, n + 1):
        terms.append(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))

def save(name, obj):
    json.dump(obj, open(os.path.join(RES, name + '.json'), 'w'), indent=1, ensure_ascii=False)
