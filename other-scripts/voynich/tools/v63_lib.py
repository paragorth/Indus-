"""v63 WHICH TWIN IS THE SLIP? shared loaders and scorers.

Real control: SCTA/LombardPress diplomatic TEI of Petrus Plaoul (4+1 witnesses), walked character by
character with flags (kept / deleted / added) and manuscript line breaks (<lb/>), so that each scribe's
uncorrected first writing ('as written') and the corrected text are both available, with the copy's own
line ends.  Voynich: ZL3b / IT2a paragraph lines as glyph tuples.
"""
import os, re, sys, glob, json, random, math, pickle
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib

ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v63_ckpt')
os.makedirs(CK, exist_ok=True)
REPO = os.path.join(CK, 'plaoulcommentary')
TEI = '{http://www.tei-c.org/ns/1.0}'
SKIP = {'note', 'rdg', 'app', 'bibl', 'reg', 'expan', 'fw', 'figure', 'corr'}
WITS = ('reims', 'sorb', 'svict', 'vat', 'cod-mmppdi')


# ------------------------------------------------------------------ Plaoul diplomatic walker
def _walk(el, flag, out):
    """append (char, flag) to out; flag in 'k','d','a'; '\n' = manuscript line break."""
    tag = el.tag.replace(TEI, '')
    if tag in SKIP or tag == 'g':
        return
    if tag == 'lb':
        if el.get('break') != 'no':
            out.append(('\n', flag))
        return
    if tag == 'choice':
        kids = {k.tag.replace(TEI, ''): k for k in el}
        k = kids.get('orig') or kids.get('abbr') or kids.get('sic')
        if k is not None:
            _walk(k, flag, out)
        return
    f = flag
    if tag == 'del':
        f = 'd'
    elif tag == 'add':
        f = 'a' if flag != 'd' else 'd'
    if el.text:
        out.extend((c, f) for c in el.text)
    for k in el:
        _walk(k, f, out)
        if k.tail:
            out.extend((c, flag) for c in k.tail)


def _tokens(chars):
    """(char,flag) stream -> list of tokens: dict(w, st, lb_before) ; st in k/d/a/m (mixed).
    mixed tokens carry 'before' (k+d letters) and 'after' (k+a letters)."""
    toks, cur, lb = [], [], False

    def flush():
        nonlocal cur
        if cur:
            letters = [(c, f) for c, f in cur if c.isalpha()]
            if letters:
                fl = set(f for _, f in letters)
                st = fl.pop() if len(fl) == 1 else 'm'
                b = ''.join(c for c, f in letters if f in 'kd').lower()
                a = ''.join(c for c, f in letters if f in 'ka').lower()
                toks.append(dict(before=norm(b), after=norm(a), st=st, lb=False))
        cur = []

    for c, f in chars:
        if c == '\n':
            flush()
            if toks:
                toks[-1]['lb'] = True   # line break AFTER this token
            continue
        if c.isspace() or not (c.isalpha()):
            if c.isspace() or c in '.,;:()[]/-':
                flush(); continue
            continue
        cur.append((c, f))
    flush()
    return toks


def norm(s):
    s = s.lower().replace('j', 'i').replace('v', 'u')
    return re.sub(r'[^a-z]', '', s)


def plaoul_witnesses(cache=True):
    """-> list of paragraphs: dict(wit, file, toks)."""
    pk = os.path.join(CK, 'plaoul_toks.pkl')
    if cache and os.path.exists(pk):
        return pickle.load(open(pk, 'rb'))
    files = sorted(f for f in glob.glob(os.path.join(REPO, 'lectio*', '*_lectio*.xml'))
                   if os.path.basename(f).split('_')[0] in WITS)
    paras = []
    for f in files:
        try:
            txt = open(f, encoding='utf-8').read()
            txt = re.sub(r'<g ref="#dbdash"/>\s*<lb([^>]*)/>', r'<lb break="no"\1/>', txt)
            root = ET.fromstring(txt.encode('utf-8'))
        except Exception:
            continue
        body = root.find('.//' + TEI + 'body')
        if body is None:
            continue
        for p in body.iter(TEI + 'p'):
            out = []
            _walk(p, 'k', out)
            t = _tokens(out)
            if len(t) >= 3:
                paras.append(dict(wit=os.path.basename(f).split('_')[0], file=os.path.basename(f), toks=t))
    pickle.dump(paras, open(pk, 'wb'))
    return paras


# ------------------------------------------------------------------ Voynich
def voynich_lines(tr='ZL3b'):
    """-> list of dict(folio, lang, illus, words=[glyph tuples]) paragraph lines; '?' words dropped later."""
    out = []
    for r in vlib.load_voynich(tr):
        ws = [tuple(vlib.glyphs(w)) for w in r['words']]
        out.append(dict(folio=r['folio'], lang=r['lang'], illus=r['illus'], quire=r['quire'], hand=r['hand'],
                        words=ws, unc=r['uncertain'], ps=r['para_start'], pe=r['para_end']))
    return out


# ------------------------------------------------------------------ string tools
def ed(a, b):
    """Levenshtein distance on tuples/strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[lb]


class GlyphLM:
    """add-k smoothed glyph trigram with word boundaries; logp per word."""
    def __init__(self, words, k=0.1):
        self.c3, self.c2 = Counter(), Counter()
        self.al = set()
        for w in words:
            s = ('<', '<') + tuple(w) + ('>',)
            self.al.update(w)
            for i in range(2, len(s)):
                self.c3[s[i - 2:i + 1]] += 1; self.c2[s[i - 2:i]] += 1
        self.V = len(self.al) + 1; self.k = k

    def logp(self, w):
        s = ('<', '<') + tuple(w) + ('>',)
        lp = 0.0
        for i in range(2, len(s)):
            lp += math.log((self.c3[s[i - 2:i + 1]] + self.k) / (self.c2[s[i - 2:i]] + self.k * self.V))
        return lp / (len(w) + 1)


def voynich_paras(tr='ZL3b', keep_lines=False):
    """paragraph streams of glyph tuples (words with '?' dropped); optional line-index per token."""
    paras, cur, li, curl, curf = [], [], [], [], None
    for k, r in enumerate(voynich_lines(tr)):
        if (r['ps'] or r['folio'] != curf) and cur:
            paras.append(dict(folio=curf, words=cur, line=curl)); cur, curl = [], []
        curf = r['folio']
        for j, w in enumerate(r['words']):
            if '?' in w or '*' in w:
                continue
            cur.append(w); curl.append((k, j, len(r['words'])))
    if cur:
        paras.append(dict(folio=curf, words=cur, line=curl))
    return paras


def plaoul_streams(mode='after'):
    """paragraph streams of letter tuples. mode 'after' = corrected text; 'before' = as first written
    (whole-word deletions kept, additions dropped)."""
    out = []
    for p in plaoul_witnesses():
        ws = []
        for t in p['toks']:
            if mode == 'after' and t['st'] == 'd': continue
            if mode == 'before' and t['st'] == 'a': continue
            w = t[mode]
            if w: ws.append(tuple(w))
        if len(ws) >= 3:
            out.append(dict(folio=p['file'], words=ws))
    return out


VS_ALPHA = list('ABCDEFGHIJKLMNOPQR')


def verbose_table(seed=7):
    rng = random.Random(seed)
    tab, used = {}, set()
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        while True:
            k = rng.choice([1, 2, 2, 3])
            g = tuple(rng.choice(VS_ALPHA) for _ in range(k))
            if g not in used:
                used.add(g); tab[ch] = g; break
    return tab


def encode_streams(streams, seed=7):
    tab = verbose_table(seed)
    out = []
    for s in streams:
        ws = [tuple(x for c in w for x in tab.get(c, ())) for w in s['words']]
        ws = [w for w in ws if w]
        out.append(dict(s, words=ws))
    return out
