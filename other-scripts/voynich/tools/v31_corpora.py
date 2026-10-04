"""v31 THE TEXTURE OF A SPELL: corpus builder.

Every corpus is a list of documents; a document is a list of lines; a line is a list of words (strings of
units). Classes: LANG (natural language), MAGIC (voces magicae extracted from grimoires and charm books),
INVENT (invented languages), GIBB (hand-written gibberish), GEN (generators, see v31_gen.py),
TEST (objects that are only classified, never trained on: Voynich, Steganographia, Martian, Lingua Ignota ...).

Sources are used as raw symbol data only (no one's interpretation):
  esotericarchives.com transcriptions (J. H. Peterson): Liber Juratus, Ars notoria (Latin column),
    Heptameron, Clavicula (several MSS), Lemegeton, Abramelin, Raziel, Romanus-Buechlein, Egyptian Secrets,
    Steganographia I, Picatrix extract, Agrippa IV, Arbatel, Almadel, Summa sacre magice, Mafteah Shelomoh
  Gaskell & Bowern 2022 data (github.com/danielgaskell/voynich): 42 gibberish samples, meaningful texts, conlangs
  ReF / CATMuS / Dalimil page chunks from data/v30_ckpt/corpora.json; Gutenberg texts in data/
  Flournoy 1900 (archive.org OCR): Helene Smith's "Martian" texts; Pitra 1882 (archive.org OCR) + Wikipedia:
    Lingua Ignota nouns
Downloads live in the scratchpad (v31/); the output is data/v31_ckpt/corpora.json.
"""
import os, re, json, glob, html, unicodedata, random, collections, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v31_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v31'
EA = os.path.join(SCR, 'ea')
GD = os.path.join(SCR, 'gaskell', 'data')

SPECIAL = {'ß': 'ss', 'æ': 'ae', 'œ': 'oe', 'ø': 'o', 'þ': 'th', 'ð': 'd', 'ł': 'l', 'ſ': 's', 'ȝ': 'g', 'ƿ': 'w'}


def norm(w):
    w = w.lower()
    w = ''.join(SPECIAL.get(c, c) for c in w)
    w = unicodedata.normalize('NFD', w)
    return ''.join(c for c in w if 'a' <= c <= 'z')


def latin_ok(text, thr=0.9):
    al = [c for c in text if c.isalpha()]
    if not al: return False
    return sum(1 for c in al if unicodedata.normalize('NFD', c)[0].lower() in 'abcdefghijklmnopqrstuvwxyzßæœøþðſ') / len(al) >= thr


def text_lines(txt, minw=1):
    out = []
    for ln in txt.split('\n'):
        ws = [norm(x) for x in re.findall(r"[^\W\d_]+", ln)]
        ws = [w for w in ws if w]
        if len(ws) >= minw: out.append(ws)
    return out


# ------------------------------------------------------------------ lexicons (for extracting voces)
def _gfile(pat):
    fs = glob.glob(os.path.join(GD, 'meaningful', 'texts', pat))
    return fs


def lexicon(lang):
    pats = {'la': ['*Latin - Literary*', '*Latin - Technical*'],
            'en': ['*English - *'], 'de': ['*German - *'], 'fr': ['*French - *'], 'it': ['*Italian - *'],
            'es': ['*Spanish - *'], 'nl': ['*Flemish - *']}[lang]
    C = collections.Counter()
    for p in pats:
        for f in _gfile(p):
            C.update(norm(x) for x in re.findall(r"[^\W\d_]+", open(f, encoding='utf-8', errors='replace').read()))
    extra = {'la': ['plain/la.txt', 'pg218.txt', 'pg23306.txt'], 'en': ['pg47342.txt'], 'de': ['pg22367.txt'],
             'it': ['pg1000.txt', 'pg45334.txt'], 'es': ['pg2000.txt']}.get(lang, [])
    for e in extra:
        C.update(norm(x) for x in re.findall(r"[^\W\d_]+", open(os.path.join(DATA, e), encoding='utf-8', errors='replace').read()))
    S = {w for w, c in C.items() if c >= 2 and w}
    big = os.path.join(SCR, 'lex', lang + '.txt')    # dwyl english-words; hermitdave FrequencyWords 50k
    if os.path.exists(big):
        for ln in open(big, encoding='utf-8', errors='replace'):
            w = norm(ln.split()[0]) if ln.strip() else ''
            if len(w) >= 5: S.add(w)
    if lang == 'la':
        S |= {w.replace('j', 'i').replace('v', 'u') for w in S}
    return S


_LEX = {}
_SK = {}


def skel(w):
    """spelling skeleton for early spellings: y->i, j->i, v->u, doubled letters collapsed, final e dropped"""
    w = w.replace('y', 'i').replace('j', 'i').replace('v', 'u')
    w = re.sub(r'(.)\1+', r'\1', w)
    return w[:-1] if len(w) > 3 and w.endswith('e') else w


def _load_lex(l):
    import pickle
    pk = os.path.join(SCR, f'lex_{l}.pkl')
    if os.path.exists(pk): return pickle.load(open(pk, 'rb'))
    L = lexicon(l); pickle.dump(L, open(pk, 'wb')); return L


def is_lex(w, langs):
    for l in langs:
        if l not in _LEX:
            _LEX[l] = _load_lex(l); _SK[l] = {skel(x) for x in _LEX[l]}
        L = _LEX[l]
        if w in L: return True
        if l == 'la' and w.replace('j', 'i').replace('v', 'u') in L: return True
        if len(w) >= 4 and skel(w) in _SK[l]: return True
    return False


_DEL = {}


def near_lex(w, langs):
    """edit distance 1 from a lexicon word (symmetric-delete index), only for words of >= 6 letters"""
    if len(w) < 6: return False
    for l in langs:
        is_lex('aa', (l,))
        if l not in _DEL:
            D = set()
            for x in _LEX[l]:
                if len(x) >= 5:
                    for i in range(len(x)): D.add(x[:i] + x[i + 1:])
            _DEL[l] = D
        dl = [w[:i] + w[i + 1:] for i in range(len(w))]
        if w in _DEL[l] or any(d in _LEX[l] or d in _DEL[l] for d in dl): return True
    return False


# ------------------------------------------------------------------ voces magicae
def html_text(raw, choose_td=True, langs=('la', 'en')):
    raw = re.sub(r'(?s)<!--.*?-->', ' ', raw)
    raw = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', raw)
    raw = re.sub(r'(?is)<(sup|em class=pg)[^>]*>.*?</(sup|em)>', ' ', raw)   # footnote markers, page numbers
    raw = re.sub(r'(?is)<font size=2 color=blue>.*?</font>', ' ', raw)
    if choose_td:
        def tr(m):
            tds = re.findall(r'(?is)<td([^>]*)>(.*?)(?=<td|</tr|$)', m.group(0))
            tds = [b for a, b in tds if 'note' not in a.lower()]
            if len(tds) <= 1: return ' '.join(tds) + '\n'
            best = max(tds, key=lambda b: len(vox_runs(clean(b), langs)) * 1000 + len(b))
            return best + '\n'
        raw = re.sub(r'(?is)<tr[^>]*>.*?</tr>', tr, raw)
    return clean(raw)


def clean(raw):
    t = re.sub(r'(?i)<br\s*/?>|</?p[^>]*>|</?h\d[^>]*>|</?div[^>]*>|</?li[^>]*>|</?blockquote[^>]*>', '\n', raw)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = html.unescape(t)
    t = re.sub(r'-\s*\|\s*', '', t)          # manuscript line-break hyphens (Juratus ME text)
    t = t.replace('|', ' ')
    t = re.sub(r'\[[^\]]{0,80}\]', ' ', t)   # editorial brackets
    return t


def tokens_with_breaks(t):
    """word tokens; a sentence-level break (. ; : ! ? newline-paragraph) is kept as None only if strong."""
    out = []
    for m in re.finditer(r"[^\W\d_]+(?:-[^\W\d_]+)*|\n\s*\n|[0-9]+", t):
        s = m.group(0)
        if s.strip() == '': out.append(None); continue
        if s[0].isdigit(): out.append(None); continue
        out.append(s)
    return out


ALL_LANGS = ('la', 'en', 'de', 'fr', 'it', 'es', 'nl')


def vox_runs(t, langs, minrun=3):
    """maximal runs of >= minrun consecutive non-lexical words (any of the 7 lexicons); runs made only of
    capitalised words (modern names, bibliography) are dropped."""
    runs, cur, caps = [], [], []
    def close():
        if len(cur) >= minrun and not all(caps):
            nl = sum(near_lex(w, ALL_LANGS) for w in cur)
            if nl / len(cur) < 1 / 3: runs.append(list(cur))
        cur.clear(); caps.clear()
    for tok in tokens_with_breaks(t):
        if tok is None: close(); continue
        w = norm(tok.replace('-', ''))
        if len(w) < 2 or is_lex(w, ALL_LANGS) or (tok.isupper() and len(tok) <= 4):
            close(); continue
        cur.append(w); caps.append(tok[0].isupper())
    close()
    return runs


def _sim(a, b):
    import difflib
    return difflib.SequenceMatcher(None, ' '.join(a), ' '.join(b)).ratio()


def dedupe(runs, back=6, thr=0.8):
    out = []
    for r in runs:
        if any(_sim(r, q) >= thr for q in out[-back:]): continue
        out.append(r)
    return out


MAGIC_SRC = {
    # name: (files, lexicon languages, dedupe parallel versions)
    'M_Juratus': (['juratus_juratus.htm'], ('la', 'en'), True),
    'M_Notoria': (['notoria_notoria.htm'], ('la', 'en'), True),
    'M_Heptameron': (['solomon_heptamer.htm'], ('la', 'en'), True),
    'M_Clavicula': (['solomon_ksol.htm', 'solomon_l1203.htm', 'solomon_sl3847.htm', 'solomon_ad36674.htm'], ('la', 'en', 'fr', 'it'), True),
    'M_Abramelin': (['abramelin_abramelin.htm'], ('en', 'fr', 'de'), True),
    'M_Raziel': (['raziel_raziel.htm'], ('la', 'en'), True),
    'M_Romanus': (['moses_romanus.htm', 'moses_egyptian.htm', 'moses_folklore.htm', 'moses_hollenz4.htm'], ('de', 'en', 'la'), True),
    'M_Agrippa4': (['agrippa_agrippa4.htm', 'solomon_arbatel.htm', 'solomon_almadel.htm'], ('la', 'en'), True),
    'M_Ganell': (['ganell_ssm.htm', 'picatrix.htm', 'gollancz_mafteah.htm', 'solomon_lemegeton.htm', 'solomon_grimhono.htm', 'solomon_petitalb.htm'], ('la', 'en', 'fr'), True),
    'M_PGM': (['../pgm/Papyri_Graecae_Magicae.txt'], ('en', 'la'), True),
    'T_Stegano': (['tritheim_stegano.htm'], ('la', 'en'), True),   # covert cipher dressed as conjurations: TEST only
}


def magic_corpus(name):
    files, langs, dd = MAGIC_SRC[name]
    runs = []
    for f in files:
        p = os.path.join(EA, f)
        if not os.path.exists(p): continue
        raw = open(p, encoding='latin-1').read()
        if 'Ã' in raw[:20000]:
            raw = open(p, encoding='utf-8', errors='replace').read()
        rs = vox_runs(html_text(raw, True, langs), langs)
        if dd: rs = dedupe(rs)
        runs += rs
    return [runs]   # one document; each run is a line


# ------------------------------------------------------------------ Gaskell & Bowern corpora
def gibberish():
    out = {}
    for f in sorted(glob.glob(os.path.join(GD, 'gibberish_transcriptions', '*.txt'))):
        k = 'B_' + os.path.basename(f).split(' - ')[-1].replace('.txt', '')
        out[k] = [text_lines(open(f, encoding='utf-8', errors='replace').read())]
    return out


def meaningful():
    out = {}
    for f in sorted(glob.glob(os.path.join(GD, 'meaningful', 'texts', '*.txt'))):
        b = os.path.basename(f)[:-4]
        kind, lang, genre = [x.strip() for x in b.split(' - ')[:3]]
        if 'LOLCat' in lang or 'Abbreviated' in lang: continue
        raw = open(f, encoding='utf-8', errors='replace').read()
        if not latin_ok(raw): continue
        cls = 'INVENT' if kind == 'Conlangs' else 'LANG'
        k = ('I_' if cls == 'INVENT' else 'L_') + re.sub(r'\W+', '', lang)[:10] + '_' + genre[:4]
        out[k] = (cls, [text_lines(raw)])
    return out


# ------------------------------------------------------------------ other sources
def gutenberg():
    import vlib
    out = {}
    for k in ['Latin-Caesar', 'Latin-Descartes', 'Italian-Manzoni', 'Italian-Dante', 'German-Kafka', 'Spanish-Cervantes']:
        L = vlib.load_ref(k, max_words=60000, skip_frac=0.02)
        out['L_pg' + k.split('-')[1][:6]] = [[[norm(w) for w in l['words'] if norm(w)] for l in L]]
    return out


def v30_pages():
    p = os.path.join(DATA, 'v30_ckpt', 'corpora.json')
    if not os.path.exists(p): return {}
    C = json.load(open(p))['corpora']
    out = {}
    for k in ['G_Bav1', 'G_Bav2', 'G_Alem', 'G_Rip', 'I_Ita', 'I_Lat', 'C_Mod', 'C_Old']:
        if k in C:
            docs = [[[norm(w) for w in pg if norm(w)][i:i + 10] for i in range(0, len(pg), 10)] for pg in C[k]]
            out['L_ms' + k] = docs
    return out


def isidore():
    t = open(os.path.join(DATA, 'plain', 'la.txt'), encoding='utf-8').read()
    return [text_lines(t)]


def martian():
    """Helene Smith's Martian texts (Flournoy 1900, ch. 'The Martian texts'): wide-spaced OCR lines in that
    section whose words are not French/English; OCR accent garbage mapped back to vowels."""
    t = open(os.path.join(SCR, 'gl', 'flournoy.txt'), encoding='utf-8', errors='replace').read().split('\n')
    a = [i for i, l in enumerate(t) if 'The  Martian  Texts' in l][-1]
    lines = []
    for l in t[a:a + 2600]:
        if not re.search(r'\S {3,}\S', l): continue
        l2 = re.sub(r'^\s*\d+\.\s*', '', l)
        l2 = l2.replace('6', 'e').replace('€', 'e').replace('^', 'e')
        ws = [norm(x) for x in re.findall(r"[^\W\d_]+", l2)]
        ws = [w for w in ws if w]
        if len(ws) < 3: continue
        lexf = sum(is_lex(w, ('fr', 'en')) for w in ws) / len(ws)
        if lexf <= 0.25: lines.append(ws)
    return [lines]


def lingua_ignota():
    ws = []
    t = open(os.path.join(SCR, 'inv', 'ignota_wp.txt'), encoding='utf-8').read()
    ws += [norm(x) for x in re.findall(r'\{\{lang\|art-DE\|([^}]*)\}\}', t)]
    # Pitra 1882, Analecta sacra VIII 497-502: capitalised single-word OCR lines of the ignota column that are
    # not Latin (the Latin plant names stand in their own column)
    p = os.path.join(SCR, 'inv', 'analectasacrasp04pitrgoog.txt')
    if os.path.exists(p):
        T = open(p, encoding='utf-8', errors='replace').read().split('\n')
        a = next(i for i, l in enumerate(T) if l.strip() == 'LINGUA  IGNOTA')
        for l in T[a:a + 1400]:
            s = l.strip()
            if re.fullmatch(r'[A-Z][a-z]{2,14}', s):
                w = norm(s)
                if not is_lex(w, ('la',)): ws.append(w)
    seen, out = set(), []
    for w in ws:
        if w and w not in seen: seen.add(w); out.append(w)
    return [[out[i:i + 8] for i in range(0, len(out), 8)]]


def build():
    C = {}
    def add(name, cls, docs, note=''):
        n = sum(len(l) for d in docs for l in d)
        C[name] = {'cls': cls, 'docs': docs, 'note': note}
        print(f'{name:22s} {cls:6s} tokens {n:7d}  e.g. {" ".join(w for l in docs[0][:3] for w in l)[:90]}', flush=True)
    for k in MAGIC_SRC:
        add(k, 'TEST' if k.startswith('T_') else 'MAGIC', magic_corpus(k))
    for k, v in gibberish().items(): add(k, 'GIBB', v)
    for k, (cls, v) in meaningful().items(): add(k, cls, v)
    for k, v in gutenberg().items(): add(k, 'LANG', v)
    for k, v in v30_pages().items(): add(k, 'LANG', v)
    add('L_Isidore', 'LANG', isidore())
    add('T_Martian', 'TEST', martian())
    add('T_LinguaIgnota', 'TEST', lingua_ignota())
    json.dump(C, open(os.path.join(CK, 'corpora.json'), 'w'))


if __name__ == '__main__':
    build()
