"""v5 'notation, not language' battery: shared corpora.

Every corpus is a list of UNITS (paragraph / chant piece); a unit is a dict
  {'id', 'stratum', 'lines': [line, ...]}  where line = [word, ...] and
  word = tuple of symbols (one symbol = one glyph / one note / one cipher glyph).

Corpora
  V-ZL, V-IT     Voynich paragraphs (P lines), glyph units from vlib.glyphs; stratum = illustration code
  chant          Gregorian chant, GregoBase dump 2019-10-24 via github.com/bacor/gregobasecorpus
                 (commit ad5060febd01536d489d68b202bd745025a974cc, file gregobase_dumps/gregobase_20191024.sql, CC0).
                 symbol = staff letter a-m as written (case folded, accidentals/custos dropped),
                 word = notes of one syllable, line = phrase between full bars ':' / '::', unit = piece,
                 stratum = mode (1-8).  This is NOTATION with known ground truth (mode -> final).
  vs-<lang>      natural-language Gutenberg texts under a fixed verbose substitution
                 (each letter -> fixed group of 1-3 symbols from an 18-symbol alphabet). Lines = typeset
                 lines (Dante: verse lines), unit = paragraph / canto block.
  V-ZL-gshuf     Voynich with glyphs shuffled within each line (word lengths kept)
  V-ZL-tri       Voynich lines regenerated word-by-word from a glyph trigram model of words (same word counts)
"""
import json, os, random, re, sys, gzip
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

DER = os.path.join(vlib.DATA, 'derived')
CHANT_SQL = os.environ.get('GREGOBASE_SQL', '')
CHANT_JSON = os.path.join(DER, 'v5_chant.json.gz')


# ---------------- Voynich ----------------
def voynich(name='ZL3b'):
    recs = vlib.load_voynich(name, ltypes=('P',))
    units, cur = [], None
    for r in recs:
        ws = [tuple(vlib.glyphs(w)) for w in r['words'] if '?' not in w]
        if not ws:
            continue
        if r['para_start'] or cur is None or cur['folio'] != r['folio']:
            cur = {'id': f"{r['folio']}.{r['n']}", 'folio': r['folio'], 'stratum': r['illus'],
                   'lang': r['lang'], 'lines': []}
            units.append(cur)
        cur['lines'].append(ws)
    return units


# ---------------- chant ----------------
def _sql_tuples(s):
    """Yield tuples of python values from a MySQL INSERT ... VALUES (...),(...); block."""
    i, n = 0, len(s)
    while True:
        i = s.find('(', i)
        if i < 0:
            return
        i += 1; vals = []; cur = None
        while i < n:
            c = s[i]
            if c == "'":
                i += 1; buf = []
                while True:
                    c = s[i]
                    if c == '\\':
                        buf.append(s[i + 1]); i += 2; continue
                    if c == "'":
                        if s[i + 1] == "'":
                            buf.append("'"); i += 2; continue
                        i += 1; break
                    buf.append(c); i += 1
                cur = ''.join(buf)
            elif c == ',':
                vals.append(cur); cur = None; i += 1
            elif c == ')':
                vals.append(cur); i += 1; break
            elif s.startswith('NULL', i):
                cur = None; i += 4
            elif c in ' \n\t':
                i += 1
            else:
                j = i
                while s[j] not in ',)':
                    j += 1
                cur = s[i:j].strip(); i = j
        yield vals
        if i >= n or s[i] == ';':
            return


BAR_RE = re.compile(r'^[,;:`\'0-9]+$')
CLEF_RE = re.compile(r'^(c|f|cb)[1-5]$')


def parse_gabc(g):
    """gabc body -> list of phrases (list of syllable note-tuples). Full bars ':' and '::' end a phrase."""
    g = re.sub(r'<[^>]*>', '', g)
    g = re.sub(r'\[[^\]]*\]', '', g)
    phrases, cur = [], []
    for m in re.finditer(r'\(([^)]*)\)', g):
        grp = m.group(1).strip()
        if not grp:
            continue
        if CLEF_RE.match(grp):
            continue
        core = grp.replace(' ', '')
        if BAR_RE.match(core):
            if ':' in core:
                if cur:
                    phrases.append(cur)
                cur = []
            continue
        # strip bars inside a group (e.g. 'f.;') -- treat trailing ':' as a phrase end
        endp = ':' in re.sub(r'[a-mA-M]', '', grp) and not re.search(r'[a-mA-M]\s*$', grp)
        notes = []
        for pm in re.finditer(r'([a-mA-M])([xy#+]?)', grp):
            if pm.group(2):
                continue  # accidental / custos, not a sounded note
            notes.append(pm.group(1).lower())
        if notes:
            cur.append(tuple(notes))
        if endp and cur:
            phrases.append(cur); cur = []
    if cur:
        phrases.append(cur)
    return phrases


def build_chant(sql_path):
    s = open(sql_path, encoding='utf-8').read()
    i = s.find("INSERT INTO `gregobase_chants`"); j = s.find("CREATE TABLE `gregobase_chant_sources`")
    region = s[i:j]
    rows = []
    for m in re.finditer(r'INSERT INTO `gregobase_chants`[^\n]*VALUES\n', region):
        k = region.find(';\n', m.end())
        rows.extend(_sql_tuples(region[m.end():k + 1]))
    out = []
    for v in rows:
        if len(v) < 14:
            continue
        cid, inc, part, mode, gabc = v[0], v[3], v[5], v[6], v[10]
        if not gabc:
            continue
        try:
            body = json.loads(gabc)
        except Exception:
            body = gabc
        if isinstance(body, list):
            body = ' '.join(x for x in body if isinstance(x, str))
        if not isinstance(body, str):
            continue
        ph = parse_gabc(body)
        if not ph:
            continue
        out.append({'id': cid, 'incipit': inc, 'part': part, 'mode': mode,
                    'phrases': [[''.join(w) for w in p] for p in ph]})
    return out


def chant(max_units=None, seed=0, min_lines=2):
    if not os.path.exists(CHANT_JSON):
        assert CHANT_SQL, 'set GREGOBASE_SQL to the GregoBase SQL dump path'
        data = build_chant(CHANT_SQL)
        with gzip.open(CHANT_JSON, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
    data = json.load(gzip.open(CHANT_JSON, 'rt', encoding='utf-8'))
    units, seen = [], set()
    for d in data:
        m = (d['mode'] or '').strip()
        if m not in list('12345678'):
            continue
        lines = [[tuple(w) for w in p] for p in d['phrases'] if len(p) >= 2]
        if len(lines) < min_lines:
            continue
        key = (d['incipit'], m, len(lines))
        if key in seen:      # GregoBase holds several versions of one chant; keep one
            continue
        seen.add(key)
        units.append({'id': d['id'], 'stratum': m, 'part': d['part'], 'lines': lines})
    rng = random.Random(seed)
    if max_units and len(units) > max_units:
        units = rng.sample(units, max_units)
    return units


# ---------------- verbose substitution of real language ----------------
VS_ALPHA = list('ABCDEFGHIJKLMNOPQR')


def verbose_table(seed=7):
    rng = random.Random(seed)
    tab = {}
    used = set()
    for ch in 'abcdefghijklmnopqrstuvwxyzàèéìíòóùúäöüßñç':
        while True:
            k = rng.choice([1, 2, 2, 3])
            g = tuple(rng.choice(VS_ALPHA) for _ in range(k))
            if g not in used:
                used.add(g); tab[ch] = g; break
    return tab


def verbose_lang(key, max_words=40000, seed=7, para_lines=None):
    tab = verbose_table(seed)
    lines = vlib.load_ref(key, max_words=max_words, skip_frac=0.05)
    units, cur = [], None
    for L in lines:
        if L['para_start'] or cur is None or (para_lines and len(cur['lines']) >= para_lines):
            cur = {'id': str(len(units)), 'stratum': '-', 'lines': []}; units.append(cur)
        ws = []
        for w in L['words']:
            t = tuple(s for c in w for s in tab.get(c, ()))
            if t:
                ws.append(t)
        if ws:
            cur['lines'].append(ws)
    return [u for u in units if u['lines']]


def plain_lang(key, max_words=40000):
    lines = vlib.load_ref(key, max_words=max_words, skip_frac=0.05)
    units, cur = [], None
    for L in lines:
        if L['para_start'] or cur is None:
            cur = {'id': str(len(units)), 'stratum': '-', 'lines': []}; units.append(cur)
        cur['lines'].append([tuple(w) for w in L['words']])
    return units


# ---------------- Voynich nulls ----------------
def glyph_shuffle(units, seed=1):
    rng = random.Random(seed)
    out = []
    for u in units:
        nl = []
        for L in u['lines']:
            g = [s for w in L for s in w]; rng.shuffle(g)
            ws, i = [], 0
            for w in L:
                ws.append(tuple(g[i:i + len(w)])); i += len(w)
            nl.append(ws)
        out.append(dict(u, lines=nl))
    return out


def trigram_words(units, seed=1):
    """Each word regenerated from a glyph trigram model of words (start/end marked).
    Keeps word structure statistics, kills any line-level or unit-level organisation."""
    rng = random.Random(seed)
    tri = defaultdict(Counter)
    for u in units:
        for L in u['lines']:
            for w in L:
                s = ('^', '^') + w + ('$',)
                for a, b, c in zip(s, s[1:], s[2:]):
                    tri[(a, b)][c] += 1
    cum = {k: (list(v.keys()), list(v.values())) for k, v in tri.items()}

    def gen():
        a, b, w = '^', '^', []
        while len(w) < 20:
            ks, vs = cum[(a, b)]
            c = rng.choices(ks, vs)[0]
            if c == '$':
                break
            w.append(c); a, b = b, c
        return tuple(w) or ('o',)
    return [dict(u, lines=[[gen() for _ in L] for L in u['lines']]) for u in units]


def all_corpora(chant_n=1500):
    zl = voynich('ZL3b')
    C = {
        'V-ZL': zl,
        'V-IT': voynich('IT2a'),
        'chant': chant(max_units=chant_n),
        'vs-Latin-Caesar': verbose_lang('Latin-Caesar'),
        'vs-Italian-Manzoni': verbose_lang('Italian-Manzoni'),
        'vs-Italian-Dante': verbose_lang('Italian-Dante', para_lines=12),
        'V-ZL-gshuf': glyph_shuffle(zl),
        'V-ZL-tri': trigram_words(zl),
    }
    return C


if __name__ == '__main__':
    if len(sys.argv) > 1:
        CHANT_SQL = sys.argv[1]
    C = all_corpora()
    for k, u in C.items():
        nl = sum(len(x['lines']) for x in u)
        nw = sum(len(L) for x in u for L in x['lines'])
        ns = sum(len(w) for x in u for L in x['lines'] for w in L)
        syms = Counter(s for x in u for L in x['lines'] for w in L for s in w)
        print(f"{k:22s} units {len(u):5d} lines {nl:6d} words/line {nw/nl:5.1f} sym/line {ns/nl:6.1f} alphabet {len(syms)}")
