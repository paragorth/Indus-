"""v58: parse downloaded candidate texts into length fingerprints.

Output data/v58_ckpt/texts.json:
  {id: {lang, genre, src, units: [{t: title, w: words, c: letters, p: [words per paragraph]}]}}
Entry = the natural unit of the work (chapter, recipe, plant entry, psalm).
Heuristic parsers; segmentation noise is part of what the alignment must tolerate.
"""
import os, re, json, glob, html, sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.dirname(HERE)
CK = os.path.join(VD, 'data', 'v58_ckpt')
SRC = os.path.join(CK, 'src')

WRE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)

import unicodedata
def wc(text):
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    ws = WRE.findall(text)
    return len(ws), sum(len(w) for w in ws)

def unit(title, paras):
    paras = [p for p in paras if p.strip()]
    ps = [wc(p) for p in paras]
    ps = [x for x in ps if x[0] > 0]
    return {'t': title[:60], 'w': sum(x[0] for x in ps), 'c': sum(x[1] for x in ps), 'p': [x[0] for x in ps]}

def strip_tags(s):
    s = re.sub(r'<[^>]+>', ' ', s)
    return html.unescape(s)

# ---------------- TEI (Perseus, First1KGreek) ----------------
NS = '{http://www.tei-c.org/ns/1.0}'

def tei_units(path, chap_test=None, books=None, chap_pat=r'<div\b[^>]*subtype="chapter"[^>]*>|<div2\b[^>]*type="chapter"[^>]*>'):
    """regex TEI splitter (robust to malformed XML): a chapter runs to the next chapter or book opening."""
    s = open(path, encoding='utf-8').read()
    s = re.sub(r'<note\b.*?</note>', ' ', s, flags=re.S)
    s = re.sub(r'<head\b.*?</head>', ' ', s, flags=re.S)
    s = re.sub(r'<bibl\b.*?</bibl>', ' ', s, flags=re.S)
    s = re.sub(r'&[a-zA-Z]+;', ' ', s)
    i0 = s.find('<body'); s = s[i0:]
    book_re = re.compile(r'<div1?\b[^>]*(?:sub)?type="book"[^>]*n="([^"]*)"')
    marks = [(m.start(), 'b', m.group(1)) for m in book_re.finditer(s)]
    marks += [(m.start(), 'c', (re.search(r'n="([^"]*)"', m.group(0)) or [None, '?'])[1]) for m in re.finditer(chap_pat, s)]
    marks.sort()
    out, book = [], None
    for k, (pos, typ, n) in enumerate(marks):
        if typ == 'b': book = n; continue
        end = marks[k + 1][0] if k + 1 < len(marks) else len(s)
        seg = s[pos:end]
        if books is not None and book not in books: continue
        paras = re.findall(r'<p\b[^>]*>(.*?)</p>', seg, flags=re.S) or [seg]
        out.append(unit('%s.%s' % (book, n), [strip_tags(x) for x in paras]))
    return [u for u in out if u['w'] >= 3]

is_chap = lambda tag, typ: typ == 'chapter'

# ---------------- Augustana HTML ----------------
def aug_pages(d):
    fs = sorted(f for f in glob.glob(os.path.join(SRC, d, '*.html')) if not f.endswith('_index.html'))
    return [open(f, encoding='utf-8', errors='replace').read() for f in fs]

def aug_anchor_units(d, min_anchors=5):
    out = []
    for s in aug_pages(d):
        i = s.find('class="contentus"'); s = s[i if i > 0 else 0:]
        s = re.sub(r'<span class="f_canusg">.*?</span>', ' ', s, flags=re.S)   # editorial refs
        s = re.sub(r'\{[^}]*\}', ' ', s)
        s = re.sub(r'\[\d+\]', ' ', s)
        parts = re.split(r'<a name="(\d+[a-z]?)">', s)
        if len(parts) < 2 * min_anchors: continue
        for k in range(1, len(parts), 2):
            body = parts[k + 1]
            heads = re.findall(r'<h3>(.*?)</h3>', body, flags=re.S)
            # first h3 blocks that are short are headings (Capitulum N., De X.)
            paras = []
            title = ''
            for h in heads:
                t = strip_tags(h).strip()
                if len(t) < 60 and (t.startswith('Cap') or t.startswith('De ') or t.startswith('Von ') or not paras and len(t) < 40):
                    title += t + ' '; continue
                paras.append(t)
            if paras: out.append(unit(title.strip() or parts[k], paras))
    return out

def hildegard():
    out = []
    for s in aug_pages('hildegard_physica'):
        i = s.find('name="inci"'); s = s[i if i > 0 else 0:]
        s = re.sub(r'<span class="f_canusg">.*?</span>', ' ', s, flags=re.S)
        s = re.sub(r'\{[^}]*\}|\[\d+\]', ' ', s)
        parts = re.split(r'<h3>\s*(?:Capitulum\s*)?[IVXLC]+\.?\s*</h3>', s)
        for p in parts[1:]:
            hs = [strip_tags(h).strip() for h in re.findall(r'<h3>(.*?)</h3>', p, flags=re.S)]
            title = hs[0] if hs else ''
            out.append(unit(title, hs[1:]))
    return [u for u in out if u['w'] >= 3]

def regimen():
    s = open(os.path.join(SRC, 'regimen_salernitanum', 'reg_sana.html'), encoding='utf-8', errors='replace').read()
    i = s.find('contentus'); t = strip_tags(s[i:])
    t = re.sub(r'\s+', ' ', t)
    parts = re.split(r'\s(?=[IVXLC]+ (?:De|Remed|Contra|Abl|Quae|Qui|[A-Z][a-z]))', t)
    return [unit(p[:30], [p.split(' ', 1)[1] if ' ' in p else p]) for p in parts[1:] if len(p) > 20]

def konrad():
    """Buch der Natur: one Augustana page per chapter (kon_BCCC.html); text after the '____' rule."""
    out = []
    for f in sorted(glob.glob(os.path.join(SRC, 'konrad_buch_der_natur', 'kon_[0-9]*.html'))):
        s = open(f, encoding='utf-8', errors='replace').read()
        i = s.find('contentus'); s = s[i if i > 0 else 0:]
        k = s.find('____'); s = s[k if k > 0 else 0:]
        k = s.find('&lt;&lt;&lt;'); s = s[:k] if k > 0 else s
        t = strip_tags(s)
        t = re.sub(r'\[\d+\]|\{[^}]*\}|_+', ' ', t)
        m = re.match(r'\s*(\d+\.)?\s*(Von [^.]*\.)?', t)
        title = os.path.basename(f) + ' ' + (m.group(2) or '')
        body = t[m.end():] if m else t
        u = unit(title, [body])
        if u['w'] >= 5: out.append(u)
    return out

# ---------------- plain-text sources ----------------
def gut_body(path):
    t = open(path, encoding='utf-8', errors='replace').read()
    m1 = re.search(r'\*\*\* ?START[^\n]*\n', t); m2 = re.search(r'\*\*\* ?END', t)
    return t[m1.end():m2.start()] if m1 and m2 else t

def split_on(lines, is_head, drop=lambda l: False):
    out, cur, title = [], None, None
    for l in lines:
        if is_head(l):
            if cur is not None: out.append(unit(title, cur))
            cur, title = [], l.strip()
            continue
        if cur is None or drop(l): continue
        if not l.strip():
            cur.append('\n')
        else:
            if cur and cur[-1] != '\n': cur[-1] += ' ' + l
            else: cur.append(l)
    if cur is not None: out.append(unit(title, cur))
    return [u for u in out if u['w'] >= 3]

def apicius_eng():
    L = gut_body(os.path.join(SRC, 'apicius_eng.txt')).split('\n')
    note = {'on': False}
    def drop(l):
        if re.match(r'\s+\[\d+\]', l): note['on'] = True; return True
        if note['on'] and (l.startswith('    ') or not l.strip()):
            if not l.strip(): note['on'] = False
            return True
        note['on'] = False
        return False
    return split_on(L, lambda l: bool(re.match(r'\[\d+\] [A-Z]', l)), drop)

def apicius_lat():
    out = []
    for i in range(1, 11):
        f = os.path.join(SRC, 'apicius_lat', 'apicius%d.shtml' % i)
        if not os.path.exists(f): continue
        s = open(f, encoding='latin-1').read()
        ps = [re.sub(r'\s+', ' ', strip_tags(p)).strip() for p in re.split(r'<[Pp][ >]', s)]
        cur, title = None, None
        for p in ps:
            if re.match(r'\d+\. [A-Z]', p) or re.match(r'\d+\.\s*$', p):
                if cur is not None: out.append(unit(title, cur))
                cur, title = [], p; continue
            if re.match(r'[IVXL]+\. ', p) or p.startswith('LIBER'):
                if cur is not None: out.append(unit(title, cur)); cur = None
                continue
            if cur is not None and p: cur.append(p)
        if cur is not None: out.append(unit(title, cur))
    return [u for u in out if u['w'] >= 3]

def forme_of_cury():
    L = gut_body(os.path.join(SRC, 'forme_of_cury.txt')).split('\n')
    start = next(i for i, l in enumerate(L) if l.startswith('FOR TO MAKE GRONDEN BENES'))
    L = L[start:]
    note = {'on': False}
    def drop(l):
        if re.match(r'\[\d+\]', l): note['on'] = True; return True
        if note['on'] and (l.startswith('    ') or not l.strip()):
            if not l.strip(): note['on'] = False
            return True
        note['on'] = False
        return False
    head = lambda l: bool(re.match(r"[A-Z][A-Z' ,]+.*\. [IVXLC]+\.?\s*$", l)) or bool(re.match(r'[IVXLC]+\. [A-Z][A-Za-z]', l))
    return split_on(L, head, drop)

def culpeper():
    L = gut_body(os.path.join(SRC, 'culpeper.txt')).split('\n')
    return split_on(L, lambda l: bool(re.match(r"\s{2,}[A-Z][A-Z'’ ,-]{3,}\.\s*$", l)))

def macer():
    t = open(os.path.join(SRC, 'macer_floridus.wiki'), encoding='utf-8').read()
    t = re.sub(r'\{\{Versus\|\d+\}\}', ' ', t)
    parts = re.split(r'\n==+([^=\n]+)==+\n', t)
    out = []
    for k in range(1, len(parts), 2):
        body = re.sub(r'<[^>]+>|\{\{[^}]*\}\}', ' ', parts[k + 1])
        out.append(unit(parts[k].strip(), [body]))
    return [u for u in out if u['w'] >= 3]

def sefaria(fn):
    d = json.load(open(os.path.join(SRC, 'sefaria', fn)))
    v = d['versions'][0]['text']
    out = []
    for i, ch in enumerate(v):
        paras = [re.sub(r'<[^>]+>', ' ', x).replace('־', ' ') for x in ch]
        out.append(unit('ch%d' % (i + 1), paras))
    return out

def ocr_caps_entries(path, start_pat=None, end_pat=None, min_w=8):
    """OCR book with entries headed by an all-capitals line (e.g. Circa instans editions).
    Footnote blocks (lines beginning '1. ' ... with 'Ms.' etc.) and page furniture dropped."""
    t = open(path, encoding='utf-8', errors='replace').read()
    if start_pat:
        m = re.search(start_pat, t); t = t[m.start():] if m else t
    if end_pat:
        m = re.search(end_pat, t); t = t[:m.start()] if m else t
    L = t.split('\n')
    def head(l):
        s = l.strip()
        return 3 <= len(s) <= 40 and re.fullmatch(r"[A-ZÀ-Ý][A-ZÀ-Ý'’ ,.\-<>]+", s) is not None and sum(c.isalpha() for c in s) >= 3
    def drop(l):
        s = l.strip()
        return (bool(re.match(r'[—\-–] ?\d+ ?[—\-–]', s)) or 'Ms.' in s or bool(re.match(r'\d+ ?(et \d+)?\. ?(Ms|Cf|Var|Sous|Le |La |Les |Il )', s))
                or (len(s) > 0 and sum(c.isalpha() for c in s) / len(s) < 0.6))
    return [u for u in split_on(L, head, drop) if u['w'] >= min_w]

def antidotarium():
    """Antidotarium Nicolai, Middle Dutch + Latin (van den Berg 1917 OCR): split at numbered entries."""
    t = open(os.path.join(SRC, 'ia', 'eenemiddelnederl00nicouoft.txt'), encoding='utf-8', errors='replace').read()
    L = t.split('\n')
    head = lambda l: bool(re.match(r'\s*\d{1,3}\.\s+\S', l)) and len(l.strip()) > 25
    drop = lambda l: bool(re.match(r'\s*[l1-9]\)', l)) or (len(l.strip()) > 0 and sum(c.isalpha() for c in l) / max(1, len(l.strip())) < 0.5)
    us = split_on(L, head, drop)
    # first line of an entry is in the head; add its words
    return [u for u in us if u['w'] >= 8]

def v21_herbals():
    d = json.load(open(os.path.join(VD, 'data', 'derived', 'v21_herbals.json')))
    out = {}
    for k in d:
        out[k] = [unit(e['id'], [' '.join(p) if isinstance(p[0], str) else ' '.join(' '.join(s) for s in p) for p in e['paras']]) for e in d[k]]
    return out

def gerard():
    """Gerard's Herball (1597) OCR pages from v13 (page = unit; an illustrated printed herbal)."""
    p = os.path.join(VD, 'data', 'derived', 'v13_gerard.json')
    d = json.load(open(p))['pages']
    out = []
    for k in sorted(d, key=lambda z: int(re.sub(r'\D', '', z) or 0)):
        ws = [w for w in d[k]['ocr'] if sum(ch.isalpha() for ch in w) >= 2]
        if len(ws) >= 5: out.append({'t': 'p' + k, 'w': len(ws), 'c': sum(len(w) for w in ws), 'p': [len(ws)]})
    return out

def main():
    T = {}
    def add(k, lang, genre, src, units):
        T[k] = {'lang': lang, 'genre': genre, 'src': src, 'units': units}
        ws = [u['w'] for u in units]
        print('%-26s %-4s %-8s n=%4d words=%7d median=%5.0f' % (k, lang, genre, len(units), sum(ws), sorted(ws)[len(ws) // 2] if ws else 0), flush=True)
    jobs = [
        ('pliny_nh_lat', 'la', 'encycl', 'Perseus phi0978.phi001.perseus-lat2', lambda: tei_units(os.path.join(SRC, 'pliny_nh_lat.xml'), is_chap)),
        ('pliny_nh_eng', 'en', 'encycl', 'Perseus phi0978.phi001.perseus-eng1 (Bostock-Riley)', lambda: tei_units(os.path.join(SRC, 'pliny_nh_eng.xml'), lambda t, y: y == 'chapter' or t == 'div2')),
        ('celsus_lat', 'la', 'medical', 'Perseus phi0836.phi002.perseus-lat4', lambda: tei_units(os.path.join(SRC, 'celsus_lat.xml'), is_chap)),
        ('celsus_eng', 'en', 'medical', 'Perseus phi0836.phi002.perseus-eng2', lambda: tei_units(os.path.join(SRC, 'celsus_eng.xml'), is_chap)),
        ('dioscorides_grc', 'grc', 'herbal', 'First1KGreek tlg0656.tlg001', lambda: tei_units(os.path.join(SRC, 'dioscorides_grc.xml'), is_chap)),
        ('theophrastus_grc', 'grc', 'botany', 'First1KGreek tlg0093.tlg001', lambda: tei_units(os.path.join(SRC, 'theophrastus_hp_grc.xml'), is_chap)),
        ('hildegard_physica', 'la', 'herbal', 'Augustana hil_phy1-9', hildegard),
        ('regimen_salern', 'la', 'regimen', 'Augustana reg_sana', regimen),
        ('konrad_bdn', 'de', 'encycl', 'Augustana kon_*', konrad),
        ('konrad_plants', 'de', 'herbal', 'Augustana kon_4*, kon_5* (trees, herbs)', lambda: [u for u in konrad() if u['t'][:5] in ('kon_4', 'kon_5')]),
        ('apicius_lat', 'la', 'recipe', 'Latin Library apicius1-5', apicius_lat),
        ('apicius_eng', 'en', 'recipe', 'Gutenberg 29728 (Vehling)', apicius_eng),
        ('forme_of_cury', 'enm', 'recipe', 'Gutenberg 8102', forme_of_cury),
        ('culpeper', 'en', 'herbal', 'Gutenberg 49513', culpeper),
        ('macer_floridus', 'la', 'herbal', 'la.wikisource De viribus herbarum', macer),
        ('psalms_he', 'he', 'psalter', 'Sefaria Psalms (Miqra)', lambda: sefaria('psalms_he.json')),
        ('psalms_en', 'en', 'psalter', 'Sefaria Psalms (English)', lambda: sefaria('psalms_en.json')),
        ('circa_instans_fr', 'fro', 'herbal', 'archive.org BIUSante_pharma_032591 (Dorveaux 1913)', lambda: ocr_caps_entries(os.path.join(SRC, 'ia', 'BIUSante_pharma_032591.txt'))),
        ('antidotarium_nl', 'dum+la', 'recipe', 'archive.org eenemiddelnederl00nicouoft', antidotarium),
        ('leechdoms_v1', 'ang', 'herbal', 'archive.org LeechdomsWortcunningStarcraftV1', lambda: ocr_caps_entries(os.path.join(SRC, 'ia', 'LeechdomsWortcunningStarcraftV1.txt'))),
        ('gerard_pages', 'en', 'herbal', 'repo data/derived/v13_gerard.json (OCR pages)', gerard),
        ('balneis_synopsis', 'la', 'baths', 'archive.org synopsiseorumqua00lomb', lambda: ocr_caps_entries(os.path.join(SRC, 'ia', 'synopsiseorumqua00lomb.txt'))),
    ]
    only = sys.argv[1:]
    old = json.load(open(os.path.join(CK, 'texts.json'))) if only and os.path.exists(os.path.join(CK, 'texts.json')) else {}
    T.update(old)
    for k, lang, genre, src, fn in jobs:
        if only and k not in only: continue
        try:
            add(k, lang, genre, src, fn())
        except Exception as e:
            print('FAIL', k, repr(e)[:200])
    if not only:
        for k, us in v21_herbals().items():
            add('v21_' + k, k.lower(), 'herbal', 'repo data/derived/v21_herbals.json', us)
    json.dump(T, open(os.path.join(CK, 'texts.json'), 'w'))

if __name__ == '__main__':
    main()
