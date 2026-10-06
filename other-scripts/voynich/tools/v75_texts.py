"""v75 IF IT IS A CODE, WHAT KIND OF CODE? Real designation systems as (section, entry-words) lists.

Every system is reduced to the same abstract form: an ordered list of entries, each entry a list of plain
words (a-z), each entry tagged with a section (the system's own division, merged into at most 8 contiguous
groups, or 6 alphabetical/ordinal bins when it has none). Kinds:
  GLOSS  herbal / medical / alchemical synonym glossaries (Alphita, Sinonoma Bartholomei, Forme of Cury glossary,
         Ruland's Lexicon alchemiae in Waite's English)
  INDEX  printed book indexes (Culpeper, Apicius, Pliny; v71 cuts)
  CATAL  library catalogue entries with shelfmarks (Syon; Christ Church, St Augustine's, Dover from James 1903)
  INGRED ingredient designations in formularies (Antidotarium Nicolai, Apicius; v73 item streams)
  APOTH  apothecary drawer and jar labels (Culpeper 1649 catalogue of simples, Pechey 1694 Latin preparation
         labels, Culpeper 1649 compound headings)
  ASTRO  star designations of the Almagest catalogue (medieval Latin version printed by Peters & Knobel 1915),
         split into northern / zodiacal / southern constellations as three sub-systems
  ALCH   alchemical symbol lists (Gessmann 1899 symbol tables; Unicode alchemical symbols block)
  NOMEN  nomenclator designation lists (Meister 1906, papal curia nomenclators; early and late halves)
  SIGNS  monastic sign-language lists (Cluny: Udalric, PL 149; Hirsau: William, PL 150)
  MODERN modern structured codes (Unicode character names by block, IANA time-zone names, Debian package
         names + short descriptions by section)
  LANG   ordinary language (prose, herbal prose, verse; v71/v72 reference texts)
Output: data/v75_ckpt/systems.json = {sid: {kind, entries: [[sec, [words]], ...]}}
Only data and symbols are taken from these sources, never anyone's reading of the Voynich.
"""
import os, re, json, sys, glob, unicodedata, html, subprocess
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
VD = os.path.dirname(HERE)
CK = os.path.join(VD, 'data', 'v75_ckpt'); SRC = os.path.join(CK, 'src')


def norm(s):
    s = s.replace('ſ', 's').replace('æ', 'ae').replace('œ', 'oe').replace('ß', 'ss')
    s = unicodedata.normalize('NFD', s.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s


def W(s, minlen=1):
    return [w for w in re.findall(r'[a-z]+', norm(s)) if len(w) >= minlen]


def bins(entries, nb=6):
    """give entries (list of word lists) ordinal sections (contiguous bins)."""
    n = len(entries)
    return [['b%d' % min(nb - 1, i * nb // max(1, n)), e] for i, e in enumerate(entries)]


def merge_secs(ents, maxs=8):
    """natural sections -> at most maxs contiguous groups by order of first appearance (by token mass)."""
    order = []
    for s, _ in ents:
        if not order or order[-1] != s: order.append(s)
    mass = Counter()
    for s, e in ents: mass[s] += len(e)
    tot = sum(mass.values()); grp = {}; acc = 0; seen = set()
    for s in order:
        if s in seen: continue
        seen.add(s); grp[s] = 'g%d' % min(maxs - 1, int(maxs * acc / max(1, tot))); acc += mass[s]
    return [[grp[s], e] for s, e in ents]


def ocr_lines(path):
    return open(path, encoding='utf-8', errors='replace').read().split('\n')


# ------------------------------------------------------------------ glossaries
def glossary_ocr(path, start_pat, end_pat=None, skip_digits=True):
    """entries start on a line that begins with a capitalised word after an entry ended with '.'"""
    L = ocr_lines(path)
    i0 = next(i for i, l in enumerate(L) if re.match(start_pat, l))
    i1 = next((i for i, l in enumerate(L) if i > i0 + 100 and end_pat and re.match(end_pat, l)), len(L))
    ents, cur, ended = [], [], True
    for l in L[i0:i1]:
        s = l.strip()
        if not s: continue
        if skip_digits and re.search(r'\d', s): continue
        if re.search(r'(Bart\.|App\.|Harl\.|MS\.|Sloane|Add\.)', s): continue
        if len(s) < 3 or sum(c.isalpha() for c in s) < 0.6 * len(s): continue
        if ended and re.match(r'^[A-Z][a-z]{2,}', s):
            if cur: ents.append(cur)
            cur = []
        cur += W(s)
        ended = s.endswith('.')
    if cur: ents.append(cur)
    return [e for e in ents if 2 <= len(e) <= 60]


def ruland():
    L = ocr_lines(os.path.join(SRC, 'a-lexicon-of-alchemy-martin-rulandus.txt'))
    ents, cur = [], None
    for l in L:
        m = re.match(r'^([A-Z][A-Z ,\.]{2,60}?)\s+-{2,}\s*(.*)', l.strip())
        if m:
            if cur: ents.append(cur)
            cur = W(m.group(1)) + W(m.group(2))
        elif cur is not None and l.strip() and not re.search(r'\d{2,}', l):
            if len(cur) < 60: cur += W(l)
    if cur: ents.append(cur)
    return [e[:40] for e in ents if len(e) >= 2]


# ------------------------------------------------------------------ catalogues
def james():
    """Christ Church (A), St Augustine's (B), Dover (C): numbered catalogue entries. Section = running heading
    (distinctio / gradus) approximated by ordinal bins inside each catalogue."""
    L = ocr_lines(os.path.join(SRC, 'ancientlibraries00jame.txt'))
    txt = '\n'.join(L)
    # boundaries: find catalogue heads
    def idx(pat, start=0):
        m = re.search(pat, txt[start:]); return start + m.start() if m else None
    a = idx(r'\n\s*1\s+(Biblia|Bibliotheca)');
    out = {}
    marks = [m.start() for m in re.finditer(r'(?i)\n[^\n]*ST\.?\s+AUGUSTINE[\'’]?S[^\n]*CATALOGUE', txt)]
    dov = [m.start() for m in re.finditer(r'(?i)\n[^\n]*DOVER[^\n]*CATALOGUE', txt)]
    return txt, a, marks, dov


def numbered_entries(lines, maxn=3000, tol=3):
    """entries that begin with an Arabic serial number at line start: '497 Collecta de phisica.'"""
    ents, cur, last = [], None, 0
    for l in lines:
        s = l.strip()
        m = re.match(r'^(\d{1,4})\s+[^\d\s]', s)
        if m:
            n = int(m.group(1))
            if 0 < n - last <= tol or (last == 0 and n < 5):
                if cur: ents.append(cur)
                cur = W(s[m.end() - 1:], 1); last = n; continue
        if cur is None: continue
        if re.search(r'\d', s) or not s: continue
        if re.match(r'(?i)(in hoc|hi hoc|/;/ hoc)', s): cur += W(s); continue
        if len(cur) < 40: cur += W(s)
    if cur: ents.append(cur)
    return ents


def syon():
    """Syon: shelfmark lines 'Donor  M 96  secundo-folio' followed by contents; section = shelfmark letter."""
    L = ocr_lines(os.path.join(SRC, 'catalogueoflibra00syonuoft.txt'))
    ents, cur, sec = [], None, None
    for l in L:
        s = l.strip()
        m = re.match(r'^(.{0,30}?)\b([A-Z])\s+(\d{1,3}|[lI]\d{0,2}|[lIo]{2,3})\^?\s+(.{0,30})$', s)
        if m and len(s) < 60:
            if cur: ents.append([sec, cur])
            sec = m.group(2); cur = W(m.group(1)) + [sec.lower()] + W(m.group(4)); continue
        if cur is None or not s or re.search(r'\d', s): continue
        if len(cur) < 40: cur += W(s)
    if cur: ents.append([sec, cur])
    return [[s, e] for s, e in ents if len(e) >= 3]


# ------------------------------------------------------------------ apothecary (TCP XML)
def tcp_text(x):
    x = re.sub(r'<g ref="char:EOLhyphen"/>', '', x)
    x = re.sub(r'<note[^>]*>.*?</note>', ' ', x, flags=re.S)
    x = re.sub(r'<gap[^>]*>(</gap>)?', '', x)
    x = re.sub(r'<[^>]+>', ' ', x)
    return html.unescape(x)


def culpeper_simples():
    t = open(os.path.join(SRC, 'A35390.xml'), encoding='utf-8').read()
    a = t.find('A CATALOGVE OF THE SIMPLES'); b = t.find('COMPOVNDS CONTAINED IN THE DISPENSATORY')
    part = t[a:b]
    ents = []
    for dm in re.finditer(r'<head>(.*?)</head>(.*?)(?=<head>|$)', part, flags=re.S):
        head = tcp_text(dm.group(1)).strip()
        if len(W(head)) > 8 or 'PREFACE' in head.upper() or 'See the' in head: continue
        for pm in re.finditer(r'<p>\s*<hi>(.*?)</hi>', dm.group(2), flags=re.S):
            lab = W(tcp_text(pm.group(1)))
            if 1 <= len(lab) <= 6: ents.append([head[:20], lab])
    return ents


def pechey_labels():
    t = open(os.path.join(SRC, 'A53916.xml'), encoding='utf-8').read()
    ents, sec = [], None
    for m in re.finditer(r'<head>(.*?)</head>', t, flags=re.S):
        h = m.group(1)
        hm = re.search(r'in Latin,?\s*<hi>(.*?)</hi>', h, flags=re.S | re.I)
        if hm:
            lab = W(tcp_text(hm.group(1)))
            if 1 <= len(lab) <= 8 and sec: ents.append([sec, lab])
        else:
            hh = tcp_text(h).strip()
            if 1 <= len(W(hh)) <= 4: sec = hh[:24]
    return ents


def culpeper_compounds():
    t = open(os.path.join(SRC, 'A35390.xml'), encoding='utf-8').read()
    a = t.find('COMPOVNDS CONTAINED IN THE DISPENSATORY'); part = t[a:]
    ents, sec = [], None
    for m in re.finditer(r'<(div[^>]*|head)>(.*?)</head>|<div type="(section|subsection)"[^>]*>', part, flags=re.S):
        pass
    # heads at two depths: section heads in CAPITALS, item heads in mixed case
    for m in re.finditer(r'<head>(.*?)</head>', part, flags=re.S):
        h = tcp_text(m.group(1)).strip(); ws = W(h)
        if not ws: continue
        letters = [c for c in h if c.isalpha()]
        if sum(c.isupper() for c in letters) > 0.8 * len(letters) and len(ws) <= 6:
            sec = h[:24]; continue
        if sec and 1 <= len(ws) <= 10: ents.append([sec, ws])
    return ents


# ------------------------------------------------------------------ astronomy
def ptolemy():
    L = ocr_lines(os.path.join(SRC, 'ptolemy_stars.txt'))
    ents, sec = [], None
    region = 'north'
    for l in L[9000:30000]:
        s = l.strip()
        if re.match(r'^(Northern|Zodiacal|Southern) Constellations', s): region = s.split()[0].lower()[:5]
        if re.fullmatch(r'[A-Z][A-Z \.]{3,30}', s) and len(s.split()) <= 3: sec = s.rstrip('.'); continue
        m = re.match(r'^([0-9IlO]{1,2})\.\s+([A-Z][a-z].*)', s)
        if m and sec:
            ws = W(m.group(2))
            if 2 <= len(ws) <= 25: ents.append([region, sec, ws])
    return ents


# ------------------------------------------------------------------ alchemy
def gessmann():
    L = ocr_lines(os.path.join(SRC, 'diegeheimsymbol01gessgoog.txt'))
    ents = []
    for i, l in enumerate(L[:700]):
        m = re.match(r'^([IVXLCl]+)\s?:\s*(.*)', l.strip())
        if m:
            ws = W(m.group(2), 2)
            if ws: ents.append(ws)
    return bins(ents)


def unicode_alch():
    ents = []
    for c in range(0x1F700, 0x1F780):
        n = unicodedata.name(chr(c), '')
        if n: ents.append(W(n.replace('ALCHEMICAL SYMBOL FOR', '')))
    return bins(ents)


# ------------------------------------------------------------------ nomenclators
GERMAN = set('der die das und ist in von zu den dem des mit auf sich nicht ein eine wir als wird sind wurde '
             'auch bei nach aus fur vor wie oder hat so es im am zum zur sie er noch nur dann aber man'.split())


def meister():
    """lines of nomenclator tables: short lines of Latin/Italian designations (proper names, offices, places)."""
    L = ocr_lines(os.path.join(SRC, 'diegeheimschrift00meis.txt'))
    ents = []
    for i, l in enumerate(L):
        s = l.strip()
        if not s or len(s) > 45: continue
        if re.match(r'(?i)(aufl|anm|vgl|s\.\s|siehe|tafel|nr\.)', s): continue
        ws = W(s, 3)
        if not (1 <= len(ws) <= 4): continue
        if any(w in GERMAN for w in ws): continue
        if sum(c.isalpha() for c in s) < 0.55 * len(s): continue
        # must sit in a run of >= 4 such short lines (a table), within 8 lines
        ents.append((i, ws))
    keep = []
    for k, (i, ws) in enumerate(ents):
        near = sum(1 for j in range(max(0, k - 4), min(len(ents), k + 5)) if abs(ents[j][0] - i) <= 10)
        if near >= 6: keep.append((i, ws))
    half = len(keep) // 2
    early = bins([ws for _, ws in keep[:half]]); late = bins([ws for _, ws in keep[half:]])
    return early, late


# ------------------------------------------------------------------ monastic signs
def signs(path, start_pat, n_max=400):
    t = open(path, encoding='utf-8', errors='replace').read()
    a = re.search(start_pat, t).start()
    seg = t[a:a + 120000]
    seg = re.sub(r'\n\s*\d+\s*\n', '\n', seg)
    seg = re.sub(r'-\s*\n\s*', '', seg)
    parts = re.split(r'(?i)\bpro\s?si-?\s?gno\b|\bprosigno\b|\bl\'ro signo\b', seg)
    ents = []
    for p in parts[1:n_max + 1]:
        ws = W(p[:600])
        ws = [w for w in ws if len(w) > 1 or w in 'aeo']
        if 2 <= len(ws): ents.append(ws[:40])
    return bins(ents)


# ------------------------------------------------------------------ modern codes
def unicode_names():
    by = []
    try:
        blocks = []
        import urllib.request
    except Exception:
        pass
    ents = []
    for c in range(0x20, 0x2FFF):
        n = unicodedata.name(chr(c), '')
        if n: ents.append(['u%02d' % (c // 0x400), W(n)])
    return ents


def zoneinfo():
    ents = []
    for root, ds, fs in os.walk('/usr/share/zoneinfo'):
        for f in fs:
            p = os.path.relpath(os.path.join(root, f), '/usr/share/zoneinfo')
            if p.startswith(('posix', 'right', 'Etc')) or '.' in f: continue
            parts = p.split('/')
            if len(parts) < 2: continue
            ents.append([parts[0], W(' '.join(parts[1:]).replace('_', ' '))])
    ents.sort()
    return [e for e in ents if e[1]]


def dpkg():
    try:
        out = subprocess.run(['dpkg-query', '-W', '-f', '${Section}\t${Package}\t${binary:Summary}\n'],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    ents = []
    for l in out.split('\n'):
        p = l.split('\t')
        if len(p) < 3: continue
        ents.append([p[0] or '-', W(p[1].replace('-', ' ').replace('.', ' ')) + W(p[2])])
    ents.sort(key=lambda x: x[0])
    return [e for e in ents if e[1]]


# ------------------------------------------------------------------ language and earlier list cuts
def v71_refs():
    R = json.load(open(os.path.join(VD, 'data', 'v71_ckpt', 'refs.json')))
    out = {}
    for k in ['culpeper_index', 'apicius_index', 'pliny_index', 'cury_glossary']:
        ents = [sum(e, []) for e in R[k]['entries']]
        out[k] = bins([e for e in ents if e])
    for k in ['caesar', 'manzoni', 'kafka', 'cervantes', 'v21_LA', 'v21_IT', 'hildegard', 'macer', 'circa_fr', 'konrad_plants']:
        paras = [p for e in R[k]['entries'] for p in e]
        out[k] = bins([p for p in paras if len(p) >= 5])
    return out


def build():
    S = {}
    def add(sid, kind, ents, note=''):
        ents = [[s, [w for w in e if w]] for s, e in ents]
        ents = [[s, e] for s, e in ents if e]
        ents = merge_secs(ents)
        n = sum(len(e) for _, e in ents)
        S[sid] = dict(kind=kind, entries=ents, note=note)
        wl = sum(len(w) for _, e in ents for w in e) / max(1, n)
        print('%-10s %-6s entries %5d tokens %6d types %5d wlen %.2f secs %d  e.g. %s' % (
            sid, kind, len(ents), n, len({w for _, e in ents for w in e}), wl, len({s for s, _ in ents}),
            ' | '.join(' '.join(e) for _, e in ents[len(ents) // 3: len(ents) // 3 + 3])[:110]), flush=True)
    A = glossary_ocr(os.path.join(SRC, 'alphitaamedicob00mowagoog.txt'), r'^\s*Absinthium|^\s*Abrotanum|^\s*Abies')
    add('ALPHITA', 'GLOSS', bins(A))
    B = glossary_ocr(os.path.join(SRC, 'sinonomabartholo01mirfuoft.txt'), r'^\s*Absinthium|^\s*Abrotanum|^\s*Acacia|^\s*Anetum')
    add('SINONOMA', 'GLOSS', bins(B))
    add('RULAND', 'GLOSS', bins(ruland()))
    V = v71_refs()
    add('CURYGLOS', 'GLOSS', V['cury_glossary'])
    for k, sid in [('culpeper_index', 'IDX_CULP'), ('apicius_index', 'IDX_APIC'), ('pliny_index', 'IDX_PLIN')]:
        add(sid, 'INDEX', V[k])
    # catalogues
    L = ocr_lines(os.path.join(SRC, 'ancientlibraries00jame.txt'))
    txt = '\n'.join(L)
    # the three catalogues follow one another; cut on the serial-number restarts
    ents = numbered_entries(L)
    # split where serial restarts at 1 after a long run
    runs, cur = [], []
    for e in ents:
        cur.append(e)
    # numbered_entries already resets 'last' only by gaps; re-scan to split on restarts
    runs = []; last = 0; cur = []
    for l in L:
        m = re.match(r'^(\d{1,4})\s+[^\d\s]', l.strip())
        if m:
            n = int(m.group(1))
            if n < 4 and last > 200: runs.append(cur); cur = []
            if 0 < n - last <= 3 or n < 4: last = n
        cur.append(l)
    runs.append(cur)
    cc = numbered_entries(L[4947:18172], tol=12)
    add('CAT_CCANT', 'CATAL', bins(cc, 8))
    add('CAT_SYON', 'CATAL', syon())
    from v73_texts import antidotarium_items, apicius_items
    add('ING_ANTID', 'INGRED', bins([e[:40] for e in antidotarium_items()]))
    add('ING_APIC', 'INGRED', bins(apicius_items()))
    add('APO_SIMP', 'APOTH', culpeper_simples())
    add('APO_PECH', 'APOTH', pechey_labels())
    add('APO_COMP', 'APOTH', culpeper_compounds())
    P = ptolemy()
    for reg in ['north', 'zodia', 'south']:
        add('AST_%s' % reg.upper()[:5], 'ASTRO', [[s, e] for r, s, e in P if r == reg])
    add('ALC_GESS', 'ALCH', gessmann())
    add('ALC_UNIC', 'ALCH', unicode_alch())
    e, l = meister()
    add('NOM_EARLY', 'NOMEN', e); add('NOM_LATE', 'NOMEN', l)
    add('SIG_CLUNY', 'SIGNS', signs(os.path.join(SRC, 'patrologiaecursu0149mign.txt'), r'De ipsis autem signis'))
    add('SIG_HIRS', 'SIGNS', signs(os.path.join(SRC, 'patrologiaecursu0150mign.txt'), r'Pro signo panis, fac unum'))
    add('MOD_UNIC', 'MODERN', unicode_names())
    add('MOD_TZ', 'MODERN', zoneinfo())
    add('MOD_DPKG', 'MODERN', dpkg())
    for k in ['caesar', 'manzoni', 'kafka', 'cervantes', 'v21_LA', 'v21_IT', 'hildegard', 'macer', 'circa_fr', 'konrad_plants']:
        add('LANG_' + k, 'LANG', V[k])
    json.dump(S, open(os.path.join(CK, 'systems.json'), 'w'))


if __name__ == '__main__':
    build()
