"""v71: reference texts of known genre as entries -> paragraphs -> words (plain letters).

Reuses the v58 parsers (data/v58_ckpt/src) with `unit` patched to keep the words,
adds narrative/argument texts (Gutenberg files in data/) and real lists/indexes
(book indexes and glossaries cut out of the same downloads).
Output: data/v71_ckpt/refs.json = {id: {genre, coarse, lang, entries: [[ [words], ...paras ], ...]}}
"""
import os, re, json, sys, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v58_texts as T

VD = os.path.dirname(HERE)
OUT = os.path.join(VD, 'data', 'v71_ckpt'); os.makedirs(OUT, exist_ok=True)
SRC = T.SRC

def words(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return [w.lower() for w in T.WRE.findall(text)]

def unit(title, paras):
    ps = [words(p) for p in paras if p.strip()]
    ps = [p for p in ps if p]
    return {'t': title[:60], 'w': sum(len(p) for p in ps), 'c': 0, 'p': [len(p) for p in ps], 'paras': ps}
T.unit = unit

def gut_body(path):
    s = open(path, encoding='utf-8', errors='replace').read()
    a = s.find('*** START'); a = s.find('\n', a) if a >= 0 else 0
    b = s.find('*** END'); b = b if b > 0 else len(s)
    return s[a:b]

def gut_paras(path, skip_frac=0.03, max_words=60000):
    """narrative: blank-line paragraphs; chapters = lines in caps or 'CAPITOLO/CHAPTER/LIBER' heads."""
    s = gut_body(path)
    blocks = [b.strip() for b in re.split(r'\n\s*\n', s) if b.strip()]
    blocks = blocks[int(len(blocks) * skip_frac):]
    ents, cur, n = [], [], 0
    for b in blocks:
        ws = words(b)
        if not ws: continue
        head = len(ws) <= 6 and (b.isupper() or re.match(r'(?i)(chapter|capitolo|cap\.|liber|kapitel|[ivxlc]+\.?$)', b))
        if head:
            if cur: ents.append(cur); cur = []
            continue
        if len(ws) < 4: continue
        cur.append(ws); n += len(ws)
        if n >= max_words: break
    if cur: ents.append(cur)
    return ents

def index_lines(path, start, end, item_re=None):
    """list: each index line / glossary item = one entry of one paragraph."""
    L = open(path, encoding='utf-8', errors='replace').read().split('\n')[start:end]
    ents = []
    if item_re is None:          # one printed line = one item; '——' continuation lines are items too
        for l in L:
            ws = words(l)
            if ws: ents.append([ws])
    else:                       # blank-line separated items
        cur = []
        for l in L + ['']:
            if not l.strip():
                if cur:
                    ws = words(' '.join(cur))
                    if ws: ents.append([ws])
                cur = []
            else: cur.append(l)
    return ents

def main():
    R = {}
    def add(k, lang, genre, coarse, ents):
        ents = [[p for p in e if p] for e in ents]
        ents = [e for e in ents if e]
        nw = sum(len(p) for e in ents for p in e)
        R[k] = {'lang': lang, 'genre': genre, 'coarse': coarse, 'entries': ents}
        print('%-18s %-8s %-10s entries %5d paras %6d words %7d' % (k, genre, coarse, len(ents), sum(len(e) for e in ents), nw), flush=True)
    def fromunits(us):
        return [u['paras'] for u in us if u.get('paras')]
    jobs = [
        # procedure (recipes, antidotaries, medical procedure)
        ('apicius_lat', 'la', 'recipe', 'procedure', T.apicius_lat),
        ('apicius_eng', 'en', 'recipe', 'procedure', T.apicius_eng),
        ('forme_of_cury', 'enm', 'recipe', 'procedure', T.forme_of_cury),
        ('antidotarium_nl', 'dum', 'antidotary', 'procedure', T.antidotarium),
        ('celsus_lat', 'la', 'medical', 'procedure', lambda: T.tei_units(os.path.join(SRC, 'celsus_lat.xml'), T.is_chap)),
        ('celsus_eng', 'en', 'medical', 'procedure', lambda: T.tei_units(os.path.join(SRC, 'celsus_eng.xml'), T.is_chap)),
        # entry (herbals, encyclopedias, bath and star catalogues)
        ('hildegard', 'la', 'herbal', 'entry', T.hildegard),
        ('culpeper', 'en', 'herbal', 'entry', T.culpeper),
        ('macer', 'la', 'herbal-verse', 'entry', T.macer),
        ('circa_fr', 'fro', 'herbal', 'entry', lambda: T.ocr_caps_entries(os.path.join(SRC, 'ia', 'BIUSante_pharma_032591.txt'))),
        ('leechdoms', 'ang', 'herbal', 'entry', lambda: T.ocr_caps_entries(os.path.join(SRC, 'ia', 'LeechdomsWortcunningStarcraftV1.txt'))),
        ('dioscorides', 'grc', 'herbal', 'entry', lambda: T.tei_units(os.path.join(SRC, 'dioscorides_grc.xml'), T.is_chap)),
        ('konrad_plants', 'de', 'herbal', 'entry', lambda: [u for u in T.konrad() if u['t'][:5] in ('kon_4', 'kon_5')]),
        ('konrad_other', 'de', 'encycl', 'entry', lambda: [u for u in T.konrad() if u['t'][:5] not in ('kon_4', 'kon_5')]),
        ('pliny_lat', 'la', 'encycl', 'entry', lambda: [u for u in T.tei_units(os.path.join(SRC, 'pliny_nh_lat.xml'), T.is_chap) if not u['t'].startswith('1.')]),
        ('theophrastus', 'grc', 'botany', 'entry', lambda: T.tei_units(os.path.join(SRC, 'theophrastus_hp_grc.xml'), T.is_chap)),
        ('balneis', 'la', 'baths', 'entry', lambda: T.ocr_caps_entries(os.path.join(SRC, 'ia', 'synopsiseorumqua00lomb.txt'))),
        ('hyginus', 'la', 'astro', 'entry', T.hyginus_astr),
        # list (indexes, tables of contents, glossaries)
        ('pliny_index', 'la', 'index', 'list', lambda: [u for u in T.tei_units(os.path.join(SRC, 'pliny_nh_lat.xml'), T.is_chap) if u['t'].startswith('1.') and u['t'] != '1.praef']),
    ]
    for k, lang, g, c, fn in jobs:
        try:
            us = fn()
            ents = fromunits(us)
            add(k, lang, g, c, ents)
        except Exception as e:
            print('FAIL', k, repr(e)[:300])
    for k, us in T.v21_herbals().items():
        add('v21_' + k, k.lower(), 'herbal', 'entry', fromunits(us))
    D = os.path.join(VD, 'data')
    for k, lang, g, fn in [('caesar', 'la', 'chronicle', 'pg218.txt'), ('descartes', 'la', 'argument', 'pg23306.txt'),
                           ('manzoni', 'it', 'novel', 'pg45334.txt'), ('kafka', 'de', 'novel', 'pg22367.txt'),
                           ('cervantes', 'es', 'novel', 'pg2000.txt')]:
        add(k, lang, g, 'narrative', gut_paras(os.path.join(D, fn)))
    # lists
    add('culpeper_index', 'en', 'index', 'list', index_lines(os.path.join(SRC, 'culpeper.txt'), 29362, 31500))
    add('cury_glossary', 'enm', 'glossary', 'list', index_lines(os.path.join(SRC, 'forme_of_cury.txt'), 4910, 7460, item_re=True))
    add('apicius_index', 'en', 'index', 'list', index_lines(os.path.join(SRC, 'apicius_eng.txt'), 15096, 19030, item_re=True))
    json.dump(R, open(os.path.join(OUT, 'refs.json'), 'w'))

if __name__ == '__main__':
    main()
