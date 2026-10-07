"""v94: build the 300+ candidate pool (data/v94_ckpt/pool/*.pkl) from downloads kept in the session scratchpad.
usage: python3 v94_texts.py DL_DIR    (DL_DIR = scratchpad/v94dl with gut/, ll/, ia/, sef/, iti/txt/, and ../v31/ea/)
Each text: paragraphs (token lists), entry starts (heading heuristics / chapters), sentence boundaries.
OCR texts are kept only if >= 60% of their tokens are attested in the clean (Gutenberg/Latin Library) vocabulary
of all texts (garbage OCR of incunabula is dropped)."""
import os, sys, re, json, html, glob, unicodedata
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v94_lib as V
import v89_lib as L

WRE = re.compile(r"[^\W\d_]+", re.UNICODE)
SRE = re.compile(r"[.;:!?׃۔؟]+")
HEAD = re.compile(r"^\s*(chapter|chap\.|cap\.|capitulum|caput|capitolo|kapitel|cap[iy]tulo|chapitre|liber|book|de |of |the |rubrica|incipit|art\.|articulus|§|[ivxlc]+\.\s|\d+\.\s)", re.I)


def strip_marks(s):
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')


def tok_para(p):
    """-> tokens, sentence boundary offsets (relative)."""
    p = strip_marks(p)
    toks, sb = [], []
    for k, sent in enumerate(SRE.split(p)):
        w = [x.lower() for x in WRE.findall(sent)]
        if w and toks: sb.append(len(toks))
        toks += w
    return toks, sb


def build(paras_txt, heads=None):
    """paras_txt: list of strings; heads: set of para indices that start an entry (else heuristic)."""
    paras, sents, ent = [], [], []
    pos = 0
    for i, p in enumerate(paras_txt):
        t, sb = tok_para(p)
        if not t: continue
        is_head = (i in heads) if heads is not None else (len(t) <= 10 and (HEAD.match(p.strip()) is not None or p.strip().isupper()))
        if is_head: ent.append(len(paras))
        paras.append(t); sents += [pos + s for s in sb]; pos += len(t)
    # merge heading paragraphs into the following paragraph's entry (heading starts the entry)
    return paras, ent, sents


def gut_body(txt):
    a = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", txt)
    b = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", txt)
    return txt[a.end() if a else 0: b.start() if b else len(txt)]


def blank_paras(txt):
    ps = re.split(r"\n\s*\n", txt.replace('\r', ''))
    return [re.sub(r"\s+", " ", p).strip() for p in ps if p.strip()]


def html_paras(h):
    h = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", h)
    h = re.sub(r"(?is)<table\b.*?</table>", " ", h) if len(re.findall(r"(?is)<table", h)) < 3 else h
    h = re.sub(r"(?i)<br\s*/?>\s*<br\s*/?>", "\n\n", h)
    h = re.sub(r"(?i)</?(p|div|h\d|li|blockquote)\b[^>]*>", "\n\n", h)
    h = re.sub(r"<[^>]+>", " ", h)
    h = html.unescape(h)
    return blank_paras(h)


def main(dl):
    out = {}
    clean_vocab = Counter()

    def add(key, meta, paras_txt, heads=None, ocr=False):
        paras, ent, sents = build(paras_txt, heads)
        n = sum(len(p) for p in paras)
        if n < 5000: return
        out[key] = (meta, paras, ent, sents, ocr)
        if not ocr:
            for p in paras: clean_vocab.update(set(p))

    # 1. v89 texts (entries as given)
    T = L.all_texts()
    for k, d in T.items():
        if len(d['units']) < 10: continue
        paras = [u['tok'] for u in d['units']]
        out['v89_' + k] = ({'lang': d['lang'], 'genre': d['genre'], 'src': 'v89 ' + d.get('src', k)}, paras, list(range(len(paras))), [], False)
    # 2. Gutenberg
    meta = {x['id']: x for x in json.load(open(os.path.join(dl, 'gut_keep.json')))}
    seen_titles = set()
    for f in sorted(glob.glob(os.path.join(dl, 'gut', 'pg*.txt'))):
        i = re.sub(r'\D', '', os.path.basename(f))
        if i not in meta or meta[i]['why'] == 'pre2': continue   # topical / language-selected only
        tt = meta[i]['title'].lower()[:25]
        if tt in seen_titles: continue
        seen_titles.add(tt)
        try: txt = open(f, encoding='utf-8', errors='replace').read()
        except Exception: continue
        m = meta.get(i, {})
        add('gut%s' % i, {'lang': m.get('lang', '?'), 'genre': 'gutenberg', 'src': 'Gutenberg %s %s' % (i, m.get('title', ''))}, blank_paras(gut_body(txt)))
    # 3. Latin Library
    for f in sorted(glob.glob(os.path.join(dl, 'll', '*.json'))):
        if f.endswith('manifest.json'): continue
        parts = json.load(open(f))
        ps, heads = [], set()
        for name, h in parts:
            heads.add(len(ps)); ps += [p for p in html_paras(h) if not re.search(r'The Latin Library|The Classics Page|Christian Latin|Medieval Latin', p)]
        add('ll_' + os.path.basename(f)[:-5], {'lang': 'la', 'genre': 'latinlibrary', 'src': 'thelatinlibrary.com ' + os.path.basename(f)}, ps, None)
    # 4. Esoteric archives (local from v31)
    for f in sorted(glob.glob(os.path.join(dl, '..', 'v31', 'ea', '*.htm'))):
        h = open(f, encoding='latin-1', errors='replace').read()
        add('ea_' + os.path.basename(f)[:-4], {'lang': 'la/en', 'genre': 'magic', 'src': 'esotericarchives.com ' + os.path.basename(f)}, html_paras(h))
    # 5. Sefaria (Mishneh Torah merged by Sefer; others whole)
    groups = {}
    for f in sorted(glob.glob(os.path.join(dl, 'sef', '*.json'))):
        if f.endswith('index.json'): continue
        d = json.load(open(f))
        cat = d.get('cat', '')
        g = cat if 'Mishneh Torah/Sefer' in cat else d['title']
        groups.setdefault(g, []).append(d)
    for g, ds in groups.items():
        ps, heads = [], set()
        for d in ds:
            def flat(x):
                if isinstance(x, str): return [x]
                return [y for z in x for y in flat(z)]
            tx = d['text']
            chs = tx if isinstance(tx, list) else [tx]
            for ch in chs:
                segs = [re.sub(r'<[^>]+>', ' ', s) for s in flat(ch)]
                if not segs: continue
                heads.add(len(ps)); ps += segs
        add('sef_' + re.sub(r'\W+', '_', g)[-60:], {'lang': 'he', 'genre': 'sefaria', 'src': 'Sefaria ' + g}, ps, heads)
    # 6. OpenITI Arabic
    for f in sorted(glob.glob(os.path.join(dl, 'iti', 'txt', '*'))):
        ps, heads, cur = [], set(), []
        for line in open(f, encoding='utf-8', errors='replace'):
            if line.startswith('#META#') or line.startswith('######'): continue
            line = re.sub(r'PageV\d+P\d+|ms\d+|@\w+', ' ', line)
            if line.startswith('### '):
                if cur: ps.append(' '.join(cur)); cur = []
                heads.add(len(ps)); ps.append(line[4:].strip('| \n')); continue
            if line.startswith('# '):
                if cur: ps.append(' '.join(cur))
                cur = [line[2:].strip()]
            else:
                cur.append(line.lstrip('~').strip())
        if cur: ps.append(' '.join(cur))
        nm = os.path.basename(f).split('.')
        add('iti_' + nm[0] + '_' + nm[1], {'lang': 'ar', 'genre': 'openiti', 'src': 'OpenITI ' + os.path.basename(f)}, ps, heads if len(heads) > 20 else None)
    # 7. archive.org OCR
    man = json.load(open(os.path.join(dl, 'ia', 'manifest.json')))
    for i, m in man.items():
        f = os.path.join(dl, 'ia', i + '.txt')
        if not os.path.exists(f): continue
        txt = open(f, encoding='utf-8', errors='replace').read()
        add('ia_' + re.sub(r'\W', '_', i)[:50], {'lang': m.get('lang'), 'genre': 'ia:' + m.get('q', '')[:40], 'src': 'archive.org %s %s %s' % (i, m.get('date'), m.get('title', '')[:80])}, blank_paras(txt), ocr=True)
    # OCR quality filter + write
    good = set(w for w, c in clean_vocab.items() if c >= 2)
    n_ok = 0; rows = []
    import shutil; shutil.rmtree(V.POOL, ignore_errors=True)
    for k, (meta, paras, ent, sents, ocr) in sorted(out.items()):
        toks = [t for p in paras for t in p]
        q = sum(t in good for t in toks) / max(1, len(toks))
        if ocr and q < 0.6:
            rows.append('%s\tDROP-ocr q=%.2f' % (k, q)); continue
        meta = dict(meta, ocr_q=round(q, 3))
        r = V.save_text(k, meta, paras, ent if len(ent) >= 10 else None, sents)
        n_ok += 1
        rows.append('%s\t%s\t%s\ttok=%d ent=%d par=%d sen=%d\t%s' % (k, meta.get('lang'), meta.get('genre'), r[0], r[1], r[2], r[3], meta.get('src', '')[:90]))
    open(os.path.join(V.CK, 'pool_manifest.tsv'), 'w').write('\n'.join(rows) + '\n')
    print('pool', n_ok, 'texts;', sum(1 for r in rows if 'DROP' in r), 'dropped')


if __name__ == '__main__':
    main(sys.argv[1])
