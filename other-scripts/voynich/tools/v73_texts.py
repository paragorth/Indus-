"""v73: real list texts used as planted item streams (one item per line).

apicius_items(): Apicius, De re coquinaria I-X (Latin Library HTML in data/v58_ckpt/src/apicius_lat):
  per recipe, the short comma-separated segments of list sentences (ingredient lists); item = first
  content word of the segment. Returns list of recipes, each a list of item words (lower case, j->i, v->u).
antidotarium_items(): Antidotarium Nicolai (Latin ingredient runs inside the van den Berg 1917 OCR):
  runs of >= 4 consecutive Latin-genitive-looking words; entry boundaries at numbered headers.
"""
import os, re, html, glob, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'data', 'v58_ckpt', 'src')

STOP = set('et in ex cum vel ad de si ut ita sic super eo eum eam aut ac atque sed per sine quod qui quae non nec ' \
           'item aliter tum deinde postea modicum satis bene unc lib sextarium ciatum acetabulum semis ana partes '
           'pondo scripulos sextarios uncias libras quantum quod ab a e ubi cuius'.split())
VERB = re.compile(r'(bis|es|et|it|unt|ur|ere|are|ire|ent|ant|abit|ebit|ito|ato|dis|tis|gas|uas|mis|nis|cis$^)$')


def _norm(s):
    s = unicodedata.normalize('NFD', s.lower()); s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.replace('j', 'i').replace('v', 'u')


def apicius_items(min_items=3):
    recs = []
    for i in range(1, 11):
        t = open(os.path.join(SRC, 'apicius_lat', 'apicius%d.shtml' % i), encoding='latin-1').read()
        t = re.sub(r'<[^>]+>', ' ', t); t = html.unescape(t)
        parts, cur = [], []
        for ln in t.split('\n'):
            st = ln.strip()
            if re.match(r'\d+\.\s', st) and sum(c.isupper() for c in st) >= 0.6 * max(1, sum(c.isalpha() for c in st)):
                parts.append(' '.join(cur)); cur = []
            else:
                cur.append(st)
        parts.append(' '.join(cur))
        for p in parts[1:]:
            items = []
            for sent in re.split(r'[.;:]', p):
                segs = [s.strip() for s in re.split(r',| et ', sent)]
                if len(segs) < 3: continue
                for s in segs:
                    ws = [w for w in re.findall(r'[a-z]+', _norm(s))]
                    if not (1 <= len(ws) <= 3): continue
                    ws = [w for w in ws if w not in STOP and not VERB.search(w) and len(w) > 2]
                    if ws: items.append(ws[0])
            if len(items) >= min_items: recs.append(items)
    return recs


if __name__ == '__main__':
    from collections import Counter
    R = apicius_items()
    n = sum(map(len, R)); c = Counter(w for r in R for w in r)
    print(len(R), n, len(c)); print(c.most_common(40)); print(R[:3])


DUTCH = set('ende van dat men die den des der het een iegen elcs elx dats saelt geven wine goet es sijn so alse met mit ' \
            'coorne corne nemt doe ofte oft na vore int si sal deel wort wert ute hoe ghe hem hare'.split())
LATEND = re.compile(r'(i|e|is|os|um|orum|arum|ae|ii|ie|ii)$')


def antidotarium_items():
    t = open(os.path.join(SRC, 'ia', 'eenemiddelnederl00nicouoft.txt'), encoding='utf-8', errors='replace').read()
    ents, cur = [], []
    for ln in t.split('\n'):
        if re.match(r'\s*\d{1,3}\.\s+\S', ln) and len(ln.strip()) > 25:
            if cur: ents.append(cur)
            cur = []
        ws = [_norm(w) for w in re.findall(r'[A-Za-zéèë]+', ln)]
        ws = [re.sub(r'[^a-z]', '', w) for w in ws]; ws = [w for w in ws if len(w) >= 4]
        if len(ws) < 5: continue
        if any(w in DUTCH for w in ws): continue
        if sum(bool(LATEND.search(w)) for w in ws) < 0.8 * len(ws): continue
        cur += ws
    if cur: ents.append(cur)
    return [e for e in ents if len(e) >= 3]
