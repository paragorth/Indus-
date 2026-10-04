#!/usr/bin/env python3
"""la28: 'the receipts fed the ledgers'. Shared loaders.

Documents in one format: dict(id, site, cls, findspot, items)
  cls   'R' receipt (LA roundel / nodule / sealing; LB W- series sealing or nodule)
        'T' tablet / ledger (LA Tablet + Lames; LB all non-W documents)
  items list of entries; each entry = dict(terms=[...], nums=[ints]) built from one line
        (LA: tokens between newlines; LB: one numbered line).
A term is a word ('W:KU-RO'), a single sign standing alone ('W:KA') or a logogram
('L:VIR+KA'); base(term) strips ligature parts ('L:VIR').
Only sign data are used, no readings.
"""
import json, os, re, unicodedata, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la28_ckpt')
os.makedirs(CK, exist_ok=True)

LA_R = {'Roundel', 'Nodule', 'Sealing'}
LA_T = {'Tablet', 'Lames (short thin tablet)'}


def base(t):
    k, v = t.split(':', 1)
    if k == 'L':
        v = v.replace("'", '')
        v = re.split(r'\+', v)[0]
    return k + ':' + v


def load_la():
    d = json.load(open(os.path.join(DATA, 'corpus.json')))
    meta = json.load(open(os.path.join(CK, 'meta.json')))
    out = []
    for r in d:
        cls = 'R' if r['support'] in LA_R else ('T' if r['support'] in LA_T else 'O')
        items, cur = [], dict(terms=[], nums=[], q=[])
        for t in r['tokens']:
            if t['t'] == 'nl':
                if cur['terms'] or cur['nums']:
                    items.append(cur)
                cur = dict(terms=[], nums=[], q=[])
            elif t['t'] == 'word':
                cur['terms'].append('W:' + '-'.join(t['s']))
            elif t['t'] == 'logo':
                cur['terms'].append('L:' + t['v'].replace("'", ''))
            elif t['t'] == 'num':
                cur['nums'].append(t['v'])
                cur['q'].append((t['v'], tuple(t.get('frac') or ())))
            elif t['t'] == 'frac':
                cur['nums'].append(0)
                cur['q'].append((0, tuple(t['v'])))
        if cur['terms'] or cur['nums']:
            items.append(cur)
        m = meta.get(r['id'], {})
        out.append(dict(id=r['id'], site=r['site'], cls=cls, support=r['support'],
                        findspot=m.get('findspot', ''), items=items))
    return out


GREEK = set('αβγδεζηθικλμνξοπρστυφχψω')
SKIP = {'supra', 'sigillum', 'vac', 'vacat', 'deest', 'vest', 'inf', 'sup', 'mut', 'lat', 'inf.', 'sup.',
        'mut.', 'vac.', 'vest.', 'v', 'v.', 'r', 'r.', 'qs', 'nihil', 'graffito', 'fr', 'frr', 'deest['}


def _clean(s):
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return s


def load_lb(sites=('KN', 'PY', 'TH', 'MY', 'MI', 'TI')):
    out = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        r = json.loads(l)
        h = r.get('heading')
        if not h:
            continue
        m = re.match(r'([A-Z]+)\s+([A-Z][a-z]*)?', h)
        if not m or m.group(1) not in sites:
            continue
        site, ser = m.group(1), m.group(2) or ''
        cls = 'R' if ser.startswith('W') else 'T'
        if ser in ('Wm', 'Wa', 'Wb', 'Wo', 'Wh', 'Wp'):   # labels / tags / uncertain: not receipts
            cls = 'O' if ser != 'Wm' else 'R'
        items = []
        txt = _clean(r.get('content') or '')
        txt = re.sub(r'supra\s+sigillum(=[A-Z0-9=]+)?', ' ', txt)
        txt = re.sub(r'CMS\s+\S+\s+\S+', ' ', txt)
        for line in re.split(r'[\r\n]+', txt):
            terms, nums = [], []
            for tok in re.split(r'[\s,/]+', line):
                if re.fullmatch(r'\.[0-9a-z]+', tok):
                    continue      # line label
                tok = tok.strip('[]⟦⟧•|?.!<>{}()')
                tok = tok.replace('[', '').replace(']', '').replace('•', '')
                if not tok or tok in GREEK or tok.lower() in SKIP or re.fullmatch(r'\.?[0-9a-z]?', tok) and not tok.isdigit():
                    continue
                if tok.isdigit():
                    nums.append(int(tok))
                elif re.fullmatch(r"[a-z0-9*]+(-[a-z0-9*]+)+|[a-z][a-z0-9]?", tok):
                    if tok.startswith('-') or tok.endswith('-'):
                        continue
                    terms.append('W:' + tok)
                elif re.fullmatch(r"[A-Z*][A-Z0-9*+;:±a-z]*", tok) and tok not in ('A', 'B') or tok in ('A', 'B'):
                    if re.fullmatch(r'[A-Z]', tok) and tok not in ('P', 'A'):
                        continue      # measure signs T V Z S M N Q
                    terms.append('L:' + tok.split(':')[0].split(';')[0])
            if terms or nums:
                items.append(dict(terms=terms, nums=nums))
        out.append(dict(id=h, site=site, cls=cls, support=ser, findspot='', items=items))
    return out


def doc_terms(doc, use_base=False):
    s = set()
    for it in doc['items']:
        for t in it['terms']:
            s.add(base(t) if use_base else t)
    return s
