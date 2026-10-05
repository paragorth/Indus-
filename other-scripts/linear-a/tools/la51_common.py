#!/usr/bin/env python3
"""la51 'The room knows what the tablet says': find-context tables and the word-context link engine.

Find-context table
  Linear A: documents -> deposit (site, building, room) -> object classes recorded in that deposit.
    Room labels for Hagia Triada come from the lineara.xyz 'findspot' field (data). Outside HT the
    deposit is site x support (x period), with the building given by excavation summaries.
    Object classes are coded per deposit in DEP_LA below with a source / confidence note.
  Linear B (control): DAMOS find_area / find_area_name per document (data/la51_ckpt/damos_find.jsonl),
    classes coded from the area names and room numbers in LB_RULES below. Area names that were given
    *because of tablets* ('Room of Chariot Tablets', 'Corridor of Sword Tablets', 'House Tablets',
    'Archive') get only ADMIN, never a commodity class, to avoid text -> context circularity.

Object classes (binary per deposit):
  STOR  storage jars / pithoi / magazine        TEXT  loom weights / spindle whorls / textile tools
  METAL bronzes, ingots, tools, weapons, metal-working   RIT  cult equipment / sanctuary
  ANIM  animal bones, horns, sacrifice debris    ADMIN mass of sealed documents / archive room
  FINE  stores of drinking/serving vessels       DOM   house / domestic        PAL  palace / villa
"""
import json, os, re, collections, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la51_ckpt')
CLASSES = ['STOR', 'TEXT', 'METAL', 'RIT', 'ANIM', 'ADMIN', 'FINE', 'DOM', 'PAL']

# ---------------------------------------------------------------- Linear A deposits
# (deposit id, site, rule, classes, confidence, note)
#   rule: function(doc, findspot) -> bool, applied in order; first match wins.
SANCT = {'Iouktas', 'Syme', 'Kophinas', 'Psykhro', 'Vrysinas', 'Traostalos', 'Skoteino Cave', 'Troullos',
         'Prassa', 'Kannia', 'Apodoulou'}
DEP_LA = [
    ('HT_VILLA_MAG', 'Haghia Triada', lambda d, f: f.startswith('Villa Magazine'), {'STOR', 'PAL'}, 'B',
     'Villa magazines; tablets partly inside pithoi (site summaries)'),
    ('HT_VILLA_R13', 'Haghia Triada', lambda d, f: f == 'Portico 11 and Room 13', {'ADMIN', 'PAL'}, 'B',
     '858 nodules + HT 1-5: sealed-document dump'),
    ('HT_VILLA_C9', 'Haghia Triada', lambda d, f: f.startswith('Corridor 9'), {'PAL'}, 'C', 'Villa corridor'),
    ('HT_CDL', 'Haghia Triada', lambda d, f: f == 'Casa del Lebete', {'METAL', 'DOM'}, 'B',
     'house named after a bronze cauldron found there'),
    ('HT_CASA7', 'Haghia Triada', lambda d, f: f == 'Casa Room 7', {'DOM'}, 'C', 'village house room 7'),
    ('HT_CASA9', 'Haghia Triada', lambda d, f: f == 'Casa Room 9', {'DOM'}, 'C', 'village house room 9'),
    ('HT_ROUNDELS', 'Haghia Triada', lambda d, f: d['support'] == 'Roundel', {'ADMIN', 'PAL'}, 'C', 'room unknown'),
    ('HT_METALOBJ', 'Haghia Triada', lambda d, f: d['support'] == 'Metal object', {'METAL'}, 'B',
     'inscribed metal objects (support = data)'),
    ('HT_VESSELS', 'Haghia Triada', lambda d, f: 'vessel' in d['support'].lower(), {'STOR'}, 'C', 'inscribed vessels'),
    ('KH_KASTELLI', 'Khania', lambda d, f: d['context'] in ('LMIB', '') and d['support'] in
     ('Tablet', 'Roundel', 'Nodule'), {'DOM', 'ADMIN'}, 'C', 'Kastelli / Katre St LM IB houses'),
    ('ZA_PALACE', 'Zakros', lambda d, f: d['support'] == 'Tablet', {'PAL', 'RIT'}, 'C',
     'palace west wing archive next to shrine treasury (rhyta)'),
    ('ZA_HOUSEA', 'Zakros', lambda d, f: d['support'] in ('Nodule', 'Sealing', 'Roundel'), {'DOM', 'ADMIN'}, 'C',
     'House A sealing deposit'),
    ('ZA_VESSELS', 'Zakros', lambda d, f: 'vessel' in d['support'].lower(), {'STOR'}, 'C', 'inscribed jars'),
    ('PH_MMII', 'Phaistos', lambda d, f: d['context'] == 'MMII', {'PAL', 'ADMIN'}, 'B', 'palace sealing deposit'),
    ('PH_OTHER', 'Phaistos', lambda d, f: True, {'PAL'}, 'C', 'palace / Chalara'),
    ('SANCT', None, lambda d, f: d['site'] in SANCT, {'RIT'}, 'B', 'peak / cave sanctuaries (libation vessels)'),
    ('PK_PETSOFAS', 'Palaikastro', lambda d, f: d['support'] == 'Stone vessel', {'RIT'}, 'C', 'stone libation vessels'),
    ('PK_TOWN', 'Palaikastro', lambda d, f: True, {'DOM'}, 'C', 'town'),
    ('ARKH_TOURK', 'Arkhalkhori', lambda d, f: d['support'] == 'Tablet', {'PAL'}, 'C', 'Arkhanes Tourkogeitonia'),
    ('ARKH_METAL', 'Arkhalkhori', lambda d, f: d['support'] == 'Metal object', {'METAL', 'RIT'}, 'C',
     'cave votive axes'),
    ('MA_PAL', 'Malia', lambda d, f: True, {'PAL'}, 'C', 'palace / quarter'),
    ('TY_HOUSES', 'Tylissos', lambda d, f: True, {'DOM', 'STOR'}, 'C', 'villas with pithoi'),
    ('PE_PAL', 'Petras', lambda d, f: True, {'PAL', 'STOR'}, 'C', 'palace magazines'),
    ('THE_AKRO', 'Thera', lambda d, f: True, {'DOM', 'STOR'}, 'C', 'Akrotiri houses'),
    ('KE_AI', 'Kea', lambda d, f: True, {'DOM'}, 'C', 'Ayia Irini'),
    ('KN_PAL', 'Knossos', lambda d, f: True, {'PAL'}, 'C', 'palace and town'),
    ('GO_PAL', 'Gournia', lambda d, f: True, {'PAL', 'DOM'}, 'C', 'small palace/town'),
    ('PYR', 'Pyrgos', lambda d, f: True, {'PAL'}, 'C', 'country house'),
]


def la_findspots():
    s = open(os.path.join(DATA, 'LinearAInscriptions.js')).read()
    out = {}
    for b in re.split(r'\n\["', s)[1:]:
        name = b.split('"')[0]
        f = re.search(r'"findspot": "([^"]*)"', b)
        if f and (f.group(1) or name not in out): out[name] = f.group(1)
    return out


def load_la():
    """Return list of docs: id, site, support, deposit, classes(frozenset), conf, terms(set)."""
    corpus = json.load(open(os.path.join(DATA, 'corpus.json')))
    fs = la_findspots()
    docs = []
    for d in corpus:
        f = fs.get(d['id'], '')
        dep = None
        for (did, site, rule, cls, conf, note) in DEP_LA:
            if site is not None and d['site'] != site: continue
            if rule(d, f): dep = (did, cls, conf); break
        terms = set()
        for w in d['words']:
            terms.add(('w:' if '-' in w else 's:') + w)
        for g in d['logograms']:
            terms.add('L:' + g.split('+')[0])
        docs.append(dict(id=d['id'], site=d['site'], support=d['support'], findspot=f,
                         deposit=dep[0] if dep else None, classes=frozenset(dep[1]) if dep else frozenset(),
                         conf=dep[2] if dep else None, terms=terms))
    # merge sides of one object (HT 6a + HT 6b -> HT 6) so a two-sided tablet counts once
    merged = collections.OrderedDict()
    for d in docs:
        k = re.sub(r'(?<=[0-9>])[a-f]$', '', d['id'])
        if k in merged and merged[k]['deposit'] == d['deposit']:
            merged[k]['terms'] |= d['terms']
        else:
            merged[k if k not in merged else d['id']] = dict(d, id=k)
    return list(merged.values())


# ---------------------------------------------------------------- Linear B deposits (control)
# keyword rules on find_area_name (lower case) -> classes ; PY numeric rooms coded separately.
LB_NAME_RULES = [
    (r'tablets|archive|clay chest|clay signet|great seal', {'ADMIN'}),
    (r'west magazine|magazine', {'STOR', 'PAL'}),
    (r'b[uü]gelkann|stirrup', {'STOR', 'PAL'}),
    (r'arsenal|armou?r', {'METAL'}),
    (r'workshop', {'METAL'}),
    (r'throne|shrine|cult|lustral|sanctuary|room of niche|pillar room|stone lamp|stone basin', {'RIT', 'PAL'}),
    (r'megaron|bath|domestic|hall of colonnades|jewel fresco|corridor|passage|entrance', {'PAL'}),
    (r'little palace', {'PAL', 'RIT'}),
    (r'pelopidou', {'DOM', 'ANIM'}),
    (r'chasm', {'ADMIN'}),
    (r'petsas', {'DOM', 'FINE'}),
    (r'house|oikonomou|pelopidou|oedipodos|epaminondou|kordatzi|soteriadou|ioakim', {'DOM'}),
    (r'oil merchant', {'STOR', 'DOM'}),
    (r'sphinx', {'DOM', 'FINE'}),
    (r'shields', {'DOM'}),
    (r'west house', {'DOM'}),
    (r'citadel house|cult cent', {'RIT'}),
    (r'ti, .*citadel', {'PAL'}),
]
# Pylos rooms (Blegen & Rawson room numbers, contents from the excavation report summaries).
PY_ROOMS = {
    '7': {'ADMIN', 'PAL'}, '8': {'ADMIN', 'PAL'},
    '23': {'STOR', 'PAL'}, '24': {'STOR', 'PAL'}, '27': {'STOR', 'PAL'}, '32': {'STOR', 'PAL'},
    '33': {'STOR', 'PAL'}, '34': {'STOR', 'PAL'}, '38': {'STOR', 'PAL'},
    '18': {'FINE', 'PAL'}, '19': {'FINE', 'PAL'}, '20': {'FINE', 'PAL'}, '21': {'FINE', 'PAL'}, '22': {'FINE', 'PAL'},
    '9': {'FINE', 'PAL'}, '60': {'FINE', 'PAL'}, '67': {'FINE', 'PAL'}, '68': {'FINE', 'PAL'},
    '92': {'METAL'}, '93': {'METAL'}, '94': {'METAL'}, '95': {'METAL'}, '96': {'METAL'}, '97': {'METAL'},
    '98': {'METAL'}, '99': {'METAL'}, '100': {'METAL'},
    '104': {'STOR'}, '105': {'STOR'},
    '6': {'RIT', 'PAL'}, '43': {'RIT', 'PAL'}, '46': {'PAL'}, '55': {'PAL'}, '64': {'PAL'},
}


def lb_classes(rec):
    site = (rec.get('heading') or '')[:2]
    name = ((rec.get('find_area_name') or '') + ' ' + (rec.get('find_area') or '')).lower()
    area = (rec.get('find_area') or '')
    cls = set()
    if site == 'PY':
        m = re.findall(r'\b(\d{1,3})\b', area + ' ' + name)
        for r in m:
            if r in PY_ROOMS: cls |= PY_ROOMS[r]
    for pat, c in LB_NAME_RULES:
        if re.search(pat, name):
            if c == {'ADMIN'}: cls = set(); cls |= c; break   # tablet-named areas: ADMIN only
            cls |= c
    return frozenset(cls)


LOGO_LB = re.compile(r'\b([A-Z][A-Z*+0-9]{1,})\b')


def lb_terms(content):
    terms = set()
    for w in re.findall(r"[a-z0-9*]+(?:-[a-z0-9*]+)+", content):
        terms.add('w:' + w)
    for g in LOGO_LB.findall(content):
        if g in ('S', 'V', 'Z', 'T', 'M', 'N', 'P', 'Q', 'L'): continue
        terms.add('L:' + g.split('+')[0])
    return terms


def load_lb():
    meta = {}
    p = os.path.join(CK, 'damos_find.jsonl')
    for l in open(p):
        r = json.loads(l)
        if 'error' in r: continue
        meta[r['id']] = r
    docs = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        r = json.loads(l)
        if 'error' in r or r['id'] not in meta: continue
        m = meta[r['id']]
        area = m.get('find_area') or ''
        if not area or area.strip() in ('-', '', 'KN, -') or area.endswith('-'): continue
        site = (r.get('heading') or '')[:2]
        docs.append(dict(id=r['heading'], site=site, support=m.get('object') or '',
                         deposit=area.strip(), classes=lb_classes(m), area_name=m.get('find_area_name'),
                         terms=lb_terms(r.get('content') or '')))
    return docs


# ---------------------------------------------------------------- link engine
def logsf_hyper(k, N, K, n):
    """log P(X >= k) for hypergeometric(N, K, n), exact via log-gamma sums."""
    from math import lgamma, exp, log
    if k <= 0: return 0.0
    hi = min(K, n)
    if k > hi: return -1e9
    def lc(a, b): return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    den = lc(N, n)
    terms = [lc(K, i) + lc(N - K, n - i) - den for i in range(k, hi + 1)]
    m = max(terms)
    return m + log(sum(exp(t - m) for t in terms))


def build_matrices(docs, min_docs=3):
    """Doc x term incidence and deposit structure. Terms need >= min_docs documents and >= 2 deposits."""
    docs = [d for d in docs if d['deposit']]
    dep_ids = sorted({d['deposit'] for d in docs})
    di = {x: i for i, x in enumerate(dep_ids)}
    dep_of = np.array([di[d['deposit']] for d in docs])
    dep_cls = {}
    for d in docs: dep_cls[d['deposit']] = d['classes']
    C = np.array([[c in dep_cls[x] for c in CLASSES] for x in dep_ids], dtype=bool)   # deposit x class
    cnt = collections.Counter(t for d in docs for t in d['terms'])
    terms = []
    for t, n in cnt.items():
        if n < min_docs: continue
        deps = {d['deposit'] for d in docs if t in d['terms']}
        if len(deps) >= 2: terms.append(t)
    terms.sort()
    ti = {t: i for i, t in enumerate(terms)}
    X = np.zeros((len(docs), len(terms)), dtype=bool)
    for j, d in enumerate(docs):
        for t in d['terms']:
            if t in ti: X[j, ti[t]] = True
    sites = np.array([d['site'] for d in docs])
    support = np.array([d['support'] for d in docs])
    return dict(docs=docs, dep_ids=dep_ids, dep_of=dep_of, C=C, terms=terms, X=X, sites=sites, support=support)


def random_predicates(rng, n, ncls=len(CLASSES)):
    """Random context predicates: conjunction of 1-2 literals (class or NOT class) OR disjunction of 2."""
    preds = set()
    while len(preds) < n:
        kind = rng.integers(3)
        a, b = rng.choice(ncls, 2, replace=False)
        na, nb = rng.integers(2), rng.integers(2)
        if kind == 0: preds.add(('one', int(a), 0, 0, 0))
        elif kind == 1: preds.add(('and', int(a), int(na), int(b), int(nb)))
        else: preds.add(('or', int(a), 0, int(b), 0))
    return sorted(preds)


def eval_pred(p, C):
    k, a, na, b, nb = p
    A = C[:, a] ^ bool(na)
    if k == 'one': return C[:, a]
    B = C[:, b] ^ bool(nb)
    return (A & B) if k == 'and' else (A | B)


def pred_name(p):
    k, a, na, b, nb = p
    A = ('NOT ' if na else '') + CLASSES[a]
    if k == 'one': return CLASSES[a]
    B = ('NOT ' if nb else '') + CLASSES[b]
    return f'{A} {"AND" if k == "and" else "OR"} {B}'
