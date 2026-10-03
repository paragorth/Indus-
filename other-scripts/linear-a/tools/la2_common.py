#!/usr/bin/env python3
"""Shared data for the la2_* geography tests (Linear A place names vs Linear B Cretan places).

Facts used from outside the Linear A corpus:
  * Linear B spellings (DAMOS, Knossos tablets) and Linear B sign values (data).
  * Locations of Linear B places that continue as classical Cretan toponyms
    (gazetteer below; grade = how secure the LB->classical place equation is:
    A = transparent and uncontested, B = standard but one step removed,
    C = probable/disputed location).
  * Approximate coordinates of Linear A find sites (modern archaeology).
No interpretation of any Linear A word is used.
"""
import json, os, re, math, unicodedata, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)

# ------------------------------------------------------------ gazetteer (LB places with locations)
GAZ = {  # lb spelling: (classical name, lat, lon, grade)
    'ko-no-so':    ('Knossos', 35.298, 25.163, 'A'),
    'pa-i-to':     ('Phaistos', 35.051, 24.814, 'A'),
    'ku-do-ni-ja': ('Kydonia (Khania)', 35.517, 24.018, 'A'),
    'tu-ri-so':    ('Tylissos', 35.299, 25.020, 'A'),
    'a-mi-ni-so':  ('Amnisos', 35.333, 25.207, 'A'),
    'ru-ki-to':    ('Lyktos', 35.205, 25.383, 'A'),
    'u-ta-no':     ('Itanos', 35.263, 26.263, 'B'),
    'a-pa-ta-wa':  ('Aptara', 35.464, 24.141, 'B'),
    'su-ki-ri-ta': ('Sybrita', 35.243, 24.666, 'B'),
    'e-ko-so':     ('Axos', 35.307, 24.840, 'B'),
    'ra-to':       ('Lato', 35.178, 25.653, 'B'),
    'se-to-i-ja':  ('Setaia/Sitia?', 35.205, 26.104, 'C'),
    'ku-ta-to':    ('Kytaion?', 35.410, 24.970, 'C'),
    'ra-su-to':    ('Lasithi area?', 35.180, 25.470, 'C'),
    'di-ka-ta':    ('Dikte (Psykhro?)', 35.163, 25.445, 'C'),
}

# ------------------------------------------------------------ Linear A find sites (approx.)
SITES = {
 'Haghia Triada': (35.059, 24.792), 'Khania': (35.517, 24.018), 'Phaistos': (35.051, 24.814),
 'Knossos': (35.298, 25.163), 'Zakros': (35.098, 26.261), 'Palaikastro': (35.198, 26.255),
 'Malia': (35.293, 25.492), 'Thera': (36.351, 25.404), 'Iouktas': (35.235, 25.132),
 'Arkhalkhori': (35.137, 25.268), 'Petras': (35.200, 26.115), 'Syme': (35.054, 25.433),
 'Kea': (37.660, 24.320), 'Tylissos': (35.299, 25.020), 'Gournia': (35.108, 25.794),
 'Miletos': (37.530, 27.280), 'Pyrgos': (35.007, 25.582), 'Milos': (36.755, 24.508),
 'Mokhilos': (35.184, 25.903), 'Mycenae': (37.731, 22.756), 'Psykhro': (35.163, 25.445),
 'Vrysinas': (35.338, 24.436), 'Apodoulou': (35.180, 24.700), 'Kophinas': (34.970, 25.130),
 'Kythera': (36.250, 23.000), 'Larani': (35.130, 24.940), 'Poros Herakleiou': (35.340, 25.150),
 'Samothrace': (40.470, 25.530), 'Troy': (39.960, 26.240), 'Zominthos': (35.280, 24.850),
 'Kamilari': (35.040, 24.800), 'Kannia': (35.050, 24.930), 'Nerokurou': (35.480, 24.050),
 'Platanos': (35.030, 24.980), 'Prassa': (35.330, 25.170), 'Selakanos': (35.050, 25.550),
 'Sitia': (35.205, 26.104), 'Skoteino Cave': (35.300, 25.340), 'Tel Haror': (31.380, 34.600),
 'Tiryns': (37.600, 22.800), 'Troullos': (35.240, 25.160), 'Traostalos': (35.150, 26.220),
 'Armenoi': (35.300, 24.470), 'Fourni': (35.240, 25.160), 'Haghios Stehanos': (36.800, 22.700),
 'Kalo Chorafi': (35.370, 24.730), 'Trypiti': (34.950, 24.970),
}

def km(a, b):
    la1, lo1 = map(math.radians, a); la2, lo2 = map(math.radians, b)
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))

SUB = str.maketrans('₂₃', '23')
def norm(sign): return sign.translate(SUB).lower()

DOC_CLASS = {'Tablet': 'tablet', 'Lames (short thin tablet)': 'tablet', '3-sided bar': 'tablet',
             '4-sided bar': 'tablet', 'Label': 'tablet', 'Nodule': 'sealing', 'Roundel': 'sealing',
             'Sealing': 'sealing'}
def dclass(sup): return DOC_CLASS.get(sup, 'object')

def la_records():
    return json.load(open(os.path.join(D, 'corpus.json')))

def la_tokens():
    """One dict per Linear A word token with slot features."""
    out = []
    for r in la_records():
        toks = r['tokens']; wi = [i for i, t in enumerate(toks) if t['t'] == 'word']
        nw = len(wi)
        for o, i in enumerate(wi):
            s = tuple(norm(x) for x in toks[i]['s'])
            if any(not re.fullmatch(r'[a-z]+[0-9]?', x) for x in s): clean = False
            else: clean = True
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            j = i + 1
            while j < len(toks) and toks[j]['t'] in ('div',): j += 1
            nx = toks[j] if j < len(toks) else None
            first_line = all(toks[k]['t'] != 'nl' for k in range(i))
            out.append({'w': '-'.join(s), 's': s, 'clean': clean, 'rec': r['id'], 'site': r['site'],
                        'dc': dclass(r['support']), 'sup': r['support'], 'ord': o, 'nw': nw,
                        'first': o == 0, 'head': o == 0 and nw >= 2 and dclass(r['support']) == 'tablet',
                        'num_next': bool(nx and nx['t'] == 'num'),
                        'one_next': bool(nx and nx['t'] == 'num' and nx['v'] == 1 and not nx['frac']),
                        'logo_next': bool(nx and nx['t'] == 'logo'),
                        'line1': first_line})
    return out

# ------------------------------------------------------------ Linear B place list (data-derived)
def lb_kn_vocab():
    from attack_anchors import load_lb
    docs = [d for d in load_lb() if d['head'].startswith('KN')]
    return Counter(w for d in docs for ln in d['lines'] for k, w in ln if k == 'w'), docs

def to_i(sign):
    m = re.fullmatch(r'([a-z]*?)([aeiou])([0-9]?)', sign)
    return (m.group(1) + 'i') if m else None

def lb_place_list(v):
    """Knossos words that are place-like by Linear B morphology alone:
       W-de (allative) exists, or for 3+ signs an ethnic W'-jo/-ja (last vowel -> i, or -ja -> -jo).
       Plus the gazetteer.  Returns {spelling: reason}."""
    P = {}
    for w in v:
        sg = w.split('-')
        if len(sg) < 2 or '*' in w: continue
        if w.endswith('-de') or w.endswith('-jo') or w.endswith('-ja') and False: pass
        if w + '-de' in v: P[w] = 'de'
        if len(sg) >= 3:
            ti = to_i(sg[-1])
            eth = []
            if ti and ti != sg[-1]: eth += ['-'.join(sg[:-1] + [ti, 'jo']), '-'.join(sg[:-1] + [ti, 'ja'])]
            if sg[-1] == 'ja': eth.append('-'.join(sg[:-1] + ['jo']))
            eth += [w + '-jo']
            if any(e in v for e in eth): P.setdefault(w, 'ethnic')
    # drop entries that are themselves the ethnic/derived form of another entry
    derived = set()
    for w in P:
        sg = w.split('-'); ti = to_i(sg[-1])
        if ti and ti != sg[-1]: derived |= {'-'.join(sg[:-1] + [ti, 'jo']), '-'.join(sg[:-1] + [ti, 'ja']), '-'.join(sg[:-1] + [ti])}
        if sg[-1] == 'ja': derived.add('-'.join(sg[:-1] + ['jo']))
    for d_ in derived: P.pop(d_, None)
    for g in GAZ: P.setdefault(g, 'gazetteer')
    # drop obvious non-places: 2-sign function words with -de
    for x in ('to-so', 'to-sa', 'o-da-a2'): P.pop(x, None)
    return P

def edit1(a, b):
    """True if sign sequences a,b differ by exactly one substitution, insertion or deletion."""
    if a == b: return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    if la == lb: return sum(x != y for x, y in zip(a, b)) == 1
    if la < lb: a, b = b, a
    for i in range(len(a)):
        if a[:i] + a[i + 1:] == b: return True
    return False
