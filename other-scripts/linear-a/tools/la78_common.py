"""LA-78 'time travel as the outside test'.

Pretend it is year X; fit hypotheses on documents first published by X; score them on
documents first published after X. No reading or sound value is used to build the
random hypotheses; frozen readings from earlier loops (la66, la72) are scored the same way.

First-publication years (data, not interpretation) come from the GORILA introductions
(vol. 1 and 2: 'edites avant 1970', with per-site edition columns; vol. 3: 'edites en 1975
et 1976' with find years; vol. 4 1982; vol. 5 1985) and the lineara.xyz source URLs for
later papers. Rules are per series and are listed in PUB_RULES with their source.
"""
import json, re, os, sys, math, random, collections, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la78_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
from la15_common import _pub_source, _logo  # noqa: E402


def _rights():
    s = open(os.path.join(DATA, 'LinearAInscriptions.js')).read()
    out = {}
    for m in re.finditer(r'\["([^"]+)",\{(.*?)\n\}\]', s, re.S):
        u = re.search(r'"imageRightsURL": "([^"]*)"', m.group(2))
        out[m.group(1)] = u.group(1) if u else ''
    return out


def series(doc_id):
    m = re.match(r'([A-Z]+(?:\(\?\))?[A-Za-z]*?)(?=[\d<?(]|$)', doc_id)
    return m.group(1) if m else doc_id


def num(doc_id):
    m = re.search(r'(\d+)', doc_id)
    return int(m.group(1)) if m else -1


# GORILA 4 (1982) concordance: earliest edition (PoM / Pugliese 1945 col. / Brice 1961 / footnoted papers)
VOL4 = {'APZa1': 1935, 'APZa2': 1935, 'ARZf1': 1956, 'ARZf2': 1956, 'CR(?)Zf1': 1981, 'HTZb158': 1945, 'HTZb159': 1945,
        'HTZb160': 1945, 'HTZb161': 1945, 'HTZd155': 1945, 'HTZd156': 1945, 'HTZd157': 1945, 'KAZf1': 1956, 'KEZb3': 1970,
        'KEZb4': 1970, 'KEZb5': 1970, 'KNZa10': 1927, 'KNZa17': 1945, 'KNZa18': 1945, 'KNZa19': 1935, 'KNZb4': 1921,
        'KNZb5': 1921, 'KNZb20': 1962, 'KNZb<27>': 1921, 'KNZb34': 1921, 'KNZb35': 1921, 'KNZb40': 1976, 'KNZc6': 1921,
        'KNZc7': 1921, 'KNZe16': 1956, 'KNZf13': 1927, 'KNZf31': 1972, 'KOZa1': 1971, 'MAZb8': 1971, 'MAZe11': 1980,
        'MIZb1': 1921, 'PHZb4': 1902, 'PHZb5': 1902, 'PHZb48': 1976, 'PKZa4': 1903, 'PKZa8': 1923, 'PKZa9': 1923,
        'PKZa10': 1923, 'PKZa11': 1923, 'PKZa12': 1923, 'PKZa14': 1972, 'PKZa15': 1972, 'PKZa16': 1981, 'PKZa17': 1981,
        'PKZa18': 1981, 'PKZc13': 1958, 'PLZf1': 1976, 'PRZa1': 1958, 'PSZa2': 1945, 'SIZg1': 1971, 'SKZb1': 1961,
        'THEZb1': 1904, 'THEZb2': 1971, 'THEZb3': 1971, 'THEZb4': 1969, 'TLZa1': 1909, 'TRAZb1': 1975, 'TYZb4': 1934,
        'TYZg1': 1921, 'VRYZa1': 1977, 'ZAZb3': 1975, 'ZAZb34': 1975, 'KNZg<21>': 1962, 'KYZg1': 1972, 'KO(?)Zf2': 1982}


# (year, certainty) ; certainty 'g' = GORILA statement, 'u' = url year, 'e' = estimate
def pub_year(doc_id, vol, url):
    s, n = series(doc_id), num(doc_id)
    y = re.search(r'(19[5-9]\d|20[0-2]\d)', url or '')
    if vol == 'G1' or (vol == 'blank' and s in ('HT', 'TY', 'ZA', 'KN', 'PK', 'PA', 'MA') and n < 160):
        if s == 'MA':
            return 1930, 'g'                       # Chapouthier 1930
        if s == 'PH' or s == 'PH(?)':
            if n <= 3: return 1945, 'g'            # Pugliese 1945
            if n <= 29: return 1958, 'g'           # Levi finds 1953-56, Bennett/Pugliese 1958
            return 1971, 'g'                       # PH 30 Raison-Pope 1971 (face b 1976)
        if s == 'KE': return 1970, 'g'             # Caskey 1970
        if s == 'PYR': return 1971, 'g'            # Morpurgo-Cadogan 1971
        if s == 'ZA' and n >= 4: return 1975, 'g'
        return 1945, 'g'                           # HT, KN, PK, PA, TY, ZA 1: Pugliese 1945 (editio princeps)
    if vol == 'blank' and s in ('PH', 'PH(?)', 'PYR', 'KE'):
        return pub_year(doc_id, 'G1', url)
    if vol == 'G2':
        if s.startswith('PH'): return 1961, 'e'    # edited before 1970; Phaistos sealings 1950s finds
        return 1945, 'e'                           # HT/KN/ZA nodules, roundels: Pugliese 1945 / Evans
    if vol == 'G3' or (vol == 'blank' and s in ('ARKH', 'KH', 'KHWa', 'KHWc') and n < 2120 and (s != 'KH' or n < 88)):
        if s.startswith('KH') and (n in (1, 2, 3, 4, 2001, 2002, 2003, 2004, 2005)): return 1973, 'g'  # Hallager 1973
        return 1975, 'g'                           # ARKH 1975, KH 1975-76, ZA 1975 (Platon-Brice)
    base = re.sub(r'[a-z]$', '', doc_id)
    if vol == 'G4' or base in VOL4:
        if base in VOL4: return VOL4[base], 'g'
        return 1982, 'e'
    if vol == 'G5': return 1985, 'g'
    if vol == 'post':
        if y: return int(y.group(1)), 'u'
        return 2000, 'e'
    # blanks not in GORILA URLs
    if s.startswith('KH'): return 1995, 'e'
    if s.startswith('PE'): return 1997, 'e'        # Petras finds 1990s
    if s.startswith('IOZa') and n >= 11: return 1995, 'e'
    if s.startswith('SY') and n >= 4: return 1995, 'e'
    return 1990, 'e'


SITE_ALIAS = {}


def load():
    C = json.load(open(os.path.join(DATA, 'corpus_ra.json')))
    src = _pub_source()
    url = _rights()
    docs = []
    for d in C:
        vol = src.get(d['id'], 'blank')
        y, cert = pub_year(d['id'], vol, url.get(d['id'], ''))
        toks = [t for t in d['tokens'] if t['t'] not in ('nl', 'div')]
        words = []
        nwi = 0
        for i, t in enumerate(toks):
            if t['t'] != 'word':
                continue
            nxt = toks[i + 1]['t'] if i + 1 < len(toks) else 'end'
            y1 = {'num': 0, 'logo': 1, 'word': 2, 'end': 3}.get(nxt, 2)
            qv = toks[i + 1].get('v') if nxt == 'num' else None
            syl = [s for s in t['s'] if not _logo(s)]
            words.append(dict(s=t['s'], syl=syl, clean=(t.get('st') == 'read'), y1=y1,
                              init=int(nwi == 0), q=qv))
            nwi += 1
        logos = [t.get('v', '').split('+')[0] for t in toks if t['t'] == 'logo' and t.get('st') == 'read']
        docs.append(dict(id=d['id'], site=d['site'] or '?', support=(d['support'] or '?').lower(), vol=vol,
                         year=y, cert=cert, words=words, logos=logos))
    return docs


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
