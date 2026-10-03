#!/usr/bin/env python3
"""LA-7 shared helpers: sign value <-> Unicode glyph map, LA word/commodity adjacency, LB (DAMOS) adjacency.

Hypothesis tested (LA-7): the depicted object of a syllabic sign still predicts the commodity
next to the word it begins. Sound values of Linear B are NOT used for any reading; transliterations
are used only as sign identifiers.
"""
import json, os, re, sys, unicodedata
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corpus import load_raw, classify

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')

# commodity families (base logogram -> family); families are what the picture classes predict
COM_FAMILY = {'GRA': 'grain', 'OLIV': 'tree', 'FIC': 'tree', 'CYP': 'spice', 'AROM': 'spice',
              'OLE': 'liquid', 'VIN': 'liquid', 'VIR': 'person', 'FEM': 'person',
              'HIDE': 'animal', 'OVIS': 'animal', 'CAP': 'animal', 'SUS': 'animal', 'BOS': 'animal',
              'TELA': 'cloth', 'NI': 'tree'}


def sign_glyph_map():
    """transliteration value -> Unicode character, from aligned words/transliteratedWords."""
    raw = load_raw()
    votes = defaultdict(Counter)
    for v in raw.values():
        for w, t in zip(v.get('words', []), v.get('transliteratedWords', [])):
            chars = [c for c in w if ord(c) != 0x1076B and 0x10600 <= ord(c) <= 0x1077F]
            comps = t.strip().split('-')
            if len(chars) == len(comps) and len(comps) >= 1:
                for c, p in zip(chars, comps):
                    votes[p][c] += 1
    out = {}
    for p, cc in votes.items():
        ch, n = cc.most_common(1)[0]
        out[p] = (ch, unicodedata.name(ch, '?'), n)
    return out


def com_of(logo):
    b = logo.split('+')[0].lstrip('*').strip("'[]")
    return b if b in COM_FAMILY else None


def la_pairs(corpus=None):
    """(first_sign, word_len, commodity, record_id) for each word immediately followed
    (same line, nothing but a divider in between) by a commodity logogram."""
    C = corpus or json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        T = [t for t in ins['tokens'] if t['t'] != 'div']
        for i, t in enumerate(T):
            if t['t'] != 'word': continue
            if i + 1 < len(T) and T[i + 1]['t'] == 'logo':
                c = com_of(T[i + 1]['v'])
                if c: out.append((t['s'][0], len(t['s']), c, ins['id'], '-'.join(t['s'])))
    return out


# ---------------- picture codes ----------------
PRED = {'plant': {'grain', 'tree', 'spice', 'plantother'}, 'vessel': {'liquid'},
        'animal': {'animal'}, 'body': {'person'}}   # fixed before any test was run


def picture_codes(strict=False):
    """sign value -> class; strict=True turns weak ('w') calls into 'abstract'."""
    K = json.load(open(os.path.join(D, 'la7_key.json')))
    P = json.load(open(os.path.join(D, 'la7_picture_codes.json')))['codes']
    out = {}
    for code, s in K['key'].items():
        v = P[code].split()
        out[s] = 'abstract' if (strict and len(v) > 1) else v[0]
    return out, K['glyph']


def ab_number(uname):
    m = re.search(r'\bAB0*(\d+)$', uname)
    return int(m.group(1)) if m else None


# ---------------- Linear B (DAMOS) ----------------
LB_FAMILY = {'GRA': 'grain', 'HORD': 'grain', 'FAR': 'grain', 'OLIV': 'tree', 'NI': 'tree', 'FIC': 'tree',
             'ARB': 'tree', 'CYP': 'spice', 'AROM': 'spice', 'CROC': 'spice', 'KO': 'spice', 'MA': 'spice',
             'KU': 'spice', 'SE': 'spice', 'SA': 'plantother', 'OLE': 'liquid', 'VIN': 'liquid',
             'ME±RI': 'liquid', 'VIR': 'person', 'MUL': 'person', 'OVIS': 'animal', 'CAP': 'animal',
             'SUS': 'animal', 'BOS': 'animal', 'EQU': 'animal', 'CERV': 'animal', 'TELA': 'cloth',
             'LANA': 'cloth', 'RI': 'cloth'}


def _strip(t):
    t = unicodedata.normalize('NFD', t)
    return ''.join(c for c in t if unicodedata.category(c) != 'Mn').strip('[]?')


def lb_value_to_ab():
    out = {}
    for cp in range(0x10000, 0x1005E):
        n = unicodedata.name(chr(cp), '')
        m = re.match(r'LINEAR B SYLLABLE B0*(\d+) (\S+)$', n)
        if m: out[m.group(2).lower()] = int(m.group(1))
    return out


WORD_RE = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')


def lb_pairs():
    """(first_sign_ab, n_signs, family, doc_id, word) for each LB word directly followed by a
    commodity ideogram on the same line (',' and '/' skipped)."""
    v2ab = lb_value_to_ab()
    out = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        for ln in (d.get('content') or '').split('\n'):
            toks = [_strip(t) for t in ln.split()]
            toks = [t for t in toks if t and t not in (',', '/') and not t.startswith('.')]
            for i, t in enumerate(toks[:-1]):
                w = t.replace('[', '').replace(']', '')
                if not WORD_RE.match(w): continue
                b = re.split(r'[:;+]', toks[i + 1])[0]
                fam = LB_FAMILY.get(b)
                if not fam: continue
                sg = w.split('-')
                if sg[0] in v2ab:
                    out.append((v2ab[sg[0]], len(sg), fam, d['id'], w))
    return out
