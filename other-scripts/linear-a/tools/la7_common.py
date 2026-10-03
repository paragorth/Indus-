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
