#!/usr/bin/env python3
"""Shared data for the la1_* fraction-as-sub-unit tests.

lb_rows(): Linear B (DAMOS) quantities as (site, commodity, unit) rows.
  A unit token (T, V, Z, S, M, N, P, Q) directly followed by a number is one row,
  credited to the last commodity ideogram on the same document. 'INT' = commodity + integer.
la_rows(): Linear A quantities with fractions as (commodity, letter) rows, same carry-over rule
  as tools/fraction_commodity.py.
Ratios used (metrological data, conventional): dry 1 = 10 T, T = 6 V, V = 4 Z;
liquid 1 = 3 S, S = 6 V, V = 4 Z.
"""
import json, os, re
from collections import Counter, defaultdict
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
UNITS = {'T', 'V', 'Z', 'S', 'M', 'N', 'P', 'Q'}
LB_COM = {'GRA', 'HORD', 'FAR', 'OLE', 'VIN', 'NI', 'OLIV', 'CYP', 'AROM', 'ME±RI', 'CROC', 'LANA',
          'AES', 'AUR', 'VIR', 'MUL', 'OVIS', 'CAP', 'SUS', 'BOS', 'TELA', 'PYC', 'KAPO', 'KA±PO',
          'SA', 'KO', 'MA', 'KU', 'PO', 'SE', 'ME', 'RI', 'TU±RO2', 'A±RE±PA', 'CORN', 'ARB'}
LA_MAIN = {'GRA', 'VIN', 'OLE', 'OLIV', 'CYP', 'VIR', 'AROM', 'HIDE', 'NI'}
NUM = re.compile(r'^\[?\]?(\d+)\[?\]?$')
DRY = {'T': Fr(1, 10), 'V': Fr(1, 60), 'Z': Fr(1, 240)}
LIQ = {'S': Fr(1, 3), 'V': Fr(1, 18), 'Z': Fr(1, 72)}
DRY_COM = {'GRA', 'HORD', 'FAR', 'NI', 'OLIV', 'CYP', 'AROM', 'CROC', 'KA±PO', 'PYC', 'SA', 'KO', 'MA', 'KU', 'SE', 'PO', 'ME', 'RI'}


def lb_rows():
    out = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        site = (d.get('heading') or '??')[:2]
        toks = (d.get('content') or '').split()
        cur = None
        for i, t in enumerate(toks):
            if t.startswith('.'): continue
            b = t.strip('[]').split('+')[0]
            if b in LB_COM and t not in UNITS:
                cur = b
                if i + 1 < len(toks) and NUM.match(toks[i + 1]): out.append((site, cur, 'INT'))
            elif t in UNITS and i + 1 < len(toks) and NUM.match(toks[i + 1]) and cur:
                out.append((site, cur, t))
    return out


def la_quantities():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        cur = None
        for t in ins['tokens']:
            if t['t'] == 'logo':
                b = t['v'].split('+')[0].lstrip('*')
                if b in LA_MAIN: cur = b
            elif t['t'] == 'word' and t['s'] == ['NI']:
                cur = 'NI'
            elif t['t'] == 'num' and t['frac']:
                out.append((ins['id'], cur, list(t['frac'])))
    return out


def la_rows():
    return [(c, f) for _, c, fr in la_quantities() if c for f in fr]


def table(rows, a=0, b=1):
    t = defaultdict(Counter)
    for r in rows: t[r[a]][r[b]] += 1
    return t
