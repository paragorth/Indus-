"""pe50: build Proto-Elamite capture histories (tablet x entity incidence) under several
entity definitions, plus Ur III analogues from data/pe50_ckpt/ur3_caps.json.

PE entity definitions (all from the CDLI transliteration; no sign readings used):
  MID2    entry middle (entry signs minus a final class sign, pe4/pe7 definition), >= 2 signs, clean lines
  MID3    the same, >= 3 signs
  FULL2   whole entry sign string incl. class sign, >= 2 signs
  VAR2    MID2 with variant signs kept (~a, ~b)
  DIRTY2  MID2 from all lines (damaged lines too), strings containing x dropped
  NSIGN   MID2 strings containing a pe6 name-spelling sign (A-graded set)
  HDR     header sign string
  DENT    distinctive entry: full entry string + exact numeral (>= 2 signs, or a numeral >= 10 units)
Ur III (DREHEM, UMMA):
  OFF     person names in fixed frames (ki X-ta, giri3 X, X i3-dab5, ...)
  ENT2    non-numeric part of a numeric line, >= 2 words (the analogue of a PE entry string)
  DENT    whole numeric line (number + string)
Output: data/pe50_ckpt/caps.json {corpus: {def: [[entities of tablet i] ...]}, meta: {...}}
"""
import json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, entries, header  # noqa
from pe4_common import FINAL  # noqa

CK = os.path.join(HERE, '..', 'data', 'pe50_ckpt')
NAMESIGN = {'M099', 'M136', 'M005', 'M246', 'M251', 'M254', 'M304', 'M262', 'M340', 'M390'}
UNIT = {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 60, 'N48': 3000}


def mid(s):
    if s[-1] in FINAL and len(s) >= 2:
        return tuple(s[:-1])
    return tuple(s)


def pe():
    T = load()
    ids = [t['id'] for t in T]
    pos = {t: i for i, t in enumerate(ids)}
    D = {k: [set() for _ in T] for k in ('MID2', 'MID3', 'FULL2', 'VAR2', 'DIRTY2', 'NSIGN', 'HDR', 'DENT')}
    for e in entries(T, require_clean=True):
        i = pos[e['tablet']]
        s = e['signs']
        m = mid(s)
        if len(m) >= 2:
            D['MID2'][i].add(m)
            if NAMESIGN & set(m):
                D['NSIGN'][i].add(m)
        if len(m) >= 3:
            D['MID3'][i].add(m)
        if len(s) >= 2:
            D['FULL2'][i].add(tuple(s))
        val = sum((c or 0) * UNIT.get(n, 0) for c, n in e['numerals'])
        if len(s) >= 2 or val >= 10:
            D['DENT'][i].add((tuple(s), tuple(map(tuple, e['numerals']))))
    for e in entries(T, require_clean=True, base_signs=False):
        m = mid(e['signs'])
        if len(m) >= 2:
            D['VAR2'][pos[e['tablet']]].add(m)
    for e in entries(T, require_clean=False):
        if 'x' in e['signs']:
            continue
        m = mid(e['signs'])
        if len(m) >= 2:
            D['DIRTY2'][pos[e['tablet']]].add(m)
    for t in T:
        h = header(t)
        if h and 'x' not in h:
            D['HDR'][pos[t['id']]].add(tuple(h))
    E = entries(T)
    ne = collections.Counter(e['tablet'] for e in E)
    meta = []
    for t in T:
        off = [l for l in t['lines'] if l['surface'] != 'obverse' and l['numerals']]
        m = re.match(r'(.+?),', t['designation'])
        meta.append({'id': t['id'], 'prov': t['provenience'].split(' (')[0], 'vol': m.group(1) if m else t['designation'],
                     'nent': ne.get(t['id'], 0), 'total': int(len(off) >= 1),
                     'hdr': ' '.join(header(t) or []), 'nlines': len(t['lines'])})
    return {k: [sorted(map(json.dumps, s)) for s in v] for k, v in D.items()}, meta


def ur3():
    U = json.load(open(os.path.join(CK, 'ur3_caps.json')))
    out, meta = {}, {}
    for a, tabs in U.items():
        d = {'OFF': [], 'ENT2': [], 'DENT': []}
        for t in tabs:
            d['OFF'].append(t['names'])
            e2, de = set(), set()
            for x in t['entries']:
                w = x.split()
                rest = [y for y in w if not re.match(r'^\d+(/\d+)?\(', y)]
                if 'x' in rest or not rest:
                    continue
                if len(rest) >= 2:
                    e2.add(' '.join(rest))
                de.add(x)
            d['ENT2'].append(sorted(e2))
            d['DENT'].append(sorted(de))
        out[a] = d
        meta[a] = [{'id': t['pid'], 'nnum': t['nnum'], 'total': t['total'], 'year': t['year'], 'nlines': t['nlines']}
                   for t in tabs]
    return out, meta


if __name__ == '__main__':
    P, pm = pe()
    U, um = ur3()
    caps = {'PE': P, **U}
    meta = {'PE': pm, **um}
    json.dump({'caps': caps, 'meta': meta}, open(os.path.join(CK, 'caps.json'), 'w'))
    for c, d in caps.items():
        for k, v in d.items():
            cnt = collections.Counter(e for s in v for e in s)
            q = collections.Counter(cnt.values())
            print(c, k, 'tabs-with', sum(1 for s in v if s), 'S', len(cnt), 'Q1', q[1], 'Q2', q[2],
                  'incid', sum(cnt.values()))
