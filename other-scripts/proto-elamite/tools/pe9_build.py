"""pe9 ('the names were drawn from a bag'): build per-tablet string corpora.

PE      : per tablet (ordered by P-number), the multi-sign entry middles (pe4/pe7
          definition: clean entry signs minus a final class sign, length >= 2),
          in line order; exact duplicates within a tablet dropped (one person listed
          twice is not a second draw).  Tablets with >= 2 strings kept.
UR3     : Ur III Drehem administrative personal names per tablet (pe4 controls,
          'ki PN-ta' / 'giri3 PN'), syllable strings, same rules.
LINB    : Linear B personnel names per tablet (DAMOS via pe7 corpora), same rules.
output  : data/pe9_corpora.json  {name: [[ [sign,...], ...], ...]}  plus tablet ids.
"""
import json, os, sys
from collections import OrderedDict, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, entries  # noqa
from pe4_common import FINAL  # noqa
DATA = os.path.join(HERE, '..', 'data')


def pe():
    E = entries(load(), require_clean=True)
    by = OrderedDict()
    for e in E:
        s = e['signs']
        mid = tuple(s[:-1]) if (s[-1] in FINAL and len(s) >= 2) else tuple(s)
        if len(mid) < 2:
            continue
        L = by.setdefault(e['tablet'], [])
        if mid not in L:
            L.append(mid)
    ids = sorted(t for t in by if len(by[t]) >= 2)
    return ids, [[list(m) for m in by[t]] for t in ids]


def ur3():
    ctl = json.load(open(os.path.join(DATA, 'pe4_controls.json')))['names']
    by = OrderedDict()
    for r in ctl:
        if len(r['attr']) >= 2:
            L = by.setdefault(r['t'], [])
            if tuple(r['attr']) not in L:
                L.append(tuple(r['attr']))
    ids = sorted(t for t in by if len(by[t]) >= 2)
    return ids, [[list(m) for m in by[t]] for t in ids]


def linb():
    C = json.load(open(os.path.join(DATA, 'pe7_corpora.json')))
    by = defaultdict(list)
    for x in C['LINB']:
        if len(x['seq']) >= 2:
            for t in x['tablets']:
                if tuple(x['seq']) not in by[t]:
                    by[t].append(tuple(x['seq']))
    ids = sorted(t for t in by if len(by[t]) >= 2)
    return ids, [[list(m) for m in by[t]] for t in ids]


if __name__ == '__main__':
    out = {}
    for k, f in (('PE', pe), ('UR3', ur3), ('LINB', linb)):
        ids, tabs = f()
        out[k] = {'ids': ids, 'tablets': tabs}
        ntok = sum(len(m) for t in tabs for m in t)
        V = len({s for t in tabs for m in t for s in m})
        print(k, 'tablets', len(tabs), 'strings', sum(map(len, tabs)), 'tokens', ntok, 'types', V)
    json.dump(out, open(os.path.join(DATA, 'pe9_corpora.json'), 'w'))
