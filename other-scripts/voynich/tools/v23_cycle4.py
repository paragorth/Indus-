"""v23 cycle 4: does the backward glyph arrow (lzma, within-word KN) hold in Currier A and B separately,
and in each illustration section?  Uses the per-page contributions saved in cycle 1 (same page order
as v23_lib.voynich) plus a fresh within-word KN arrow trained inside each language group."""
import os, sys, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v23_lib as L
import vlib


def meta(name):
    lines = vlib.load_voynich(name, drop_uncertain=True)
    m, order = {}, []
    for l in lines:
        if l['folio'] not in m:
            m[l['folio']] = (l.get('lang') or 'x', l.get('illus') or 'x'); order.append(l['folio'])
    # reproduce the page filter of L.voynich
    C = L.voynich(name)
    kept = []
    pages = {}
    for l in lines:
        ws = [L.U(w) for w in l['words']]; ws = [w for w in ws if w and '?' not in w and '*' not in w]
        pages[l['folio']] = pages.get(l['folio'], 0) + len(ws)
    kept = [f for f in order if pages[f] >= 20]
    assert len(kept) == len(C), (len(kept), len(C))
    return kept, m, C


out = {}
for name, key in (('ZL3b', 'ZL'), ('IT2a', 'IT')):
    folios, m, C = meta(name)
    d = json.load(open(os.path.join(L.CK, f'c1_{key}.json')))
    groups = defaultdict(list)
    for i, f in enumerate(folios):
        lang, ill = m[f]
        groups['lang ' + lang].append(i); groups['illus ' + ill].append(i)
    res = {}
    for g, idx in sorted(groups.items()):
        if len(idx) < 8: continue
        r = {}
        for s in ('Z_lzma', 'Z_zlib', 'G4_inword', 'G6_stream'):
            a = [d['signed'][s]['a'][i] for i in idx]
            z, p = L.signflip(a)
            r[s] = round(float(z), 2)
        # within-word KN arrow trained inside the group only
        words = [[w for w in L.tokens(C[i])] for i in idx]
        kr = L.kn_arrow(words, 4)
        a = [b - f for f, b, n in kr]
        r['G4_inword_ingroup'] = round(float(L.signflip(a)[0]), 2)
        r['n'] = len(idx)
        res[g] = r
        print(key, g, r, flush=True)
    out[key] = res
json.dump(out, open(os.path.join(L.CK, 'c4_groups.json'), 'w'))
