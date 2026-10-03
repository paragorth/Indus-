#!/usr/bin/env python3
"""LA-7 cycle 3: ligature adjuncts. A syllabic sign written INSIDE a commodity logogram (OLE+KI,
GRA+PA, VIR+KA ...) is the tightest possible sign-commodity link. If pictures still carry meaning,
the adjunct's picture class should match the base commodity (vessel on liquids, plant on plant
products). Null: permute picture labels among the 100 coded signs. Linear B adjuncts (OLE+PA,
TELA+TE, SUS+SI ...) are the control: there they are known to be sound abbreviations.
Also: SI-before-CYP check across sites and in Linear B."""
import json, os, random, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la7_common import *
import la7_test as T
T.N = 5000

lab, glyph = picture_codes(False)
labs, _ = picture_codes(True)
labab = {ab_number(glyph[s]): c for s, c in lab.items() if ab_number(glyph[s]) is not None}
labsab = {ab_number(glyph[s]): c for s, c in labs.items() if ab_number(glyph[s]) is not None}
C = json.load(open(os.path.join(D, 'corpus.json')))
la = []
for r in C:
    for t in r['tokens']:
        if t['t'] == 'logo' and '+' in t['v']:
            parts = [p.strip("'[] ") for p in t['v'].split('+')]
            b = com_of(parts[0])
            if not b: continue
            for a in parts[1:]:
                if a in lab: la.append((a, COM_FAMILY[b], t['v']))
print('LA adjunct tokens', len(la), 'types', len(set(x[2] for x in la)))
print(Counter((x[2], lab[x[0]]) for x in la).most_common())
tok = [(a, f) for a, f, _ in la]
typ = list({v: (a, f) for a, f, v in la}.values())
for nm, L in (('full', lab), ('strict', labs)):
    print('LA adjuncts', nm, 'tokens', T.perm_test(tok, L))
    print('LA adjuncts', nm, 'types ', T.perm_test(typ, L))

v2ab = lb_value_to_ab()
lb = []
for line in open(os.path.join(D, 'damos_items.jsonl')):
    d = json.loads(line)
    for t in (d.get('content') or '').split():
        t = re.sub(r'[\[\]?]', '', t)
        import unicodedata as u
        t = ''.join(c for c in u.normalize('NFD', t) if u.category(c) != 'Mn')
        if '+' not in t: continue
        parts = re.split(r'\+', t)
        b = re.split(r'[:;]', parts[0])[0]
        if b not in LB_FAMILY: continue
        for a in parts[1:]:
            a = re.split(r'[:;]', a)[0].lower()
            if a in v2ab and v2ab[a] in labab: lb.append((v2ab[a], LB_FAMILY[b], t))
print('\nLB adjunct tokens', len(lb), 'types', len(set(x[2] for x in lb)))
print(Counter((x[2], labab[x[0]]) for x in lb).most_common(30))
tok = [(a, f) for a, f, _ in lb]
typ = list({v: (a, f) for a, f, v in lb}.values())
for nm, L in (('full', labab), ('strict', labsab)):
    print('LB adjuncts', nm, 'tokens', T.perm_test(tok, L))
    print('LB adjuncts', nm, 'types ', T.perm_test(typ, L))

# SI before CYP: sites, and LB
print('\nSI (single-sign word) -> next commodity, by site:')
site = {r['id']: r['site'] for r in C}
print(Counter((site[d], c) for s, k, c, d, w in la_pairs(mode='window') if s == 'SI' and k == 1))
print('all single-sign SI tokens by site:', Counter(site[r['id']] for r in C for t in r['tokens'] if t['t'] == 'word' and t['s'] == ['SI']))
print('CYP window pairs by site:', Counter(site[d] for s, k, c, d, w in la_pairs(mode='window') if c == 'CYP'))
lbp = lb_pairs()
print('LB single-sign si before:', Counter(f for a, k, f, d, w in lbp if w == 'si'))
print('LB single-sign words before spice:', Counter(w for a, k, f, d, w in lbp if k == 1 and f == 'spice'))
