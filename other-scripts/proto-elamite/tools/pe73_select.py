#!/usr/bin/env python3
"""pe73 cycle 1: list the CDLI readings each B-grade result depends on (claims), and a blind validation set.
Writes data/pe73_ckpt/claims.json (with the transliterated values = KEY), data/pe73_ckpt/sheet_key.txt
(tablet + line positions only, no values: the sheet read from photos), data/pe73_ckpt/val_q.json (questions only)
and data/pe73_ckpt/val_key.json (answers; not opened until scoring)."""
import json, os, random, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, '..', 'data'); CK = os.path.join(DATA, 'pe73_ckpt'); os.makedirs(CK, exist_ok=True)
C = {t['id']: t for t in json.load(open(os.path.join(DATA, 'pe_corpus.json')))}
import pe70_common as pc
SEAL = {r['id']: r for r in pc.get('pe')}

def num(l):
    return ' '.join('%d(%s)' % (n, s) for n, s in l['numerals'])

def signed(l):
    return bool(l['signs']) or bool(l['numerals'])

claims = []
def add(pid, res, kind, line=None, what=None):
    t = C[pid]; d = dict(pid=pid, des=t['designation'], result=res, kind=kind)
    if line is not None:
        l = t['lines'][line]
        d.update(idx=line, surface=l['surface'], col=l['column'], label=l['label'], raw=l['raw'], value=num(l), signs=l['signs'])
    if what: d['what'] = what
    claims.append(d)

# pe42: every numeral-bearing line of the five capacity tablets
for p in ['P008229', 'P008241', 'P008784', 'P009107', 'P009296']:
    for i, l in enumerate(C[p]['lines']):
        if l['numerals']:
            add(p, 'pe42', 'NUM', i)
# pe69 per-line rule hits: the M288 line and the count line above it
rows = [r.rstrip('\n').split('\t') for r in open(os.path.join(DATA, 'pe69_decoded_slots.tsv')) if not r.startswith('#')][1:]
for r in rows:
    if r[-1].startswith('LAST'):
        p, ln = r[0], int(r[1])
        lines = C[p]['lines']
        # line index: pe69 'line' = index into lines (check that it carries M288)
        i = ln if ln < len(lines) and 'M288' in ' '.join(lines[ln]['signs']) else None
        if i is None:
            for j, l in enumerate(lines):
                if l['label'] == str(ln) and 'M288' in l['signs']:
                    i = j; break
        if i is None:
            print('no line', p, ln); continue
        add(p, 'pe69', 'NUM', i, 'M288 line')
        k = i - 1
        while k >= 0 and not lines[k]['numerals']:
            k -= 1
        if k >= 0:
            add(p, 'pe69', 'NUM', k, 'count line above M288')
# pe63 dossiers D1 D2 D3 and TSF: closing line and M288 lines; D1 M009 M003 closing
D = {'D1': ['P0087%02d' % i for i in range(17, 32)], 'D2': ['P0087%02d' % i for i in range(96, 100)] + ['P0088%02d' % i for i in range(0, 3)],
     'D3': ['P009190', 'P009211', 'P009220', 'P009237', 'P009238', 'P009286', 'P009309'], 'D7': ['P393079', 'P393080', 'P393082']}
for dz, ps in D.items():
    for p in ps:
        if p not in C: print('missing', p); continue
        L = C[p]['lines']
        last = max(i for i, l in enumerate(L) if signed(l))
        add(p, 'pe63/' + dz, 'LAST', last, 'closing line')
        for i, l in enumerate(L):
            if 'M288' in l['signs'] and l['numerals']:
                add(p, 'pe63/' + dz, 'NUM', i, 'M288 line')
                k = i - 1
                while k >= 0 and not L[k]['numerals']:
                    k -= 1
                if k >= 0: add(p, 'pe63/' + dz, 'NUM', k, 'count line above M288')
for p in [q for q in C if C[q]['designation'].startswith('TSF')][:0]:
    pass
# pe70: PES0329 / PES0334 tablets: header and seal impression
for p, r in SEAL.items():
    if set(r['seals']) & {'PES0329', 'PES0334'}:
        add(p, 'pe70', 'SEAL', what=','.join(r['seals']))
        L = C[p]['lines']
        if L: add(p, 'pe70', 'HDR', 0, 'header line (M157?)')
# pe71: |M153+X| and |M153+M342| tokens: line and whether it is the last signed line
for p, t in C.items():
    L = t['lines']
    sl = [i for i, l in enumerate(L) if signed(l)]
    for i, l in enumerate(L):
        for s in l['signs']:
            if s.startswith('|M153+X') or s.startswith('|M153+M342'):
                add(p, 'pe71', 'M153', i, '%s; last signed line=%s; numerals=%s' % (s, i == sl[-1], bool(l['numerals'])))
# de-duplicate
seen = set(); out = []
for c in claims:
    k = (c['pid'], c['kind'], c.get('idx'))
    if k in seen:
        for o in out:
            if (o['pid'], o['kind'], o.get('idx')) == k and c['result'] not in o['result']:
                o['result'] += ',' + c['result']
        continue
    seen.add(k); out.append(c)
json.dump(out, open(os.path.join(CK, 'claims.json'), 'w'), indent=0)
tabs = sorted({c['pid'] for c in out})
with open(os.path.join(CK, 'sheet_key.txt'), 'w') as f:
    for p in tabs:
        f.write('%s %s\n' % (p, C[p]['designation']))
        for c in out:
            if c['pid'] == p:
                f.write('   %-5s %-12s %s %s %s\n' % (c['kind'], c['result'], c.get('surface', ''), c.get('col', ''), c.get('label', '')))
print(len(out), 'claims on', len(tabs), 'tablets')
from collections import Counter
print(Counter(c['kind'] for c in out), Counter(c['result'].split(',')[0].split('/')[0] for c in out))
# blind validation set: 24 random tablets with numerals, outside the key set, MDP volumes
rng = random.Random('pe73val')
pool = [p for p, t in C.items() if p not in tabs and t['designation'].startswith('MDP') and sum(bool(l['numerals']) for l in t['lines']) >= 3 and len(t['lines']) <= 12]
val = rng.sample(sorted(pool), 24)
Q, K = [], []
targets = ['M288', 'M157', 'M009', 'M003', 'M388', 'M218']
for p in val:
    L = C[p]['lines']
    nl = [i for i, l in enumerate(L) if l['numerals']]
    for i in nl:
        Q.append(dict(pid=p, q='NUM', surface=L[i]['surface'], col=L[i]['column'], label=L[i]['label']))
        K.append(num(L[i]))
    i = rng.choice([j for j, l in enumerate(L) if l['signs']] or [0])
    s = rng.choice(targets)
    Q.append(dict(pid=p, q='SIGN', surface=L[i]['surface'], col=L[i]['column'], label=L[i]['label'], sign=s)); K.append(s in L[i]['signs'])
    Q.append(dict(pid=p, q='HDR')); K.append(bool(L) and L[0]['signs'] == ['M157'] and not L[0]['numerals'])
json.dump(Q, open(os.path.join(CK, 'val_q.json'), 'w'), indent=0)
json.dump(K, open(os.path.join(CK, 'val_key.json'), 'w'))
print('validation', len(val), 'tablets', len(Q), 'questions')
