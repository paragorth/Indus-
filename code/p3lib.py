"""Part 3 shared loader. Reuses the frozen pipeline's direction correction and class induction."""
import re, json, random, math
from collections import Counter, defaultdict
import numpy as np

DATA = '/home/claude/p3/repro/data'
sql = open(f'{DATA}/population-script.sql').read()

def sql_rows(table):
    m = re.search(r'INSERT INTO ' + table + r'[^;]*?VALUES(.*?);', sql, re.S | re.I)
    return re.findall(r'\(([^()]*)\)', m.group(1)) if m else []

def split_row(r):
    return [x.strip().strip('"') for x in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', r)]

# --- objects
OBJ = {}
for r in sql_rows('SEAL'):
    p = split_row(r)
    OBJ[int(p[0])] = dict(site_id=p[1], material=p[2], cisi=p[3], museum=p[4])
site_name = {}
for r in sql_rows('SITE'):
    p = split_row(r)
    if len(p) >= 2: site_name[p[0]] = p[1]
for o in OBJ.values():
    o['site'] = site_name.get(o['site_id'], '?')

icon = {}
m = re.search(r'INSERT INTO ICONOGRAPHY\(SEALID[^;]*?VALUES(.*?);', sql, re.S)
for r in re.findall(r'\(([^()]*)\)', m.group(1)):
    p = r.split(',', 1)
    icon[int(p[0])] = p[1].strip().strip('"').lower()

seq = defaultdict(dict)
for r in sql_rows('GLYPHSEQUENCE'):
    s, g, i = [int(x) for x in r.split(',')]
    seq[s][i] = g

FOREIGN_SITES = {'Gonur Depe', 'Altyn Depe', 'Salut', 'Failaka', 'Ur', 'Kish', 'Susa', 'Nippur', 'Bahrain'}

# --- texts in reading order (direction correction), with metadata
TEXTS = []   # dicts: id, site, material, cisi, icon, text(tuple), register
for s, d in seq.items():
    t = tuple(reversed([d[i] for i in sorted(d)]))
    if len(t) < 1: continue
    o = OBJ.get(s, {})
    TEXTS.append(dict(id=s, site=o.get('site', '?'), material=o.get('material', 'NULL'),
                      cisi=o.get('cisi', ''), icon=icon.get(s), text=t,
                      seal=(s in icon)))

def register(x):
    if x['site'] in FOREIGN_SITES: return 'FOREIGN'
    if x['site'] == 'Mohenjo-daro': return 'MD_SEAL' if x['seal'] else 'MD_OTHER'
    if x['site'] == 'Harappa': return 'H_SEAL' if x['seal'] else 'H_OTHER'
    return 'OTHER_SEAL' if x['seal'] else 'OTHER_OTHER'
for x in TEXTS: x['register'] = register(x)

# deduplicated home corpus (exact sequence identity), as in pipeline
seen = set(); HOME = []; FOREIGN = []
for x in TEXTS:
    if len(x['text']) < 2: continue
    if x['text'] in seen: continue
    seen.add(x['text'])
    (FOREIGN if x['site'] in FOREIGN_SITES else HOME).append(x)
Mse = [x['text'] for x in HOME if x['register'] == 'MD_SEAL']

# --- class induction, identical to pipeline (seed 11, 80/20 split)
random.seed(11)
M = Mse[:]; random.shuffle(M)
h = int(0.8 * len(M)); Mtr, Mte = M[:h], M[h:]
def induce(train):
    freq = Counter(); fin = Counter(); ini = Counter(); nxt = defaultdict(Counter)
    for t in train:
        for s in t: freq[s] += 1
        fin[t[-1]] += 1; ini[t[0]] += 1
        for a, b in zip(t, t[1:]): nxt[a][b] += 1
    TERM = {s for s in freq if freq[s] >= 10 and fin[s] / freq[s] >= 0.60}
    INIT = {s for s in freq if freq[s] >= 10 and ini[s] / freq[s] >= 0.50 and s not in TERM}
    PRE = {s for s in freq if s not in TERM | INIT and freq[s] >= 10 and sum(nxt[s].values()) >= 8
           and sum(nxt[s][x] for x in TERM) / sum(nxt[s].values()) >= 0.6}
    CONN = {s for s in freq if freq[s] >= 15 and s not in TERM | INIT | PRE
            and fin[s] / freq[s] <= 0.05 and ini[s] / freq[s] <= 0.05}
    return TERM, INIT, PRE, CONN, freq
TERM, INIT, PRE, CONN, freqTr = induce(Mtr)

SHORT_NUM = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 16: 6, 17: 7, 18: 8}   # short-stroke numerals
TALL_NUM = {31: 1, 32: 2, 33: 3}
NUMFAM = {3, 4, 5}
def cls(s):
    if s in TERM: return 'T'
    if s in INIT: return 'I'
    if s in PRE: return 'P'
    if s in CONN: return 'C'
    if s in NUMFAM or s in (16, 17, 18): return 'N'
    return 'X'

BRIDGE = json.load(open(f'{DATA}/bridge.json'))
def shape(g):
    v = BRIDGE.get(str(g))
    return (v['desc'][:60] + ' [' + ','.join(v['mahadevan']) + ']') if v else '?'

def texts(reg=None, dedup=True, minlen=2):
    src = HOME + FOREIGN if dedup else TEXTS
    return [x for x in src if len(x['text']) >= minlen and (reg is None or x['register'] == reg)]
