"""S-DARK-70: DO LINE BREAKS FALL AT WORD BOUNDARIES?

Shared loader + unit definitions for loop 70. Cycle scripts: tools/dark_loop70_c1.py (split rates per unit), _c2.py (break-learned
segmentation), _c3.py (replication: Wells vs IM77 on linked objects, three merge levels, Harappa tablets, held-out sites).

Data conventions (checked 3 Oct 2026):
- data/raw/inscriptions.csv: `text` is the Wells string in PHYSICAL order left->right, '/' = line/register break on one face, 000 = unread,
  '[' / ']' = damaged end. Canonical reading order (merged-corpus-canonical.json `seq_*`) = the whole string reversed with 000 dropped,
  i.e. the segment AFTER '/' (B) is read first, then the segment before it (A), each reversed (verified on Ad-7 and on IM77 1001 =
  M-1005 through loop24_pairs.json). S-DARK-8.2: a trigram model cannot tell the Wells segment order (tie), so results are given for
  the stored (B then A) order and for the order-free version (break sign pair taken either way). Multi-sided objects (id n.1, n.2 ...)
  are SEPARATE texts (S-DARK-37 arrow C) and are kept apart as 'side boundaries'.
- merged-corpus-canonical.json `seq` fields DROP the line break (lists of ints); the raw->strong / raw->all sign maps are learned from
  its zipped seq_raw/seq_strong/seq_all columns (both are functions) and applied to the raw text here, so the 5,680-row csv is used.
- data/im77/im77_corpus_lines.csv: one row per line; (text_no, side) with lines 1..3 is a multi-line side; line order 1 -> 2 is the
  supported reading order (S-DARK-8.2). Mahadevan numbers; units are mapped W -> M through bridge_extended.json + S-DARK-27 proposals.
"""
import csv, json, collections, re, random, math, sys
ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'

BR = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
PROP = json.load(open(ROOT + 'data/derived/dark/bridge_proposals.json'))
W2M = {int(w): set(ms) for w, ms in BR.items()}
M2W = collections.defaultdict(set)
for w, ms in BR.items():
    for m in ms: M2W[m].add(int(w))
PROPM2W = collections.defaultdict(set)
for p in PROP['proposals']:
    PROPM2W[p['M']].add(p['W'])
    W2M.setdefault(p['W'], set()).add(p['M'])
M2W_ALL = collections.defaultdict(set)
for m, ws in M2W.items(): M2W_ALL[m] |= ws
for m, ws in PROPM2W.items(): M2W_ALL[m] |= ws

WID = json.load(open(OUT + 'loop38_glyph_widths.json'))['signs']
def width(w):
    s = WID.get(str(w)); return s['w_rel'] if s else None
MEDW = sorted(v['w_rel'] for v in WID.values())[len(WID) // 2]

# ---- merge maps learned from the canonical corpus ----
_CAN = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
MAP = {'seq_raw': {}, 'seq_strong': {}, 'seq_all': {}}
for r in _CAN:
    for a, b, c in zip(r['seq_raw'], r['seq_strong'], r['seq_all']):
        MAP['seq_strong'][a] = b; MAP['seq_all'][a] = c
def lv(seq, level):
    m = MAP[level]; return [m.get(a, a) for a in seq]

# ---- frame sets (S310/S331, as tools/dark_loop61.py) ----
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
HEADS = set(CL) | {595}
FISH = {235, 240, 233, 231, 220}
NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
NUMALL = set(range(3, 8)) | set(range(12, 21)) | set(range(25, 30)) | set(range(32, 40)) | {1, 2, 31, 55, 56}
# S-DARK-26.4 deduplicated frozen pairs (unordered adjacency) + 26.1 genuine compounds
FROZEN = [(33, 705), (255, 435), (60, 550), (176, 740), (4, 390), (3, 900), (17, 575), (415, 798), (407, 845), (35, 171),
          (220, 415), (503, 615), (405, 501), (413, 575), (13, 840)]
FROZEN_SET = {frozenset(p) for p in FROZEN}
# S303 named qualifier -> head pairs (W numbers)
S303 = {(760, 740), (100, 740), (176, 740), (33, 520), (220, 520), (233, 520), (550, 527), (555, 527), (142, 617), (806, 154), (806, 158), (3, 156)}

def otype(t):
    t = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t, 'other')
BIG = ('Mohenjo-daro', 'Harappa', 'Mohenjodaro')

# ---------------- Wells ----------------
def parse_wells_text(t):
    """-> (segments in canonical reading order, each a list of ints with 0 = unread, damaged flag). None if unparsable."""
    damaged = '[' in t or ']' in t
    core = t.strip('+[] ')
    segs = [s for s in core.split('/') if s.strip()]
    try:
        segs = [[int(x) for x in s.split('-') if x.strip() != ''] for s in segs]
    except ValueError:
        return None
    segs = [list(reversed(s)) for s in reversed(segs)]   # canonical: later physical segment first, each reversed
    return segs, damaged

def load_wells():
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    objs = []
    for r in rows:
        t = r['text'] or ''
        p = parse_wells_text(t)
        if p is None: continue
        segs, dam = p
        oid, face = (r['id'].split('.') + ['1'])[:2]
        objs.append(dict(src='wells', id=r['id'], obj=oid, face=int(face), cisi=r['cisi'], site=r['site'], type=r['type'], ot=otype(r['type']),
                         complete=r['complete'] == 'Y' and not dam, dir=r['dir.'], segs=segs, H=r['horizontal(mm)'], V=r['vertical(mm)'],
                         nsides=r['sides']))
    return objs

# ---------------- IM77 ----------------
def load_im77():
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[(r['text_no'], r['side'])].append(r)
    objs = []
    for (tn, side), v in by.items():
        v = [x for x in v if x['line'] != '9']
        if not v: continue
        v.sort(key=lambda x: int(x['line']))
        segs = [[int(s) for s in x['signs_clean'].split()] for x in v]
        o = v[0]
        objs.append(dict(src='im77', id=f'{tn}.{side}', obj=tn, face=int(side), site=o['site'], ot={'seal': 'seal', 'sealing': 'sealing',
                         'miniature tablet': 'tablet', 'copper tablet': 'tablet', 'pottery graffito': 'pot'}.get(o['object_type'], 'other'),
                         type=o['object_type'], complete=all(0 not in s for s in segs), dirs=[x['direction'] for x in v], segs=segs,
                         doubt=[x['doubtful_positions'] for x in v]))
    return objs

def flat(o):
    """full sequence and the set of gap indices (gap g = between sign g and g+1, 0-based) that are line breaks."""
    seq = []; breaks = set()
    for i, s in enumerate(o['segs']):
        if i > 0: breaks.add(len(seq) - 1)
        seq.extend(s)
    return seq, breaks

# ---------------- qualifier sets (S-DARK-61 build_qual) from a training set of sequences ----------------
def build_qual(seqs):
    left = collections.defaultdict(collections.Counter)
    for s in seqs:
        s = list(s)
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
    Q = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        Q[c] = q
    return Q

def parse(s, QUAL):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if not s: return lab
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK:
            lab[1] = 'MARKER'; i = 2
            if s[0] == 920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
    while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CL or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CL:
        c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
        if c == 520:
            if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
            while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
        elif c == 740:
            if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
    for k in range(i, j - 1):
        if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
    for k in range(i, j):
        if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
    return lab

# ---------------- unit gap classification (W numbering) ----------------
def gap_units(seq, lab, QUAL, loyal=None):
    """For each gap g (between seq[g] and seq[g+1]) return the set of unit classes the gap is INSIDE of."""
    n = len(seq); U = [set() for _ in range(n - 1)]
    for g in range(n - 1):
        a, b = seq[g], seq[g + 1]
        if a == 0 or b == 0: U[g].add('UNREAD'); continue
        la, lb = lab[g], lab[g + 1]
        # opener phrase: opener + marker (+ marked jar)
        if la == 'OPENER' and lb == 'MARKER' or la == 'MARKER' and lb == 'MARKER': U[g].add('opener_phrase')
        if (a, b) in ((920, 60), (817, 2), (861, 2), (692, 60)) or (a == 60 and b in MJAR and g >= 1 and seq[g - 1] == 920): U[g].add('opener_phrase_fixed')
        # qualifier + head (parser TITLE -> CLOSER, or TITLE -> TITLE inside the closing phrase)
        if la == 'TITLE' and lb in ('CLOSER', 'TITLE'): U[g].add('qual_head')
        if lb in HEADS and a in QUAL.get(b, ()): U[g].add('qual_head_loyal60')
        if (a, b) in S303: U[g].add('qual_head_S303')
        if loyal and (a, b) in loyal: U[g].add('qual_head_loyal80')
        # head + suffix
        if la in ('CLOSER', 'SUFFIX') and lb == 'SUFFIX': U[g].add('head_suffix')
        # frozen pairs (unordered)
        if frozenset((a, b)) in FROZEN_SET: U[g].add('frozen_pair')
        # fish words: numeral + fish, fish + fish, marker 2 + fish, fish + arrow
        if (a in NUM and b in FISH) or (a in FISH and b in FISH) or (a == 2 and b in FISH): U[g].add('fish_word')
        if a in FISH and b == 520: U[g].add('fish_arrow')
        # numeral + item (COUNT pair in the middle, not followed by head)
        if la == 'COUNT' and lb == 'COUNT': U[g].add('num_item')
        if a in NUM and b not in NUM and lb == 'NAME' and la == 'COUNT': U[g].add('num_item')
        # middle NAME-NAME gap (the 'single-sign element' hypothesis says these are all word edges)
        if la == 'NAME' and lb == 'NAME' and a not in NUM and b not in NUM: U[g].add('name_name')
        # any-unit summary
        if U[g] - {'name_name'}: U[g].add('ANY_UNIT')
    return U

def loyal_pairs(seqs, minn=5, thr=0.8):
    """penultimate -> head pairs where the penultimate sign (when penultimate before a head) picks one head >= thr of the time."""
    pen = collections.defaultdict(collections.Counter)
    for s in seqs:
        s = list(s)
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in HEADS: pen[s[-2]][s[-1]] += 1
    out = set()
    for a, c in pen.items():
        tot = sum(c.values())
        if tot >= minn:
            h, k = c.most_common(1)[0]
            if k / tot >= thr: out.add((a, h))
    return out

def to_W(seq_m):
    """IM77 Mahadevan sequence -> list of candidate W sets (0 stays 0)."""
    return [({0} if m == 0 else set(M2W_ALL.get(m, ()))) for m in seq_m]

def wells_view_of_im77(seq_m):
    """Pick one W per M (bridge first, commonest W-frame member preferred) for frame parsing; None when unmapped."""
    out = []
    for m in seq_m:
        if m == 0: out.append(0); continue
        ws = M2W.get(m) or PROPM2W.get(m)
        if not ws: out.append(-m); continue   # negative = unmapped M (unique, non-frame)
        pref = [w for w in ws if w in HEADS or w in OPEN or w in MARK or w in NUM or w in FISH or w in SUF]
        out.append(sorted(pref)[0] if pref else sorted(ws)[0])
    return out

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)
