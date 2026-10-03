#!/usr/bin/env python3
"""Loop 71, cycle 3: where are the seals that stamped the Lothal sealings? Lothal / Mohenjo-daro / elsewhere / nowhere.
Every distinct Lothal sealing die (cycle-1 assignment; Wells and, separately, IM77) is compared with every seal text
in the same transcription (Wells SEAL objects; IM77 object_type 'seal'):
  - complete die (no lost sign, >= 3 signs): exact text equality;
  - fragment die: fits inside a seal text with lost signs as wildcards and >= 3 agreeing signs.
Control: the die's signs shuffled within the die (lost-sign positions kept), 1,000x; expected number of dies that
match somewhere, and at Lothal, by chance. Emblem check for matched pairs (Wells symbol field / IM77 field symbol).
Usage: python3 tools/dark_loop71_c3.py [seq_raw|seq_strong|seq_all] [nperm]"""
import sys, csv, collections, random, re
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop71_common import *
from dark_loop37 import IM77 as IM77_PATH
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'; NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rnd = random.Random(713)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

def site_class(s):
    if s == 'Lothal': return 'Lothal'
    if s in ('Mohenjo-daro', 'Mohenjodaro'): return 'Mohenjo-daro'
    return 'elsewhere'

def matcher(seals):
    """seals: list of (text tuple, site, id, emblem). Returns f(die tokens) -> list of hits."""
    by_exact = collections.defaultdict(list)
    for t, s, i, e in seals: by_exact[t].append((s, i, e))
    def f(t):
        t = list(t); nz = [x for x in t if x]
        if 0 not in t and len(t) >= 3: return by_exact.get(tuple(t), [])
        if len(nz) < 3: return None          # too short to test
        i, j = 0, len(t)
        while i < j and t[i] == 0: i += 1
        while j > i and t[j - 1] == 0: j -= 1
        core = t[i:j]; hits = []
        for st, s, sid, e in seals:
            for off in range(len(st) - len(core) + 1):
                if all(x == 0 or x == y for x, y in zip(core, st[off:off + len(core)])): hits.append((s, sid, e)); break
        return hits
    return f

def analyse(name, dies, seals, uses):
    """dies: list of die token tuples (distinct); uses[d] = sealings carrying it"""
    f = matcher(seals)
    P(f'\n-- {name}: {len(seals)} seal texts; {len(dies)} distinct Lothal sealing dies')
    res = collections.Counter(); rows = []
    for d in dies:
        h = f(d)
        if h is None: res['too short'] += 1; continue
        cl = sorted({site_class(s) for s, i, e in h})
        key = 'nowhere' if not h else ('Lothal' if 'Lothal' in cl else ('Mohenjo-daro' if cl == ['Mohenjo-daro'] else '+'.join(cl)))
        res[key] += 1
        if h: rows.append((d, h))
    tested = sum(v for k, v in res.items() if k != 'too short')
    P(f'   tested {tested} (>= 3 legible signs): ' + ', '.join(f'{k} {v}' for k, v in res.most_common()))
    nl = len({i for t, s, i, e in seals if s == 'Lothal'})
    P(f'   seal objects at Lothal in this transcription: {nl}; complete dies tested {sum(1 for d in dies if 0 not in d and len(d) >= 3)}, '
      f'fragments tested {sum(1 for d in dies if 0 in d and len([x for x in d if x]) >= 3)}')
    for d, h in rows:
        kind = 'EXACT complete text' if (0 not in d and len(d) >= 3) else f'fragment-compatible'
        P(f'      {fmt(d)} (on {uses[d]} sealings; {kind}; {len(h)} seals) -> ' + '; '.join(f'{s} {i} [{e}]' for s, i, e in h[:6]) + (f' (+{len(h)-6})' if len(h) > 6 else ''))
    # control: shuffle signs within each die
    nm, nl, nmd = [], [], []
    for _ in range(NP):
        m = l = md = 0
        for d in dies:
            t = list(d); pos = [k for k, x in enumerate(t) if x]; vals = [t[k] for k in pos]; rnd.shuffle(vals)
            for k, v in zip(pos, vals): t[k] = v
            if tuple(t) == tuple(d) and len(set(vals)) > 1:
                pass
            h = f(t)
            if h is None: continue
            if h: m += 1
            if any(site_class(s) == 'Lothal' for s, i, e in h): l += 1
            if any(site_class(s) == 'Mohenjo-daro' for s, i, e in h): md += 1
        nm.append(m); nl.append(l); nmd.append(md)
    om = sum(1 for d, h in rows); ol = sum(1 for d, h in rows if any(site_class(s) == 'Lothal' for s, i, e in h))
    omd = sum(1 for d, h in rows if any(site_class(s) == 'Mohenjo-daro' for s, i, e in h))
    P(f'   matched anywhere {om} vs shuffled {sum(nm)/NP:.2f} (P = {pval(om, nm):.4f}); at Lothal {ol} vs {sum(nl)/NP:.2f} (P = {pval(ol, nl):.3f}); '
      f'at Mohenjo-daro {omd} vs {sum(nmd)/NP:.2f} (P = {pval(omd, nmd):.4f})')
    return res, rows

# ---------------- Wells ----------------
mp = merge_maps()[LV]
seals = []; dies_raw = []
Wl = wells_lothal(LV)
for r in csv.DictReader(open(RAW)):
    if not r['type'].startswith('SEAL'): continue
    t = [int(x) for x in re.findall(r'\d{3}', r['text'])][::-1]
    t = tuple(mp.get(x, x) for x in t if x not in (0, 999))
    if len(t) >= 3: seals.append((t, r['site'], r['cisi'] or r['id'], r['symbol']))
imps = [(f['id'], f['toks']) for k, fs in Wl.items() for f in fs]
lab = assign_dies(imps)
uses = collections.Counter()
for c, fs in Wl.items():
    for d in {lab[f['id']] for f in fs if [x for x in f['toks'] if x]}: uses[toks_ := (tuple(d[1:]) if d[0] == 'u' else d)] += 1
dies = sorted(uses, key=lambda d: -uses[d])
resW, rowsW = analyse('Wells', dies, seals, uses)
# emblem check for the Wells matches: sealing emblem vs seal emblem
P('   emblem check (sealing symbol field -> matched seal symbols):')
for d, h in rowsW:
    em = collections.Counter(f['symbol'] for c, fs in Wl.items() for f in fs if (lab[f['id']][1:] if lab[f['id']][0] == 'u' else lab[f['id']]) == d)
    P(f'      {fmt(d)}: sealing {dict(em)} | seals {[e for s, i, e in h[:8]]}')

# ---------------- IM77 ----------------
imseals = []; raw = collections.defaultdict(lambda: collections.defaultdict(list)); fsym = {}
seal_sides = collections.defaultdict(lambda: collections.defaultdict(list)); seal_meta = {}
for r in csv.DictReader(open(IM77_PATH)):
    if r['line'] == '9' or not r['signs_clean'].strip(): continue
    sg = [int(x) for x in r['signs_clean'].split()]
    if r['object_type'] == 'seal':
        seal_sides[r['text_no']][int(r['side'])].extend(sg); seal_meta[r['text_no']] = (r['site'], r['fs_description'][:24])
    if r['object_type'] == 'sealing' and r['site'] == 'Lothal':
        raw[r['text_no']][int(r['side'])].extend(sg)
for k, sides in seal_sides.items():
    for s, t in sides.items():
        t = tuple(x for x in t if x)
        if len(t) >= 3: imseals.append((t, seal_meta[k][0], k, seal_meta[k][1]))
iimps = [(f'{k}.{s}', t) for k, sides in raw.items() for s, t in sides.items()]
ilab = assign_dies(iimps)
iuses = collections.Counter()
for k, sides in raw.items():
    for d in {ilab[f'{k}.{s}'] for s in sides if [x for x in sides[s] if x]}: iuses[tuple(d[1:]) if d[0] == 'u' else d] += 1
idies = sorted(iuses, key=lambda d: -iuses[d])
resM, rowsM = analyse('IM77', idies, imseals, iuses)

# ---------------- weighted by use: share of IMPRESSIONS whose seal is found ----------------
for name, rows, uses_ in (('Wells', rowsW, uses), ('IM77', rowsM, iuses)):
    tot = sum(uses_.values()); hit = collections.Counter()
    for d, h in rows:
        cl = {site_class(s) for s, i, e in h}
        hit['Lothal' if 'Lothal' in cl else ('Mohenjo-daro' if 'Mohenjo-daro' in cl else 'elsewhere')] += uses_[d]
    P(f'   {name}: of {tot} die-uses on sealings, seal found at Lothal {hit["Lothal"]}, Mohenjo-daro {hit["Mohenjo-daro"]}, elsewhere {hit["elsewhere"]}, '
      f'not found {tot - sum(hit.values())} (incl. dies too short to test)')
open(OUT + f'loop71_c3_{LV}.txt', 'w').write('\n'.join(LOG) + '\n')
