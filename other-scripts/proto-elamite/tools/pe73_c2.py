#!/usr/bin/env python3
"""pe73 cycle 2: re-run the B results with (a) the photo/copy corrections from cycle 1 and (b) every
editorial restoration '[...]' and every '?' numeral removed (the editor's inference is not on the clay).
A: pe42 capacity closers (N01/N14-only tablets that close under N14 = 6 N01 and not under N14 = 10).
B: per-head 60 rule (pe69 LAST60/LAST120 hits; pe63 D3 6/6).
C: pe71 |M153+X| / |M153+M342| on the last signed line, copy-confirmed tokens only.
Output: data/pe73_ckpt/c2.json"""
import json, os, random, re, collections
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data'); CK = os.path.join(DATA, 'pe73_ckpt')
C = {t['id']: t for t in json.load(open(os.path.join(DATA, 'pe_corpus.json')))}
OUT = {}
NUMRE = re.compile(r'(\[?)(\d+)\((N\d+[A-Z]?)\)(\]?)([#?!]*)')

def num_part(raw):
    return raw.split(',', 1)[1] if ',' in raw else raw

def restored(l):
    """True if any numeral of the line is inside [ ] or carries '?'"""
    s = num_part(l['raw']); inb = False
    for m in re.finditer(r'\[|\]|\d+\(N\d+[A-Z]?\)[#?!]*', s):
        t = m.group(0)
        if t == '[': inb = True
        elif t == ']': inb = False
        elif inb or '?' in t: return True
    return False

def damaged_num(l):
    return '#' in num_part(l['raw'])

# ---------------- A: pe42 capacity closers
def nv(nums, n14):
    v = 0
    for n, s in nums:
        if s == 'N01': v += n
        elif s == 'N14': v += n14 * n
        else: return None
    return v

def cases(strict, fix):
    out = []
    for p, t in C.items():
        L = t['lines']
        ob = [l for l in L if l['surface'] == 'obverse' and l['numerals']]
        rv = [l for l in L if l['surface'] == 'reverse' and l['numerals']]
        if len(ob) < 2 or len(rv) != 1: continue
        if any(l['lacuna'] for l in L if l['surface'] == 'obverse'): continue
        allnum = [x for l in ob + rv for x in l['numerals']]
        if not allnum or any(s not in ('N01', 'N14') for _, s in allnum): continue
        if strict and any(restored(l) for l in ob + rv): continue
        ent = [l['numerals'] for l in ob]; tot = rv[0]['numerals']
        if fix and p in fix: ent = fix[p](ent)
        if ent is None: continue
        out.append((p, ent, tot))
    return out

def closers(cs, perm=None):
    tots = [c[2] for c in cs] if perm is None else perm
    n = 0; ids = []
    for (p, ent, _), tot in zip(cs, tots):
        s6 = sum(nv(e, 6) for e in ent); s10 = sum(nv(e, 10) for e in ent)
        if s6 == nv(tot, 6) and s10 != nv(tot, 10):
            n += 1; ids.append(p)
    return n, ids

def test_A(cs, label, reps=4000):
    n, ids = closers(cs)
    rng = random.Random(73); tots = [c[2] for c in cs]; ge = 0; mean = 0
    for _ in range(reps):
        sh = tots[:]; rng.shuffle(sh); k = closers(cs, sh)[0]; mean += k; ge += k >= n
    return dict(label=label, tablets=len(cs), closers=n, ids=ids, null_mean=round(mean / reps, 2), p=round((ge + 1) / (reps + 1), 4))

FIX = {  # photo/copy corrections (cycle 1): P009296 line 3 shows 3 wedges only (N14 is the editor's restoration)
    'P009296': lambda ent: [ent[0], [[3, 'N01']], ent[2]],
}
A = [test_A(cases(False, None), 'as transliterated (restorations kept)'),
     test_A(cases(False, FIX), 'photo/copy-corrected (P009296 restored N14 removed)'),
     test_A(cases(True, None), 'strict: tablets with any restored or ? numeral dropped')]
FIX2 = dict(FIX); FIX2['P008784'] = lambda ent: [ent[0], [[3, 'N01']], ent[2]]
A.append(test_A(cases(False, FIX2), 'worst case: also P008784 line 3 read 3(N01) (copy ambiguous 2 or 3)'))
OUT['A_pe42'] = A
for a in A: print('A', a)

# ---------------- B: per-head rule hits
rows = [r.rstrip('\n').split('\t') for r in open(os.path.join(DATA, 'pe69_decoded_slots.tsv')) if not r.startswith('#')][1:]
claims = json.load(open(os.path.join(CK, 'claims.json')))
def find_line(p, ln):
    L = C[p]['lines']
    if ln < len(L) and 'M288' in L[ln]['signs']: return ln
    for j, l in enumerate(L):
        if l['label'] == str(ln) and 'M288' in l['signs']: return j
hits = []
for r in rows:
    if not r[-1].startswith('LAST'): continue
    p, ln = r[0], int(r[1]); i = find_line(p, ln)
    if i is None: continue
    L = C[p]['lines']; k = i - 1
    while k >= 0 and not L[k]['numerals']: k -= 1
    m288, cnt = L[i], L[k]
    hits.append(dict(pid=p, line=ln, cls=r[-1], m288=m288['raw'], count=cnt['raw'],
                     restored=restored(m288) or restored(cnt), damaged=damaged_num(m288) or damaged_num(cnt)))
B = dict(hits=len(hits), restored=sum(h['restored'] for h in hits), damaged=sum(h['damaged'] for h in hits),
         clean=sum(not h['restored'] and not h['damaged'] for h in hits),
         restored_list=[(h['pid'], h['line'], h['count'], h['m288']) for h in hits if h['restored']])
# pe69 headline 34/49 vs 5.5: bare 'M288 n' lines after PERSON-class lines.  Upper-bound check: if every restored hit
# is turned into a miss, the hit rate falls to (34 - r) / 49 against the same chance level.
r_bare = sum(1 for h in hits if h['restored'] and h['m288'].split(',')[0].strip().strip('[]#') == 'M288')
B['pe69_headline_after_dropping_restored'] = '%d/49 (was 34/49; chance 5.5)' % (34 - r_bare)
# D3 dossier 6/6
d3 = ['P009190', 'P009211', 'P009220', 'P009237', 'P009238', 'P009286', 'P009309']
B['D3'] = []
for p in d3:
    L = C[p]['lines']
    for i, l in enumerate(L):
        if 'M288' in l['signs'] and i > 0:
            c = L[i - 1]
            B['D3'].append(dict(pid=p, count=c['raw'], m288=l['raw'], restored=restored(l) or restored(c), damaged=damaged_num(l) or damaged_num(c)))
OUT['B_rate'] = B
print('B', {k: v for k, v in B.items() if k not in ('restored_list', 'D3')})
for d in B['D3']: print('  D3', d)
for x in B['restored_list']: print('  restored hit', x)

# ---------------- C: M153 last line, copy-confirmed tokens only
ks = [l.split(None, 5) for l in open(os.path.join(CK, 'ksign_answers.txt')) if l.startswith('P')]
conf = {(k[0]) for k in ks if k[1] == 'M153' and 'A' in (k[3], k[4]) and 'D' not in (k[3], k[4])}
def m153_tokens(pids=None):
    obs = exp = 0; n = 0
    for p, t in C.items():
        if pids is not None and p not in pids: continue
        L = t['lines']; signed = [i for i, l in enumerate(L) if l['signs']]
        un = [i for i, l in enumerate(L) if l['signs'] and not l['numerals']]
        for i, l in enumerate(L):
            for s in l['signs']:
                if (s.startswith('|M153+X') or s.startswith('|M153+M342')) and not l['numerals']:
                    n += 1; obs += i == signed[-1]; exp += (1 / len(un)) if signed[-1] in un else 0
    return n, obs, round(exp, 2)
def binom_p(n, k, e):
    # Poisson-binomial approximated by binomial at the mean rate
    from math import comb
    q = e / n if n else 0
    return sum(comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(k, n + 1))
Call = m153_tokens(); Cc = m153_tokens(conf)
OUT['C_m153'] = dict(all=dict(n=Call[0], last=Call[1], exp=Call[2], p=binom_p(*Call)), copy_confirmed=dict(n=Cc[0], last=Cc[1], exp=Cc[2], p=binom_p(*Cc), tablets=sorted(conf)))
print('C', OUT['C_m153'])
json.dump(OUT, open(os.path.join(CK, 'c2.json'), 'w'), indent=1)
