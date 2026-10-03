#!/usr/bin/env python3
"""Loop 51 (origin tracing of the foreign-found texts): shared loaders and scorers.

Corpus: data/derived/merged-corpus-canonical.json (seq_raw / seq_strong / seq_all; reading order, 0 dropped).
Foreign = find-spots outside the Indus culture area (Mesopotamia, Susa/Luristan/Tepe Yahya, Gulf and Oman,
Central Asia). Miri Qalat (Makran) and Shortughai (Indus colony) are flagged as borderland.
Equivalences: merge level (seq_*), optional W->M collapse (bridge_extended + loop27 bridge + S-DARK-27 proposals),
optional composite decomposition from xlits (56 = 55 55, 91 = 90 90, 93 = 1 90).
"""
import json, csv, re, math, random, collections, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

FOREIGN_REGION = {'Mesopotamia', 'Persian Gulf', 'Central Asia'}
FOREIGN_IRAN = {'Luristan', 'Susa', 'Tepe Yahya', 'Murda Sang'}          # Iranian Plateau region, outside Indus
BORDER = {'Miri Qalat', 'Shortughai'}                                    # Makran / Indus colony: flagged
GULF = {'Dilmun', 'Failaka', 'Hajar', 'Janabiyah', 'Kalba', 'Karzakan', "Qala'at al-Bahrain", "Ra's al-Junayz",
        'Saar', 'Salut'}
MESO = {'Girsu', 'Kish', 'Nippur', 'Tell Umma', 'Tello', 'Ur'}
CASIA = {'Altyn Depe', 'Gonur Depe', 'Shortughai'}
HOME_SITES = ['Mohenjo-daro', 'Harappa', 'Lothal', 'Kalibangan', 'Dholavira', 'Chanhu-daro']

C = json.load(open('data/derived/merged-corpus-canonical.json'))
CSV = list(csv.DictReader(open('data/raw/inscriptions.csv')))
BR = json.load(open('data/derived/bridge_extended.json'))
PROP = json.load(open('data/derived/dark/bridge_proposals.json'))
L27 = json.load(open('data/derived/dark/loop27_sets.json'))

def region_of(site):
    for r in CSV:
        if r['site'] == site: return r['region']
    return '?'
REGION = {}
for r in CSV: REGION.setdefault(r['site'], r['region'])

def is_foreign(site):
    reg = REGION.get(site, '?')
    return (reg in FOREIGN_REGION or site in FOREIGN_IRAN) and site != 'Unknown'

def otype(t):
    t = t.split(':')[0]
    return 'seal' if t == 'SEAL' else 'tablet' if t == 'TAB' else 'sealing' if t == 'TAG' else 'pot' if t == 'POT' else 'other'

def shape_class(r):
    sh = r.get('shape', '-'); t = r['type'].split(':')
    if t[0] == 'SEAL':
        if sh in ('circular',) or (len(t) > 1 and t[1] == 'C'): return 'round'
        if sh == 'cylindrical' or (len(t) > 1 and t[1] == 'CY'): return 'cylinder'
        return 'square'
    return otype(r['type'])

# ---- W -> M collapse
W2M = {}
for w, ms in BR.items():
    if ms: W2M[int(w)] = ms[0]
for w, ms in L27['bridge'].items():
    if ms and int(w) not in W2M: W2M[int(w)] = ms[0]
PROPOSED = set()
for row in PROP['proposals']:
    w, m = int(row['W']), int(row['M'])
    if w not in W2M: W2M[w] = m; PROPOSED.add(w)
COMPOSITE = {56: (55, 55), 91: (90, 90), 93: (1, 90)}   # xlits.csv

def mapper(mode):
    """mode: 'W' (level signs as they are), 'M' (collapse through the bridge), 'MC' (bridge + composites split)"""
    if mode == 'W': return lambda s: tuple(s)
    def m(s):
        out = []
        for x in s:
            parts = COMPOSITE.get(x, (x,)) if mode == 'MC' else (x,)
            for y in parts: out.append(('M', W2M[y]) if y in W2M else ('W', y))
        return tuple(out)
    return m

def load(level='seq_raw'):
    """returns list of dict(id, site, type, ot, shape, seq, complete, foreign, border, group)"""
    out = []
    for i, r in enumerate(C):
        s = tuple(r[level])
        if not s: continue
        site = r['site']
        grp = ('gulf' if site in GULF else 'meso' if site in MESO else 'iran' if site in FOREIGN_IRAN
               else 'casia' if site in CASIA else 'home')
        out.append(dict(idx=i, id=r['cisi'], site=site, typ=r['type'], ot=otype(r['type']), shape=shape_class(r),
                        seq=s, raw=tuple(r['seq_raw']), complete=r['complete'] == 'Y', dir=r['dir.'],
                        symbol=r.get('symbol', ''), foreign=is_foreign(site), border=site in BORDER, group=grp))
    return out

def home_label(site):
    return site if site in HOME_SITES else 'other-Indus'

def dedup(objs):
    """collapse identical (site, seq) copies to one record carrying n copies"""
    seen = {}
    for o in objs:
        k = (o['site'], o['seq'])
        if k in seen: seen[k]['copies'] += 1
        else:
            o = dict(o); o['copies'] = 1; seen[k] = o
    return list(seen.values())

# ---- edit distance with early exit
def edit(a, b, cap=None):
    la, lb = len(a), len(b)
    if cap is not None and abs(la - lb) > cap: return cap + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        if cap is not None and min(cur) > cap: return cap + 1
        prev = cur
    return prev[lb]

def nn(q, pool, exclude_idx=None, cap=None, rnd=None):
    """nearest neighbour normalised edit distance of q in pool (list of (idx, mapped_seq)); capped = infinite.
    Ties are broken at random (rnd) so that the largest site does not win every tie."""
    best = None; args = []
    for idx, s in pool:
        if idx == exclude_idx: continue
        d = edit(q, s, cap)
        if cap is not None and d > cap: continue
        dn = d / max(len(q), len(s))
        if best is None or dn < best - 1e-12: best, args = dn, [(idx, s)]
        elif abs(dn - best) <= 1e-12: args.append((idx, s))
    if best is None: return 1.0, None, 0
    arg = (rnd or random).choice(args)
    return best, arg, len(args)

# ---- per-site boundary bigram model with pooled backoff (Dirichlet smoothing)
START, END = ('#S',), ('#E',)
class SiteModel:
    def __init__(self, texts_by_site, k=10.0, kuni=10.0):
        self.k = k; self.kuni = kuni
        self.sites = sorted(texts_by_site)
        self.big = {s: collections.Counter() for s in self.sites}; self.ctx = {s: collections.Counter() for s in self.sites}
        self.uni = {s: collections.Counter() for s in self.sites}; self.ntok = {s: 0 for s in self.sites}
        self.pb = collections.Counter(); self.pc = collections.Counter(); self.pu = collections.Counter(); self.pn = 0
        for s, texts in texts_by_site.items():
            for t in texts: self.add(s, t, +1)
        self.V = len(self.pu) + 1
    def add(self, s, t, sign):
        seq = (START,) + tuple(t) + (END,)
        for a, b in zip(seq, seq[1:]):
            self.big[s][(a, b)] += sign; self.ctx[s][a] += sign; self.pb[(a, b)] += sign; self.pc[a] += sign
        for x in t:
            self.uni[s][x] += sign; self.pu[x] += sign
        self.ntok[s] += sign * len(t); self.pn += sign * len(t)
    def p_pool_uni(self, x): return (self.pu[x] + 0.5) / (self.pn + 0.5 * self.V)
    def p_pool(self, a, b): return (self.pb[(a, b)] + 1.0 * self.p_pool_uni(b)) / (self.pc[a] + 1.0)
    def p_site_uni(self, s, x): return (self.uni[s][x] + self.kuni * self.p_pool_uni(x)) / (self.ntok[s] + self.kuni)
    def p_site(self, s, a, b): return (self.big[s][(a, b)] + self.k * self.p_pool(a, b)) / (self.ctx[s][a] + self.k)
    def loglr(self, s, t, exclude=None):
        """log P(t | site s) - log P(t | pool); exclude = text to remove from the counts first (leave-one-out)"""
        if exclude is not None: self.add(s, exclude, -1)
        seq = (START,) + tuple(t) + (END,)
        lr = 0.0
        for a, b in zip(seq, seq[1:]):
            lr += math.log(self.p_site(s, a, b)) - math.log(self.p_pool(a, b))
        if exclude is not None: self.add(s, exclude, +1)
        return lr
    def loglr_uni(self, s, t, exclude=None):
        if exclude is not None: self.add(s, exclude, -1)
        lr = sum(math.log(self.p_site_uni(s, x)) - math.log(self.p_pool_uni(x)) for x in t)
        if exclude is not None: self.add(s, exclude, +1)
        return lr

# ---- frame slots (S289 closer paradigm; S310/S331 openers; as loop 40)
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; SUF = {400, 90}
CLS = {740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700}
PERSON = {90, 91, 93, 71}
NUMS = {1, 2, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
def frame(s):
    return dict(opener=s[0] in OPEN, marker=len(s) > 1 and s[0] in OPEN and s[1] in MARK,
                closer=(s[-1] in CLS) or (len(s) > 1 and s[-1] in SUF and s[-2] in CLS),
                suffix=s[-1] in SUF, person=any(x in PERSON for x in s), person_first=s[0] in PERSON,
                num=any(x in NUMS for x in s))

def pval(obs, null, side='hi'):
    n = len(null)
    if side == 'hi': return (sum(1 for v in null if v >= obs) + 1) / (n + 1)
    return (sum(1 for v in null if v <= obs) + 1) / (n + 1)

def fmt(x, d=3): return f'{x:.{d}f}' if isinstance(x, float) else str(x)
