"""S-DARK-2: ARROW-IN-THE-DARK loop 2 (extends tools/strat_dark_pairs.py).

PAIR arrows: random (text-pair relation) x (object-pair fact) over random pairs of objects within the same site.
  Statistic: mutual information.  Null: object-level permutation of the fact within site (pair dependence kept).
  Train: Mohenjo-daro + Harappa pairs (p < 0.001, then refined to 20k permutations for a Bonferroni check across
  all arrows fired in the cycle); replication: held-out sites (all non-MD/H sites, p < 0.01) and seq_all (p < 0.01).
SET arrows: find-groups (site, area-section, block-house, room-grid) of >= 3 objects; random set statistic averaged
  over groups passing a random filter.  Null: re-deal objects among the groups of the same sizes within site.
  Train: Mohenjo-daro groups; replication: Harappa groups (other city), outside groups (K/L/other; low power), seq_all.
  SETxFACT arrows: Spearman correlation across groups between a text-set statistic and an object-set homogeneity;
  null: permute object facts within site.
Usage: python3 tools/strat_dark_pairs2.py CYCLE SEED [NPAIR NSET] [control]
Writes data/derived/dark/loop2_cycle<CYCLE>.txt and .json.
"""
import json, csv, random, sys, collections, math, time
import numpy as np
import multiprocessing as mp

CYCLE = int(sys.argv[1]); SEED = int(sys.argv[2])
NPAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
NSET = int(sys.argv[4]) if len(sys.argv) > 4 else 500
CONTROL = 'control' in sys.argv[5:]
OUT = 'data/derived/dark/loop2_cycle%d%s' % (CYCLE, '_control' if CONTROL else '')
rng = random.Random(SEED); nrng = np.random.default_rng(SEED)

C = json.load(open('data/derived/merged-corpus-canonical.json'))
raw = {}
for r in csv.DictReader(open('data/raw/inscriptions.csv')):
    if r['cisi'] and r['cisi'] != '-' and r['cisi'] not in raw: raw[r['cisi']] = r
NUM = {1: 1, 3: 3, 4: 4, 5: 5, 16: 6, 17: 7, 18: 8, 31: 1, 32: 2, 33: 3, 34: 4}
OPEN = {817, 861, 820, 920, 692}
CLOSE = {740, 400, 700, 520, 390, 90, 407, 741, 595}
JAR = 740

def num(x):
    try:
        v = float(x); return v if v > 0 else None
    except Exception:
        return None

def depth_cm(x):
    if not x: return None
    x = x.strip().replace('- -', '')
    if not x: return None
    try:
        tok = x.replace('-', '').split()
        v = float(tok[0]); u = tok[1] if len(tok) > 1 else 'ft'
        return v * 30.48 if u.startswith('ft') else (v * 100 if u.startswith('m') else v)
    except Exception:
        return None

def clean(v):
    return None if v in ('', '-', '--', 'None', '?', '- -', None) else v

OBJ = []
for r in C:
    if not r.get('seq_raw') or len(r['seq_raw']) < 2: continue
    x = raw.get(r['cisi'], {})
    f = {'site': r['site'], 'type': r['type'].split(':')[0], 'typefull': r['type'],
         'emblem': (r.get('symbol') or '').split(':')[0] or None, 'material': clean(x.get('material')) or clean(r.get('material')),
         'shape': clean(x.get('shape')) or clean(r.get('shape')), 'boss': x.get('boss'), 'area': r.get('area-section'),
         'room': None, 'time': clean(x.get('time')) or clean(r.get('time')), 'period': clean(r.get('period')),
         'h': num(x.get('horizontal(mm)')), 'v': num(x.get('vertical(mm)')), 'depth': depth_cm(x.get('depth')),
         'cult': x.get('cult'), 'color': x.get('color'), 'complete': r.get('complete'), 'dir': r.get('dir.'),
         'cond': x.get('condition'), 'sides': x.get('sides')}
    if r.get('area-section') != '--' and (r.get('block-house') != '--' or r.get('room-grid') != '--'):
        f['room'] = '%s|%s|%s' % (r['area-section'], r['block-house'], r['room-grid'])
    for k, v in list(f.items()): f[k] = clean(v)
    f['site'] = r['site']
    OBJ.append({'cisi': r['cisi'], 'raw': tuple(r['seq_raw']), 'all': tuple(r['seq_all']), 'f': f})

SIGNFREQ = collections.Counter(s for o in OBJ for s in o['raw'])
RARE = {s for s, c in SIGNFREQ.items() if c < 10}
TOPSIGNS = [s for s, _ in SIGNFREQ.most_common(60)]
FACTS = ['type', 'emblem', 'material', 'shape', 'boss', 'area', 'room', 'time', 'period', 'h', 'v', 'depth', 'cult', 'color',
         'complete', 'dir', 'cond', 'sides', 'typefull']
NUMERIC = {'h': 5.0, 'v': 5.0, 'depth': 60.0}

if CONTROL:  # shuffle every fact within site (room: within site x type) before anything else: calibrates the whole procedure
    for key in FACTS:
        bysite = collections.defaultdict(list)
        for o in OBJ: bysite[(o['f']['site'], o['f']['type'] if key == 'room' else None)].append(o)
        for os_ in bysite.values():
            vals = [o['f'][key] for o in os_]; rng.shuffle(vals)
            for o, v in zip(os_, vals): o['f'][key] = v

# ---------------------------------------------------------------- text features
def feats(s):
    A = set(s); op = s[0] if s[0] in OPEN else None
    mid = s[1:-1] if op else s[:-1]
    nums = [NUM[x] for x in s if x in NUM]
    return dict(s=s, A=A, op=op, mid=tuple(mid), close=s[-1], nums=nums, numset=set(nums), sum=sum(s), big=set(zip(s, s[1:])),
                rbig=set(zip(s[1:], s)), rare=A & RARE, L=len(s), freq=np.mean([SIGNFREQ[x] for x in s]))

def lcs_len(s, t):
    best = 0; prev = [0] * (len(t) + 1)
    for a in s:
        cur = [0] * (len(t) + 1)
        for j, b in enumerate(t):
            if a == b: cur[j + 1] = prev[j] + 1; best = max(best, cur[j + 1])
        prev = cur
    return best

def same_order(s, t, shared):
    ps = [s.index(x) for x in shared]; pt = [t.index(x) for x in shared]
    return all((ps[i] < ps[j]) == (pt[i] < pt[j]) for i in range(len(ps)) for j in range(i + 1, len(ps)))

# relation registry: name -> (cycle introduced, function(p, F_s, F_t) -> hashable)
REL = {}
def rel(cycle, name=None, params=None):
    def deco(fn):
        REL[name or fn.__name__] = (cycle, fn, params or (lambda r: {}))
        return fn
    return deco

# -- base family (strat_dark_pairs.py)
@rel(0)
def jacc(p, s, t): return min(3, int(4 * len(s['A'] & t['A']) / len(s['A'] | t['A'])))
@rel(0)
def first(p, s, t): return s['s'][0] == t['s'][0]
@rel(0)
def last(p, s, t): return s['close'] == t['close']
@rel(0)
def edit1(p, s, t): return s['L'] == t['L'] and sum(a != b for a, b in zip(s['s'], t['s'])) == 1
@rel(0)
def nest(p, s, t):
    a, b = s['s'], t['s']
    if len(a) == len(b): return False
    if len(a) > len(b): a, b = b, a
    return any(a == b[i:i + len(a)] for i in range(len(b) - len(a) + 1))
@rel(0)
def disjoint(p, s, t): return not (s['A'] & t['A'])
@rel(0)
def samelen(p, s, t): return s['L'] == t['L']
@rel(0)
def lensum(p, s, t): return min(12, s['L'] + t['L'])
@rel(0, params=lambda r: {'s': r.choice(TOPSIGNS)})
def sharesign(p, s, t): return p['s'] in s['A'] and p['s'] in t['A']
@rel(0, params=lambda r: {'s': r.choice(TOPSIGNS)})
def xorsign(p, s, t): return (p['s'] in s['A']) != (p['s'] in t['A'])
@rel(0)
def samenum(p, s, t): return bool(s['numset'] & t['numset'])
@rel(0)
def sharepair(p, s, t): return bool(s['big'] & t['big'])
@rel(0, params=lambda r: {'k': r.randint(1, 3)})
def posk(p, s, t):
    k = p['k']; return s['L'] > k and t['L'] > k and s['s'][k] == t['s'][k]
@rel(0)
def mirror(p, s, t): return s['s'][::-1] == t['s'] or s['s'][0] == t['close']
@rel(0)
def sumw(p, s, t): return min(5, abs(s['sum'] - t['sum']) // 300)

# -- cycle 1 additions (the task list)
@rel(1)
def complementary(p, s, t): return (not (s['A'] & t['A'])) and len(s['A']) + len(t['A']) >= 6
@rel(1)
def samemid_diffclose(p, s, t): return len(s['mid']) > 0 and s['mid'] == t['mid'] and s['close'] != t['close']
@rel(1)
def prefix1(p, s, t):
    a, b = s['s'], t['s']
    return (len(b) == len(a) + 1 and b[1:] == a) or (len(a) == len(b) + 1 and a[1:] == b)
@rel(1)
def suffix1(p, s, t):
    a, b = s['s'], t['s']
    return (len(b) == len(a) + 1 and b[:-1] == a) or (len(a) == len(b) + 1 and a[:-1] == b)
@rel(1)
def numeq(p, s, t): return 'none' if not (s['nums'] and t['nums']) else (max(s['nums']) == max(t['nums']))
@rel(1)
def numdiff(p, s, t): return -1 if not (s['nums'] and t['nums']) else min(4, abs(max(s['nums']) - max(t['nums'])))
@rel(1, params=lambda r: {'m': r.choice([2, 3, 4, 5, 7, 12])})
def summod(p, s, t): return (s['sum'] - t['sum']) % p['m'] == 0
@rel(1)
def sameopen_diffmid(p, s, t): return s['op'] is not None and s['op'] == t['op'] and s['mid'] != t['mid']
@rel(1, params=lambda r: {'k': r.randint(1, 3)})
def sharek(p, s, t): return len(s['A'] & t['A']) == p['k']
@rel(1)
def revmatch(p, s, t): return s['s'][::-1] == t['s'] or bool(s['big'] & t['rbig'])
@rel(1)
def sameclose_diffmid(p, s, t): return s['close'] == t['close'] and s['mid'] != t['mid']
@rel(1)
def bothopen(p, s, t): return (s['op'] is not None, t['op'] is not None)
@rel(1)
def bothjar(p, s, t): return (s['close'] == JAR, t['close'] == JAR)
@rel(1)
def lendiff(p, s, t): return min(4, abs(s['L'] - t['L']))
@rel(1)
def ident(p, s, t): return s['s'] == t['s']

# -- cycle 2 additions
@rel(2)
def closerclass(p, s, t): return (s['close'] in CLOSE, t['close'] in CLOSE)
@rel(2)
def sharemidsign(p, s, t): return bool(set(s['mid']) & set(t['mid']))
@rel(2)
def second(p, s, t): return s['L'] > 2 and t['L'] > 2 and s['s'][1] == t['s'][1]
@rel(2)
def lcs2(p, s, t): return min(3, lcs_len(s['s'], t['s']))
@rel(2)
def numsum(p, s, t): return 'none' if not (s['nums'] and t['nums']) else (sum(s['nums']) == sum(t['nums']))
@rel(2)
def revbigram(p, s, t): return bool(s['big'] & t['rbig'])
@rel(2)
def sharerare(p, s, t): return bool(s['rare'] & t['rare'])
@rel(2)
def anypos(p, s, t): return any(a == b for a, b in zip(s['s'], t['s']))
@rel(2)
def orderkept(p, s, t):
    sh = list(s['A'] & t['A'])
    return 'lt2' if len(sh) < 2 else same_order(s['s'], t['s'], sh)
@rel(2)
def freqbin(p, s, t): return min(3, int(abs(s['freq'] - t['freq']) // 150))
@rel(2)
def bothrare(p, s, t): return (bool(s['rare']), bool(t['rare']))

# -- cycle 3 additions
@rel(3)
def openunit(p, s, t): return s['op'] is not None and t['op'] is not None and s['s'][:2] == t['s'][:2]
@rel(3)
def bothnum(p, s, t): return (bool(s['nums']), bool(t['nums']))
@rel(3)
def closepair(p, s, t): return tuple(sorted((s['close'], t['close']))) if (s['close'] in CLOSE and t['close'] in CLOSE) else 'other'
@rel(3)
def midlen(p, s, t): return min(3, abs(len(s['mid']) - len(t['mid'])))
@rel(3)
def midjacc(p, s, t):
    A, B = set(s['mid']), set(t['mid'])
    return 'none' if not (A and B) else min(3, int(4 * len(A & B) / len(A | B)))
@rel(3)
def samelast2(p, s, t): return s['L'] > 2 and t['L'] > 2 and s['s'][-2:] == t['s'][-2:]
@rel(3)
def sharenumsign(p, s, t): return bool({x for x in s['A'] if x in NUM} & {x for x in t['A'] if x in NUM})
@rel(3)
def rotation(p, s, t):
    a, b = s['s'], t['s']
    return len(a) == len(b) and len(a) > 2 and a != b and any(a[i:] + a[:i] == b for i in range(1, len(a)))
@rel(3)
def frameclass(p, s, t):
    def cls(F): return (F['op'] is not None) * 2 + (F['close'] in CLOSE)
    return (cls(s), cls(t))
@rel(3)
def singlediff(p, s, t):  # one sign substituted anywhere, lengths equal, bins by position of the difference
    if s['L'] != t['L']: return -1
    d = [i for i, (a, b) in enumerate(zip(s['s'], t['s'])) if a != b]
    return -1 if len(d) != 1 else ('first' if d[0] == 0 else 'last' if d[0] == s['L'] - 1 else 'mid')

# -- cycle 4: random conjunction / disjunction of two boolean relations
BOOLREL = ['first', 'last', 'edit1', 'nest', 'disjoint', 'samelen', 'sharesign', 'samenum', 'sharepair', 'posk', 'complementary',
           'samemid_diffclose', 'prefix1', 'suffix1', 'summod', 'sameopen_diffmid', 'sharek', 'revmatch', 'sameclose_diffmid',
           'ident', 'sharemidsign', 'second', 'revbigram', 'sharerare', 'anypos', 'openunit', 'samelast2', 'sharenumsign', 'rotation']
def _combo_params(r):
    a, b = r.sample(BOOLREL, 2)
    return {'a': a, 'pa': REL[a][2](r), 'b': b, 'pb': REL[b][2](r), 'op': r.choice(['and', 'or', 'xor'])}
@rel(4, params=_combo_params)
def combo(p, s, t):
    x = bool(REL[p['a']][1](p['pa'], s, t)); y = bool(REL[p['b']][1](p['pb'], s, t))
    return (x and y) if p['op'] == 'and' else (x or y) if p['op'] == 'or' else (x != y)

ACTIVE = [k for k, v in REL.items() if v[0] <= CYCLE]

# ---------------------------------------------------------------- encoding + MI
def encode(objs, key):
    vals = [o['f'][key] for o in objs]
    if key in NUMERIC:
        return np.array([v if v is not None else np.nan for v in vals], float)
    u = {v: i for i, v in enumerate(sorted({v for v in vals if v is not None}, key=str))}
    return np.array([u[v] if v is not None else -1 for v in vals], int)

def pairfact(key, codes, a, b):
    fa, fb = codes[a], codes[b]
    if key in NUMERIC:
        ok = ~(np.isnan(fa) | np.isnan(fb)); d = np.where(ok, np.abs(fa - fb), 0.0); y = np.minimum(3, d // NUMERIC[key]).astype(int)
    else:
        ok = (fa >= 0) & (fb >= 0); y = (fa == fb).astype(int)
    return y, ok

def mi_codes(x, y, nx, ny):
    t = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float) / len(x)
    px = t.sum(1, keepdims=True); py = t.sum(0, keepdims=True); nz = t > 0
    return float((t[nz] * np.log(t[nz] / (px @ py)[nz])).sum())

def site_perm(codes, site_idx):
    c = codes.copy()
    for idx in site_idx:
        c[idx] = codes[nrng.permutation(idx)]
    return c

class Pool:
    def __init__(self, objs, npairs, minsite=20):
        self.objs = objs
        self.F = {'raw': [feats(o['raw']) for o in objs], 'all': [feats(o['all']) for o in objs]}
        bysite = collections.defaultdict(list)
        for i, o in enumerate(objs): bysite[o['f']['site']].append(i)
        self.site_idx = [np.array(v) for v in bysite.values()]
        st = collections.defaultdict(list)
        for i, o in enumerate(objs): st[(o['f']['site'], o['f']['type'])].append(i)
        self.strict_idx = [np.array(v) for v in st.values()]
        sites = [s for s in bysite if len(bysite[s]) >= minsite]; w = [len(bysite[s]) for s in sites]
        P = []
        for _ in range(npairs):
            s = rng.choices(sites, w)[0]; P.append(rng.sample(bysite[s], 2))
        P = np.array(P); self.a, self.b = P[:, 0], P[:, 1]
        self.codes = {k: encode(objs, k) for k in FACTS}
        self.sites = sites
    def relation(self, kind, p, level):
        fn = REL[kind][1]; F = self.F[level]
        vals = [str(fn(p, F[i], F[j])) for i, j in zip(self.a, self.b)]
        u = {v: i for i, v in enumerate(sorted(set(vals)))}
        return np.array([u[v] for v in vals]), len(u)
    def evaluate(self, kind, p, fact, level, nperm, early=True, rcache=None, strict=False):
        strata = self.strict_idx if (strict and fact not in ('type', 'typefull')) else self.site_idx
        if rcache is None: rcache = self.relation(kind, p, level)
        r, nr = rcache
        y, ok = pairfact(fact, self.codes[fact], self.a, self.b)
        if ok.sum() < 200 or nr < 2: return None
        ra = r[ok]; ya = y[ok]
        if len(set(ya.tolist())) < 2 or len(set(ra.tolist())) < 2: return None
        ny = 4 if fact in NUMERIC else 2
        o = mi_codes(ra, ya, nr, ny); ge = 0
        for k in range(nperm):
            c2 = site_perm(self.codes[fact], strata)
            y2, _ = pairfact(fact, c2, self.a, self.b)
            ge += mi_codes(ra, y2[ok], nr, ny) >= o
            if early and ge >= 5 and k >= 20: return o, (ge + 1) / (k + 2), int(ok.sum())
        return o, (ge + 1) / (nperm + 1), int(ok.sum())

_PTR = None
def _train_eval(x):
    """One relation spec, several facts: relation computed once."""
    global nrng
    g, (kind, p, items) = x; nrng = np.random.default_rng(SEED * 100000 + g)
    rc = _PTR.relation(kind, p, 'raw')
    out = []
    for h, fact in items:
        out.append((h, _PTR.evaluate(kind, p, fact, 'raw', 500, rcache=rc)))
    print('  train group %d done (%s, %d facts)' % (g, kind, len(items)), flush=True)
    return out

_PTE = None
def _follow_eval(x):
    global nrng
    h, (kind, p, fact), r = x; nrng = np.random.default_rng(SEED * 100000 + 50000 + h)
    r2 = _PTE.evaluate(kind, p, fact, 'raw', 1000)
    r3 = _PTR.evaluate(kind, p, fact, 'all', 500)
    rstrict = _PTR.evaluate(kind, p, fact, 'raw', 1000, strict=True)
    ok = (r2 is not None and r2[1] < 0.01) and (r3 is not None and r3[1] < 0.01)
    rfine = _PTR.evaluate(kind, p, fact, 'raw', 20000 if ok else 1000, early=False) if ok else None
    return r2, r3, rstrict, rfine

def make_arrow():
    kind = rng.choice(ACTIVE); p = REL[kind][2](rng); return kind, p, rng.choice(FACTS)

# ---------------------------------------------------------------- SET arrows
def set_stats_registry():
    S = {}
    def closers(G): return [F['close'] for F in G]
    S['distinct_closers_frac'] = lambda G: len(set(closers(G))) / len(G)
    S['all_middles_differ'] = lambda G: float(len({F['mid'] for F in G if F['mid']}) == sum(1 for F in G if F['mid'])) if any(F['mid'] for F in G) else None
    def run(G):
        v = sorted({max(F['nums']) for F in G if F['nums']})
        return None if len(v) < 2 else float(v == list(range(v[0], v[0] + len(v))))
    S['counts_form_run'] = run
    S['openers_all_same'] = lambda G: float(len({F['s'][0] for F in G}) == 1)
    S['first_sign_distinct_frac'] = lambda G: len({F['s'][0] for F in G}) / len(G)
    S['jar_frac'] = lambda G: np.mean([F['close'] == JAR for F in G])
    S['opener_frac'] = lambda G: np.mean([F['op'] is not None for F in G])
    def mj(G):
        v = [len(G[i]['A'] & G[j]['A']) / len(G[i]['A'] | G[j]['A']) for i in range(len(G)) for j in range(i + 1, len(G))]
        return float(np.mean(v))
    S['mean_pair_jaccard'] = mj
    S['any_identical_pair'] = lambda G: float(len({F['s'] for F in G}) < len(G))
    S['distinct_texts_frac'] = lambda G: len({F['s'] for F in G}) / len(G)
    S['sign_richness'] = lambda G: len(set().union(*[F['A'] for F in G])) / sum(F['L'] for F in G)
    S['len_sd'] = lambda G: float(np.std([F['L'] for F in G]))
    S['mean_len'] = lambda G: float(np.mean([F['L'] for F in G]))
    S['all_same_closer'] = lambda G: float(len(set(closers(G))) == 1)
    S['num_frac'] = lambda G: np.mean([bool(F['nums']) for F in G])
    S['distinct_numvals'] = lambda G: len({max(F['nums']) for F in G if F['nums']})
    S['rare_frac'] = lambda G: np.mean([bool(F['rare']) for F in G])
    S['shared_sign_all'] = lambda G: float(bool(set.intersection(*[F['A'] for F in G])))
    S['max_sign_share'] = lambda G: max(collections.Counter(x for F in G for x in F['A']).values()) / len(G)
    S['closer_entropy'] = lambda G: float(-sum((c / len(G)) * math.log(c / len(G)) for c in collections.Counter(closers(G)).values()))
    S['any_nested_pair'] = lambda G: float(any(nest({}, G[i], G[j]) for i in range(len(G)) for j in range(i + 1, len(G))))
    S['any_edit1_pair'] = lambda G: float(any(edit1({}, G[i], G[j]) for i in range(len(G)) for j in range(i + 1, len(G))))
    S['any_samemid_diffclose'] = lambda G: float(any(samemid_diffclose({}, G[i], G[j]) for i in range(len(G)) for j in range(i + 1, len(G))))
    S['mean_sum_mod7'] = lambda G: float(np.mean([F['sum'] % 7 for F in G]))
    S['sumsame_pairs'] = lambda G: float(np.mean([G[i]['sum'] == G[j]['sum'] for i in range(len(G)) for j in range(i + 1, len(G))]))
    S['second_sign_distinct'] = lambda G: (len({F['s'][1] for F in G if F['L'] > 2}) / sum(1 for F in G if F['L'] > 2)) if sum(1 for F in G if F['L'] > 2) >= 2 else None
    S['any_prefix1_pair'] = lambda G: float(any(prefix1({}, G[i], G[j]) for i in range(len(G)) for j in range(i + 1, len(G))))
    S['closer_in_class_frac'] = lambda G: np.mean([F['close'] in CLOSE for F in G])
    S['mean_freq'] = lambda G: float(np.mean([F['freq'] for F in G]))
    return S
SETSTATS = set_stats_registry()
SETNAMES = list(SETSTATS)
SIZEBANDS = {'all': (3, 10 ** 6), 'small': (3, 5), 'mid': (6, 15), 'large': (16, 10 ** 6)}
TYPEFILT = ['any', 'SEAL', 'nonSEAL']
HOMOG = ['type', 'emblem', 'material', 'shape', 'time', 'period', 'boss', 'cult']

def groups_for(objs, filt):
    """Return (list of index arrays per group, pool index array) for a site set; dedup option drops repeated texts within group."""
    g = collections.defaultdict(list)
    for i, o in enumerate(objs):
        f = o['f']
        if f['room'] is None: continue
        if filt['type'] == 'SEAL' and f['type'] != 'SEAL': continue
        if filt['type'] == 'nonSEAL' and f['type'] == 'SEAL': continue
        g[(f['site'], f['room'])].append(i)
    lo, hi = SIZEBANDS[filt['band']]
    G = {k: v for k, v in g.items() if lo <= len(v) <= hi}
    return G

class SetPool:
    def __init__(self, objs):
        self.objs = objs
        self.F = {'raw': [feats(o['raw']) for o in objs], 'all': [feats(o['all']) for o in objs]}
        self.codes = {k: encode(objs, k) for k in HOMOG}
    def stat_over(self, groups, name, level, dedup):
        fn = SETSTATS[name]; F = self.F[level]; vals = []
        for idx in groups:
            G = [F[i] for i in idx]
            if dedup:
                seen = set(); G2 = []
                for x in G:
                    if x['s'] not in seen: seen.add(x['s']); G2.append(x)
                G = G2
                if len(G) < 3: continue
            v = fn(G)
            if v is not None: vals.append(float(v))
        return vals
    def homog_over(self, groups, key, codes):
        out = []
        for idx in groups:
            c = codes[idx]; c = c[c >= 0]
            out.append(np.nan if len(c) < 2 else max(collections.Counter(c.tolist()).values()) / len(c))
        return np.array(out)
    def redeal(self, G):
        """Re-deal objects among groups of the same sizes within site AND within object type (each group keeps its type mix)."""
        groups = [np.array(v) for v in G.values()]
        strata = collections.defaultdict(list)
        for gi, idx in enumerate(groups):
            for i in idx: strata[(self.objs[i]['f']['site'], self.objs[i]['f']['type'])].append(gi)
        out = [np.empty(len(idx), int) for idx in groups]; fill = [0] * len(groups)
        for key, slots in strata.items():
            pool = np.array([i for gi, idx in enumerate(groups) for i in idx if (self.objs[i]['f']['site'], self.objs[i]['f']['type']) == key])
            pool = pool[nrng.permutation(len(pool))]
            for gi, i in zip(slots, pool): out[gi][fill[gi]] = i; fill[gi] += 1
        return out
    def evaluate(self, name, filt, level, dedup, nperm, early=True):
        G = groups_for(self.objs, filt)
        if len(G) < 4: return None
        groups = [np.array(v) for v in G.values()]
        obs = self.stat_over(groups, name, level, dedup)
        if len(obs) < 4: return None
        o = float(np.mean(obs)); ge = 0; le = 0; null = []
        for k in range(nperm):
            v = self.stat_over(self.redeal(G), name, level, dedup)
            if not v: continue
            m = float(np.mean(v)); null.append(m); ge += m >= o; le += m <= o
            if early and min(ge, le) >= 5 and k >= 20: break
        n = len(null); p2 = 2 * min((ge + 1) / (n + 1), (le + 1) / (n + 1))
        return o, min(1.0, p2), len(obs), (float(np.mean(null)) if null else None), ('+' if ge < le else '-')
    def evaluate_fact(self, name, filt, level, dedup, key, nperm, early=True):
        G = groups_for(self.objs, filt)
        if len(G) < 6: return None
        groups = [np.array(v) for v in G.values()]
        tv = []
        F = self.F[level]; fn = SETSTATS[name]
        for idx in groups:
            v = fn([F[i] for i in idx]); tv.append(np.nan if v is None else float(v))
        tv = np.array(tv)
        def corr(codes):
            hv = self.homog_over(groups, key, codes); ok = ~(np.isnan(tv) | np.isnan(hv))
            if ok.sum() < 6 or np.std(tv[ok]) == 0 or np.std(hv[ok]) == 0: return None
            from scipy.stats import spearmanr
            return float(spearmanr(tv[ok], hv[ok]).statistic)
        o = corr(self.codes[key])
        if o is None: return None
        bysite = collections.defaultdict(list)
        for i, ob in enumerate(self.objs): bysite[(ob['f']['site'], None if key == 'type' else ob['f']['type'])].append(i)
        site_idx = [np.array(v) for v in bysite.values()]
        ge = 0; n = 0
        for k in range(nperm):
            c = corr(site_perm(self.codes[key], site_idx))
            if c is None: continue
            n += 1; ge += abs(c) >= abs(o)
            if early and ge >= 5 and k >= 20: break
        return o, (ge + 1) / (n + 1), int((~np.isnan(tv)).sum())

def make_set_arrow():
    kind = 'set' if rng.random() < 0.6 else 'setfact'
    return dict(kind=kind, stat=rng.choice(SETNAMES), band=rng.choice(list(SIZEBANDS)), type=rng.choice(TYPEFILT),
                dedup=rng.random() < 0.5, key=rng.choice(HOMOG))

# ---------------------------------------------------------------- run
def main():
    t0 = time.time(); log = []
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.append(s)
    say('S-DARK-2 cycle %d seed %d control=%s | active relations %d: %s' % (CYCLE, SEED, CONTROL, len(ACTIVE), ','.join(ACTIVE)))
    TR = [o for o in OBJ if o['f']['site'] in ('Mohenjo-daro', 'Harappa')]; TE = [o for o in OBJ if o['f']['site'] not in ('Mohenjo-daro', 'Harappa')]
    PTR = Pool(TR, 40000); PTE = Pool(TE, 20000)
    say('objects %d | train %d (pairs %d) | held-out %d (pairs %d; sites %s)' % (len(OBJ), len(TR), len(PTR.a), len(TE), len(PTE.a), ','.join(PTE.sites)))
    # ---- PAIR arrows (train stage in parallel)
    global _PTR
    _PTR = PTR
    specs = [make_arrow() for _ in range(NPAIR)]
    groups = collections.OrderedDict()
    for h, (kind, p, fact) in enumerate(specs):
        groups.setdefault((kind, json.dumps(p, sort_keys=True)), []).append((h, fact))
    jobs = [(g, (kind, json.loads(pj), items)) for g, ((kind, pj), items) in enumerate(groups.items())]
    say('  %d distinct relation specs for %d arrows' % (len(jobs), NPAIR))
    trains = [None] * NPAIR
    with mp.get_context('fork').Pool(4) as pool:
        for out in pool.imap_unordered(_train_eval, jobs, chunksize=1):
            for h, r in out: trains[h] = r
    arrows = [dict(kind=k, p=p, fact=f, train=r) for (k, p, f), r in zip(specs, trains)]; surv = []
    global _PTE
    _PTE = PTE
    hits = [(h, spec, r) for h, (spec, r) in enumerate(zip(specs, trains)) if r and r[1] <= 0.0021]
    with mp.get_context('fork').Pool(4) as pool:
        follow = pool.map(_follow_eval, hits, chunksize=1)
    for (h, (kind, p, fact), r), (r2, r3, rstrict, rfine) in zip(hits, follow):
        ok = (r2 is not None and r2[1] < 0.01) and (r3 is not None and r3[1] < 0.01)
        bonf = rfine is not None and rfine[1] < 0.05 / NPAIR
        strict_ok = rstrict is not None and rstrict[1] < 0.01
        surv.append(dict(kind=kind, p=p, fact=fact, train=r, fine=rfine, strict=rstrict, held=r2, all=r3, ok=ok, bonf=bonf, strict_ok=strict_ok))
        say('  [%d] train hit %s %s x %s MI %.4f p %.4f n %d | fine p %s (bonf %s) | within-site-x-type null p %.4f | held %s | seq_all %s | %s' % (
            h, kind, p, fact, r[0], r[1], r[2], ('%.5f' % rfine[1]) if rfine else 'na', bonf, rstrict[1] if rstrict else -1,
            tuple(round(x, 4) for x in r2[:2]) if r2 else None, tuple(round(x, 4) for x in r3[:2]) if r3 else None,
            'REPLICATED' if ok else 'train-only'))
    tested = sum(1 for a in arrows if a['train'])
    say('PAIR: fired %d, testable %d, train p<0.002: %d, replicated(held-out & seq_all): %d, Bonferroni-at-%d: %d, survive within-site-x-type null: %d, all four: %d  [%.0fs]' % (
        NPAIR, tested, len(surv), sum(s['ok'] for s in surv), NPAIR, sum(s['bonf'] for s in surv), sum(s['strict_ok'] for s in surv),
        sum(s['ok'] and s['bonf'] and s['strict_ok'] for s in surv), time.time() - t0))
    say('  expected train hits under null ~ %.1f (0.002 x testable)' % (0.002 * tested))
    fam = collections.Counter((s['kind'], s['fact']) for s in surv)
    say('  hit families: %s' % dict(fam.most_common(12)))
    # ---- SET arrows
    SMD = SetPool([o for o in OBJ if o['f']['site'] == 'Mohenjo-daro']); SH = SetPool([o for o in OBJ if o['f']['site'] == 'Harappa'])
    SO = SetPool([o for o in OBJ if o['f']['site'] not in ('Mohenjo-daro', 'Harappa')])
    say('SET: MD groups %d | H groups %d | outside groups %d (any filter)' % (len(groups_for(SMD.objs, {'type': 'any', 'band': 'all'})),
        len(groups_for(SH.objs, {'type': 'any', 'band': 'all'})), len(groups_for(SO.objs, {'type': 'any', 'band': 'all'}))))
    sarrows = []; ssurv = []
    for h in range(NSET):
        a = make_set_arrow(); filt = {'type': a['type'], 'band': a['band']}
        if a['kind'] == 'set':
            r = SMD.evaluate(a['stat'], filt, 'raw', a['dedup'], 500)
            a['train'] = r; sarrows.append(a)
            if not r or r[1] > 0.0041: continue
            r2 = SH.evaluate(a['stat'], filt, 'raw', a['dedup'], 1000); r3 = SMD.evaluate(a['stat'], filt, 'all', a['dedup'], 500)
            r4 = SO.evaluate(a['stat'], filt, 'raw', a['dedup'], 1000)
            rfine = SMD.evaluate(a['stat'], filt, 'raw', a['dedup'], 5000, early=False)
            ok = r2 is not None and r2[1] < 0.05 and r2[4] == r[4] and r3 is not None and r3[1] < 0.01
            a.update(held=r2, all=r3, outside=r4, fine=rfine, ok=ok, bonf=rfine is not None and rfine[1] < 0.05 / NSET); ssurv.append(a)
            say('  [%d] set hit %s band=%s type=%s dedup=%s | MD obs %.3f null %.3f dir %s p %.4f (fine %.5f) | H %s | outside %s | seq_all %s | %s' % (
                h, a['stat'], a['band'], a['type'], a['dedup'], r[0], r[3], r[4], r[1], rfine[1] if rfine else -1,
                (round(r2[0], 3), round(r2[1], 4), r2[4]) if r2 else None, (round(r4[0], 3), round(r4[1], 3), r4[4]) if r4 else None,
                (round(r3[1], 4), r3[4]) if r3 else None, 'REPLICATED' if ok else 'MD-only'))
        else:
            r = SMD.evaluate_fact(a['stat'], filt, 'raw', False, a['key'], 500)
            a['train'] = r; sarrows.append(a)
            if not r or r[1] > 0.0041: continue
            r2 = SH.evaluate_fact(a['stat'], filt, 'raw', False, a['key'], 1000); r3 = SMD.evaluate_fact(a['stat'], filt, 'all', False, a['key'], 500)
            ok = r2 is not None and r2[1] < 0.05 and np.sign(r2[0]) == np.sign(r[0]) and r3 is not None and r3[1] < 0.01
            a.update(held=r2, all=r3, ok=ok, bonf=False); ssurv.append(a)
            say('  [%d] setfact hit %s ~ homog(%s) band=%s type=%s | MD rho %.3f p %.4f n %d | H %s | seq_all %s | %s' % (
                h, a['stat'], a['key'], a['band'], a['type'], r[0], r[1], r[2], tuple(round(x, 3) for x in r2[:2]) if r2 else None,
                tuple(round(x, 3) for x in r3[:2]) if r3 else None, 'REPLICATED' if ok else 'MD-only'))
    stested = sum(1 for a in sarrows if a['train'])
    say('SET: fired %d, testable %d, MD p<0.004: %d, replicated(Harappa same direction & seq_all): %d, Bonferroni: %d  [%.0fs]' % (
        NSET, stested, len(ssurv), sum(s['ok'] for s in ssurv), sum(s['bonf'] for s in ssurv), time.time() - t0))
    say('  expected MD hits under null ~ %.1f (0.004 x testable)' % (0.004 * stested))
    json.dump(dict(cycle=CYCLE, seed=SEED, control=CONTROL, active=ACTIVE, pair_arrows=arrows, pair_surv=surv, set_arrows=sarrows, set_surv=ssurv),
              open(OUT + '.json', 'w'), default=str)
    open(OUT + '.txt', 'w').write('\n'.join(log) + '\n')

if __name__ == '__main__':
    main()
