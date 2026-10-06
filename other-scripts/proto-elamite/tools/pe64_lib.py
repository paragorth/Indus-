#!/usr/bin/env python3
"""pe64 RECONSTRUCT THE RATION LADDER FROM EVERY PER-HEAD RATE: shared library.

Every pairing of a count line with a capacity line gives a per-unit rate r = capacity / count.
A rate-detection RULE picks a subset of these pairs (scope, target filter, how N01-only lines are read,
minimum count, source-sign filter, target-sign filter, exactness).  Its score is the number of distinct
tablets that show its most common rate.  Millions of rules are scored on real data and on nulls:
  N1 'repair'  : counts re-dealt among count lines and capacities among capacity lines of the same tablet
  N2 'swap'    : every capacity value replaced by a capacity value from another tablet, same notation shape
                 (keeps the PE number system's own 'nice' numbers, breaks the count -> capacity link)
Corpora share one shape: tablet = {'id', 'ord', 'hdr', 'lines': [{'s': [signs], 'cnt': int|None,
 'cap': Fraction-as-float|None, 'amb': bool, 'shape': str}]}
Signs are opaque to the method.  Truth (Ur III grade words, proto-cuneiform answer-key classes) is used
only to score controls.
"""
import os, sys, re, json, math, random, hashlib
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe64_ckpt')
os.makedirs(CK, exist_ok=True)

CAPV = {"N39C": 1, "N30D": 2, "N30C": 4, "N24": 12, "N39B": 24, "N01": 120, "N14": 720, "N45": 7200,
        "N34": 21600, "N48": 216000}
CNTV = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}
CAPSET = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N28', 'N29B', 'N39N'}


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ PE
def pe_corpus():
    fn = os.path.join(CK, 'pe.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    src = json.load(open(os.path.join(DATA, 'pe63_ckpt', 'pe_tabs.json')))
    out = []
    for t in src:
        L = []
        for l in t['lines']:
            d = {'s': l['s'], 'cnt': None, 'cap': None, 'amb': False, 'shape': ''}
            if l['sys'] and l['v'] is not None and l['raw']:
                pairs = re.findall(r'(\d+)\((\w+)\)', l['raw'])
                codes = {c for _, c in pairs}
                d['shape'] = '+'.join(sorted(codes))
                if l['sys'] in ('C',) and all(c in CAPV for c in codes):
                    d['cap'] = float(sum(int(n) * CAPV[c] for n, c in pairs))
                elif l['sys'] == 'SDB' and all(c in CNTV for c in codes):
                    d['cnt'] = int(sum(int(n) * CNTV[c] for n, c in pairs))
                    if codes <= {'N01', 'N14'}:          # written the same in both systems
                        d['amb'] = True
                        d['cap'] = float(sum(int(n) * CAPV[c] for n, c in pairs))
            L.append(d)
        hdr = t['lines'][0]['s'][0] if t['lines'][0]['s'] else 'NONE'
        out.append({'id': t['id'], 'ord': t['ord'], 'hdr': hdr, 'lines': L})
    json.dump(out, open(fn, 'w'))
    return out


# ------------------------------------------------------------------ Ur III control
def ur3_corpus():
    """Ur III ration lists rebuilt with the rate HIDDEN: each '<n> <grade> <r>-ta' line becomes a count line
    (n, grade words) followed by a capacity line (n*r, the word 'sze-bi'); the tablet's other real capacity
    lines (ur3cap) are inserted at random positions as distractors. Words are opaque to the method."""
    fn = os.path.join(CK, 'ur3.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    ta = json.load(open(os.path.join(DATA, 'pe58_ckpt', 'ur3ta.json')))
    cap = json.load(open(os.path.join(DATA, 'pe58_ckpt', 'ur3cap.json')))
    capby = defaultdict(list)
    for c in cap:
        capby[c['tab']].append(c)
    by = defaultdict(list)
    for x in ta:
        by[x['tab']].append(x)
    rng = random.Random(seed('pe64-ur3'))
    out = []
    for tab, xs in sorted(by.items()):
        L = []
        for x in xs:
            if x['n'] <= 0:
                continue
            L.append({'s': x['grade'][:4], 'cnt': int(x['n']), 'cap': None, 'amb': False, 'shape': 'cnt',
                      'truth_rate': x['rate']})
            L.append({'s': ['sze-bi'], 'cnt': None, 'cap': float(x['n'] * x['rate']), 'amb': False, 'shape': 'cap'})
        for c in capby.get(tab, [])[:8]:
            L.insert(rng.randrange(len(L) + 1), {'s': c['words'][:4], 'cnt': None, 'cap': float(c['q']),
                                                 'amb': False, 'shape': 'cap'})
        if L:
            out.append({'id': tab, 'ord': int(re.sub(r'\D', '', tab)), 'hdr': 'NONE', 'lines': L})
    json.dump(out, open(fn, 'w'))
    return out


# ------------------------------------------------------------------ proto-cuneiform control
def pc_corpus():
    fn = os.path.join(CK, 'pc.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    import pe59_lib as L59
    capm, cntm = L59.pc_maps()
    out = []
    for t in L59.build_pc():
        L = []
        for l in t['lines']:
            d = {'s': l['signs'][:12], 'cnt': None, 'cap': None, 'amb': False, 'shape': ''}
            if l.get('numclean') and l['nums']:
                cls = L59.ncls(l['nums'])
                a = L59.value(l['nums'], capm)
                b = L59.value(l['nums'], cntm['S'])
                d['shape'] = cls
                if cls == 'CAP' and a:
                    d['cap'] = float(a)
                elif cls == 'AMB' and b and a:
                    d['cnt'] = int(b) if b == int(b) else None
                    d['cap'] = float(a); d['amb'] = True
                elif cls in ('OTH', 'AMB') and b and b == int(b):
                    d['cnt'] = int(b)
            if d['s'] or d['cnt'] or d['cap']:
                L.append(d)
        if len(L) >= 2:
            hdr = L[0]['s'][0] if L[0]['s'] else 'NONE'
            out.append({'id': t['id'], 'ord': int(re.sub(r'\D', '', t['id']) or 0), 'hdr': hdr, 'lines': L})
    json.dump(out, open(fn, 'w'))
    return out


def load(name):
    return {'PE': pe_corpus, 'U3': ur3_corpus, 'PC': pc_corpus}[name]()


# ------------------------------------------------------------------ nulls and plants
def null_repair(T, rng):
    T = json.loads(json.dumps(T))
    for t in T:
        ci = [i for i, l in enumerate(t['lines']) if l['cnt'] and not l['amb']]
        ki = [i for i, l in enumerate(t['lines']) if l['cap'] is not None and not l['amb']]
        for idx, key in ((ci, 'cnt'), (ki, 'cap')):
            vals = [t['lines'][i][key] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                t['lines'][i][key] = v
    return T


def null_swap(T, rng):
    T = json.loads(json.dumps(T))
    pool = defaultdict(list)
    for ti, t in enumerate(T):
        for l in t['lines']:
            if l['cap'] is not None:
                pool[(l['shape'], l['amb'])].append((ti, l['cap'], l['cnt']))
    for ti, t in enumerate(T):
        for l in t['lines']:
            if l['cap'] is not None:
                P = pool[(l['shape'], l['amb'])]
                for _ in range(20):
                    tj, cap, cnt = P[rng.randrange(len(P))]
                    if tj != ti or len(P) < 3:
                        break
                l['cap'] = cap
                if l['amb']:
                    l['cnt'] = cnt
    return T


def plant_ladder(T, rng, rungs=(20, 40, 80), per=10, tag='PL'):
    """On a swap-null corpus: choose 3 marker signs (frequent count-line signs); on `per` tablets each,
    set the capacity line right after a count line carrying the marker to count * rung."""
    T = json.loads(json.dumps(T))
    sc = Counter(s for t in T for i, l in enumerate(t['lines'][:-1]) if l['cnt']
                 and t['lines'][i + 1]['cap'] is not None and not t['lines'][i + 1]['amb'] for s in set(l['s']))
    cand = [s for s, c in sc.most_common(80)[3:] if c >= per]
    markers = rng.sample(cand, len(rungs))
    truth = {}
    for m, r in zip(markers, rungs):
        tabs = [ti for ti, t in enumerate(T) if any(l['cnt'] and m in l['s'] and
                t['lines'][i + 1]['cap'] is not None and not t['lines'][i + 1]['amb']
                for i, l in enumerate(t['lines'][:-1]))]
        rng.shuffle(tabs)
        for ti in tabs[:per]:
            t = T[ti]
            for i, l in enumerate(t['lines'][:-1]):
                if l['cnt'] and m in l['s'] and t['lines'][i + 1]['cap'] is not None and not t['lines'][i + 1]['amb']:
                    t['lines'][i + 1]['cap'] = float(l['cnt'] * r)
                    break
        truth[m] = r
    return T, truth


# ------------------------------------------------------------------ pair table
SCOPES = ['next', 'next2', 'prev', 'win5', 'tablet', 'last', 'dossier']


def dossier_nb(T):
    by = sorted(range(len(T)), key=lambda i: T[i]['ord'])
    nb = defaultdict(list)
    for a, b in zip(by, by[1:]):
        if abs(T[a]['ord'] - T[b]['ord']) <= 3 and T[a]['hdr'] == T[b]['hdr'] and T[a]['hdr'] != 'NONE':
            nb[a].append(b); nb[b].append(a)
    return nb


def pairs(T):
    """All count -> capacity pairings with feature columns."""
    nb = dossier_nb(T)
    rows = []
    for ti, t in enumerate(T):
        L = t['lines']
        caps = [j for j, l in enumerate(L) if l['cap'] is not None]
        lastcap = caps[-1] if caps else None
        for i, l in enumerate(L):
            if not l['cnt'] or l['cnt'] < 1:
                continue
            for j in caps:
                if j == i:
                    continue
                d = j - i
                sc = 0
                if d == 1: sc |= 1
                if 1 <= d <= 2: sc |= 2
                if d == -1: sc |= 4
                if abs(d) <= 5: sc |= 8
                sc |= 16
                if j == lastcap and j > i: sc |= 32
                rows.append((ti, i, ti, j, sc))
            for tj in nb.get(ti, []):
                L2 = T[tj]['lines']
                for j in (i - 1, i, i + 1):
                    if 0 <= j < len(L2) and L2[j]['cap'] is not None and not L2[j]['amb']:
                        rows.append((ti, i, tj, j, 64))
    return rows


class PairTable:
    def __init__(self, T, rows=None):
        rows = pairs(T) if rows is None else rows
        self.T = T
        self.rows = rows
        n = len(rows)
        self.tab = np.array([r[0] for r in rows], dtype=np.int32)
        self.scope = np.array([r[4] for r in rows], dtype=np.int32)
        sl = [T[r[0]]['lines'][r[1]] for r in rows]
        tl = [T[r[2]]['lines'][r[3]] for r in rows]
        self.cnt = np.array([l['cnt'] for l in sl], dtype=np.float64)
        self.cap = np.array([l['cap'] for l in tl], dtype=np.float64)
        self.samb = np.array([l['amb'] for l in sl], dtype=bool)
        self.tamb = np.array([l['amb'] for l in tl], dtype=bool)
        self.rate = self.cap / self.cnt
        self.rkey = np.round(self.rate * 1000).astype(np.int64)
        self.exact = np.abs(self.rate - np.round(self.rate)) < 1e-9
        self.ssig = [frozenset(l['s']) for l in sl]
        self.tsig = [frozenset(l['s']) for l in tl]
        self.t288 = np.array(['M288' in s for s in self.tsig], dtype=bool)
        self.sidx = defaultdict(list)
        self.tidx = defaultdict(list)
        for k in range(n):
            for s in self.ssig[k]:
                self.sidx[s].append(k)
            for s in self.tsig[k]:
                self.tidx[s].append(k)
        self.sidx = {s: np.array(v, dtype=np.int64) for s, v in self.sidx.items()}
        self.tidx = {s: np.array(v, dtype=np.int64) for s, v in self.tidx.items()}

    def revalue(self, T2):
        """same pairs, values taken from a null corpus with identical line structure"""
        P = object.__new__(PairTable)
        P.__dict__.update(self.__dict__)
        P.T = T2
        sl = [T2[r[0]]['lines'][r[1]] for r in self.rows]
        tl = [T2[r[2]]['lines'][r[3]] for r in self.rows]
        P.cnt = np.array([l['cnt'] or 0 for l in sl], dtype=np.float64)
        P.cap = np.array([l['cap'] if l['cap'] is not None else 0 for l in tl], dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            P.rate = np.where(P.cnt > 0, P.cap / np.maximum(P.cnt, 1e-9), -1)
        P.rkey = np.round(P.rate * 1000).astype(np.int64)
        P.exact = np.abs(P.rate - np.round(P.rate)) < 1e-9
        return P


# ------------------------------------------------------------------ rules
def rule_space(P, K=60, rng=None, n_random=0):
    """Family A: exhaustive grid. Family B: random OR-sets of source signs with random other settings."""
    srcs = [s for s, _ in sorted(((s, len(np.unique(P.tab[v]))) for s, v in P.sidx.items()), key=lambda x: -x[1])[:K]]
    tgts = [s for s, _ in sorted(((s, len(np.unique(P.tab[v]))) for s, v in P.tidx.items()), key=lambda x: -x[1])[:K]]
    R = []
    for sc in range(7):
        for tf in range(3):
            for amb in range(2):
                for cmin in (1, 2, 3):
                    for ex in range(2):
                        for s in [None] + srcs:
                            for g in [None] + tgts[:20]:
                                R.append((sc, tf, amb, cmin, ex, (s,) if s else (), g))
    if n_random and rng is not None:
        allsrc = [s for s, v in P.sidx.items() if len(v) >= 2]
        for _ in range(n_random):
            k = rng.randint(2, 5)
            R.append((rng.randrange(7), rng.randrange(3), rng.randrange(2), rng.choice((1, 2, 3)), rng.randrange(2),
                      tuple(sorted(rng.sample(allsrc, k))), rng.choice([None] * 3 + tgts[:20])))
    return R


def rule_mask(P, rule, cache):
    sc, tf, amb, cmin, ex, srcs, g = rule
    key = (sc, tf, amb, cmin, ex)
    base = cache.get(key)
    if base is None:
        m = (P.scope & (1 << sc)) > 0
        if tf == 1: m &= P.t288
        if tf == 2: m &= ~P.t288
        if not amb: m &= ~P.tamb
        m &= ~(P.samb & P.tamb)          # never read one N01 line as both count and capacity
        m &= P.cnt >= cmin
        m &= P.rate > 0
        if ex: m &= P.exact
        base = np.flatnonzero(m)
        cache[key] = base
    idx = base
    if srcs:
        sel = np.unique(np.concatenate([P.sidx.get(s, np.zeros(0, np.int64)) for s in srcs]))
        idx = np.intersect1d(idx, sel, assume_unique=True)
    if g:
        idx = np.intersect1d(idx, P.tidx.get(g, np.zeros(0, np.int64)), assume_unique=True)
    return idx


def score(P, idx):
    """(k_top, rate_top, m_tabs): distinct tablets showing the commonest rate (one vote per tablet per rate)"""
    if len(idx) == 0:
        return 0, 0, 0
    tr = np.unique(P.tab[idx].astype(np.int64) * 10**12 + P.rkey[idx])
    rk = tr % 10**12
    vals, cts = np.unique(rk, return_counts=True)
    j = int(np.argmax(cts))
    return int(cts[j]), int(vals[j]), int(len(np.unique(P.tab[idx])))


def rate_tabs(P, idx):
    """rate key -> set of tablets"""
    d = defaultdict(set)
    for k in idx:
        d[int(P.rkey[k])].add(int(P.tab[k]))
    return d


def split(T, tag):
    A, B = [], []
    for i, t in enumerate(T):
        (A if int(hashlib.sha256((tag + t['id']).encode()).hexdigest(), 16) % 2 == 0 else B).append(i)
    return A, B


def count_at(P, idx, rk):
    """distinct tablets where the selected pairs show rate key rk"""
    if len(idx) == 0:
        return 0
    sel = idx[P.rkey[idx] == rk]
    return int(len(np.unique(P.tab[sel])))
