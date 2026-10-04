#!/usr/bin/env python3
"""LA-30 'the totals are in a hidden currency' (CONTEXT idea #1).

Model: a written total T = sum_i r[type_i] * q_i, where q_i = integer + sum of fraction
values, type_i = the commodity (logogram) the entry is counted in, r = exchange rates into
a common unit of value. Plain sum = all r = 1.

Differs from la6 (pairwise ratios between commodities inside entries, no totals): here
only written totals constrain the rates, and the rates are free per type.

la_sections(): KU-RO (and PO-TO-KU-RO) sections, same cutting rule as totals_test.py
  (entries = numbers since the previous KU-RO / KI-RO / PO-TO-KU-RO or the top).
  Type of a number = the last logogram (or NI / single *NNN sign used as a logogram)
  since the last word; 'bare' if a word intervened.
ur3_sections(): Ur III merchant accounts (CDLI ATF, scratchpad, not committed):
  goods lines followed by 'ku3-bi <silver>'; the section total is the sum of the scribe's
  own silver values, and the true price of each line is known.
Scorer: vectorised count of sections with |S - T| <= tol.
"""
import json, os, re
import numpy as np
from fractions import Fraction as Fr
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la30_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
MARK = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}
# conventional (site) fraction values, data only
SITE = {'J': 1/2, 'E': 1/4, 'F': 1/8, 'K': 1/16, 'D': 1/5, 'B': 1/3, 'A': 1/6, 'H': 1/6, 'JE': 3/4,
        'L2': 3/20, 'X': 3/10, 'Y': 1/4, 'L': 1/10, 'L4': 1/40, 'L3': 1/30, 'L6': 1/60, 'W': 1/24, 'DD': 1/10}
FGRID = [1/2, 1/3, 1/4, 1/5, 1/6, 1/8, 1/10, 1/12, 1/16, 1/20, 1/24, 1/30, 2/3, 3/4, 3/10, 3/20, 3/8]
# simple-rational rate grid p/q, p,q <= 12
RGRID = sorted({p / q for p in range(1, 13) for q in range(1, 13)})


def base_of(v):
    parts = [p.strip("'[]? ") for p in v.split('+')]
    for p in parts:
        q = p.lstrip('*')
        if q in ('GRA', 'OLE', 'OLIV', 'VIN', 'VINb', 'CYP', 'VIR', 'AROM', 'HIDE', 'CAP'):
            return 'VIN' if q == 'VINb' else q
    return parts[0]


def la_sections(terms=('KU-RO',), gran='full'):
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        T = ins['tokens']
        last = 0
        for i, t in enumerate(T):
            if t['t'] != 'word':
                continue
            w = '-'.join(t['s'])
            if w not in MARK:
                continue
            if w in terms:
                tot = None
                for j in range(i + 1, min(i + 4, len(T))):
                    if T[j]['t'] == 'num':
                        tot = T[j]; break
                    if T[j]['t'] == 'word':
                        break
                ents = []
                cur = 'bare'
                for k in range(last, i):
                    x = T[k]
                    if x['t'] == 'word':
                        if x['s'] == ['NI']:
                            cur = 'NI'
                        elif len(x['s']) == 1 and x['s'][0].startswith('*') and k + 1 < len(T) and T[k + 1]['t'] == 'num':
                            cur = x['s'][0]
                        else:
                            cur = 'bare'
                    elif x['t'] == 'logo':
                        cur = x['v'].strip("[]?' ") if gran == 'full' else base_of(x['v'])
                    elif x['t'] == 'num':
                        ents.append((cur, x['v'], list(x['frac'])))
                if tot is not None and ents:
                    out.append({'id': ins['id'], 'term': w, 'entries': ents,
                                'total': (tot['v'], list(tot['frac']))})
            last = i + 1
    # disambiguate repeated ids
    cnt = Counter()
    for s in out:
        cnt[s['id']] += 1
        s['key'] = s['id'] + '#%d' % cnt[s['id']]
    return out


class Design:
    """Turns sections into arrays: I[s,t] integer sums, C[s,t,l] fraction-letter counts,
    TI[s], TC[s,l]."""

    def __init__(self, secs, types=None, letters=None):
        self.secs = secs
        self.types = types or sorted({e[0] for s in secs for e in s['entries']})
        self.letters = letters or sorted({f for s in secs for q in s['entries'] + [s['total']]
                                          for f in (q[2] if len(q) == 3 else q[1])})
        ti = {t: k for k, t in enumerate(self.types)}
        li = {l: k for k, l in enumerate(self.letters)}
        n, K, L = len(secs), len(self.types), len(self.letters)
        self.I = np.zeros((n, K)); self.C = np.zeros((n, K, L))
        self.TI = np.zeros(n); self.TC = np.zeros((n, L))
        for a, s in enumerate(secs):
            for ty, v, fr in s['entries']:
                self.I[a, ti[ty]] += v
                for f in fr:
                    self.C[a, ti[ty], li[f]] += 1
            self.TI[a] = s['total'][0]
            for f in s['total'][1]:
                self.TC[a, li[f]] += 1
        self.ntypes_per_sec = (self.I + self.C.sum(2) > 0).sum(1)

    def S_T(self, R, F):
        """R: (m,K) rates, F: (m,L) fraction values -> S (m,n), T (m,n)."""
        Q = self.I[None] + np.einsum('nkl,ml->mnk', self.C, F)  # (m,n,K)
        S = np.einsum('mnk,mk->mn', Q, R)
        T = self.TI[None] + F @ self.TC.T
        return S, T


def balanced(S, T, mode):
    d = np.abs(S - T)
    if mode == 'exact':
        return d <= 1e-6
    if mode == 'round':  # within rounding: half a unit, or 1 % of a large total
        return d <= np.maximum(0.5, 0.01 * T) + 1e-9
    if mode == 'rel2':
        return d <= 0.02 * np.abs(T) + 1e-9
    raise ValueError(mode)


def rng_rates(rng, m, K, p1=0.5, grid=None, logrange=None):
    if logrange is not None:
        lo, hi = logrange
        R = np.exp(rng.uniform(np.log(lo), np.log(hi), (m, K)))
        return R
    g = np.array(grid if grid is not None else RGRID)
    R = g[rng.integers(0, len(g), (m, K))]
    R[rng.random((m, K)) < p1] = 1.0
    return R


# ---------------------------------------------------------------- Ur III control
NUMRE = re.compile(r"(\d+)/(\d+)\(disz\)|(\d+)\((disz|u|gesz2|gesz'u|asz|barig|ban2)\)")
SYS = {'disz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'asz': 1}


def parse_number(s):
    """Return (disz-type count incl. fractions, asz count, capacity in sila3 from barig/ban2, seen_cap)."""
    tot = Fr(0); asz = Fr(0); cap = Fr(0); seen_cap = False
    for m in NUMRE.finditer(s):
        if m.group(1):
            tot += Fr(int(m.group(1)), int(m.group(2)))
        else:
            n, u = int(m.group(3)), m.group(4)
            if u == 'barig':
                cap += 60 * n; seen_cap = True
            elif u == 'ban2':
                cap += 10 * n; seen_cap = True
            elif u == 'disz':
                tot += n
            else:
                asz += n * SYS[u]  # asz, u, gesz2, gesz'u: gur counts in a capacity context
    return tot + asz, asz, cap, seen_cap


def igi(s):
    m = re.search(r'igi-(\d+)\(disz\)-gal2', s)
    return Fr(1, int(m.group(1))) if m else Fr(0)


def silver_gin(s):
    """ku3-bi line -> silver in gin2 (None if unparsable)."""
    s = re.sub(r'[#!?\[\]<>]', '', s)
    s = s.replace('ku3-bi', '').replace('ku3-babbar', '').replace('ku3', '').strip()
    if 'x' in s.split() or '...' in s or ' n ' in ' %s ' % s:
        return None
    la2 = None
    if ' la2 ' in ' %s ' % s:
        s, la2 = s.split('la2', 1)
    val = Fr(0); ok = False
    # split by unit words
    for unit, mult in (('ma-na', Fr(60)), ('gin2', Fr(1)), ('sze', Fr(1, 180))):
        if unit in s:
            part, s = s.split(unit, 1)
            v = parse_number(part)[0]
            v += igi(part)
            val += v * mult; ok = True
    if s.strip():
        v = igi(s)
        if v:
            val += v; ok = True
        elif NUMRE.search(s):
            return None
    if la2:
        sub = Fr(0)
        for unit, mult in (('ma-na', Fr(60)), ('gin2', Fr(1)), ('sze', Fr(1, 180))):
            if unit in la2:
                part, la2 = la2.split(unit, 1)
                v = parse_number(part)[0]
                sub += (v + igi(part)) * mult
        if sub == 0:
            return None
        val -= sub
    return float(val) if ok and val > 0 else None


def goods(s):
    """goods line -> (commodity key, quantity) or None. Capacity in sila3, weight in ma-na,
    otherwise a count."""
    s = re.sub(r'[#!?\[\]<>]', '', s).strip()
    if '...' in s or s.startswith('ku3') or s.startswith('szunigin') or s.startswith('szu-nigin2'):
        return None
    toks = s.split()
    num = []; rest = []
    for i, t in enumerate(toks):
        if NUMRE.fullmatch(t) or re.fullmatch(r"\d+/\d+\(disz\)", t):
            if rest:
                return None
            num.append(t)
        else:
            rest.append(t)
    if not num or not rest:
        return None
    v, asz, cap, seen = parse_number(' '.join(num))
    disz = v - asz
    words = rest
    gur = 'gur' in words or (('sila3' in words or seen) and asz > 0)
    if 'gur' in words:
        words = words[:words.index('gur')]
        if not words:
            return None
    unit = 'n'
    if gur:
        unit = 'cap'; q = 300 * asz + cap + disz
        if words and words[0] == 'sila3':
            words = words[1:]
    elif words[0] == 'sila3':
        unit = 'cap'; words = words[1:]; q = v + cap
    elif words[0] == 'gu2':
        unit = 'wt'; words = words[1:]; q = 60 * v
        if words and words[0] == 'ma-na':
            return None
    elif words[0] == 'ma-na':
        unit = 'wt'; words = words[1:]; q = v
    elif words[0] == 'gin2':
        unit = 'wt'; words = words[1:]; q = v / 60
    elif seen:
        unit = 'cap'; q = cap + v
    else:
        q = v
    if not words or q <= 0:
        return None
    key = ' '.join(words[:2]) + '|' + unit
    return key, float(q)


def ur3_sections():
    p = os.path.join(CK, 'ur3_sections.json')
    if os.path.exists(p):
        return json.load(open(p))
    docs = json.load(open(os.path.join(CK, 'ur3_kubi_texts.json')))
    out = []
    for pid, L in docs.items():
        ents = []
        for i in range(1, len(L)):
            if not L[i].startswith('ku3-bi'):
                continue
            g = goods(L[i - 1]); sv = silver_gin(L[i])
            if g and sv:
                ents.append((g[0], g[1], sv))
        # inline 'N gu4 ku3-bi N gin2'
        for l in L:
            if ' ku3-bi ' in l and not l.startswith('ku3-bi'):
                a, b = l.split(' ku3-bi ', 1)
                g = goods(a); sv = silver_gin('ku3-bi ' + b)
                if g and sv:
                    ents.append((g[0], g[1], sv))
        if len(ents) >= 2:
            out.append({'id': pid, 'entries': ents})
    json.dump(out, open(p, 'w'))
    return out
