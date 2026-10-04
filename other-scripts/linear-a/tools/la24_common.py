#!/usr/bin/env python3
"""LA-24 shared code: THE CLERK RAN AN APPORTIONMENT ALGORITHM.

Hypothesis: the amounts in a list were produced by dividing a total among recipients by shares
(share classes = small integer weights), then rounding to the available units. Each list is
coded two ways and the shorter code wins (MDL):

  baseline  an adaptive per-list code that already knows the corpus amount distribution, the
            list's scale (log-normal around the running mean) and repeats (a cache).
  model     1 flag bit + choice of share alphabet S (subsets of {1,2,3,4,5,6,8} of size 1-3 with gcd 1, plus {1,2,4,8}, {1,2,3,6}, {1,2,3,4}: 50 alphabets) + choice of
            rule/grid + code of the unit share u (simplest p/q, Elias) or of the total T
            (largest remainder) + log2|S| bits per reproduced entry + misses coded by the
            baseline plus log2 C(n, m) to say which.
  rules     EXACT (x = u w; fractions absorb the remainder), FLOOR (D'Hondt-like divisor),
            ROUND (Sainte-Lague-like divisor), CEIL (Adams-like divisor), HAMIL (largest
            remainder, T = sum).

Datasets: Linear A lists (runs of >= 3 quantities of one commodity, cut at KU-RO/KI-RO),
Linear B lists (DAMOS lines of one commodity), Ur III ration lists (CDLI ATF, texts with
sze-ba; amounts in sila3; ATF dump kept in the scratchpad, not committed), planted lists.
Fraction letters are valued by a value set V (a hypothesis); the baseline codes the letters
themselves and does not depend on V.
"""
import ctypes, json, math, os, random, re, sys
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from itertools import combinations
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la24_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
UR_ATF = os.environ.get('UR_ATF', '')

# ------------------------------------------------------------------ fraction value sets
CONV = {'J': Fr(1, 2), 'E': Fr(1, 4), 'F': Fr(1, 8), 'K': Fr(1, 16), 'D': Fr(1, 5), 'B': Fr(1, 3),
        'A': Fr(1, 6), 'H': Fr(1, 6), 'JE': Fr(3, 4), 'L2': Fr(1, 16), 'DD': Fr(1, 10), 'X': Fr(1, 16),
        'L': Fr(1, 4), 'W': Fr(1, 16), 'Y': Fr(1, 8), 'L4': Fr(1, 32), 'L6': Fr(1, 16)}
# la1 / attack-1 larger-first binary example (FINDINGS, attack 1)
LA1BIN = {'L': Fr(7, 16), 'E': Fr(3, 8), 'J': Fr(5, 16), 'A': Fr(1, 4), 'F': Fr(1, 4), 'H': Fr(1, 4),
          'JE': Fr(1, 4), 'B': Fr(1, 8), 'D': Fr(1, 8), 'K': Fr(1, 8), 'Y': Fr(1, 8), 'L2': Fr(1, 16),
          'L6': Fr(1, 16), 'L4': Fr(1, 32), 'DD': Fr(1, 10), 'X': Fr(1, 16), 'W': Fr(1, 16)}
POOL = [Fr(1, k) for k in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)] + \
       [Fr(2, 3), Fr(3, 4), Fr(3, 8), Fr(5, 8), Fr(5, 16), Fr(3, 16), Fr(7, 16), Fr(2, 5), Fr(3, 10), Fr(5, 6)]


def random_V(rng):
    return {k: rng.choice(POOL) for k in CONV}


def val(amount, V):
    i, lets = amount
    return float(i + sum((V.get(l, Fr(1, 16)) for l in lets), Fr(0)))

# ------------------------------------------------------------------ share alphabets
ALPH = []
for _k in range(1, 4):
    for _c in combinations((1, 2, 3, 4, 5, 6, 8), _k):
        if math.gcd(*_c) == 1:
            ALPH.append(_c)
ALPH += [(1, 2, 4, 8), (1, 2, 3, 6), (1, 2, 3, 4)]
NS = len(ALPH)
_Sflat = np.array([w for a in ALPH for w in a], dtype=np.int32)
_Slen = np.array([len(a) for a in ALPH], dtype=np.int32)
_Soff = np.concatenate([[0], np.cumsum(_Slen)[:-1]]).astype(np.int32)
_LOGK = np.log2(_Slen.astype(float))
RULES = ['EXACT', 'FLOOR', 'ROUND', 'CEIL', 'HAMIL']

_lib = ctypes.CDLL(os.path.join(HERE, 'la24_core.so'))
_P = np.ctypeslib.ndpointer
_lib.fit_all.argtypes = [_P(np.float64), ctypes.c_int, _P(np.int32), _P(np.int32), _P(np.int32), ctypes.c_int,
                         ctypes.c_int, ctypes.c_double, _P(np.int32), _P(np.float64), _P(np.float64)]


def fit_all(x, rule, g):
    x = np.ascontiguousarray(x, dtype=np.float64)
    cov = np.zeros(NS, np.int32); cost = np.zeros(NS); ub = np.zeros(NS)
    _lib.fit_all(x, len(x), _Sflat, _Soff, _Slen, NS, rule, g, cov, cost, ub)
    return cov, cost, ub


_LG = np.array([math.lgamma(i + 1) for i in range(400)]) / math.log(2)


def log2comb(n, m):
    return _LG[n] - _LG[m] - _LG[n - m]


def model_best(x, ent_cost, grids):
    """Best model code length for one list. x: values; ent_cost: baseline bits per entry.
    Returns (bits, info dict)."""
    n = len(x)
    combos = [(0, grids[0])] + [(r, g) for r in (1, 2, 3, 4) for g in grids]
    hyp = 1.0 + math.log2(NS) + math.log2(len(combos))
    # grids finer than 1 can only reproduce entries with fractions; skip them for integer lists
    if grids[0] == 1.0 and max(grids[1:]) < 1 and all(abs(v - round(v)) < 1e-9 for v in x):
        combos = [c for c in combos if c[1] == grids[0]]
    med = float(np.median(ent_cost))
    best = (float('inf'), None)
    for r, g in combos:
        cov, cost, ub = fit_all(x, r, g)
        m = n - cov
        ok = cov >= 3
        if r == 4: ok &= (m == 0)
        if not ok.any(): continue
        bits = hyp + cost + cov * _LOGK + m * med + log2comb(n, m)
        bits = np.where(ok, bits, np.inf)
        s = int(np.argmin(bits))
        if bits[s] < best[0]:
            best = (float(bits[s]), {'rule': RULES[r], 'grid': g, 'S': ALPH[s], 'u': float(ub[s]),
                                     'cover': int(cov[s]), 'n': n})
    return best

# ------------------------------------------------------------------ baseline coder
class Baseline:
    """Adaptive code: P(x_i) = a*cache + b*scale + c*marg (i > 0); marg for i = 0.
    Tokens are hashable amount keys; numeric value (for scale) from a reference value set."""

    def __init__(self, lists, numval, sigma=0.6):
        cnt = Counter(t for L in lists for t in L)
        self.N = sum(cnt.values()); self.cnt = cnt
        self.toks = list(cnt); self.tid = {t: i for i, t in enumerate(self.toks)}
        self.pm = np.array([(cnt[t] + 0.5) / (self.N + 0.5 * len(cnt) + 1.0) for t in self.toks])
        self.pesc = 1.0 / (self.N + 0.5 * len(cnt) + 1.0)
        self.lv = np.log(np.array([max(numval(t), 1e-3) for t in self.toks]))
        self.numval = numval; self.sigma = sigma
        self.w = np.array([0.34, 0.33, 0.33])
        self.fit_weights(lists)

    def marg(self, t):
        i = self.tid.get(t)
        if i is not None: return self.pm[i]
        v = max(self.numval(t), 1e-3)
        return self.pesc * 2.0 ** (-(2 * math.floor(math.log2(max(1, int(v)))) + 1)) * 0.25

    def scale(self, t, m):
        k = np.exp(-(self.lv - m) ** 2 / (2 * self.sigma ** 2)) * self.pm
        Z = k.sum()
        i = self.tid.get(t)
        if i is None: return 1e-9
        return k[i] / Z

    def comps(self, L):
        out = []
        for i, t in enumerate(L):
            pm = self.marg(t)
            if i == 0: out.append((0.0, 0.0, pm, True)); continue
            cache = sum(1 for s in L[:i] if s == t) / i
            m = np.mean([math.log(max(self.numval(s), 1e-3)) for s in L[:i]])
            out.append((cache, self.scale(t, m), pm, False))
        return out

    def fit_weights(self, lists, iters=30):
        C = [c for L in lists for c in self.comps(L) if not c[3]]
        A = np.array([c[:3] for c in C])
        w = self.w.copy()
        for _ in range(iters):
            r = A * w; r /= r.sum(1, keepdims=True)
            w = r.mean(0)
        self.w = w

    def costs(self, L):
        out = []
        for c in self.comps(L):
            p = c[2] if c[3] else float(np.dot(self.w, c[:3]))
            out.append(-math.log2(max(p, 1e-300)))
        return np.array(out)

# ------------------------------------------------------------------ Linear A lists
TOTALW = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}
MAINC = {'GRA', 'OLE', 'OLIV', 'VIN', 'NI', 'CYP', '*304', 'VIR', '*308', 'AROM', 'HIDE', '*86', '*305', 'CAP', '*307'}


def la_lists(min_n=3):
    from la6_common import base_of
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        T = ins['tokens']
        key = 'BARE'; run = []; runkey = None; pending_total = False

        def close(total=None):
            nonlocal run
            if len(run) >= min_n and all(a[0] > 0 or a[1] for a in run):
                out.append({'id': ins['id'], 'site': ins['id'][:2], 'key': runkey, 'amts': list(run),
                            'total': total})
            run = []
        for i, t in enumerate(T):
            if t['t'] == 'word':
                w = '-'.join(t['s'])
                if w in TOTALW:
                    if w == 'KU-RO' or w == 'PO-TO-KU-RO':
                        pending_total = True
                    else:
                        close(); pending_total = False
                    key = 'BARE'; continue
                if w == 'NI': key = 'NI'; continue
                key = 'BARE'
            elif t['t'] == 'logo':
                b = base_of(t['v'])
                nd = T[i + 1] if i + 1 < len(T) else None
                key = b if (nd is not None and nd['t'] == 'num') or b in MAINC else 'BARE'
            elif t['t'] == 'num':
                a = (int(t['v']), tuple(sorted(t['frac'])))
                if pending_total:
                    close(a); pending_total = False; continue
                if run and key != runkey: close()
                runkey = key; run.append(a)
        close()
    return out


def la_numval(V=CONV):
    return lambda a: val(a, V)

# ------------------------------------------------------------------ Linear B lists
def lb_lists(min_n=3):
    from la6_common import lb_docs
    out = []
    for d in lb_docs():
        run = []; rk = None
        for l in d['lines'] + [None]:
            if l is not None and len(l) == 1:
                k, v = next(iter(l.items()))
                if v > 0 and (rk is None or k == rk):
                    run.append(v); rk = k; continue
            if len(run) >= min_n:
                out.append({'id': d['id'], 'site': d['site'], 'key': rk, 'amts': [(x, ()) for x in run], 'total': None})
            run = []; rk = None
            if l is not None and len(l) == 1:
                k, v = next(iter(l.items()))
                if v > 0: run = [v]; rk = k
    return out

# ------------------------------------------------------------------ Ur III ration lists
_UNIT = {'asz': 300, 'barig': 60, 'ban2': 10}
_numtok = re.compile(r"^(\d+)\(([a-z0-9']+)(?:@[a-z])?\)$")


def ur_qty(toks):
    """Leading capacity quantity in sila3, or None."""
    v = 0; pend = 0; seen = False; j = 0
    for j, t in enumerate(toks):
        t = t.strip('#!?*')
        m = _numtok.match(t)
        if m:
            n, u = int(m.group(1)), m.group(2)
            if u in ('barig', 'ban2'): v += n * _UNIT[u]; seen = True
            elif u == 'asz': pend += n
            elif u in ('disz', 'u'): pend += n * (10 if u == 'u' else 1)
            else: return None
            continue
        if t == 'sila3': v += pend; pend = 0; seen = True; continue
        if t == 'gur': v += pend * 300; pend = 0; seen = True; continue
        break
    if pend:
        return None
    if not seen or v <= 0: return None
    return v, toks[j:] if j < len(toks) else []


def ur_lists(min_n=3, max_lists=None, seed=0):
    if not UR_ATF or not os.path.exists(UR_ATF): return []
    texts = {}; pid = None; buf = []
    for raw in open(UR_ATF, encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            if pid and buf: texts[pid] = buf
            pid = raw.split()[0][1:]; buf = []; continue
        buf.append(raw.rstrip('\n'))
    if pid and buf: texts[pid] = buf
    out = []
    for pid, lines in texts.items():
        if not any('sze-ba' in l for l in lines): continue
        if not any(l.startswith('#atf: lang sux') for l in lines): continue
        run = []
        for l in lines + ['@end']:
            m = re.match(r"^\d+'?\.\s*(.*)$", l.strip())
            q = None
            if m:
                body = m.group(1)
                bad = ('[' in body or ']' in body or '...' in body or '-ta' in body or '/' in body
                       or 'szu-nigin' in body or 'szunigin' in body)
                toks = body.split()
                if not bad:
                    r = ur_qty(toks)
                    if r is not None and r[0] < 600: q = r[0]
            if q is not None:
                run.append(q); continue
            if m is None and l.startswith('$') is False and l.startswith('@') is False:
                continue
            if len(run) >= min_n:
                out.append({'id': pid, 'site': 'UR', 'key': 'sila', 'amts': [(x, ()) for x in run], 'total': None})
            run = []
    if max_lists and len(out) > max_lists:
        rng = random.Random(seed); out = rng.sample(out, max_lists)
    return out

# ------------------------------------------------------------------ scoring and nulls
def score_lists(lists, base, numval, grids):
    """Return list of (gain, info) for each list."""
    res = []
    for L in lists:
        toks = L['amts'] if isinstance(L, dict) else L
        x = np.array([numval(t) for t in toks])
        bc = base.costs(toks)
        mb, info = model_best(x, bc, grids)
        res.append((float(bc.sum() - mb), info, float(bc.sum())))
    return res


def band_pool(base, numval):
    """tokens sorted by value, with a 'has letters' flag, for band replacement."""
    toks = base.toks
    vals = np.array([numval(t) for t in toks])
    frac = np.array([bool(t[1]) if isinstance(t, tuple) else False for t in toks])
    o = np.argsort(vals)
    return [toks[i] for i in o], vals[o], frac[o], base.pm[o]


def null_band(toks, pool, numval, rng, band=0.25):
    """N3b: map each distinct value to a random corpus token within +-band in log value, same
    letter status, different from the original when possible; repeats are kept."""
    ptoks, pvals, pfrac, pw = pool
    mp = {}
    for t in toks:
        if t in mp: continue
        v = numval(t); f = bool(t[1]) if isinstance(t, tuple) else False
        lo, hi = np.searchsorted(pvals, v * math.exp(-band)), np.searchsorted(pvals, v * math.exp(band), 'right')
        cand = [i for i in range(lo, hi) if pfrac[i] == f and ptoks[i] != t]
        mp[t] = ptoks[rng.choice(cand)] if cand else t
    return [mp[t] for t in toks]


def null_shuffle(lists, rng):
    allt = [t for L in lists for t in L]
    rng.shuffle(allt)
    out = []; k = 0
    for L in lists:
        out.append(allt[k:k + len(L)]); k += len(L)
    return out


def null_marg(lists, base, rng):
    p = base.pm / base.pm.sum()
    idx = np.arange(len(base.toks))
    return [[base.toks[i] for i in rng.choice(idx, size=len(L), p=p)] for L in lists]


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for r, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * n / r); q[i] = prev
    return q

# ------------------------------------------------------------------ planting
def apportion(T, w, rule, g):
    W = sum(w); u = T / W
    if rule == 'EXACT': return [u * x for x in w]
    if rule == 'FLOOR': return [g * math.floor(u * x / g + 1e-9) for x in w]
    if rule == 'ROUND': return [g * math.floor(u * x / g + 0.5) for x in w]
    if rule == 'CEIL': return [g * math.ceil(u * x / g - 1e-9) for x in w]
    q = [u * x / g for x in w]; f = [math.floor(v + 1e-9) for v in q]
    R = int(round(T / g - sum(f)))
    order = sorted(range(len(w)), key=lambda i: -(q[i] - f[i]))
    for i in order[:R]: f[i] += 1
    return [g * v for v in f]


INV_CONV = {Fr(1, 2): ('J',), Fr(1, 4): ('E',), Fr(3, 4): ('JE',), Fr(0): ()}


def to_amount(v):
    fv = Fr(v).limit_denominator(16)
    i = int(fv); fr = fv - i
    if fr not in INV_CONV: return None
    return (i, INV_CONV[fr])


def plant_list(rng, n):
    rule = rng.choice(RULES)
    k = rng.choice([1, 2, 2, 3, 3])
    S = list(rng.choice([a for a in ALPH if len(a) == k]))
    w = [rng.choice(S) for _ in range(n)]
    for _ in range(50):
        if rule == 'EXACT':
            u = rng.choice([1, 2, 3, 4, 5, 6, 8, 10]) + rng.choice([0, 0, 0.5, 0.25])
            vals = [u * x for x in w]
        else:
            T = rng.randint(2 * n, 40 * n)
            vals = apportion(T, w, rule, 1.0)
        amts = [to_amount(v) for v in vals]
        if all(a is not None and (a[0] > 0 or a[1]) for a in amts):
            return {'amts': amts, 'rule': rule, 'S': S, 'w': w}
    return None


# ------------------------------------------------------------------ cycle 1b: one clerk, one rule
COMBOS = None


def combos_for(grids):
    return [(0, grids[0])] + [(r, g) for r in (1, 2, 3, 4) for g in grids]


def list_matrix(x, ent_cost, grids):
    """Model bits for every (rule/grid combo, alphabet), WITHOUT the hypothesis cost (paid once
    per corpus when one clerk's rule and alphabet are shared). inf where the fit is invalid."""
    n = len(x)
    cmb = combos_for(grids)
    M = np.full((len(cmb), NS), np.inf)
    med = float(np.median(ent_cost))
    intl = grids[0] == 1.0 and max(grids[1:]) < 1 and all(abs(v - round(v)) < 1e-9 for v in x)
    for ci, (r, g) in enumerate(cmb):
        if intl and g != grids[0]: continue
        cov, cost, ub = fit_all(x, r, g)
        m = n - cov
        ok = cov >= 2
        if r == 4: ok &= (m == 0)
        bits = 1.0 + cost + cov * _LOGK + m * med + log2comb(n, m)
        M[ci] = np.where(ok, bits, np.inf)
    return M


def null_band_w(toks, pool, numval, rng, band=0.25):
    """N3c: as N3b but the replacement is drawn with probability proportional to its corpus
    frequency (so the null keeps the corpus preference for common, round amounts)."""
    ptoks, pvals, pfrac, pw = pool
    mp = {}
    for t in toks:
        if t in mp: continue
        v = numval(t); f = bool(t[1]) if isinstance(t, tuple) else False
        lo, hi = np.searchsorted(pvals, v * math.exp(-band)), np.searchsorted(pvals, v * math.exp(band), 'right')
        cand = [i for i in range(lo, hi) if pfrac[i] == f and ptoks[i] != t]
        if cand:
            w = np.array([pw[i] for i in cand]); w = w / w.sum()
            mp[t] = ptoks[cand[int(np.searchsorted(np.cumsum(w), rng.random()))] if len(cand) > 1 else cand[0]]
        else:
            mp[t] = t
    return [mp[t] for t in toks]
