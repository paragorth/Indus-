#!/usr/bin/env python3
"""LA-44 shared code: RESURRECT THE ANCESTOR BY SIMULATING HISTORY.

Each simulation draws a random proto-language (consonant and vowel inventory, syllable structure, morphology type,
lexicon size and Zipf use), generates a word list, lets it evolve under a random sequence of regular sound changes
plus borrowing (early loans that get inflected, late loans that do not), and writes the descendant in a CV syllabary
(13 consonant series x 5 vowels, the la10 sign alphabet) under a random spelling convention (which codas and cluster
consonants are dropped, which dead vowel is used, random series reassignments).  The written type list is summarised
by a fixed panel (la10 fingerprint + value-free extras); approximate Bayesian computation (rejection and random-forest
ABC) reads off the posterior over the ancestor's typology for a target.

Morphology classes (the main inference target):
  0 ISOL      bare roots (+ compounds)
  1 AGG_SUF   agglutinative suffix slots
  2 AGG_PRE   agglutinative prefix slots
  3 AGG_BOTH  prefix and suffix slots
  4 FUS_SUF   inflection classes, one obligatory portmanteau ending (Greek-like)
  5 TEMPL     3-consonant roots interdigitated with vowel patterns (+ optional affixes)
Syllable classes: 0 CV, 1 CVC (codas, no onset clusters), 2 CCVC (onset clusters, codas optional).
"""
import os, sys, math, random, collections, json, bisect
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ['LA10_V2'] = '1'
import la10_common as T10

MORPH = ['ISOL', 'AGG_SUF', 'AGG_PRE', 'AGG_BOTH', 'FUS_SUF', 'TEMPL']
SYL = ['CV', 'CVC', 'CCVC']
CK = os.path.join(HERE, '..', 'data', 'la44_ckpt')
os.makedirs(CK, exist_ok=True)

# ---------------------------------------------------------------- segment universe
# consonants: (name, place, manner, voice, weight). place 0 lab 1 cor 2 pal 3 vel 4 labvel 5 glot 6 uvul
# manner 0 stop 1 fric 2 affr 3 nasal 4 lat 5 rhot 6 glide
CU = [('p', 0, 0, 0, 10), ('b', 0, 0, 1, 6), ('t', 1, 0, 0, 10), ('d', 1, 0, 1, 6), ('k', 3, 0, 0, 10), ('g', 3, 0, 1, 6),
      ('q', 6, 0, 0, 2), ('kw', 4, 0, 0, 2), ('gw', 4, 0, 1, 1), ('?', 5, 0, 0, 3), ('c', 2, 0, 0, 2),
      ('f', 0, 1, 0, 3), ('v', 0, 1, 1, 2), ('s', 1, 1, 0, 9), ('z', 1, 1, 1, 3), ('sh', 2, 1, 0, 4), ('x', 3, 1, 0, 3),
      ('h', 5, 1, 0, 6), ('gh', 3, 1, 1, 2), ('ts', 1, 2, 0, 3), ('ch', 2, 2, 0, 4), ('dz', 1, 2, 1, 1), ('j_', 2, 2, 1, 3),
      ('m', 0, 3, 1, 10), ('n', 1, 3, 1, 10), ('ny', 2, 3, 1, 3), ('ng', 3, 3, 1, 4), ('l', 1, 4, 1, 8), ('ly', 2, 4, 1, 1),
      ('r', 1, 5, 1, 7), ('w', 4, 6, 1, 7), ('y', 2, 6, 1, 7)]
NC = len(CU)
CB = 100  # consonant code = CB + index
CFEAT = {(p, m, v): i for i, (_, p, m, v, _) in enumerate(CU)}
# vowels: (name, height 0 high..3 low, back 0 front 1 central 2 back, round)
VU = [('a', 3, 1, 0), ('i', 0, 0, 0), ('u', 0, 2, 1), ('e', 1, 0, 0), ('o', 1, 2, 1), ('E', 2, 0, 0), ('O', 2, 2, 1),
      ('1', 0, 1, 0), ('@', 1, 1, 0), ('y', 0, 0, 1), ('2', 1, 0, 1), ('ae', 3, 0, 0)]
NVU = len(VU)
FRONTV = {i for i, v in enumerate(VU) if v[2] == 0}
def isv(x): return x < CB
def cman(x): return CU[x - CB][2]
def cplace(x): return CU[x - CB][1]
def obstr(x): return cman(x) <= 2
def sonor(x): return cman(x) >= 3 or CU[x - CB][0] == 's'


def wchoice(rnd, items, w):
    return rnd.choices(items, w)[0]


# ---------------------------------------------------------------- proto-language
class Lang:
    pass


def draw_inventory(rnd, L):
    nc = rnd.randint(6, 26)
    idx = list(range(NC)); w = [c[4] * rnd.uniform(0.3, 1.7) for c in CU]
    inv = set()
    inv.add(rnd.choice([0, 2, 4])); inv.add(rnd.choice([23, 24]))
    while len(inv) < nc:
        inv.add(wchoice(rnd, idx, w))
    L.cons = sorted(CB + i for i in inv)
    nv = rnd.randint(3, 8)
    vs = [0, 1, 2]
    rest = [3, 4] + rnd.sample(range(5, NVU), NVU - 5)
    if nv == 4 and rnd.random() < 0.5: rest = [rnd.choice([3, 4, 7, 8])] + rest
    for v in rest:
        if len(vs) >= nv: break
        if v not in vs: vs.append(v)
    L.vows = vs
    # use frequencies: Dirichlet-ish
    L.cw = [rnd.gammavariate(0.8, 1) * CU[c - CB][4] for c in L.cons]
    L.vw = [rnd.gammavariate(1.0, 1) for _ in L.vows]
    L.vw[0] *= 1.5


def draw_phonotactics(rnd, L, syl=None):
    L.syl = rnd.randrange(3) if syl is None else syl
    L.p0i = rnd.uniform(0, 0.5); L.p0m = rnd.uniform(0, 0.25)
    if L.syl == 0: L.pcl = rnd.uniform(0, 0.03); L.pcoda = rnd.uniform(0, 0.05); L.pfin = rnd.uniform(0, 0.05)
    elif L.syl == 1: L.pcl = rnd.uniform(0, 0.03); L.pcoda = rnd.uniform(0.15, 0.6); L.pfin = rnd.uniform(0.15, 0.85)
    else: L.pcl = rnd.uniform(0.12, 0.5); L.pcoda = rnd.uniform(0.0, 0.5); L.pfin = rnd.uniform(0.0, 0.85)
    L.rootlen = rnd.uniform(1.0, 3.0)


def cons(rnd, L, pos='on'):
    c = wchoice(rnd, L.cons, L.cw)
    return c


def cluster_onset(rnd, L):
    obs = [c for c in L.cons if obstr(c)]; son = [c for c in L.cons if cman(c) in (3, 4, 5, 6)]
    s = [c for c in L.cons if CU[c - CB][0] == 's']
    if s and rnd.random() < 0.3 and obs: return [s[0], rnd.choice(obs)]
    if obs and son: return [rnd.choice(obs), rnd.choice(son)]
    return [cons(rnd, L)]


def syllable(rnd, L, initial, final):
    out = []
    if rnd.random() >= (L.p0i if initial else L.p0m):
        if rnd.random() < L.pcl: out += cluster_onset(rnd, L)
        else: out.append(cons(rnd, L))
    out.append(wchoice(rnd, L.vows, L.vw))
    if rnd.random() < (L.pfin if final else L.pcoda):
        son = [c for c in L.cons if sonor(c)]
        out.append(rnd.choice(son) if son and rnd.random() < 0.6 else cons(rnd, L))
    return out


def root(rnd, L, nsyl=None):
    if nsyl is None:
        nsyl = max(1, int(round(rnd.gauss(L.rootlen, 0.7))))
    out = []
    for i in range(nsyl): out += syllable(rnd, L, i == 0, i == nsyl - 1)
    return out


def affix(rnd, L, side):
    shape = wchoice(rnd, ['V', 'CV', 'VC', 'CVC', 'CVCV'], [2, 5, 3, 2, 1] if side == 'suf' else [2, 6, 2, 2, 1])
    v = lambda: wchoice(rnd, L.vows, L.vw); c = lambda: cons(rnd, L)
    return [v() if ch == 'V' else c() for ch in shape]


def ending(rnd, L):
    shape = wchoice(rnd, ['V', 'VC', 'VCV', 'VCVC', 'CV', 'VV'], [3, 4, 1.5, 1, 1, 0.7])
    v = lambda: wchoice(rnd, L.vows, L.vw)
    son = [c for c in L.cons if sonor(c)] or L.cons
    c = lambda: rnd.choice(son) if rnd.random() < 0.6 else cons(rnd, L)
    return [v() if ch == 'V' else c() for ch in shape]


def draw_morph(rnd, L, morph=None):
    L.morph = rnd.randrange(6) if morph is None else morph
    m = L.morph
    L.pcomp = rnd.uniform(0, 0.25) if m == 0 else rnd.uniform(0, 0.05)
    L.pre_slots = []; L.suf_slots = []
    if m in (1, 3):
        for _ in range(rnd.randint(1, 3 if m == 1 else 2)):
            L.suf_slots.append(([affix(rnd, L, 'suf') for _ in range(rnd.randint(2, 6))], rnd.uniform(0.2, 0.9)))
    if m in (2, 3):
        for _ in range(rnd.randint(1, 3 if m == 2 else 2)):
            L.pre_slots.append(([affix(rnd, L, 'pre') for _ in range(rnd.randint(2, 6))], rnd.uniform(0.2, 0.9)))
    if m == 4:
        L.nclass = rnd.randint(2, 4)
        L.endings = [[ending(rnd, L) for _ in range(rnd.randint(4, 12))] for _ in range(L.nclass)]
        L.pzero = rnd.uniform(0, 0.2); L.pstemC = rnd.uniform(0.3, 0.95)
        L.deriv = [affix(rnd, L, 'suf') for _ in range(rnd.randint(2, 5))]; L.pderiv = rnd.uniform(0, 0.3)
    if m == 5:
        L.patterns = []
        for _ in range(rnd.randint(4, 10)):
            pat = ['C1']
            for k in ('C2', 'C3'):
                if rnd.random() < 0.8: pat.append(wchoice(rnd, L.vows, L.vw))
                if rnd.random() < 0.15: pat.append(wchoice(rnd, L.vows, L.vw))
                pat.append(k)
            if rnd.random() < 0.6: pat.append(wchoice(rnd, L.vows, L.vw))
            L.patterns.append(pat)
        L.tpre = [affix(rnd, L, 'pre') for _ in range(rnd.randint(1, 4))]; L.ptpre = rnd.uniform(0, 0.5)
        L.tsuf = [affix(rnd, L, 'suf') for _ in range(rnd.randint(1, 5))]; L.ptsuf = rnd.uniform(0, 0.5)


def make_roots(rnd, L, nroots, donor=None):
    rs = []
    for _ in range(nroots):
        src = donor if (donor is not None and rnd.random() < L.pborrow) else L
        if L.morph == 5 and src is L:
            r = [cons(rnd, L) for _ in range(3)]
        else:
            r = root(rnd, src)
            if L.morph == 5:  # loans into a templatic language: consonantal skeleton of the loan
                cs = [x for x in r if not isv(x)] + [cons(rnd, L) for _ in range(3)]
                r = cs[:3]
        if L.morph == 4 and rnd.random() < L.pstemC and isv(r[-1]) and len(r) > 2:
            r = r[:-1]
        rs.append(r)
    return rs


def zipf_w(n, s):
    return [1.0 / (k + 1) ** s for k in range(n)]


_ZC = {}
def zpick(rnd, n, s):
    key = (n, s)
    cw = _ZC.get(key)
    if cw is None:
        cw = []; acc = 0.0
        for k in range(n): acc += 1.0 / (k + 1) ** s; cw.append(acc)
        _ZC[key] = cw
    return bisect.bisect_left(cw, rnd.random() * cw[-1])


def make_form(rnd, L, roots, rw, ri, cls):
    r = roots[ri]; m = L.morph
    if m == 5:
        pat = rnd.choice(L.patterns); w = []
        for p in pat:
            if p == 'C1': w.append(r[0])
            elif p == 'C2': w.append(r[1])
            elif p == 'C3': w.append(r[2])
            else: w.append(p)
        if rnd.random() < L.ptpre: w = rnd.choice(L.tpre) + w
        if rnd.random() < L.ptsuf: w = w + rnd.choice(L.tsuf)
        return w
    w = list(r)
    if rnd.random() < L.pcomp: w = w + roots[bisect.bisect_left(rw, rnd.random() * rw[-1])]
    if m == 4:
        if rnd.random() < L.pderiv: w = w + rnd.choice(L.deriv)
        if rnd.random() >= L.pzero:
            e = L.endings[cls[ri]]
            w = w + e[zpick(rnd, len(e), 0.8)]
        return w
    for slot, p in L.pre_slots:
        if rnd.random() < p: w = slot[zpick(rnd, len(slot), 0.7)] + w
    for slot, p in L.suf_slots:
        if rnd.random() < p: w = w + slot[zpick(rnd, len(slot), 0.7)]
    return w


# ---------------------------------------------------------------- sound change
def find_c(place, manner, voice):
    return CFEAT.get((place, manner, voice))


def nv(w): return sum(1 for x in w if isv(x))


def sc_factory(rnd, L):
    """returns (name, function list->list) for one random regular sound change; may update L inventories."""
    kind = rnd.choice(['apocope', 'finC', 'syncope', 'clsimp', 'epenth', 'lenit', 'palat', 'cmerge', 'vmerge', 'initC',
                       'prothesis', 'vassim', 'vshift', 'hloss', 'nasloss', 'paragoge', 'metath', 'hiatus'])
    vows = list(L.vows); conss = list(L.cons)
    if kind == 'apocope':
        Vs = set(rnd.sample(vows, rnd.randint(1, len(vows))))
        def f(w):
            if w and isv(w[-1]) and w[-1] in Vs and nv(w) >= 2: return w[:-1]
            return w
    elif kind == 'finC':
        cl = rnd.choice(['all', 'stop', 'son', 'nas_s'])
        ok = {'all': lambda c: True, 'stop': lambda c: cman(c) == 0, 'son': sonor,
              'nas_s': lambda c: cman(c) == 3 or CU[c - CB][0] == 's'}[cl]
        def f(w):
            while w and not isv(w[-1]) and ok(w[-1]) and nv(w) >= 1: w = w[:-1]
            return w
    elif kind == 'syncope':
        Vs = set(rnd.sample(vows, rnd.randint(1, len(vows))))
        pos = rnd.choice([1, 2])
        def f(w):
            vi = [i for i, x in enumerate(w) if isv(x)]
            if len(vi) >= 3 and pos < len(vi) - 1:
                i = vi[pos]
                if w[i] in Vs and i > 0 and not isv(w[i - 1]) and i + 1 < len(w) and not isv(w[i + 1]):
                    return w[:i] + w[i + 1:]
            return w
    elif kind == 'clsimp':
        mode = rnd.choice(['del1', 'del2', 'assim'])
        def f(w):
            out = []; i = 0
            while i < len(w):
                if i + 1 < len(w) and not isv(w[i]) and not isv(w[i + 1]):
                    if mode == 'del1': i += 1; continue
                    if mode == 'del2': out.append(w[i]); i += 2; continue
                    out.append(w[i + 1]); i += 2; continue
                out.append(w[i]); i += 1
            return out
    elif kind == 'epenth':
        mode = rnd.choice(['fixed', 'copy']); ev = rnd.choice(vows)
        where = rnd.choice(['all', 'init', 'final'])
        def f(w):
            out = []
            for i, x in enumerate(w):
                out.append(x)
                if i + 1 < len(w) and not isv(x) and not isv(w[i + 1]):
                    if where == 'init' and any(isv(y) for y in w[:i]): continue
                    if where == 'final': continue
                    nxt = next((y for y in w[i + 1:] if isv(y)), ev)
                    out.append(ev if mode == 'fixed' else nxt)
            if where == 'final' and out and not isv(out[-1]): out.append(ev)
            return out
    elif kind == 'lenit':
        to = rnd.choice(['voice', 'fric'])
        mp = {}
        for c in conss:
            if cman(c) == 0:
                p, m, v = CU[c - CB][1:4]
                t = find_c(p, 0, 1) if to == 'voice' else (find_c(p, 1, v) or find_c(p, 1, 0))
                if t is not None: mp[c] = CB + t
        def f(w):
            return [mp.get(x, x) if 0 < i < len(w) - 1 and isv(w[i - 1]) and isv(w[i + 1]) else x for i, x in enumerate(w)]
        L.cons = sorted(set(L.cons) | set(mp.values())); L.cw = [1.0] * len(L.cons)
    elif kind == 'palat':
        mp = {}
        for c in conss:
            if cman(c) == 0 and cplace(c) in (1, 3):
                mp[c] = CB + (find_c(2, 2, CU[c - CB][3]) or find_c(2, 2, 0))
        def f(w):
            return [mp.get(x, x) if i + 1 < len(w) and w[i + 1] in FRONTV and isv(w[i + 1]) else x for i, x in enumerate(w)]
        L.cons = sorted(set(L.cons) | set(mp.values())); L.cw = [1.0] * len(L.cons)
    elif kind == 'cmerge':
        a, b = rnd.sample(conss, 2) if len(conss) > 2 else (conss[0], conss[0])
        def f(w): return [b if x == a else x for x in w]
        L.cons = [c for c in L.cons if c != a] or L.cons; L.cw = [1.0] * len(L.cons)
    elif kind == 'vmerge':
        if len(vows) > 2:
            a = rnd.choice(vows[1:]); b = min((v for v in vows if v != a),
                                              key=lambda v: abs(VU[v][1] - VU[a][1]) + abs(VU[v][2] - VU[a][2]))
        else: a = b = vows[0]
        def f(w): return [b if x == a else x for x in w]
        L.vows = [v for v in L.vows if v != a] or L.vows; L.vw = [1.0] * len(L.vows)
    elif kind == 'initC':
        cl = rnd.choice(['glot', 'glide', 'rand'])
        S = {'glot': {c for c in conss if cplace(c) == 5}, 'glide': {c for c in conss if cman(c) == 6},
             'rand': set(rnd.sample(conss, min(len(conss), rnd.randint(1, 3))))}[cl]
        def f(w): return w[1:] if w and w[0] in S and len(w) > 1 else w
    elif kind == 'prothesis':
        ev = rnd.choice(vows)
        def f(w):
            if len(w) > 1 and not isv(w[0]) and (not isv(w[1]) or cman(w[0]) == 5): return [ev] + w
            return w
    elif kind == 'vassim':
        Vs = set(rnd.sample(vows, rnd.randint(1, max(1, len(vows) // 2))))
        def f(w):
            out = list(w); prev = None
            for i, x in enumerate(out):
                if isv(x):
                    if prev is not None and x in Vs: out[i] = prev
                    prev = out[i]
            return out
    elif kind == 'vshift':
        k = min(len(vows), rnd.randint(2, 3)); cyc = rnd.sample(vows, k)
        mp = {cyc[i]: cyc[(i + 1) % k] for i in range(k)}
        def f(w): return [mp.get(x, x) if isv(x) else x for x in w]
    elif kind == 'hloss':
        S = {c for c in conss if cplace(c) == 5 or CU[c - CB][0] in ('x', 'gh')}
        def f(w): return [x for x in w if x not in S] or w
        L.cons = [c for c in L.cons if c not in S] or L.cons; L.cw = [1.0] * len(L.cons)
    elif kind == 'nasloss':
        def f(w):
            return [x for i, x in enumerate(w) if not (not isv(x) and cman(x) == 3 and i + 1 < len(w) and not isv(w[i + 1]))]
    elif kind == 'paragoge':
        ev = rnd.choice(vows)
        def f(w): return w + [ev] if w and not isv(w[-1]) else w
    elif kind == 'metath':
        def f(w):
            out = list(w)
            for i in range(len(out) - 1):
                if not isv(out[i]) and not isv(out[i + 1]) and cman(out[i]) in (4, 5): out[i], out[i + 1] = out[i + 1], out[i]
            return out
    else:  # hiatus
        mode = rnd.choice(['del1', 'del2'])
        def f(w):
            out = []; i = 0
            while i < len(w):
                if i + 1 < len(w) and isv(w[i]) and isv(w[i + 1]):
                    out.append(w[i + 1] if mode == 'del1' else w[i]); i += 2; continue
                out.append(w[i]); i += 1
            return out
    return kind, f


# ---------------------------------------------------------------- spelling
SERI = {s: i for i, s in enumerate(T10.SER)}   # '' P T D K Q M N S Z R W J
VI = {v: i for i, v in enumerate(T10.VOW)}     # A E I O U


def default_series(c, rnd, conv):
    n, p, m, v, _ = CU[c - CB]
    if m == 3: return SERI['M'] if p == 0 else SERI['N']
    if m in (4, 5): return SERI['R']
    if m == 6: return SERI['W'] if p == 4 else SERI['J']
    if p == 5: return 0
    if m == 0:
        if p == 0: return SERI['P']
        if p == 1: return SERI['D'] if (v and conv['dvoice']) else SERI['T']
        if p == 2: return SERI['Z'] if conv['pal_z'] else SERI['K']
        if p == 4: return SERI['Q'] if conv['qser'] else SERI['K']
        return SERI['K']
    if m == 1:
        if p == 1: return SERI['S'] if not v else (SERI['Z'] if conv['pal_z'] else SERI['S'])
        if p == 2: return SERI['S'] if conv['sh_s'] else SERI['Z']
        if p == 0: return SERI['P'] if conv['f_p'] else SERI['W']
        if p == 3: return SERI['K'] if conv['x_k'] else 0
        return 0
    # affricates
    return SERI['Z'] if conv['pal_z'] else (SERI['T'] if p == 1 else SERI['K'])


def default_vowel(v, conv):
    n, h, b, r = VU[v]
    if n == 'a' or n == 'ae': return VI['A'] if n == 'a' or conv['ae_a'] else VI['E']
    if n in ('i', 'y'): return VI['I']
    if n in ('e', 'E', '2'): return VI['E']
    if n in ('o', 'O'): return VI['O']
    if n == 'u': return VI['U']
    if n == '1': return VI['I'] if conv['schwa_i'] else VI['U']
    return VI['E'] if conv['schwa_i'] else VI['A']   # @


def draw_spelling(rnd):
    conv = {k: rnd.random() < 0.5 for k in ('dvoice', 'pal_z', 'qser', 'sh_s', 'f_p', 'x_k', 'ae_a', 'schwa_i')}
    conv['noise'] = rnd.uniform(0, 0.25)          # P(a consonant gets a random series)
    conv['vnoise'] = rnd.uniform(0, 0.15)         # P(a vowel gets a random script vowel)
    for k in ('om_cl_son', 'om_cl_obs', 'om_med_son', 'om_med_obs', 'om_fin_son', 'om_fin_obs'):
        conv[k] = rnd.random() < 0.5
    conv['dead'] = rnd.choice(['echo', 'echo', 'fixed', 'prev'])
    conv['deadv'] = rnd.choice([VI['E'], VI['E'], VI['I'], VI['A'], VI['U'], VI['O']])
    return conv


def make_maps(rnd, conv):
    cmap = {}
    for i in range(NC):
        c = CB + i
        cmap[c] = default_series(c, rnd, conv) if rnd.random() >= conv['noise'] else rnd.randrange(len(T10.SER))
    vmap = {}
    for v in range(NVU):
        vmap[v] = default_vowel(v, conv) if rnd.random() >= conv['vnoise'] else rnd.randrange(5)
    return cmap, vmap


def spell(w, conv, cmap, vmap):
    vi = [i for i, x in enumerate(w) if isv(x)]
    if not vi: return ()
    sylls = []
    prev = 0
    for k, i in enumerate(vi):
        seg = w[prev:i]
        if k == 0: on, cod_prev = seg, []
        else:
            if len(seg) <= 1: on, cod_prev = seg, []
            else: on, cod_prev = seg[-1:], seg[:-1]
            sylls[-1][2] = cod_prev
        sylls.append([list(on), w[i], []])
        prev = i + 1
    sylls[-1][2] = list(w[prev:])
    out = []
    nsy = len(sylls)
    for k, (on, v, cod) in enumerate(sylls):
        sv = vmap[v]
        nxt = vmap[sylls[k + 1][1]] if k + 1 < nsy else None
        for c in on[:-1]:
            if conv['om_cl_son' if sonor(c) else 'om_cl_obs']: continue
            out.append(cmap[c] * 5 + (sv if conv['dead'] in ('echo', 'prev') else conv['deadv']))
        if on: out.append(cmap[on[-1]] * 5 + sv)
        else: out.append(sv)
        final = k == nsy - 1
        for c in cod:
            key = ('om_fin_' if final else 'om_med_') + ('son' if sonor(c) else 'obs')
            if conv[key]: continue
            if conv['dead'] == 'echo': dv = nxt if nxt is not None else sv
            elif conv['dead'] == 'prev': dv = sv
            else: dv = conv['deadv']
            out.append(cmap[c] * 5 + dv)
    return tuple(out)


# ---------------------------------------------------------------- one simulation
def simulate(seed, n_target, fixed=None):
    """-> (params dict, list of written types) ; fixed may pin 'morph', 'syl'."""
    fixed = fixed or {}
    rnd = random.Random(seed)
    L = Lang()
    draw_inventory(rnd, L)
    draw_phonotactics(rnd, L, fixed.get('syl'))
    draw_morph(rnd, L, fixed.get('morph'))
    L.pborrow = 0.0 if rnd.random() < 0.3 else rnd.uniform(0, 0.4)
    D = Lang(); draw_inventory(rnd, D); D.cons, D.cw = L.cons, L.cw; draw_phonotactics(rnd, D)
    nroots = max(20, int(n_target * rnd.uniform(0.3, 1.5)))
    roots = make_roots(rnd, L, nroots, donor=D)
    zs = rnd.uniform(0.5, 1.3); rw = []; acc = 0.0
    for k in range(nroots): acc += 1.0 / (k + 1) ** zs; rw.append(acc)
    cls = [rnd.randrange(getattr(L, 'nclass', 1)) for _ in range(nroots)]
    forms = set(); goal = int(n_target * 1.8); tries = 0
    ridx = list(range(nroots))
    while len(forms) < goal and tries < 8 * n_target:
        tries += 1
        ri = bisect.bisect_left(rw, rnd.random() * rw[-1])
        forms.add(tuple(make_form(rnd, L, roots, rw, ri, cls)))
    forms = [list(f) for f in forms]
    rnd.shuffle(forms)
    nsc = rnd.randint(0, 14)
    hist = []
    for _ in range(nsc):
        kind, f = sc_factory(rnd, L); hist.append(kind)
        forms = [f(w) for w in forms]
    plate = 0.0 if rnd.random() < 0.5 else rnd.uniform(0, 0.2)
    nlate = int(plate * len(forms))
    for i in range(nlate):
        forms[i] = root(rnd, D)
    conv = draw_spelling(rnd); cmap, vmap = make_maps(rnd, conv)
    seen = set(); types = []
    for w in forms:
        t = spell(w, conv, cmap, vmap)
        if len(t) >= 2 and t not in seen:
            seen.add(t); types.append(t)
            if len(types) >= n_target: break
    P = dict(seed=seed, morph=L.morph, syl=L.syl, nc0=len(set(L.cons)), nv0=len(L.vows), rootlen=L.rootlen,
             pcl=L.pcl, pcoda=L.pcoda, pfin=L.pfin, p0i=L.p0i, pborrow=L.pborrow, plate=plate, nsc=nsc,
             hist=hist, dead=conv['dead'], om_fin=int(conv['om_fin_son']) + int(conv['om_fin_obs']),
             om_med=int(conv['om_med_son']) + int(conv['om_med_obs']), noise=conv['noise'], nroots=nroots, zs=zs,
             ntypes=len(types))
    return P, types


# ---------------------------------------------------------------- summary panel
def extras(types):
    n = len(types)
    uni = collections.Counter(a for t in types for a in t); N = sum(uni.values())
    H = -sum(c / N * math.log(c / N) for c in uni.values())
    ini = collections.Counter(t[0] for t in types); fin = collections.Counter(t[-1] for t in types)
    Hi = -sum(c / n * math.log(c / n) for c in ini.values()); Hf = -sum(c / n * math.log(c / n) for c in fin.values())
    fr = sorted(uni.values(), reverse=True)[:40]
    xs = [math.log(i + 1) for i in range(len(fr))]; ys = [math.log(c) for c in fr]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-9, sum((x - mx) ** 2 for x in xs))
    S = set(types)
    lens = [len(t) for t in types]; ml = sum(lens) / n
    sdl = (sum((l - ml) ** 2 for l in lens) / n) ** 0.5
    pre2 = sum(1 for t in types if len(t) >= 4 and t[2:] in S) / n
    suf2 = sum(1 for t in types if len(t) >= 4 and t[:-2] in S) / n
    # families: types sharing first 2 signs / last 2 signs with another type
    f2 = collections.Counter(t[:2] for t in types if len(t) >= 3); l2 = collections.Counter(t[-2:] for t in types if len(t) >= 3)
    fam_i = sum(1 for t in types if len(t) >= 3 and f2[t[:2]] >= 2) / n
    fam_f = sum(1 for t in types if len(t) >= 3 and l2[t[-2:]] >= 2) / n
    # distinct finals given same stem (alternation richness)
    by = collections.defaultdict(set)
    for t in types:
        if len(t) >= 3: by[t[:-1]].add(t[-1])
    alt = sum(len(v) for v in by.values() if len(v) >= 2) / n
    byi = collections.defaultdict(set)
    for t in types:
        if len(t) >= 3: byi[t[1:]].add(t[0])
    alti = sum(len(v) for v in byi.values() if len(v) >= 2) / n
    # adjacent MI (value-free)
    pairs = collections.Counter((t[i], t[i + 1]) for t in types for i in range(len(t) - 1)); NP = sum(pairs.values())
    l1 = collections.Counter(); r1 = collections.Counter()
    for (a, b), c in pairs.items(): l1[a] += c; r1[b] += c
    mi = sum(c / NP * math.log(c * NP / (l1[a] * r1[b])) for (a, b), c in pairs.items())
    # concentration of finals in top 3
    top3f = sum(c for _, c in fin.most_common(3)) / n; top3i = sum(c for _, c in ini.most_common(3)) / n
    return dict(H=H, Hi_rel=Hi / H, Hf_rel=Hf / H, zslope=slope, nsign=len(uni), mlen=ml, sdlen=sdl, pre2=pre2,
                suf2=suf2, fam_i=fam_i, fam_f=fam_f, alt=alt, alti=alti, mi=mi, top3f=top3f, top3i=top3i)


def panel(types):
    f = T10.fingerprint(types)
    f.update(extras(types))
    return f


PANEL = None
def panel_names():
    global PANEL
    if PANEL is None:
        PANEL = list(T10.STAT_NAMES) + list(extras([(0, 1), (1, 2), (2, 0, 1)]).keys())
    return PANEL


# value-free subset (no LB values needed for the target): signs as anonymous labels
VALUE_FREE = ['dbl', 'gap_exP', 'fin10_ex', 'init10_ex', 'pos', 'pre', 'suf', 'len2', 'len3', 'len4', 'len5p',
              'H', 'Hi_rel', 'Hf_rel', 'zslope', 'nsign', 'mlen', 'sdlen', 'pre2', 'suf2', 'fam_i', 'fam_f', 'alt', 'alti',
              'mi', 'top3f', 'top3i']


# ---------------------------------------------------------------- targets
def cyp_types():
    w = json.load(open(os.path.join(HERE, '..', 'data', 'la38_ckpt', 'cypriot_idalion.json')))
    return T10.encode_types(set(tuple(s.upper() for s in x) for x in w))


def targets(seed=0):
    rnd = random.Random(seed)
    la = T10.la_types(); lb = T10.lb_types(); cy = cyp_types()
    return la, lb, cy
