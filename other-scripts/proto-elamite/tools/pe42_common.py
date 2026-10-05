"""pe42 shared code: LEARN FROM THE SCRIBES' MISTAKES.

Correct totals only say what the scribe meant; wrong totals show how he added.
For every tablet whose written total(s) do not close under any value set, we
enumerate single mental-arithmetic error processes and ask which processes
explain the real discrepancies better than chance (nulls) and better than in
the controls.

Number maps are anchored at N01 = 1 (ladder of codes, each code's value):
  PE   dec2/dec6/sex2/sex6 (count, fraction N08 = 1/2 or 1/6), cap (a-priori
       capacity chain N14 = 6 N01, small units below N01).
  PC   S (N14 10, N34 60, N45 600, N48 3600, N50 36000), B (N14 10, N34 60,
       N51 120, N48 7200?) and SE (N14 6, N45 60, N34 180, N48 1800).
  UR3  sex only (Ur III counts written sexagesimally; dec offered as decoy).

Error families (one error per tablet; all other total pairs must close):
  OMIT(i)      entry i left out             OMITH: i repeats a neighbour's signs
                                             (eye-skip); OMITO: other omissions
  DOUBLE(i)    entry i counted twice
  SYS(i,m')    entry i (or the total) valued in another map m'
  WHOLE(m')    all lines valued in another map that is not a base map
  CONV(i,f)    entry i multiplied by f in {3,5,6,10,1/2,1/3,1/5,1/6,1/10}
  CARRY(u,k)   k carried units into ladder code u dropped (total = S - k*u)
  XCARRY(u)    one extra carry into u (S + u) at a carry position
  NOCARRY      every carry dropped (digit sums mod ratio)
  DIGIT(u,+-1) a +-1 slip of code u at a NON-carry position
  FOREIGN(w)   T = S +- w, w any entry value from another tablet of the class
"""
import collections, itertools, json, os, random, re, sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe42_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa: E402
from pe24_common import system, tot_class, member_variants, has_gap, annotation  # noqa: E402
from pe30_common import line_ok, lost_numeral  # noqa: E402

FRAC = ('N08', 'N08A', 'N8B', 'N02')


def _cnt(big, fr):
    d = {'N01': Fr(1), 'N51': Fr(120)}
    d.update({k: Fr(v) for k, v in big.items()})
    for c in FRAC:
        d[c] = fr
    return d


DEC = {'N14': 10, 'N45': 100, 'N34': 300, 'N48': 3000}
SEX = {'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}
PE_MAPS = {
    'dec2': _cnt(DEC, Fr(1, 2)), 'dec6': _cnt(DEC, Fr(1, 6)),
    'sex2': _cnt(SEX, Fr(1, 2)), 'sex6': _cnt(SEX, Fr(1, 6)),
    'cap': {'N39C': Fr(1, 120), 'N30D': Fr(1, 60), 'N30C': Fr(1, 30), 'N24': Fr(1, 10),
            'N39B': Fr(1, 5), 'N01': Fr(1), 'N51': Fr(1), 'N14': Fr(6), 'N45': Fr(60),
            'N34': Fr(180), 'N48': Fr(1800), 'N08': Fr(1, 240), 'N08A': Fr(1, 240),
            'N8B': Fr(1, 240), 'N02': Fr(1, 240)},
}
PE_BASE = {'cnt': ['dec2', 'dec6', 'sex2', 'sex6'], 'cap': ['cap']}
PE_ALT = ['dec2', 'dec6', 'sex2', 'sex6', 'cap']

PC_MAPS = {
    'S': {'N01': Fr(1), 'N14': Fr(10), 'N34': Fr(60), 'N45': Fr(600), 'N48': Fr(3600), 'N50': Fr(36000)},
    'B': {'N01': Fr(1), 'N14': Fr(10), 'N34': Fr(60), 'N51': Fr(120), 'N48': Fr(7200)},
    'SE': {'N01': Fr(1), 'N14': Fr(6), 'N45': Fr(60), 'N34': Fr(180), 'N48': Fr(1800)},
}
PC_BASE = {'pc': ['S', 'B', 'SE']}
PC_ALT = ['S', 'B', 'SE']

UR_MAPS = {
    'sex': {'N01': Fr(1), 'N14': Fr(10), 'N34': Fr(60), 'N45': Fr(600), 'N48': Fr(3600), 'N50': Fr(36000)},
    'dec': {'N01': Fr(1), 'N14': Fr(10), 'N34': Fr(100), 'N45': Fr(1000), 'N48': Fr(10000), 'N50': Fr(100000)},
}
UR_BASE = {'ur': ['sex']}
UR_ALT = ['sex', 'dec']

CONVF = (Fr(3), Fr(5), Fr(6), Fr(10), Fr(1, 2), Fr(1, 3), Fr(1, 5), Fr(1, 6), Fr(1, 10))
FAMS = ['OMITH', 'OMITO', 'DOUBLE', 'SYS', 'WHOLE', 'CONV', 'CARRY', 'XCARRY', 'NOCARRY', 'DIGIT', 'FOREIGN', 'CODE']


class Corpus:
    def __init__(self, name, maps, basemaps, alts):
        self.name, self.maps, self.basemaps, self.alts = name, maps, basemaps, alts


CORP = {'PE': Corpus('PE', PE_MAPS, PE_BASE, PE_ALT),
        'PC': Corpus('PC', PC_MAPS, PC_BASE, PC_ALT),
        'UR3': Corpus('UR3', UR_MAPS, UR_BASE, UR_ALT)}


# ------------------------------------------------------------------ values
def val(nums, m, fallback=None):
    t = Fr(0)
    for n, c in nums:
        c = c.split('@')[0]
        if c in m:
            t += Fr(n) * m[c]
        elif fallback is not None and c in fallback:
            t += Fr(n) * fallback[c]
        else:
            return None
    return t


def ladder(m):
    return sorted(m.items(), key=lambda kv: kv[1])


def canon(x, m):
    """Express x >= 0 as numerals in map m (greedy over a de-duplicated ladder)."""
    out = []
    seen = set()
    for c, v in sorted(m.items(), key=lambda kv: -kv[1]):
        if v in seen:
            continue
        seen.add(v)
        k = int(x // v)
        if k:
            out.append([k, c]); x -= k * v
    return out if x == 0 else None


def column_add(entries, m):
    """Column addition on the ladder of m. Returns (carry_into {code: k},
    nocarry total value). Codes of equal value are merged."""
    lad = []
    for c, v in ladder(m):
        if lad and lad[-1][1] == v:
            lad[-1][0].append(c)
        else:
            lad.append([[c], v])
    D = [0] * len(lad)
    idx = {c: i for i, (cs, _) in enumerate(lad) for c in cs}
    for e in entries:
        for n, c in e:
            c = c.split('@')[0]
            if c not in idx:
                return None, None
            D[idx[c]] += Fr(n)
    carry_into = {}
    nocarry = Fr(0); cin = 0
    for i, (cs, v) in enumerate(lad):
        s = D[i] + cin
        if i + 1 < len(lad):
            r = lad[i + 1][1] / v
            if r.denominator != 1:
                return None, None
            r = int(r)
            cout = int(s // r)
            nocarry += (D[i] % r) * v
        else:
            cout = 0
            nocarry += D[i] * v
        if cout:
            carry_into[lad[i + 1][0][0]] = (cout, lad[i + 1][1])
        cin = cout
    return carry_into, nocarry


# ------------------------------------------------------------------ PE cases
def sic_flags():
    """(id, surface, column, label) of lines flagged by the editor: '!' in the
    line, or a following '# sic' / erasure / written-over / mistake comment."""
    out = set(); cur = None; surf = 'obverse'; col = 1; last = None
    for raw in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        raw = raw.rstrip('\n')
        if raw.startswith('&'):
            cur = raw[1:].split()[0]; surf = 'obverse'; col = 1; last = None
        elif raw.startswith('@'):
            w = raw[1:].split()
            if not w:
                continue
            if w[0] == 'column':
                col = int(re.sub(r'\D', '', w[1]) or 1) if len(w) > 1 else 1
            elif w[0] in ('obverse', 'reverse', 'top', 'bottom', 'left', 'right', 'edge', 'seal'):
                surf = w[0]; col = 1
        elif raw.startswith('#'):
            if last and re.search(r'sic|eras|written over|mistake|error', raw, re.I):
                out.add(last)
        elif re.match(r"^\S+\.\s", raw):
            lab = raw.split('.')[0] if not re.match(r"^\d+'?\.[a-z0-9]", raw) else raw.split(' ')[0].rstrip('.')
            last = (cur, surf, col, lab)
            if '!' in raw:
                out.add(last)
    return out


def _line(l, flags=None, tid=None):
    sg = [s for s in l['signs'] if is_sign(s)]
    fl = bool(flags and (tid, l['surface'], l.get('column', 1), str(l.get('label'))) in flags)
    return {'nums': [[n, c] for n, c in l['numerals']], 'signs': sg, 'flag': fl}


def build_pe():
    T = load()
    flags = sic_flags()
    broken = set(); cur = None
    for raw in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        if raw.startswith('&'):
            cur = raw[1:].split()[0]
        elif raw.startswith('$') and re.search(r'broken|missing|not given|traces', raw):
            broken.add(cur)
    out = []
    for t in T:
        if has_gap(t) or t['id'] in broken:
            continue
        lines = [l for l in t['lines'] if l['numerals'] and not (
            l['surface'] in ('top', 'left', 'seal') and not any(is_sign(s) for s in l['signs']))]
        lines = [l for l in lines if not annotation(l)]
        O = [l for l in lines if l['surface'] == 'obverse']
        R = [l for l in lines if l['surface'] != 'obverse']
        if not R or not O:
            continue
        t1 = (not any(l['lacuna'] or '...' in l['raw'] for l in t['lines'])
              and all(line_ok(l, 1) for l in lines))
        if not t1:
            if any(lost_numeral(l) for l in t['lines']):
                continue
            if any(l['lacuna'] and not l['numerals'] and l['surface'] in ('obverse', 'reverse')
                   and re.fullmatch(r'[\s\[\].]*', l['raw'].replace('...', '')) for l in t['lines']):
                continue
            if not all(line_ok(l, 2) for l in lines):
                continue
        if any('@' in c for l in lines for _, c in l['numerals']):
            continue
        if len(R) == 1:
            pairs = [(R[0], O)]
        else:
            cl = [tot_class(r, O) for r in R]
            if len(set(cl)) != len(cl):
                continue
            pairs = [(r, O) for r in R]
        per_pair = [[(tot_class(tl, ent), m, tl) for m in member_variants(ent, tot_class(tl, ent))]
                    for tl, ent in pairs]
        hyps = []
        for combo in itertools.product(*per_pair):
            if any(len(m) < 2 for _, m, _ in combo):
                continue
            if any(cls == 'cap' and all(system(l) in ('cnt', 'amb') for l in m) for cls, m, _ in combo):
                continue
            if any(cls not in ('cnt', 'cap') for cls, _, _ in combo):
                continue
            hyps.append([{'cls': cls, 'T': _line(tl, flags, t['id']),
                          'E': [_line(l, flags, t['id']) for l in m]} for cls, m, tl in combo])
        if hyps:
            out.append({'id': t['id'], 'tier': 1 if t1 else 2, 'hyps': hyps})
    return out


# ------------------------------------------------------------------ PC cases
PC_OK = {'N01', 'N14', 'N34', 'N45', 'N48', 'N50', 'N51'}


def build_pc():
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    out = []
    for t in T:
        L = [l for l in t['lines'] if l['numerals']]
        if any(l['lacuna'] or '...' in l['raw'] or '[' in l['raw'] for l in t['lines']):
            continue
        if any(n is None or c not in PC_OK for l in L for n, c in l['numerals']):
            continue
        O = [l for l in L if l['surface'] == 'obverse']
        R = [l for l in L if l['surface'] == 'reverse']
        if len(R) != 1 or len(O) < 2:
            continue
        def ln(l):
            return {'nums': [[n, c] for n, c in l['numerals']],
                    'signs': [s for s in l['signs'] if s != 'x'], 'flag': '!' in l['raw']}
        out.append({'id': t['id'], 'tier': 1, 'period': t.get('period', ''),
                    'hyps': [[{'cls': 'pc', 'T': ln(R[0]), 'E': [ln(l) for l in O]}]]})
    return out


# ------------------------------------------------------------------ Ur III cases
def build_ur():
    """Sumerian (mostly Ur III) count lists from the CDLI dump: exactly one
    szu-nigin2 line, 2-40 numeral-initial count lines before it, no break."""
    from pe24_ur3 import parse_ur_line
    LINE = re.compile(r"^(\d+'?)\.\s+(.*)$")
    NUMSTART = re.compile(r"^(\d+(?:/\d+)?)\(")
    m = UR_MAPS['sex']
    txt = open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace').read()
    out = []
    for d in re.split(r'\n(?=&P)', txt):
        if 'szu-nigin' not in d:
            continue
        if re.search(r'\n\$ .*(broken|missing|traces)', d):
            continue
        rows = []
        for raw in d.split('\n'):
            mm = LINE.match(raw.strip())
            if mm:
                rows.append((mm.group(2).replace('@c', '').replace('@v', ''), '!' in mm.group(2)))
        num = [r for r, _ in rows if NUMSTART.match(re.sub(r'^szu-nigin2?\s+', '', re.sub(r'[#?!]', '', r)))]
        if any('[' in r or '...' in r or re.search(r'(^|\s)n\(', r) for r in num):
            continue
        if any('...' in r and not NUMSTART.match(r) for r, _ in rows) and False:
            continue
        tl = [k for k, (r, _) in enumerate(rows) if re.sub(r'[#?!]', '', r).startswith('szu-nigin')]
        if len(tl) != 1:
            continue
        mem = []; bad = False
        for r, fl in rows[:tl[0]]:
            t = re.sub(r'[#?!]', '', r).strip()
            if not NUMSTART.match(t):
                continue
            p = parse_ur_line(t)
            if p is None or p[3] != 'n' or p[0] <= 0 or p[0].denominator != 1:
                bad = True; break
            mem.append((p[0], [w for w in p[2].split() if not NUMSTART.match(w)], fl))
        p = parse_ur_line(re.sub(r'[#?!]', '', rows[tl[0]][0]).strip())
        if bad or p is None or p[3] != 'n' or not (2 <= len(mem) <= 40) or p[0] <= 0 or p[0].denominator != 1:
            continue
        H = [[{'cls': 'ur', 'T': {'nums': canon(p[0], m), 'signs': p[2].split()[:2], 'flag': rows[tl[0]][1]},
               'E': [{'nums': canon(v, m), 'signs': w[:2], 'flag': fl} for v, w, fl in mem]}]]
        out.append({'id': d[1:8], 'tier': 1, 'hyps': H})
    return out


def load_cases(name):
    p = os.path.join(CK, 'cases_%s.json' % name)
    if not os.path.exists(p):
        C = {'PE': build_pe, 'PC': build_pc, 'UR3': build_ur}[name]()
        json.dump(C, open(p, 'w'))
    return json.load(open(p))


# ------------------------------------------------------------------ evaluation
def pair_vals(p, m):
    T = val(p['T']['nums'], m)
    E = [val(e['nums'], m) for e in p['E']]
    if T is None or any(x is None for x in E):
        return None
    return T, E


def closes(case, corp):
    for h in case['hyps']:
        for mset in mapsets(h, corp):
            if all(pv is not None and pv[0] == sum(pv[1]) for pv in
                   (pair_vals(p, corp.maps[mn]) for p, mn in zip(h, mset))):
                return True
    return False


def mapsets(h, corp):
    """One map per pair; count pairs of one tablet share their map."""
    cls = [p['cls'] for p in h]
    opts = {c: corp.basemaps[c] for c in set(cls)}
    keys = sorted(opts)
    for combo in itertools.product(*[opts[k] for k in keys]):
        d = dict(zip(keys, combo))
        yield [d[c] for c in cls]


def neighbours_same(E, i):
    s = E[i]['signs']
    if not s:
        return False
    return any(0 <= j < len(E) and E[j]['signs'] and (E[j]['signs'] == s or E[j]['signs'][-1] == s[-1])
               for j in (i - 1, i + 1))


def instances(p, mname, corp, pool):
    """Yield (family, detail, predicted_total_value) for one pair under map mname,
    plus counts of instances per family (for likelihood normalisation)."""
    m = corp.maps[mname]
    pv = pair_vals(p, m)
    if pv is None:
        return [], {}
    T, E = pv
    S = sum(E)
    out = []; n = collections.Counter()
    for i, v in enumerate(E):
        fam = 'OMITH' if neighbours_same(p['E'], i) else 'OMITO'
        out.append((fam, ('i', i), S - v)); n[fam] += 1
        out.append(('DOUBLE', ('i', i), S + v)); n['DOUBLE'] += 1
        for f in CONVF:
            out.append(('CONV', ('i', i, str(f)), S + (f - 1) * v)); n['CONV'] += 1
        for a in corp.alts:
            if a == mname:
                continue
            w = val(p['E'][i]['nums'], corp.maps[a], fallback=m)
            n['SYS'] += 1
            if w is not None and w != v:
                out.append(('SYS', ('i', i, mname, a), S - v + w))
    # total read in another map: compare T' with S
    for a in corp.alts:
        if a == mname:
            continue
        w = val(p['T']['nums'], corp.maps[a], fallback=m)
        n['SYS'] += 1
        if w is not None and w != T:
            out.append(('SYS', ('T', mname, a), S - (w - T)))  # equivalent: T == S + (T - w)
        # whole tablet in a non-base map
        if a not in corp.basemaps[p['cls']]:
            pa = pair_vals(p, corp.maps[a]) if all(val(x['nums'], corp.maps[a]) is not None for x in p['E'] + [p['T']]) else None
            n['WHOLE'] += 1
            if pa is not None and pa[0] == sum(pa[1]):
                out.append(('WHOLE', (mname, a), T))
    # code-level system slip: one numeral code read with another map's value
    codes0 = sorted({c.split('@')[0] for x in p['E'] + [p['T']] for _, c in x['nums']})
    for c in codes0:
        for a in corp.alts:
            va = corp.maps[a].get(c)
            if a == mname or va is None or va == m.get(c):
                continue
            mm = dict(m); mm[c] = va
            Ta = val(p['T']['nums'], mm); Sa = sum(val(e['nums'], mm) for e in p['E'])
            n['CODE'] += 3
            out.append(('CODE', ('T', c, a), S - (Ta - T)))   # total's c misread: T_written(c@a) == S
            out.append(('CODE', ('E', c, a), Sa))             # entries' c misread
            if Ta == Sa:
                out.append(('CODE', ('A', c, a), T))
    ci, noc = column_add([e['nums'] for e in p['E']], m)
    carrycodes = set()
    if ci is not None:
        for c, (k, v) in ci.items():
            carrycodes.add(c)
            for j in range(1, k + 1):
                out.append(('CARRY', (c, j), S - j * v)); n['CARRY'] += 1
            out.append(('XCARRY', (c,), S + v)); n['XCARRY'] += 1
        if ci:
            out.append(('NOCARRY', (), noc)); n['NOCARRY'] += 1
    seen = set()
    codes = {c.split('@')[0] for x in p['E'] + [p['T']] for _, c in x['nums']}
    for c, v in ladder(m):
        if v in seen:
            continue
        seen.add(v)
        if c in carrycodes or not (c in codes or any(m.get(d) is not None and (m[d] / v in (10, 6, 3, 2, 5) or v / m[d] in (10, 6, 3, 2, 5)) for d in codes)):
            continue
        for s in (1, -1):
            out.append(('DIGIT', (c, s), S + s * v)); n['DIGIT'] += 1
    # foreign: membership test only
    if pool is not None:
        P = pool.get(mname, set())
        n['FOREIGN'] += 2 * max(1, len(P))
        d = T - S
        if d in P or -d in P:
            out.append(('FOREIGN', (str(d),), T))
    return out, n


def explain(case, corp, pool):
    """Return (closes, {family: [details]}, {family: max instance count}).
    A family explains the tablet if for some hypothesis/map set one pair is
    fixed by one instance and every other pair closes exactly."""
    if closes(case, corp):
        return True, {}, {}
    expl = collections.defaultdict(list); N = collections.Counter()
    for h in case['hyps']:
        for mset in mapsets(h, corp):
            pvs = [pair_vals(p, corp.maps[mn]) for p, mn in zip(h, mset)]
            if any(pv is None for pv in pvs):
                continue
            ok = [pv[0] == sum(pv[1]) for pv in pvs]
            for k, (p, mn) in enumerate(zip(h, mset)):
                if ok[k] or not all(ok[j] for j in range(len(h)) if j != k):
                    continue
                inst, n = instances(p, mn, corp, pool)
                for f, c in n.items():
                    N[f] = max(N[f], c)
                T = pvs[k][0]
                for fam, det, pred in inst:
                    if pred == T:
                        expl[fam].append((k, mn) + tuple(det))
    return False, dict(expl), dict(N)


def make_pool(cases, corp):
    """Entry values of every case, per map, keyed by case index (for FOREIGN)."""
    per = []
    for c in cases:
        d = collections.defaultdict(set)
        for h in c['hyps']:
            for p in h:
                for mn in corp.basemaps[p['cls']]:
                    for e in p['E']:
                        v = val(e['nums'], corp.maps[mn])
                        if v is not None:
                            d[mn].add(v)
        per.append(d)
    return per


def pool_excluding(per, k, cache={}):
    key = (id(per), k)
    tot = collections.defaultdict(collections.Counter)
    ck = ('all', id(per))
    if ck not in cache:
        for d in per:
            for mn, s in d.items():
                tot[mn].update(s)
        cache[ck] = tot
    tot = cache[ck]
    own = per[k]
    return {mn: {v for v, c in cnt.items() if c - (1 if v in own.get(mn, ()) else 0) > 0}
            for mn, cnt in tot.items()}


def evaluate(cases, corp, foreign=True):
    per = make_pool(cases, corp) if foreign else None
    R = []
    for k, c in enumerate(cases):
        pool = pool_excluding(per, k) if foreign else None
        cl, ex, N = explain(c, corp, pool)
        R.append({'id': c['id'], 'closes': cl, 'expl': {f: len(v) > 0 for f, v in ex.items()},
                  'det': {f: v[:6] for f, v in ex.items()}, 'N': N})
    return R


def fam_counts(R):
    fail = [r for r in R if not r['closes']]
    c = collections.Counter()
    for r in fail:
        for f, b in r['expl'].items():
            if b:
                c[f] += 1
    c['_fail'] = len(fail); c['_any'] = sum(any(r['expl'].values()) for r in fail)
    return c


# ------------------------------------------------------------------ nulls & plants
def _pref(p, corp):
    return corp.basemaps[p['cls']][0]


def null_noise(cases, corp, rng, R=None):
    """Failing tablets: keep entries, replace one pair's total by S + d, where d
    is the real discrepancy of a random OTHER failing tablet of the same class
    (under its preferred map). Closing tablets unchanged."""
    R = R or evaluate(cases, corp, foreign=False)
    disc = collections.defaultdict(list)
    for c, r in zip(cases, R):
        if r['closes']:
            continue
        p = c['hyps'][0][0]
        pv = pair_vals(p, corp.maps[_pref(p, corp)])
        if pv:
            disc[p['cls']].append((c['id'], pv[0] - sum(pv[1])))
    out = []
    for c, r in zip(cases, R):
        if r['closes']:
            out.append(c); continue
        c2 = json.loads(json.dumps(c))
        h = c2['hyps'][0]
        p = h[0]
        mn = _pref(p, corp); m = corp.maps[mn]
        pv = pair_vals(p, m)
        D = [d for i, d in disc[p['cls']] if i != c['id']]
        for _ in range(50):
            if not pv or not D:
                break
            d = rng.choice(D)
            newT = sum(pv[1]) + d
            nm = canon(newT, m) if newT > 0 else None
            if nm:
                for hh in c2['hyps']:
                    hh[0]['T']['nums'] = nm
                break
        out.append(c2)
    return out


def null_shuffle_entries(cases, corp, rng, R=None):
    """Entry lines re-dealt across failing tablets of the same class (keep count)."""
    R = R or evaluate(cases, corp, foreign=False)
    fail = [k for k, r in enumerate(R) if not r['closes']]
    out = [json.loads(json.dumps(c)) for c in cases]
    bycls = collections.defaultdict(list)
    for k in fail:
        bycls[out[k]['hyps'][0][0]['cls']].append(k)
    for cls, ks in bycls.items():
        pool = [e for k in ks for e in out[k]['hyps'][0][0]['E']]
        rng.shuffle(pool); j = 0
        for k in ks:
            n = len(out[k]['hyps'][0][0]['E'])
            new = pool[j:j + n]; j += n
            out[k]['hyps'] = [[dict(out[k]['hyps'][0][0], E=new)] + out[k]['hyps'][0][1:]]
    return out


def null_totals(cases, corp, rng, R=None):
    """Totals reassigned among failing tablets of the same class and size bin."""
    R = R or evaluate(cases, corp, foreign=False)
    fail = [k for k, r in enumerate(R) if not r['closes']]
    out = [json.loads(json.dumps(c)) for c in cases]
    bins = collections.defaultdict(list)
    for k in fail:
        p = out[k]['hyps'][0][0]
        pv = pair_vals(p, corp.maps[_pref(p, corp)])
        if not pv:
            continue
        b = int(float(pv[0]).__floor__().bit_length())
        bins[(p['cls'], min(b // 2, 8))].append(k)
    for key, ks in bins.items():
        Ts = [out[k]['hyps'][0][0]['T'] for k in ks]
        perm = Ts[:]; rng.shuffle(perm)
        for k, t in zip(ks, perm):
            for hh in out[k]['hyps']:
                hh[0]['T'] = t
    return out


def plant(cases, corp, rng, fam, R=None, nmax=40):
    """Take closing tablets and plant ONE error of family fam (random instance)."""
    R = R or evaluate(cases, corp, foreign=False)
    per = make_pool(cases, corp)
    out = []; truth = []
    ks = [k for k, r in enumerate(R) if r['closes']]
    rng.shuffle(ks)
    for k in ks:
        if len(out) >= nmax:
            break
        c = cases[k]
        # use the first hypothesis / map set that closes
        hit = None
        for h in c['hyps']:
            for mset in mapsets(h, corp):
                pvs = [pair_vals(p, corp.maps[mn]) for p, mn in zip(h, mset)]
                if all(pv is not None and pv[0] == sum(pv[1]) for pv in pvs):
                    hit = (h, mset); break
            if hit:
                break
        if not hit:
            continue
        h, mset = hit
        j = rng.randrange(len(h)); p = h[j]; mn = mset[j]; m = corp.maps[mn]
        if fam == 'FOREIGN':
            vals = sorted(v for kk, d in enumerate(per) if kk != k for v in d.get(mn, ()))
            if not vals:
                continue
            pred = sum(pair_vals(p, m)[1]) + rng.choice([1, -1]) * rng.choice(vals)
        else:
            inst, _ = instances(p, mn, corp, None)
            inst = [x for x in inst if x[0] == fam or (fam == 'OMIT' and x[0].startswith('OMIT'))]
            if not inst:
                continue
            pred = rng.choice(inst)[2]
        if pred is None or pred <= 0 or pred == pair_vals(p, m)[0]:
            continue
        nm = canon(pred, m)
        if nm is None:
            continue
        c2 = json.loads(json.dumps(c))
        c2['hyps'] = [json.loads(json.dumps(h))]
        c2['hyps'][0][j]['T']['nums'] = nm
        c2['id'] = c['id'] + '+' + fam
        # must not close any more under any base map
        if closes(c2, corp):
            continue
        out.append(c2); truth.append(fam)
    return out, truth
