"""pe5: is the Proto-Elamite corpus one continuous ledger?  Shared code.

A quantity record = (tablet, line index, role, system, notation, value, signs).
role: 'T' = written total (sole numeric line off the obverse of a tablet with
>= 2 obverse numeric entries; Ur III: line with szu-nigin2), 'E' = entry.

Value sets for PE are hypotheses (FINDINGS attack 2); the notation key
(canonical multiset of numeral codes) is value-free.
"""
import csv, json, os, random, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign, norm_code  # noqa: E402

PEDATA = os.path.join(HERE, '..', 'data')
SCRATCH = os.environ.get('PE5_SCRATCH', '')   # holds cdli.atf / cat.csv for controls

CAP = {'N39C', 'N30D', 'N30C', 'N24', 'N39B', 'N39A', 'N28', 'N29B', 'N39N'}
FRAC = {'N02', 'N08', 'N08A', 'N8B', 'N8A'}
CNT_OK = {'N01', 'N14', 'N45', 'N34', 'N48', 'N50'} | FRAC

# counting value sets (FINDINGS attack 2 grid: D3 is grade B, others tie within 1 tablet)
VSETS = {
    'D3':  {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300, 'N48': 3000, 'N50': 30000},
    'DEC': {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 1000, 'N48': 10000, 'N50': 100000},
    'SEX': {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000},
    'SEXr': {'N01': 1, 'N14': 10, 'N45': 60, 'N34': 600, 'N48': 3600, 'N50': 36000},
}
# capacity, a-priori notation set (N39C units), and a proto-cuneiform-like alternative
CSETS = {
    'NOT': {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120,
            'N14': 720, 'N45': 7200, 'N34': 21600},
    'PCS': {'N39C': 1, 'N30D': 2, 'N30C': 3, 'N24': 6, 'N39B': 30, 'N01': 150,
            'N14': 900, 'N45': 9000, 'N34': 27000},
}
FRACV = {'N02': 0.5, 'N08': 0.5, 'N08A': 0.25, 'N8A': 0.25, 'N8B': 0.2}


def pe_system(nums):
    codes = {c for _, c in nums}
    if any(not isinstance(n, int) for n, _ in nums):
        return None
    if not codes or any('@' in c or c in ('n', 'N1B', 'N14B') for c in codes):
        return None                              # modified / damaged notation: skip
    if codes & CAP and codes <= CAP | {'N01', 'N14', 'N45', 'N34'} | FRAC:
        return 'C'
    if codes <= CNT_OK:
        return 'S'
    return None


def pe_value(nums, sysname, vs, cs):
    tab = vs if sysname == 'S' else cs
    v = 0.0
    for n, c in nums:
        if c in FRACV and sysname == 'S':
            v += n * FRACV[c]
        elif c in tab:
            v += n * tab[c]
        else:
            return None
    return round(v, 4)


def notation(nums):
    d = Counter()
    for n, c in nums:
        d[c] += n
    return tuple(sorted(d.items()))


def pe_records(vs='D3', cs='NOT', T=None):
    vs = VSETS[vs] if isinstance(vs, str) else vs
    cs = CSETS[cs] if isinstance(cs, str) else cs
    """Return list of records and tablet meta (header, set of final signs)."""
    if T is None:
        T = load()
    recs, meta = [], {}
    for t in T:
        lines = t['lines']
        damaged_tab = any('broken' in l.get('raw', '') for l in lines)
        num = [(i, l) for i, l in enumerate(lines) if l['numerals']]
        obv = [x for x in num if x[1]['surface'] == 'obverse']
        off = [x for x in num if x[1]['surface'] in ('reverse',) ]
        tot_idx = off[0][0] if len(off) == 1 and len(obv) >= 2 else None
        hdr = None
        if lines and not lines[0]['numerals']:
            s = [base(x) for x in lines[0]['signs'] if is_sign(x)]
            hdr = s[0] if s else None
        finals = Counter()
        for i, l in num:
            if l['surface'] == 'top':
                continue                          # edge marks (attack 2)
            sysn = pe_system(l['numerals'])
            if sysn is None:
                continue
            sg = [base(x) for x in l['signs'] if is_sign(x)]
            if sg:
                finals[sg[-1]] += 1
            val = pe_value(l['numerals'], sysn, vs, cs)
            recs.append({'tab': t['id'], 'i': i, 'role': 'T' if i == tot_idx else 'E',
                         'sys': sysn, 'key': notation(l['numerals']), 'val': val,
                         'nums': l['numerals'], 'signs': sg,
                         'clean': not (l['damaged'] or l['lacuna'] or 'x' in l['signs']),
                         'obv': l['surface'] == 'obverse'})
        meta[t['id']] = {'hdr': hdr, 'finals': finals, 'damaged': damaged_tab,
                         'design': t.get('designation', '')}
    return recs, meta


# ---------------- Ur III control ----------------
UR_UNITS = {"disz": 1, "asz": 1, "u": 10, "gesz2": 60, "gesz'u": 600, "szar2": 3600,
            "szar'u": 36000, "szar2'u": 36000}
UR_MAX = {1: 9, 10: 5, 60: 9, 600: 5, 3600: 9, 36000: 5}
NONCOUNT = re.compile(r'\b(gur|sila3|ban2|barig|gin2|ma-na|iku|bur3|esze3|sar|sze-gur|gu2)\b')
NUMTOK = re.compile(r"^(\d+)\(([a-z'0-9]+)\)$")


def parse_ur_line(txt):
    """Leading count numerals of an Ur III line -> (value, digit dict) or None."""
    toks = txt.replace('#', '').replace('?', '').replace('!', '').split()
    tot = False
    if toks and (toks[0].startswith('szu-nigin') or toks[0].startswith('szunigin')):
        tot = True
        toks = toks[1:]
    digs = Counter()
    k = 0
    while k < len(toks):
        m = NUMTOK.match(toks[k])
        if not m or m.group(2) not in UR_UNITS:
            break
        digs[UR_UNITS[m.group(2)]] += int(m.group(1))
        k += 1
    if not digs or k == len(toks):
        return None
    if toks[k] == 'la2':                       # subtractive: n la2 m
        m = NUMTOK.match(toks[k + 1]) if k + 1 < len(toks) else None
        if not m or m.group(2) not in UR_UNITS:
            return None
        digs[1] -= int(m.group(1)) * UR_UNITS[m.group(2)]
        k += 2
    rest = ' '.join(toks[k:])
    if NONCOUNT.search(rest) or '[' in txt or 'x' in toks[:k]:
        return None
    val = sum(u * n for u, n in digs.items())
    if val <= 0:
        return None
    return val, tot, rest


def ur3_records(prov='Puzri', period='Ur III'):
    """Parse Ur III tablets of one provenience from the CDLI dump in SCRATCH."""
    csv.field_size_limit(10 ** 9)
    keep = {}
    for r in csv.DictReader(open(os.path.join(SCRATCH, 'cat.csv'), encoding='utf-8', errors='replace')):
        if r['period'].startswith(period) and r['provenience'].startswith(prov):
            keep['P%06d' % int(r['id_text'])] = r.get('dates_referenced', '')
    recs, meta = [], {}
    cur, lines = None, []

    def flush():
        if cur is None or not lines:
            return
        ent = [x for x in lines if not x[1]]
        tots = [x for x in lines if x[1]]
        tots_ok = len(tots) >= 1 and len(ent) >= 2
        for j, (val, istot, rest, surf) in enumerate(lines):
            role = 'T' if (istot and tots_ok) else ('E' if not istot else 'X')
            if role == 'X':
                continue
            recs.append({'tab': cur, 'i': j, 'role': role, 'sys': 'S', 'val': float(val),
                         'key': val, 'signs': rest.split()[:3], 'clean': True, 'obv': surf == 'obverse'})
        meta[cur] = {'hdr': None, 'date': keep.get(cur, ''), 'finals': Counter(
            x[2].split()[0] for x in lines if x[2])}

    surf = 'obverse'
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            flush()
            pid = raw[1:8]
            cur = pid if pid in keep else None
            lines = []
            surf = 'obverse'
            continue
        if cur is None:
            continue
        if raw.startswith('@'):
            w = raw[1:].split()[0] if raw[1:].split() else ''
            if w in ('obverse', 'reverse', 'left', 'top', 'bottom', 'edge', 'seal', 'envelope'):
                surf = w
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.strip())
        if not m:
            continue
        p = parse_ur_line(m.group(1))
        if p:
            lines.append((p[0], p[1], p[2], surf))
    flush()
    return recs, meta


# ---------------- proto-cuneiform control ----------------
PC_S = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000}


def pc_records():
    T = json.load(open(os.path.join(PEDATA, 'pe2_pc_corpus.json')))
    recs, meta = [], {}
    for t in T:
        num = [(i, l) for i, l in enumerate(t['lines']) if l['numerals']]
        obv = [x for x in num if x[1]['surface'] == 'obverse']
        off = [x for x in num if x[1]['surface'] == 'reverse']
        tot_idx = off[0][0] if len(off) == 1 and len(obv) >= 2 else None
        for i, l in num:
            codes = {norm_code(c) for _, c in l['numerals']}
            if not codes <= set(PC_S):
                continue
            v = sum(n * PC_S[norm_code(c)] for n, c in l['numerals'])
            recs.append({'tab': t['id'], 'i': i, 'role': 'T' if i == tot_idx else 'E', 'sys': 'S',
                         'val': float(v), 'key': notation([[n, norm_code(c)] for n, c in l['numerals']]),
                         'signs': l['signs'], 'clean': not (l['damaged'] or l['lacuna']),
                         'obv': l['surface'] == 'obverse'})
        meta[t['id']] = {'hdr': None, 'period': t.get('period', ''), 'finals': Counter()}
    return recs, meta


# ---------------- matcher ----------------
def index_vals(recs, roles=('E',)):
    idx = defaultdict(set)
    for r in recs:
        if r['role'] in roles and r['val'] is not None:
            idx[(r['sys'], r['val'])].add(r['tab'])
    return idx


def digits_of(r, vs):
    """Decompose a value back into place-value digits for perturbation."""
    if 'nums' in r:
        return [(n, c) for n, c in r['nums']]
    v = int(r['val'])
    out = []
    for u in sorted(UR_MAX, reverse=True):
        q, v = divmod(v, u)
        if q:
            out.append((q, u))
    return out


def perturb(r, rng, vs, cs):
    """Same-shape perturbation: one digit +-1, staying within 1..max; recompute value."""
    d = digits_of(r, vs)
    for _ in range(20):
        k = rng.randrange(len(d))
        n, c = d[k]
        n2 = n + rng.choice((-1, 1))
        mx = UR_MAX.get(c, 9) if isinstance(c, int) else 9
        if n2 < 1 or n2 > mx:
            continue
        d2 = list(d)
        d2[k] = (n2, c)
        if isinstance(c, int):
            return float(sum(a * b for a, b in d2))
        return pe_value(d2, r['sys'], vs, cs)
    return None


def match_stat(totals, idx, minval=0):
    """# totals whose (sys, val) occurs as an entry on ANOTHER tablet."""
    hit = []
    for r in totals:
        if r['val'] is None or r['val'] < minval:
            continue
        tabs = idx.get((r['sys'], r['val']), set()) - {r['tab']}
        if tabs:
            hit.append((r, tabs))
    return hit
