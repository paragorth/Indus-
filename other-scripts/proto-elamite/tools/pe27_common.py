"""pe27 NUMBERS IN TWO LANGUAGES: exact ratio scans between quantities written in different
number systems on the same tablet ("bilingual numbers": a count and its grain / fodder / ration
equivalent).  Shared code.

  pe_tablets(cfg)   per tablet: list of quantities (system, value as Fraction, signs, line, surface)
  ur3_tablets()     Ur III control: counts of people / animals and capacity amounts (sila) on the
                    same tablet, with the written '-ta' rates kept apart as the truth (never scanned)
  hits(tabs, X, Y)  per tablet: set of ratios Y/X over all pairs of an X line and a Y line
  scan(...)         H(r) = number of tablets with >= 1 pair at exact ratio r; permutation null
                    (Y line-sets re-dealt among tablets that carry both systems), search-corrected
No sign readings from anyone are used.  Values: PE capacity in N39C units (a-priori set fixed in
attack_arith from the writing); counts sexagesimal unless the tablet is decimal (N14 >= 6 in a line).
"""
import json, os, re, sys, random, math
from fractions import Fraction as Fr
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe27_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa: E402

C_CODES = {'N39B', 'N30C', 'N24', 'N30D', 'N39C'}
CNT_CODES = {'N01', 'N14', 'N34', 'N45', 'N48'}
B_CODES = {'N51', 'N54', 'N46', 'N51G', 'N54G'}
CAPV = {'apriori': {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720},
        'alt_d': {'N39C': 1, 'N30D': 2, 'N30C': 6, 'N24': 12, 'N39B': 60, 'N01': 300, 'N14': 1800},
        'alt_b': {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 144, 'N14': 1440}}
SEX = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}
DEC = {'N01': 1, 'N14': 10, 'N45': 100, 'N48': 1000, 'N34': 60}
# bisexagesimal (proto-cuneiform-style) values for the B group, in units of N01-equivalents
BV = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 120, 'N48': 1200, 'N51': 1, 'N54': 10, 'N46': 120,
      'N51G': 1, 'N54G': 10}
CAP_CLASS = {'M297', 'M002', 'M036', 'M243', 'M106', 'M075', 'M010', 'M379', 'M050'}


def _clean_line(l):
    if l.get('lacuna') or '...' in l['raw']:
        return False
    tail = l['raw'].split(',')[-1]
    if re.search(r'[\[?]', tail):
        return False
    return all(isinstance(n, int) and n > 0 and not c.startswith('n') for n, c in l['numerals'])


def pe_tablets(capset='apriori', amb='sign', allow_damage=True):
    """amb: how to treat N01/N14-only lines on tablets that also have capacity lines:
       'sign' -> capacity if final sign is a capacity-class sign, else count; 'drop' -> skip them."""
    cv = CAPV[capset]
    out = []
    for t in load():
        if t['object_type'] != 'tablet':
            continue
        lines = t['lines']
        has_cap = any({c.split('@')[0] for _, c in l['numerals']} & C_CODES for l in lines)
        dec = any(sum(n for n, c in l['numerals'] if c == 'N14' and isinstance(n, int)) >= 6
                  for l in lines if not ({c.split('@')[0] for _, c in l['numerals']} & C_CODES))
        cntv = DEC if dec else SEX
        Q = []
        for i, l in enumerate(lines):
            if not l['numerals'] or not _clean_line(l):
                continue
            if not allow_damage and l.get('damaged'):
                continue
            codes = [c for _, c in l['numerals']]
            bc = {c.split('@')[0] for c in codes}
            at = any('@' in c for c in codes)
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            fin = sg[-1] if sg else '-'
            if bc & C_CODES:
                if not bc <= set(cv):
                    continue
                sysn = 'CAP@' if at else 'CAP'
                v = sum(n * cv[c.split('@')[0]] for n, c in l['numerals'])
            elif bc & B_CODES:
                if not bc <= set(BV):
                    continue
                sysn = 'B'
                v = sum(n * BV[c.split('@')[0]] for n, c in l['numerals'])
            elif bc <= CNT_CODES:
                if has_cap and bc <= {'N01', 'N14'}:
                    if fin in CAP_CLASS:
                        if amb == 'drop':
                            continue
                        sysn = 'CAP@' if at else 'CAP'
                        v = sum(n * cv[c.split('@')[0]] for n, c in l['numerals'])
                    else:
                        if amb == 'drop':
                            continue
                        sysn = 'CNT@' if at else 'CNT'
                        v = sum(n * cntv[c.split('@')[0]] for n, c in l['numerals'])
                else:
                    sysn = 'CNT@' if at else 'CNT'
                    v = sum(n * cntv[c.split('@')[0]] for n, c in l['numerals'])
            else:
                continue
            Q.append({'sys': sysn, 'v': Fr(v), 'fin': fin, 'first': sg[0] if sg else '-',
                      'line': i, 'surf': l['surface'], 'nums': [[n, c] for n, c in l['numerals']],
                      'raw': l['raw']})
        if Q:
            out.append({'id': t['id'], 'site': t['provenience'].split(' (')[0], 'Q': Q})
    return out


# ---------------------------------------------------------------- Ur III control
UR_INT = {'disz': 1, 'asz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600, "szar'u": 36000}
UR_CAP = {'barig': 60, 'ban2': 10}
NUMTOK = re.compile(r"^(\d+)\(([a-z']+\d?)(?:@[a-z])?\)$")
NOUNS = {'gurusz', 'geme2', 'erin2', 'dumu', 'szu-gi4', 'lu2', 'udu', 'u8', 'masz2', 'ud5', 'gu4',
         'ab2', 'sila4', 'masz', 'kir11', 'amar', 'ansze', 'dusu2', 'udu-nita2', 'gukkal',
         'ug3-IL2', 'dumu-munus', 'nita2', 'munus', 'kinkin2', 'ninda'}


def _toks(txt):
    return re.sub(r'[_\[\]#?!<>*]', '', txt).split()


def ur_int(toks, k):
    v, j, seen = 0, k, False
    while j < len(toks):
        m = NUMTOK.match(toks[j])
        if m and m.group(2) in UR_INT:
            v += int(m.group(1)) * UR_INT[m.group(2)]
            seen = True
            j += 1
        else:
            break
    return (v if seen else None), j


def ur_cap(toks):
    """'[n gur-units] [n barig] [n ban2] [n sila] (sila3|gur)' -> sila, or None."""
    k = 0
    while k < len(toks) and not NUMTOK.match(toks[k]):
        k += 1
    if k == len(toks):
        return None
    j = k
    run = []
    while j < len(toks) and NUMTOK.match(toks[j]):
        run.append(toks[j]); j += 1
    after = toks[j:j + 4]
    has_gur = any(a == 'gur' or a.startswith('gur-') for a in after)
    has_sila = bool(after) and after[0].startswith('sila3')
    ints, cap, sila, phase = 0, 0, 0, 0
    for t in run:
        m = NUMTOK.match(t)
        n, u = int(m.group(1)), m.group(2)
        if u in UR_CAP:
            phase = 1
            cap += n * UR_CAP[u]
        elif u in UR_INT:
            if phase == 0:
                ints += n * UR_INT[u]
            else:
                sila += n * UR_INT[u]
        else:
            return None
    if phase == 0:
        if ints and has_sila:
            return ints
        if ints and has_gur:
            return ints * 300
        return None
    if sila and not has_sila:
        return None
    if ints and not has_gur:
        return None
    v = ints * 300 + cap + sila
    return v if v > 0 else None


def ur3_tablets(rebuild=False):
    fn = os.path.join(CK, 'ur3_tabs.json')
    if os.path.exists(fn) and not rebuild:
        d = json.load(open(fn))
        for t in d:
            for q in t['Q']:
                q['v'] = Fr(q['v'])
        return d
    import csv
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III'):
            keep['P%06d' % int(row['id_text'])] = row['provenience'].split(' (')[0]
    texts = defaultdict(list)
    cur = None
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            pid = raw[1:8]
            cur = pid if pid in keep else None
            continue
        if cur and raw[:1].isdigit():
            m = re.match(r"^\d+'?\.\s+(.*)$", raw.rstrip())
            if m:
                texts[cur].append(m.group(1))
    out = []
    for pid, lines in texts.items():
        Q, rates = [], []
        for i, body in enumerate(lines):
            if '...' in body or ' x ' in f' {body} ' or ('[' in body and ']' not in body):
                continue
            tk = _toks(body)
            if not tk or any(t == 'n' or t.startswith('n(') for t in tk):
                continue
            if '-ta' in body:
                mm = re.search(r'((?:\d+\((?:barig|ban2|disz|u|asz|gesz2)\)\s*)+(?:sila3|gur)?)-ta\b', body)
                if mm:
                    rt = mm.group(1).strip()
                    if not re.search(r'(sila3|gur)$', rt):
                        rt = rt + (' sila3' if not re.search(r'(barig|ban2)\)$', rt) else '')
                    r = ur_cap(_toks(rt) + (['sila3'] if re.search(r'(barig|ban2)\)$', rt) else []))
                    if r:
                        rates.append(r)
            # count line (may carry a rate in the same line: then only the count is used)
            v, j = ur_int(tk, 0)
            if v and j < len(tk) and tk[j] in NOUNS:
                Q.append({'sys': 'CNT', 'v': v, 'fin': tk[j], 'first': tk[j], 'line': i,
                          'surf': 'o', 'raw': body})
                continue
            if '-ta' in body:
                continue
            tk2 = tk[1:] if tk[0] in ('sze-bi', 'ziz2-bi', 'sze') else tk
            c = ur_cap(tk2)
            if c and abs(c - round(c)) < 1e-9:
                noun = [t for t in tk if not NUMTOK.match(t)]
                Q.append({'sys': 'CAP', 'v': int(c), 'fin': noun[0] if noun else '-',
                          'first': tk[0], 'line': i, 'surf': 'o', 'raw': body})
        sy = {q['sys'] for q in Q}
        if {'CNT', 'CAP'} <= sy:
            out.append({'id': pid, 'site': keep[pid], 'Q': Q, 'rates': sorted(set(rates))})
    json.dump(out, open(fn, 'w'), default=str)
    for t in out:
        for q in t['Q']:
            q['v'] = Fr(q['v'])
    return out


# ---------------------------------------------------------------- scanning
def pairs_ratios(QX, QY, adj=False):
    """Set of exact ratios y/x over all pairs (adj: only pairs on neighbouring lines)."""
    rs = set()
    for a in QX:
        for b in QY:
            if adj and abs(a['line'] - b['line']) != 1:
                continue
            rs.add(b['v'] / a['v'])
    return rs


def split_sys(tabs, X, Y):
    """Tablets carrying both systems -> (list of X line-lists, list of Y line-lists, ids)."""
    xs, ys, ids = [], [], []
    for t in tabs:
        qx = [q for q in t['Q'] if q['sys'] == X]
        qy = [q for q in t['Q'] if q['sys'] == Y]
        if qx and qy:
            xs.append(qx); ys.append(qy); ids.append(t['id'])
    return xs, ys, ids


def H_of(xs, ys, perm=None, adj=False):
    H = Counter()
    for i in range(len(xs)):
        qy = ys[perm[i]] if perm is not None else ys[i]
        for r in pairs_ratios(xs[i], qy, adj=adj and perm is None):
            H[r] += 1
    return H


def scan(xs, ys, nperm=200, seed=0, minH=3):
    """Real H(r) vs permutation null.  Score = Poisson upper-tail -log10 p with the null mean;
    search correction = distribution of the max score over r in each permutation (scored against
    the null means of the other permutations)."""
    rng = np.random.default_rng(seed)
    n = len(xs)
    Hr = H_of(xs, ys)
    Hp = []
    for k in range(nperm):
        Hp.append(H_of(xs, ys, perm=rng.permutation(n)))
    keys = set(Hr)
    for h in Hp:
        keys |= set(h)
    keys = list(keys)
    M = np.array([[h.get(r, 0) for r in keys] for h in Hp], float)
    tot = M.sum(0)
    from scipy.stats import poisson

    def score(obs, mu):
        mu = np.maximum(mu, 0.05)
        return -np.log10(np.maximum(poisson.sf(obs - 1, mu), 1e-300))
    real = np.array([Hr.get(r, 0) for r in keys], float)
    mu = tot / nperm
    s_real = np.where(real >= minH, score(real, mu), 0)
    mx_null = []
    for k in range(nperm):
        mu_k = (tot - M[k]) / (nperm - 1)
        s = np.where(M[k] >= minH, score(M[k], mu_k), 0)
        mx_null.append(s.max())
    mx_null = np.array(mx_null)
    order = np.argsort(-s_real)
    top = []
    for j in order[:25]:
        if s_real[j] <= 0:
            break
        top.append({'r': str(keys[j]), 'rf': float(keys[j]), 'H': int(real[j]),
                    'mu': round(float(mu[j]), 2), 'score': round(float(s_real[j]), 2),
                    'p_corr': float((1 + (mx_null >= s_real[j]).sum()) / (1 + nperm))})
    return {'n_tab': n, 'max_real': float(s_real.max()) if len(s_real) else 0.0,
            'null_max_q95': float(np.quantile(mx_null, 0.95)), 'null_max_med': float(np.median(mx_null)),
            'top': top, 'p_global': float((1 + (mx_null >= s_real.max()).sum()) / (1 + nperm))}


def plant(tabs, X, Y, r, frac, rng):
    """Copy of tabs where on a fraction of tablets carrying both systems one Y line = r * one X line."""
    out = []
    for t in tabs:
        Q = [dict(q) for q in t['Q']]
        qx = [q for q in Q if q['sys'] == X]
        qy = [q for q in Q if q['sys'] == Y]
        if qx and qy and rng.random() < frac:
            a = qx[rng.integers(len(qx))]
            b = qy[rng.integers(len(qy))]
            b['v'] = a['v'] * Fr(r)
        out.append({'id': t['id'], 'site': t['site'], 'Q': Q})
    return out


