"""pe23 FORENSIC ACCOUNTING: digit fingerprints of counted / measured / estimated / allocated numbers.

Shared code.
  pe_entries()    PE entries with (system, value, notation) under a fixed value set
  ur3_entries()   Ur III control lines with KNOWN number type (cached; CDLI bulk ATF in scratchpad)
  canon()         canonical (greedy) notation of a value in a denomination list
  fingerprint()   5 fingerprints of a list of values (+ tablet ids)
  smooth_null()   values redrawn from a smooth local (log-uniform) distribution, same notation
Fingerprints (all computed on the canonical notation, so real and null are treated alike):
  ROUND  share of values written with ONE denomination (among v >= 2nd denomination)
  LOW    mean index of the lowest denomination used (among v >= 2nd denomination)
  FIVE   share of values divisible by 5 (among v >= 10)
  LEAD1  share whose leading (highest) denomination count is 1 (among v >= 2nd denomination)
  REP    share of entries whose value recurs elsewhere on the same tablet (same group)
  CONC   log Simpson concentration of values (sum p^2), real minus null
"""
import json, os, re, csv, math, random
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe23_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

# ---------------------------------------------------------------- PE value sets
CAPSET = {'N39B', 'N30C', 'N24', 'N30D', 'N39C'}
CNTSET = {'N01', 'N14', 'N45', 'N34'}
FRACSET = {'N08', 'N08A', 'N8B', 'N8A', 'N02', 'N51', 'N54', 'N23', 'N51G', 'N46', 'N48', 'N1B',
           'N14B', 'N91', 'N54G', 'N39A', 'N29B', 'N39N'}
# counting: grade-B value set (attack 2); alternative 'N34 = 1000' set for sensitivity
CNT_VALS = {'B': {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300},
            'ALT': {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 1000}}
CNT_DEN = {'B': [1, 10, 100, 300], 'ALT': [1, 10, 100, 1000]}
# capacity: a-priori set from the writing (attack 2) and a x2-different alternative
CAP_VALS = {'B': {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720},
            'ALT': {'N39C': 1, 'N30D': 2, 'N30C': 6, 'N24': 12, 'N39B': 60, 'N01': 300, 'N14': 1800}}
OFFICES = {
    'GRAIN': {'M010', 'M106', 'M002', 'M243', 'M075', 'M081', 'M265', 'M266', 'M296', 'M112'},
    'CLASS': {'M387', 'M388', 'M218', 'M124', 'M009', 'M066', 'M057'},
    'BARE': {'M054', 'M367', 'M001', 'M370', 'M032', '|M036+1(N30D)|', 'M206', 'M269', 'M059', 'M102'},
}


def base(s):
    return re.sub(r'~[a-z0-9]+', '', s.replace('#', '').replace('?', '').replace('!', ''))


def is_sign(s):
    return s.startswith('M') or s.startswith('|')


def pe_entries(vset='B'):
    """One record per clean entry line (signs + numerals)."""
    corp = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    out = []
    for t in corp:
        lines = t['lines']
        # header = first line with signs and no numerals
        hdr = None
        for l in lines[:2]:
            if l['signs'] and not l['numerals']:
                hdr = base(l['signs'][0])
                break
        # tablet capacity share (to treat ambiguous N01/N14-only lines)
        syst = []
        for l in lines:
            codes = {c.split('@')[0] for _, c in l['numerals']}
            syst.append('cap' if codes & CAPSET else 'oth')
        capshare = syst.count('cap') / max(1, sum(1 for l in lines if l['numerals']))
        site = t['provenience'].split(' (')[0]
        nent = sum(1 for l in lines if l['signs'] and l['numerals'])
        for i, l in enumerate(lines):
            if not l['numerals']:
                continue
            if l['lacuna'] or '...' in l['raw']:
                continue
            tail = l['raw'].split(',')[-1]
            if re.search(r'[\[?]', tail):
                continue
            nums = l['numerals']
            if any(n is None or c.startswith('n') for n, c in nums):
                continue
            codes = [c for _, c in nums]
            bcodes = {c.split('@')[0] for c in codes}
            signs = [base(s) for s in l['signs'] if is_sign(s)]
            if any(c in FRACSET for c in bcodes):
                sysname = 'FRAC'
                val = None
            elif bcodes & CAPSET:
                if not bcodes <= set(CAP_VALS[vset]):
                    continue
                sysname = 'CAP'
                val = sum(n * CAP_VALS[vset][c.split('@')[0]] for n, c in nums)
            elif bcodes <= CNTSET:
                sysname = 'CNT' if capshare <= 0.5 else 'AMBCAP'
                val = sum(n * CNT_VALS[vset][c.split('@')[0]] for n, c in nums)
            else:
                continue
            out.append({'vset': vset, 'tab': t['id'], 'site': site, 'hdr': hdr, 'surface': l['surface'],
                        'signs': signs, 'final': signs[-1] if signs else None,
                        'first': signs[0] if signs else None, 'sys': sysname, 'val': val,
                        'hatched': any('@' in c for c in codes), 'nsign': len(signs),
                        'line': i, 'nent': nent, 'raw': l['raw']})
    return out


def pe_den(sysname, vset='B'):
    if sysname == 'CAP':
        return sorted(CAP_VALS[vset].values())
    return CNT_DEN[vset]


# ---------------------------------------------------------------- Ur III control
UR_INT = {'disz': 1, 'asz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600, "szar'u": 36000}
UR_CAP = {'barig': 60, 'ban2': 10}
NUMTOK = re.compile(r"^(\d+)\(([a-z']+\d?)(?:@[a-z])?\)$")
ANIMALS = {'udu', 'u8', 'masz2', 'ud5', 'gu4', 'ab2', 'sila4', 'masz', 'kir11', 'amar',
           'ansze', 'dusu2', 'szeg9-bar', 'maszda3', 'dara3', 'udu-nita2', 'gukkal', 'sila4-nita2'}
PEOPLE = {'gurusz', 'geme2', 'erin2', 'dumu', 'szu-gi4'}
UR_CNT_DEN = [1, 10, 60, 600, 3600, 36000]
UR_CAP_DEN = [1, 10, 60, 300, 3000, 18000, 180000]


def _clean(txt):
    return re.sub(r'[_\[\]#?!<>]', '', txt).split()


def ur_int(toks, k):
    """Integer numeral run starting at toks[k] (with 'la2' subtraction). Returns (value, next k)."""
    v, j, neg = 0, k, False
    seen = False
    while j < len(toks):
        m = NUMTOK.match(toks[j])
        if m and m.group(2) in UR_INT:
            v += (-1 if neg else 1) * int(m.group(1)) * UR_INT[m.group(2)]
            seen = True
            j += 1
        elif toks[j] == 'la2' and seen and not neg:
            neg = True
            j += 1
        else:
            break
    return (v if seen else None), j


def ur_cap(toks):
    """Capacity amount: '[int] [la2 int] [n barig] [n ban2] [n sila3] gur' or '[int] sila3'. In sila."""
    toks = [t[:-3] if t.endswith('-ta') else t for t in toks]
    for k, t in enumerate(toks):
        if NUMTOK.match(t):
            break
    else:
        return None
    j = k
    run = []
    while j < len(toks) and (NUMTOK.match(toks[j]) or toks[j] == 'la2'):
        run.append(toks[j]); j += 1
    after = toks[j:j + 4]
    ints, cap, sila, sign, phase = 0, 0, 0, 1, 0
    for t in run:
        if t == 'la2':
            sign = -1
            continue
        m = NUMTOK.match(t)
        n, u = int(m.group(1)), m.group(2)
        if u in UR_CAP:
            phase = 1
            sign = 1
            cap += n * UR_CAP[u]
        elif u in UR_INT:
            if phase == 0:
                ints += sign * n * UR_INT[u]
            else:
                sila += n * UR_INT[u]
        else:
            return None
    has_gur = any(a.startswith('gur') for a in after)
    has_sila = bool(after) and after[0].startswith('sila3')
    if ints and phase == 0:
        if after and after[0].startswith('gur') or has_gur and not has_sila:
            return ints * 300
        if has_sila:
            return ints
        return None
    if ints and not has_gur:
        return None
    if sila and not has_sila:
        return None
    v = ints * 300 + cap + sila
    return v if v > 0 else None


def ur3_entries(rebuild=False):
    fn = os.path.join(CK, 'ur3.json')
    if os.path.exists(fn) and not rebuild:
        return json.load(open(fn))
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
        prov = keep[pid]
        has_gan = any('GAN2' in l for l in lines)
        for body in lines:
            if '...' in body or ' x ' in f' {body} ' or '[' in body and ']' not in body:
                continue
            toks = _clean(body)
            if not toks or toks[0] in ('szu-nigin2', 'szunigin', 'szu-nigin', 'szunigin2'):
                continue
            if any(t == 'n' or t.startswith('n(') for t in toks):
                continue
            cls = None
            val = None
            if toks[0] == 'sze-bi' and has_gan and 'gur' in toks and '-ta' not in body:
                cls, val = 'ESTIMATE', ur_cap(toks[1:])
            elif 'GAN2' in toks and re.search(r'gur-ta\b', body):
                i = toks.index('GAN2')
                cls, val = 'TARGET', ur_cap(toks[i + 1:])
            elif re.search(r'(sila3|ban2|barig)\)?-ta\b|sila3-ta\b|gur-ta\b', body) and not has_gan:
                it = max([i for i, t in enumerate(toks) if t.endswith('-ta')] or [len(toks) - 1])
                s = it
                while s > 0 and (NUMTOK.match(toks[s - 1]) or toks[s - 1] in ('la2', 'sila3', 'gur')):
                    s -= 1
                cls, val = 'RATION', ur_cap(toks[s:it + 1])
            elif NUMTOK.match(toks[0]) and not has_gan and '-ta' not in body:
                v, j = ur_int(toks, 0)
                nxt = toks[j] if j < len(toks) else ''
                if v and nxt in ANIMALS and prov.startswith('Puzri'):
                    cls, val = 'LIVESTOCK', v
                elif v and nxt in PEOPLE:
                    cls, val = 'PEOPLE', v
                elif 'gur' in toks and 'sze' in toks:
                    cls, val = 'GRAIN', ur_cap(toks)
            if cls and val and val > 0 and abs(val - round(val)) < 1e-9:
                out.append({'tab': pid, 'prov': prov, 'cls': cls, 'val': int(round(val)), 'raw': body})
    json.dump(out, open(fn, 'w'))
    return out


def ur_den(cls):
    return UR_CNT_DEN if cls in ('LIVESTOCK', 'PEOPLE') else UR_CAP_DEN


# ---------------------------------------------------------------- notation fingerprints
def canon_arrays(vals, den):
    """Vectorised greedy decomposition. Returns (nlev, lowidx, leadcount)."""
    v = np.asarray(vals, dtype=np.int64).copy()
    D = np.asarray(den, dtype=np.int64)
    L = len(D)
    counts = np.zeros((len(v), L), dtype=np.int64)
    rem = v.copy()
    for k in range(L - 1, -1, -1):
        counts[:, k] = rem // D[k]
        rem = rem % D[k]
    nz = counts > 0
    nlev = nz.sum(1)
    lowidx = np.argmax(nz, axis=1)
    hi = L - 1 - np.argmax(nz[:, ::-1], axis=1)
    lead = counts[np.arange(len(v)), hi]
    return nlev, lowidx, lead


FEATS = ['ROUND', 'LOW', 'FIVE', 'LEAD1', 'REP', 'CONC']


def fingerprint(vals, tabs, den):
    vals = np.asarray(vals, dtype=np.int64)
    nlev, low, lead = canon_arrays(vals, den)
    big = vals >= den[1]
    ten = vals >= 10
    f = {}
    f['ROUND'] = float((nlev[big] == 1).mean()) if big.any() else np.nan
    f['LOW'] = float(low[big].mean()) if big.any() else np.nan
    f['FIVE'] = float((vals[ten] % 5 == 0).mean()) if ten.any() else np.nan
    f['LEAD1'] = float((lead[big] == 1).mean()) if big.any() else np.nan
    # repeats on the same tablet
    key = Counter(zip(tabs, vals.tolist()))
    f['REP'] = float(np.mean([key[(t, v)] > 1 for t, v in zip(tabs, vals.tolist())]))
    c = Counter(vals.tolist())
    p = np.array(list(c.values()), float) / len(vals)
    f['CONC'] = float(np.log((p ** 2).sum()))
    f['nbig'] = int(big.sum())
    return f


def smooth_null(vals, rng, spread=1.5):
    """Each value redrawn log-uniformly in [v/spread, v*spread], rounded to an integer >= 1."""
    v = np.asarray(vals, float)
    u = rng.uniform(-1, 1, len(v)) * math.log(spread)
    w = np.maximum(1, np.rint(v * np.exp(u))).astype(np.int64)
    return w


def null_fp(vals, tabs, den, rng, reps=200, spread=1.5):
    arr = defaultdict(list)
    for _ in range(reps):
        f = fingerprint(smooth_null(vals, rng, spread), tabs, den)
        for k in FEATS:
            arr[k].append(f[k])
    return {k: np.array(v) for k, v in arr.items()}


def effects(vals, tabs, den, rng, reps=200, spread=1.5):
    """Real minus null mean, and z, per feature."""
    f = fingerprint(vals, tabs, den)
    nf = null_fp(vals, tabs, den, rng, reps, spread)
    eff, z = {}, {}
    for k in FEATS:
        a = nf[k][~np.isnan(nf[k])]
        if np.isnan(f[k]) or len(a) < 10:
            eff[k], z[k] = np.nan, np.nan
            continue
        eff[k] = f[k] - a.mean()
        z[k] = (f[k] - a.mean()) / (a.std() + 1e-9)
    return f, eff, z


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for r, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * n / r)
        q[i] = prev
    return q
