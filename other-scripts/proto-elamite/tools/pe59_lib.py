"""pe59 THE CRACK ATTEMPT: shared library.

One explicit working reading of the Proto-Elamite administrative system (READING below), assembled
from the surviving B-grade (and strongest C-grade) results of earlier loops, written as a
machine-checkable model, frozen by hash, and turned into a generative grammar that writes whole
tablets (header, entries with role / commodity class / quantity / unit system, totals).

Scored on held-out tablets against simpler models and against the same grammar with sign roles
shuffled.  The same pipeline runs on proto-cuneiform with its known readings as calibration.

Tablet format (both corpora):
  {'id','site','lines':[{'surf','signs':[base forms],'clean':bool,'nums':[(n,code)],'numclean':bool,
                          'role': 'H' header | 'E' entry | 'T' total | 'G' edge tag | 'N' other}]}
"""
import os, sys, re, json, math, hashlib, random, csv
from collections import Counter, defaultdict
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe59_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

from common import load, base, is_sign  # noqa: E402

# ------------------------------------------------------------------ THE READING (pe59-R1)
READING = {
    'name': 'pe59-R1 working reading of the Proto-Elamite administrative system',
    'systems': {
        'CAP': {'grade': 'B-', 'src': 'pe27 (M106, M288 rates), pe42 (5 tablets close with N14=6 N01), attack2',
                'unit': 'N39C',
                'values': {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720,
                           'N45': 7200, 'N34': 21600, 'N48': 216000},
                'litres_per_N39C': [0.6, 0.8], 'litres_alias': [5, 7], 'litres_grade': 'C (pe16, pe55)'},
        'CNT': {'grade': 'B', 'src': 'attack2 (N14=10 pinned; 1/10/100/300), pe6/pe8 (system per tablet)',
                'maps': {'dec': {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300, 'N48': 3000, 'N51': 120},
                         'sex': {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N51': 120}},
                'fraction_values': ['1/2', '1/6'], 'fraction_codes': ['N08', 'N08A', 'N8B', 'N02']},
        'AMB_rule': {'grade': 'B', 'src': 'pe6, pe8, pe42, attack2',
                     'rule': 'one system per tablet; plain N01/N14/N45/N34 take the tablet system '
                             '(capacity tablet: N14 = 6 N01 = 720 N39C; count tablet: N14 = 10 N01)'},
    },
    'header': {
        'OPEN_GEN': {'signs': ['M157'], 'grade': 'A', 'src': 'test d (37% of headers)', 'gloss': 'general opener'},
        'OPEN_327': {'signs': ['|M327+M342|', 'M327'], 'prefix_match': '|M327+', 'grade': 'B', 'src': 'test d, pe34 (M342 header-maker C)',
                     'gloss': 'second opener family'},
        'OPEN_CAP': {'signs': ['M136', 'M388'], 'grade': 'C', 'src': 'test d (M136 leans capacity), pe42 (M388 tablets read in capacity)',
                     'gloss': 'capacity-account opener'},
        'OPEN_OTHER': {'signs': ['|M377+M320+M377|', 'M005', 'M305', 'M247'], 'grade': 'B', 'src': 'test d', 'gloss': 'other opener'},
    },
    'edge_tag': {'grade': 'B', 'src': 'pe23', 'rule': 'top-edge 1(N34) (or 2(N01)) is a constant document-type tag, '
                 'commoner on M157 tablets, never a sum'},
    'entry': {'grade': 'A', 'src': 'test a', 'frame': '[prefix] + [name-string from the tablet pool] + [final class sign] , numeral'},
    'roles': {
        'MEASURED': {'grade': 'A/B', 'src': 'test b, pe6, pe15 GRAIN office',
                     'signs': ['M297', 'M002', 'M036', 'M243', 'M075', 'M106', 'M010', 'M379', 'M050',
                               'M081', 'M265', 'M266', 'M296', 'M112'],
                     'gloss': 'commodity measured in capacity (grain-class)'},
        'COUNTED': {'grade': 'A/B', 'src': 'test b, pe6',
                    'signs': ['M263', 'M346', 'M264', 'M003', 'M032', 'M373', 'M102', 'M362', 'M317', 'M149',
                              'M046', 'M072'],
                    'gloss': 'counted, never measured (heads, animals, objects)'},
        'FRACLINE': {'grade': 'B', 'src': 'test b, pe51', 'signs': ['M376'], 'gloss': 'line-level class that takes fractions'},
        'ALLOT': {'grade': 'B', 'src': 'pe27, pe28, pe32', 'signs': ['M288'],
                  'gloss': 'standard allotment line: 60 N39C = 2(N39B) 1(N24) (= 1/2 N01 capacity) per unit of the '
                           'preceding person line; sometimes 1 N01 per unit'},
        'PERSON': {'grade': 'B', 'src': 'pe28, pe32 (closed ration class)',
                   'signs': ['M388', 'M054', 'M124', 'M371', 'M057', 'M128', 'M218', 'M220', 'M228', 'M230',
                             'M301', 'M066', 'M320', 'M370', '|M370+M388|'],
                   'gloss': 'rationed person-like unit'},
        'PREFIX': {'grade': 'A', 'src': 'test a', 'signs': ['M387', 'M157', 'M370', 'M124', 'M305', 'M038', 'M111'],
                   'gloss': 'entry-initial qualifier'},
        'TRIO': {'grade': 'B-', 'src': 'pe51', 'signs': ['M387', 'M036', 'M297'], 'gloss': 'capacity triangle'},
        'NAMEFRAME': {'grade': 'C', 'src': 'pe45', 'final': ['M056', 'M010', 'M001'], 'initial': ['M157', 'M210']},
    },
    'rates': {'M288_per_unit_N39C': [60, 120], 'grade': 'B/B-', 'src': 'pe27, pe28'},
    'weights': {'grade': 'B', 'src': 'pe52 frozen file (log quantity shift vs same-tablet siblings)', 'file': 'pe52_frozen_weights.json'},
    'fixed_allotment': {'grade': 'B-', 'src': 'pe44', 'rule': 'a final sign repeated on a tablet tends to repeat its numeral'},
    'two_line_records': {'grade': 'A/B', 'src': 'pe12, pe14', 'rule': 'name M288 n / M376 1/4 records on ~20 tablets'},
    'totals': {'grade': 'A', 'src': 'pe38 (no total sign), pe30 (no exclusion rule), pe25 (same hand)',
               'rule': 'a single numeric line on the reverse is the sum of the entries in the tablet system; it has no '
                       'dedicated sign'},
    'name_pool': {'grade': 'B', 'src': 'pe9, pe45', 'rule': 'name signs drawn from a freely reused tablet pool'},
    'outposts': {'grade': 'C', 'src': 'pe48', 'Yahya': ['M056', 'M044', 'M219', 'M136']},
    'herd': {'grade': 'B', 'src': 'pe20', 'reference_count': 'M362', 'series': 'plain vs ~a'},
}

CAPSET = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N28', 'N29B', 'N39N', 'N24A', 'N28C', 'N29A', 'N30A',
          'N26', 'N27'}
BIS = {'N51', 'N51G', 'N54', 'N54G', 'N56', 'N46', 'N52'}
FRAC = {'N02', 'N03', 'N04', 'N05', 'N07A', 'N07B', 'N08', 'N08A', 'N8A', 'N8B', 'N09'}
AMB = {'N01', 'N14', 'N45', 'N34', 'N48', 'N50'}
CLS = ['CAP', 'AMB', 'BIS', 'FRAC', 'MOD', 'OTH']


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def norm_code(c):
    c = str(c).replace('N1@', 'N01@')
    return 'N01' if c == 'N1' else c


def ncls(nums):
    codes = {norm_code(c) for _, c in nums}
    if any('@' in c for c in codes):
        return 'MOD'
    if codes & CAPSET:
        return 'CAP'
    if codes & BIS:
        return 'BIS'
    if codes & FRAC:
        return 'FRAC'
    if codes <= AMB:
        return 'AMB'
    return 'OTH'


def nkey(nums):
    d = Counter()
    for n, c in nums:
        d[norm_code(c)] += n
    return '|'.join('%s:%s' % (c, d[c]) for c in sorted(d))


def unkey(k):
    out = []
    for p in k.split('|'):
        c, n = p.split(':')
        out.append((Fr(n), c))
    return out


# ------------------------------------------------------------------ value maps
def pe_maps():
    cap = {c: Fr(v) for c, v in READING['systems']['CAP']['values'].items()}
    cnt = {}
    for nm, m in READING['systems']['CNT']['maps'].items():
        for fr in (Fr(1, 2), Fr(1, 6)):
            d = {c: Fr(v) for c, v in m.items()}
            for c in READING['systems']['CNT']['fraction_codes']:
                d[c] = fr
            cnt['%s%d' % (nm, fr.denominator)] = d
    return cap, cnt


PC_S = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000}
PC_B = {'N01': 1, 'N14': 10, 'N34': 60, 'N51': 120}
# SE (grain capacity): plain codes 1/6/60/180, small units by the archaic chain (assumed, Englund)
PC_SE = {'N39A': Fr(1, 5), 'N24': Fr(1, 10), 'N01': 1, 'N14': 6, 'N45': 60, 'N34': 180, 'N48': 1800}


def pc_maps():
    cap = {c: Fr(v) for c, v in PC_SE.items()}
    cnt = {'S': {c: Fr(v) for c, v in PC_S.items()}, 'B': {c: Fr(v) for c, v in PC_B.items()}}
    return cap, cnt


def value(nums, m):
    t = Fr(0)
    for n, c in nums:
        c = norm_code(c)
        if c not in m:
            return None
        t += Fr(n) * m[c]
    return t


def canon(x, m):
    out = []
    seen = set()
    for c, v in sorted(m.items(), key=lambda kv: -kv[1]):
        if v in seen or v <= 0:
            continue
        seen.add(v)
        k = int(x // v)
        if k:
            out.append((k, c)); x -= k * v
    return out if x == 0 and out else None


# ------------------------------------------------------------------ corpora
def _roles(lines):
    """Assign observable line roles."""
    num_idx = [i for i, l in enumerate(lines) if l['nums']]
    first_num = num_idx[0] if num_idx else len(lines)
    rev_num = [i for i in num_idx if lines[i]['surf'] == 'reverse']
    obv_num = [i for i in num_idx if lines[i]['surf'] == 'obverse']
    tot = rev_num[0] if (len(rev_num) == 1 and len(obv_num) >= 2) else None
    for i, l in enumerate(lines):
        if l['surf'] in ('top', 'bottom', 'left', 'right', 'edge') and l['nums'] and not l['signs']:
            l['role'] = 'G'
        elif i < first_num and not l['nums'] and l['surf'] == 'obverse':
            l['role'] = 'H' if l['signs'] else 'N'
        elif i == tot:
            l['role'] = 'T'
        elif l['nums']:
            l['role'] = 'E'
        else:
            l['role'] = 'N'
    return lines


def build_pe():
    fn = os.path.join(CK, 'pe_tabs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    T = load()
    out = []
    for t in T:
        L = []
        for l in t['lines']:
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            clean = not ('x' in l['signs'] or l['lacuna'] or '...' in l['raw'] or '[' in l['raw'])
            nums = [[n, norm_code(c)] for n, c in l['numerals']]
            numclean = bool(nums) and all(n is not None and c != 'n' and isinstance(n, int) for n, c in nums) \
                and not re.search(r'\[|\.\.\.', l['raw'].split(',')[-1])
            L.append({'surf': l['surface'], 'signs': sg, 'clean': clean, 'nums': nums if numclean or not nums else
                      [[n, c] for n, c in nums if isinstance(n, int) and c != 'n'], 'numclean': numclean,
                      'had_nums': bool(nums)})
        for l in L:
            if l['had_nums'] and not l['nums']:
                l['nums'] = [[0, 'n']]
        _roles(L)
        site = t['provenience'].split(' (')[0]
        out.append({'id': t['id'], 'site': site, 'lines': L})
    json.dump(out, open(fn, 'w'))
    return out


def pc_admin_ids():
    fn = os.path.join(CK, 'pc_admin_ids.json')
    if os.path.exists(fn):
        return set(json.load(open(fn)))
    csv.field_size_limit(10 ** 9)
    ids = set()
    for r in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), errors='replace')):
        if r['period'].startswith('Uruk') and r['genre'].startswith('Admin') and r['id_text'].isdigit():
            ids.add('P%06d' % int(r['id_text']))
    json.dump(sorted(ids), open(fn, 'w'))
    return ids


def build_pc():
    fn = os.path.join(CK, 'pc_tabs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    adm = pc_admin_ids()
    P = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    out = []
    for t in P:
        if t['id'] not in adm:
            continue
        L = []
        for l in t['lines']:
            sg = [base(s) for s in l['signs'] if s not in ('x', 'X')]
            clean = not ('x' in l['signs'] or 'X' in l['signs'] or l['lacuna'] or '...' in l['raw'] or '[' in l['raw'])
            nums = [[n, norm_code(c)] for n, c in l['numerals']]
            numclean = bool(nums) and all(isinstance(n, int) and c != 'n' for n, c in nums) and clean
            L.append({'surf': l['surface'], 'signs': sg, 'clean': clean,
                      'nums': nums if (numclean or not nums) else [[0, 'n']], 'numclean': numclean,
                      'had_nums': bool(nums)})
        _roles(L)
        out.append({'id': t['id'], 'site': t.get('provenience', '').split(' (')[0], 'lines': L})
    json.dump(out, open(fn, 'w'))
    return out


def split(T, tag='pe59'):
    tr, ho = [], []
    for t in T:
        (tr if int(hashlib.sha256((tag + t['id']).encode()).hexdigest(), 16) % 2 == 0 else ho).append(t)
    return tr, ho


# ------------------------------------------------------------------ role assignment
def pe_roles():
    R = READING['roles']
    H = READING['header']
    w = json.load(open(os.path.join(DATA, READING['weights']['file'])))['signs']
    roles = {
        'MEASURED': set(R['MEASURED']['signs']), 'COUNTED': set(R['COUNTED']['signs']),
        'FRACLINE': set(R['FRACLINE']['signs']), 'ALLOT': set(R['ALLOT']['signs']),
        'PERSON': set(R['PERSON']['signs']), 'PREFIX': set(R['PREFIX']['signs']),
        'H_GEN': set(H['OPEN_GEN']['signs']), 'H_327': set(H['OPEN_327']['signs']),
        'H_CAP': set(H['OPEN_CAP']['signs']), 'H_OTH': set(H['OPEN_OTHER']['signs']),
        'OUT_Yahya': set(READING['outposts']['Yahya']),
    }
    weights = {s: v['tab'] for s, v in w.items() if v['both']}
    return {'sets': roles, 'weights': weights, 'h327_prefix': '|M327+', 'allot_sign': 'M288'}


# proto-cuneiform calibration reading: from the known Sumerological readings of the signs (answer key),
# not from data.  Grain measured in the SE capacity system; grain products / rations in B; animals,
# people, textiles and vessels counted in S.
PC_KNOWN = {
    'MEASURED': {'SZE', 'ZIZ2', '|SZE+NAM2|'},
    'BISCLASS': {'GAR', 'NINDA2', 'KU6'},
    'COUNTED': {'UDU', 'U8', 'UDUNITA', 'SILA4', 'MASZ', 'MASZ2', 'UD5', 'GU4', 'AB2', 'SAL', 'KUR', 'ERIM', 'TUG2',
                'DUG', 'ANSZE', 'SZAH2', 'KISZ', 'SIG2', 'KUR2', 'SAG', 'GURUSZ', 'AMAR'},
    'PERSON': {'SAL', 'KUR', 'ERIM', 'SAG', 'GURUSZ', 'KUR2'},
}


def pc_roles():
    s = {'MEASURED': set(PC_KNOWN['MEASURED']), 'COUNTED': set(PC_KNOWN['COUNTED']),
         'FRACLINE': set(), 'ALLOT': set(PC_KNOWN['BISCLASS']), 'PERSON': set(PC_KNOWN['PERSON']),
         'PREFIX': set(), 'H_GEN': set(), 'H_327': set(), 'H_CAP': set(), 'H_OTH': set(), 'OUT_Yahya': set()}
    return {'sets': s, 'weights': {}, 'h327_prefix': None, 'allot_sign': None}


def shuffle_roles(roles, T, rng):
    """Same grammar, sign roles shuffled: every role set is replaced by a frequency-matched random set of
    signs (drawn without overlap from the corpus vocabulary); weights are re-dealt among signs; the
    allotment sign is replaced by a frequency-matched sign."""
    freq = Counter(s for t in T for l in t['lines'] for s in l['signs'])
    vocab = sorted(freq, key=lambda s: -freq[s])
    rank = {s: i for i, s in enumerate(vocab)}

    def match(s, used):
        r = rank.get(s, len(vocab) - 1)
        for d in range(0, len(vocab)):
            for j in (r + d, r - d):
                if 0 <= j < len(vocab):
                    c = vocab[j]
                    if c not in used and rng.random() < 0.5:
                        used.add(c)
                        return c
        return vocab[rng.randrange(len(vocab))]

    perm = {}
    used = set()
    allsigns = set().union(*roles['sets'].values()) | set(roles['weights']) | (
        {roles['allot_sign']} if roles['allot_sign'] else set())
    # include signs matched by the 327 prefix
    for s in sorted(allsigns, key=lambda s: rank.get(s, 10 ** 6)):
        perm[s] = match(s, used)
    new = {'sets': {k: {perm[s] for s in v} for k, v in roles['sets'].items()},
           'weights': {perm[s]: w for s, w in roles['weights'].items()},
           'h327_prefix': None, 'allot_sign': perm.get(roles['allot_sign']) if roles['allot_sign'] else None}
    if roles['h327_prefix']:
        fam = [s for s in vocab if s.startswith(roles['h327_prefix'])]
        new['sets']['H_327'] |= {match(s, used) for s in fam}
    return new


def header_role(sign, roles):
    S = roles['sets']
    if sign in S['H_GEN']:
        return 'H_GEN'
    if sign in S['H_327'] or (roles['h327_prefix'] and sign.startswith(roles['h327_prefix'])):
        return 'H_327'
    if sign in S['H_CAP']:
        return 'H_CAP'
    if sign in S['H_OTH']:
        return 'H_OTH'
    return 'H_X'


def line_role(signs, roles):
    """Role of an entry line from its final sign (then any sign)."""
    S = roles['sets']
    if not signs:
        return 'BARE'
    f = signs[-1]
    if roles['allot_sign'] and f == roles['allot_sign'] or f in S['ALLOT']:
        return 'ALLOT'
    if f in S['FRACLINE']:
        return 'FRACLINE'
    if f in S['MEASURED']:
        return 'MEASURED'
    if f in S['COUNTED']:
        return 'COUNTED'
    if f in S['PERSON']:
        return 'PERSON'
    if any(s in S['MEASURED'] for s in signs):
        return 'MEASURED_IN'
    if any(s in S['COUNTED'] for s in signs):
        return 'COUNTED_IN'
    return 'OTHER'


LROLES = ['ALLOT', 'FRACLINE', 'MEASURED', 'COUNTED', 'PERSON', 'MEASURED_IN', 'COUNTED_IN', 'OTHER', 'BARE']
HROLES = ['H_GEN', 'H_327', 'H_CAP', 'H_OTH', 'H_X', 'NONE']


def tablet_type(t):
    """Observable tablet system: CAPT if any capacity numeral; CNTT if count-only evidence
    (fractions, bisexagesimal, or a plain group not writable in capacity notation: N01 >= 6, N14 >= 10 ...)."""
    cl = [ncls(l['nums']) for l in t['lines'] if l['nums'] and l['nums'][0][1] != 'n' and l['role'] in 'ETG']
    if 'CAP' in cl:
        return 'CAPT'
    for l in t['lines']:
        if not l['nums'] or l['nums'][0][1] == 'n' or l['role'] not in 'ET':
            continue
        c = ncls(l['nums'])
        if c in ('FRAC', 'BIS'):
            return 'CNTT'
        if c == 'AMB':
            d = Counter()
            for n, cc in l['nums']:
                d[cc] += n
            if d['N01'] >= 6:
                return 'CNTT'
    return None
