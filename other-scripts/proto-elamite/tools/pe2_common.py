"""Shared code for the PE <-> proto-cuneiform distributional bridge (pe2_*).

Both corpora use one line schema: {'surface','signs','numerals':[[k,code]],...}.
A sign token is reduced to its base form (variants ~a,~b dropped, compounds kept).
"""
import json, math, os, random, re
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')

# Numeral-notation classes (codes are CDLI/Englund sign-list codes, i.e. notation, not readings)
CAP = {'N39A', 'N39B', 'N39C', 'N24', 'N24A', 'N24B', "N24'", 'N28', 'N28C', 'N29A', 'N29B',
       'N30A', 'N30C', 'N30D', 'N26', 'N27', 'N39N'}
BIS = {'N51', 'N51G', 'N54', 'N54G', 'N56', 'N46', 'N52'}
FRAC = {'N02', 'N03', 'N04', 'N05', 'N07A', 'N07B', 'N08', 'N08A', 'N8A', 'N8B', 'N09'}
PLAIN = {'N01', 'N14', 'N34', 'N45', 'N48', 'N50', 'N1B', 'N14B'}
SYS = ['C', 'C*', 'B', 'F', 'SDB', 'O']   # capacity, modified capacity, bisexagesimal, fractions, plain, other


def norm_code(c):
    c = c.replace('N1@', 'N01@')
    if c == 'N1':
        c = 'N01'
    return c


def system_of(nums):
    codes = {norm_code(c) for _, c in nums}
    if not codes:
        return None
    base = {c.split('@')[0] for c in codes}
    mod = any('@' in c for c in codes)
    if base & CAP:
        return 'C*' if mod else 'C'
    if base & BIS:
        return 'B'
    if base & FRAC:
        return 'F'
    if base <= PLAIN and not mod:
        return 'SDB'
    return 'O'


def magnitude(nums):
    return math.log1p(sum(k for k, _ in nums if isinstance(k, int)))


def base(sign):
    if sign.startswith('|'):
        parts = re.split(r'([+.x&])', sign.strip('|'))
        return '|' + ''.join(re.sub(r'~[A-Za-z0-9]+', '', p) for p in parts) + '|'
    return re.sub(r'~[A-Za-z0-9]+', '', sign)


def load_pe():
    T = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    out = []
    for t in T:
        lines = []
        for i, l in enumerate(t['lines']):
            sg = [base(s) for s in l['signs'] if s.startswith('M') or s.startswith('|')]
            lines.append({'surface': l['surface'], 'signs': sg, 'nums': [[n, norm_code(c)] for n, c in l['numerals']],
                          'x': ('x' in l['signs']) or l['lacuna']})
        out.append({'id': t['id'], 'site': t['provenience'], 'lines': lines})
    return out


def load_pc():
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    out = []
    for t in T:
        lines = []
        for l in t['lines']:
            sg = [base(s) for s in l['signs'] if s not in ('x', 'X')]
            lines.append({'surface': l['surface'], 'signs': sg, 'nums': l['numerals'],
                          'x': ('x' in l['signs']) or l['lacuna']})
        out.append({'id': t['id'], 'site': t['provenience'], 'period': t['period'], 'lines': lines})
    return out


FEATS = (['pos_sole', 'pos_init', 'pos_fin', 'pos_mid'] + ['sys_' + s for s in SYS] +
         ['ctx_' + s for s in SYS] + ['hdr', 'rev', 'mag'])
GROUPS = {'pos': FEATS[0:4], 'sys': FEATS[4:10], 'ctx': FEATS[10:16], 'hdr': ['hdr'], 'rev': ['rev'], 'mag': ['mag']}


def profiles(T, min_n=20, sys_override=None):
    """Return {sign: raw feature vector (dict)} and token counts.
    sys_override: optional {(tablet_idx, line_idx): system} to use shuffled systems."""
    acc = defaultdict(lambda: defaultdict(float))
    n_tok = Counter(); n_ent = Counter()
    for ti, t in enumerate(T):
        ent_sys = []
        for li, l in enumerate(t['lines']):
            if l['signs'] and l['nums']:
                s = sys_override.get((ti, li)) if sys_override else system_of(l['nums'])
                ent_sys.append((li, s))
        tab_c = Counter(s for _, s in ent_sys)
        tot = len(ent_sys)
        sysmap = dict(ent_sys)
        for li, l in enumerate(t['lines']):
            sg = l['signs']
            if not sg:
                continue
            for j, s in enumerate(sg):
                a = acc[s]; n_tok[s] += 1
                a['hdr'] += 0 if l['nums'] else 1
                a['rev'] += 1 if l['surface'] != 'obverse' else 0
                if li in sysmap:
                    n_ent[s] += 1
                    if len(sg) == 1: a['pos_sole'] += 1
                    elif j == 0: a['pos_init'] += 1
                    elif j == len(sg) - 1: a['pos_fin'] += 1
                    else: a['pos_mid'] += 1
                    sy = sysmap[li]
                    a['sys_' + sy] += 1
                    a['mag'] += magnitude(l['nums'])
                    # tablet context: systems of the OTHER entries on this tablet
                    oth = tot - 1
                    if oth > 0:
                        for k in SYS:
                            a['ctx_' + k] += (tab_c[k] - (1 if k == sy else 0)) / oth
                        a['ctx_n'] += 1
    P = {}
    for s, a in acc.items():
        if n_ent[s] < min_n:
            continue
        v = {}
        for f in GROUPS['pos'] + GROUPS['sys']:
            v[f] = a[f] / n_ent[s]
        for f in GROUPS['ctx']:
            v[f] = a[f] / a['ctx_n'] if a['ctx_n'] else 0.0
        v['hdr'] = a['hdr'] / n_tok[s]
        v['rev'] = a['rev'] / n_tok[s]
        v['mag'] = a['mag'] / n_ent[s]
        P[s] = v
    return P, n_ent


def matrix(P, signs, feats):
    X = np.array([[P[s][f] for f in feats] for s in signs], float)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    return (X - mu) / sd


def cos_sim(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def top_signs(P, n_ent, k):
    return [s for s, _ in sorted(((s, n_ent[s]) for s in P), key=lambda x: -x[1])][:k]


def split_tablets(T, seed):
    rng = random.Random(seed)
    idx = list(range(len(T))); rng.shuffle(idx)
    h = len(idx) // 2
    return [T[i] for i in idx[:h]], [T[i] for i in idx[h:]]


# Conventional CDLI sign names of proto-cuneiform commodity / person signs (sign names
# are the sign-list labels; they are used only as the outside 'function' key).
PC_FUNC = {
    'SZE': 'grain', 'ZIZ2': 'grain', 'GAR': 'grain-product', 'NINDA': 'grain-product', 'KU7': 'grain-product',
    'KASZ': 'beer', 'DUG': 'vessel',
    'UDU': 'small-cattle', 'U8': 'small-cattle', 'UDUNITA': 'small-cattle', 'SILA4': 'small-cattle',
    'MASZ': 'small-cattle', 'MASZ2': 'small-cattle', 'UD5': 'small-cattle', 'KIR11': 'small-cattle', 'ESZGAR': 'small-cattle',
    'GU4': 'large-cattle', 'AB2': 'large-cattle', 'AMAR': 'large-cattle',
    'SAL': 'person', 'KUR': 'person', 'ERIM': 'person', 'GURUSZ': 'person', 'SAG': 'person', 'N57': 'person',
    'KU6': 'fish', 'TUG2': 'textile', 'SIG2': 'textile', 'GADA': 'textile', 'GA': 'dairy', 'NI': 'dairy',
    'MUSZEN': 'bird', 'GISZ': 'wood', 'NAGA': 'plant',
}
