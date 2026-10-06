"""pe66 THE KILL SWEEP: shared code.

Every grade-C Proto-Elamite guess in FINDINGS.md is put to its own stated kill
test (or the strongest test the existing corpus allows), with fresh seeds,
held-out tablet halves, permutation nulls and frequency-matched DECOY signs run
through the identical test, so that 'survive' means more than 'the test is easy'.

Fresh seeds: every random draw in pe66 uses seeds 66000+ (never used before).
"""
import collections, hashlib, json, math, os, re, sys
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe66_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
from common import load, base, is_sign, system_of  # noqa: E402

CAPSET = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N28', 'N29B', 'N39N'}
# capacity in N39C units (pe59 ladder: N39C 1 N30D 2 N30C 4 N24 12 N39B 24 N01 120 N14 720)
CAPV = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720,
        'N45': 7200, 'N34': 21600, 'N48': 216000}
# counts: 1/10/100/300 (grade B), fractions dropped (value None)
CNTV = {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300, 'N48': 3000}

T = load()
TAB = {t['id']: t for t in T}


def site(t):
    p = t.get('provenience', '')
    for k in ('Susa', 'Yahya', 'Malyan', 'Sialk', 'Sofalin'):
        if k in p:
            return k
    return 'other'


def signs_of(l):
    return [s for s in l['signs'] if is_sign(s)]


def clean_num(l):
    tail = l['raw'].split(',')[-1] if ',' in l['raw'] else l['raw']
    return (not l['lacuna'] and '...' not in l['raw'] and not re.search(r'[\[?]', tail)
            and all(n is not None and c != 'n' and '@' not in c for n, c in l['numerals']))


def val(nums, table):
    v = 0
    for n, c in nums:
        if c not in table:
            return None
        v += n * table[c]
    return v


def lsys(l):
    s = system_of(l['numerals'])
    return s


def entries(t):
    """Numeral lines with >= 1 sign. Each: dict(i, surf, signs, bsigns, final, nums, sys, cnt, cap, clean)."""
    out = []
    for i, l in enumerate(t['lines']):
        sg = signs_of(l)
        if not l['numerals'] or not sg:
            continue
        cl = clean_num(l)
        out.append(dict(i=i, surf=l['surface'], signs=sg, bsigns=[base(s) for s in sg],
                        final=base(sg[-1]), nums=l['numerals'], sys=lsys(l),
                        cnt=val(l['numerals'], CNTV) if cl else None,
                        cap=val(l['numerals'], CAPV) if cl else None, clean=cl))
    return out


ENT = {t['id']: entries(t) for t in T}


def header(t):
    """First line if it carries signs and no numeral (the tablet header)."""
    for l in t['lines']:
        if l['surface'] != 'obverse':
            continue
        sg = signs_of(l)
        if sg and not l['numerals']:
            return [base(s) for s in sg]
        return None
    return None


# ------------------------------------------------------------- frequencies
TOK = collections.Counter()
TABN = collections.defaultdict(set)
for t in T:
    for l in t['lines']:
        for s in signs_of(l):
            TOK[base(s)] += 1
            TABN[base(s)].add(t['id'])
            if s != base(s):
                TOK[s] += 0  # variants counted under their own key below
VTOK = collections.Counter(s for t in T for l in t['lines'] for s in signs_of(l))


def decoys(target, k, seed, exclude=(), counter=None, lo=0.67, hi=1.5, pool_filter=None):
    """k signs of matched token frequency (within [lo, hi] x), excluding the guess set."""
    counter = counter or TOK
    f = counter[target]
    ex = set(exclude) | {target}
    pool = [s for s, n in counter.items() if lo * f <= n <= hi * f and s not in ex and s.startswith(('M', '|'))]
    if pool_filter:
        pool = [s for s in pool if pool_filter(s)]
    if len(pool) < k:  # widen
        pool = sorted([s for s in counter if s not in ex and s.startswith(('M', '|'))],
                      key=lambda s: abs(math.log((counter[s] + 1) / (f + 1))))[:max(k, 3 * k)]
    rng = np.random.default_rng(seed)
    pool = sorted(pool)
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[i] for i in idx]


def half(seed, ids=None):
    """Deterministic random half of tablet ids (set A)."""
    ids = sorted(ids or TAB)
    return {i for i in ids if int(hashlib.sha256(('%d|%s' % (seed, i)).encode()).hexdigest(), 16) % 2 == 0}


def pval_ge(obs, null):
    null = np.asarray(null)
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def row(code, method, result, verdict):
    def clean(x):
        x = re.sub(r'\|([^|\s]+)\|', r'[\1]', str(x).replace('\n', ' '))
        return x.replace('|', '/')
    return '| %s | %s | %s | %s |' % (code, clean(method), clean(result), clean(verdict))


def dump(obj, name):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, Fr):
            return float(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=conv, indent=1)
