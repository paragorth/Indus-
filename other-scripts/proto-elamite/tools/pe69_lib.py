"""pe69 TRY HARD TO KILL THE TEAM RULE, THEN USE IT: shared library.

The pe68 C+ rule: an M288-final line carries 60 N39C x (sum of the count lines since the previous M288).
Here every M288-final line on a non-joined tablet is turned into a 'slot' with its neighbourhood, so that
many rules of the same complexity (aggregation of nearby count lines x one multiplier) can be scored on
the same footing, and nulls can re-sample the numbers.

Values follow pe68 exactly: count lines (plain N01/N14/N45/N34 only, clean) in the pe59 'sex2' count map
(N14 = 10); M288 lines in the pe59 capacity map (N39C 1, N30D 2, N30C 4, N24 12, N39B 24, N01 120, N14 720).
"""
import os, sys, json, math, random, hashlib
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe69_ckpt')
os.makedirs(CK, exist_ok=True)

import pe59_lib as P
from pe68_lib import JOINED_IDS

CAP, CNTM = P.pe_maps()
CM = CNTM['sex2']
PERSON = set(P.READING['roles']['PERSON']['signs'])


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def shape(nums):
    return tuple(sorted({c for _, c in nums}))


def line_kind(l):
    """M288 (final), CNT (clean plain count line), CAPL (clean capacity line), BROKEN (number unreadable), OTH."""
    if l['role'] != 'E':
        return 'NONE'
    if not l['numclean']:
        return 'BROKEN_M288' if (l['signs'] and l['signs'][-1] == 'M288') else 'BROKEN'
    if l['signs'] and l['signs'][-1] == 'M288':
        return 'M288'
    c = P.ncls(l['nums'])
    if c == 'AMB':
        return 'CNT'
    if c == 'CAP':
        return 'CAPL'
    return 'OTH'


def tablets(include_joined=False):
    out = []
    for t in P.build_pe():
        if t['id'] in JOINED_IDS and not include_joined:
            continue
        L = []
        for i, l in enumerate(t['lines']):
            k = line_kind(l)
            if k == 'NONE':
                continue
            v = None
            if k == 'CNT':
                v = P.value(l['nums'], CM)
            elif k == 'M288':
                v = P.value(l['nums'], CAP)
                if v is None:
                    k = 'BROKEN_M288'
            L.append({'i': i, 'k': k, 'v': v, 'nums': l['nums'], 'shape': shape(l['nums']) if l['numclean'] else None,
                      'fin': l['signs'][-1] if l['signs'] else None, 'signs': l['signs'], 'surf': l['surf']})
        out.append({'id': t['id'], 'site': t['site'], 'E': L})
    return out


def slots(T):
    """one slot per M288 line (clean or broken): the entry lines between the previous M288 (any) and this one
    ('before'), and between this one and the next M288 ('after')."""
    S = []
    for t in T:
        E = t['E']
        idx = [j for j, e in enumerate(E) if e['k'] in ('M288', 'BROKEN_M288')]
        for n, j in enumerate(idx):
            a = idx[n - 1] + 1 if n else 0
            b = idx[n + 1] if n + 1 < len(idx) else len(E)
            S.append({'id': t['id'], 'j': j, 'line': E[j]['i'], 'm': E[j]['v'], 'm_shape': E[j]['shape'],
                      'm_signs': E[j]['signs'], 'before': E[a:j], 'after': E[j + 1:b], 'first_on_tablet': n == 0})
    return S


# ------------------------------------------------------------------ aggregations ("run definitions")
def _clean_run(run):
    """None if any line in the run is not a clean count line (unknown sum)."""
    if not run or any(e['k'] != 'CNT' for e in run):
        return None
    return run


def _cnt_only(run):
    r = [e for e in run if e['k'] == 'CNT']
    return r or None


AGG = {
    'SUM_BEFORE': lambda s: (lambda r: sum(e['v'] for e in r) if r else None)(_clean_run(s['before'])),
    'SUM_BEFORE_LENIENT': lambda s: (lambda r: sum(e['v'] for e in r) if r else None)(
        _cnt_only(s['before']) if not any(e['k'] == 'BROKEN' for e in s['before']) else None),
    'LAST': lambda s: s['before'][-1]['v'] if s['before'] and s['before'][-1]['k'] == 'CNT' else None,
    'FIRST': lambda s: (lambda r: r[0]['v'] if r else None)(_clean_run(s['before'])),
    'MAX': lambda s: (lambda r: max(e['v'] for e in r) if r else None)(_clean_run(s['before'])),
    'NLINES': lambda s: (lambda r: Fr(len(r)) if r else None)(_clean_run(s['before'])),
    'SUM_LAST2': lambda s: (lambda r: sum(e['v'] for e in r[-2:]) if r and len(r) >= 2 else None)(_clean_run(s['before'])),
    'SUM_EXCL_LAST': lambda s: (lambda r: sum(e['v'] for e in r[:-1]) if r and len(r) >= 2 else None)(_clean_run(s['before'])),
    'SUM_PERSON': lambda s: (lambda r: (lambda q: sum(e['v'] for e in q) if q else None)(
        [e for e in r if e['fin'] in PERSON]) if r else None)(_clean_run(s['before'])),
    'SUM_NONPERSON': lambda s: (lambda r: (lambda q: sum(e['v'] for e in q) if q else None)(
        [e for e in r if e['fin'] not in PERSON]) if r else None)(_clean_run(s['before'])),
    'SUM_AFTER': lambda s: (lambda r: sum(e['v'] for e in r) if r else None)(_clean_run(s['after'])),
    'NEXT': lambda s: s['after'][0]['v'] if s['after'] and s['after'][0]['k'] == 'CNT' else None,
    'SUM_BEFORE_PLUS_NEXT': lambda s: (lambda r, n: (sum(e['v'] for e in r) + n) if (r and n is not None) else None)(
        _clean_run(s['before']), s['after'][0]['v'] if s['after'] and s['after'][0]['k'] == 'CNT' else None),
}
TEAM = ('SUM_BEFORE', 60)
MULTS = list(range(1, 241))
MULTS_PRIOR = [60]  # the multiplier was fixed before pe68 (pe27)


def nclean(s):
    r = _clean_run(s['before'])
    return len(r) if r else 0


def feats(S):
    """array view: for each slot, M288 value and each aggregation value (np.nan if undefined)."""
    m = np.array([float(s['m']) if s['m'] is not None else np.nan for s in S])
    A = {k: np.array([float(x) if (x := f(s)) is not None else np.nan for s in S]) for k, f in AGG.items()}
    return m, A


def hits(m, A, rules, mask=None):
    out = {}
    for a, k in rules:
        x = A[a] * k
        ok = np.isfinite(x) & np.isfinite(m)
        if mask is not None:
            ok &= mask
        out[(a, k)] = (int((np.abs(m[ok] - x[ok]) < 1e-9).sum()), int(ok.sum()))
    return out


def expected_redeal(m, A, rules, mask=None):
    """analytic expectation of hits if each defined slot's M288 value were drawn from the pool of clean M288 values
    (the pe68 re-deal null), and Poisson z."""
    pool = m[np.isfinite(m)]
    vc = Counter(np.round(pool, 6).tolist())
    n = len(pool)
    out = {}
    for a, k in rules:
        x = A[a] * k
        ok = np.isfinite(x) & np.isfinite(m)
        if mask is not None:
            ok &= mask
        e = sum(vc.get(round(v, 6), 0) / n for v in x[ok])
        h = int((np.abs(m[ok] - x[ok]) < 1e-9).sum())
        out[(a, k)] = {'h': h, 'n': int(ok.sum()), 'e': e, 'z': (h - e) / math.sqrt(e + 0.25)}
    return out


# ------------------------------------------------------------------ nulls on the numbers
def shape_pools(T):
    """written-shape pools: for count lines and for M288 lines, the counts of each numeral sign seen with
    each exact shape (set of numeral sign codes)."""
    pools = {'CNT': defaultdict(list), 'M288': defaultdict(list)}
    for t in T:
        for e in t['E']:
            if e['k'] in ('CNT', 'M288'):
                pools[e['k']][e['shape']].append(e['nums'])
    return pools


def resample_shapes(T, pools, rng, which=('CNT', 'M288'), only_nonlast=False):
    """copy of T with the numbers of each clean count / M288 line replaced by those of a random line of the same
    kind and the same written shape (same numeral signs, other counts). only_nonlast: for count lines, only those
    not directly before an M288 line are re-sampled (keeps the per-line pairing, scrambles the rest of the team)."""
    out = []
    for t in T:
        E2 = []
        E = t['E']
        for j, e in enumerate(E):
            e2 = dict(e)
            if e['k'] in which:
                skip = only_nonlast and e['k'] == 'CNT' and j + 1 < len(E) and E[j + 1]['k'] in ('M288', 'BROKEN_M288')
                if not skip:
                    nums = rng.choice(pools[e['k']][e['shape']])
                    e2['nums'] = nums
                    e2['v'] = P.value(nums, CM if e['k'] == 'CNT' else CAP)
            E2.append(e2)
        out.append({'id': t['id'], 'site': t['site'], 'E': E2})
    return out


def redeal_m(m, rng, groups=None):
    """permute clean M288 values across slots (pe68 null); groups: optional array of stratum labels."""
    m2 = m.copy()
    ok = np.isfinite(m)
    if groups is None:
        idx = np.where(ok)[0]
        m2[idx] = m[rng.permutation(idx)]
    else:
        for g in set(groups[ok].tolist()):
            idx = np.where(ok & (groups == g))[0]
            m2[idx] = m[rng.permutation(idx)]
    return m2
