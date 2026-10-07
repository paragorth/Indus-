"""pe78: fluctuation scaling (Taylor's law) of the accounts.

Pivot from the brief (herd-demography simulator): pe20, pe37, pe53 and pe54 already scored random demographic
class assignments, kill-off ratios, lambing curves and flock histories on the PE herd signs.  Here the animals
(and workers, and grain) are simulated only to learn how a living population's counts FLUCTUATE, not which sign
is which class.  Ecology's Taylor's law: across groups, var = a * mean^b.  For each count key (the entry's last
sign) the groups are the tablets where it is written >= 2 times; (log mean, log var) per group gives a key's
slope b, intercept a and scatter.  The slope does not depend on the unit (decimal vs sexagesimal readings change
it little), so it can be read without knowing the number values.

Data: PE (common.load(), damage-aware), proto-cuneiform (pe2_pc_corpus.json), Ur III lines with known kind
(pe23_ckpt/ur3.json: LIVESTOCK / PEOPLE / GRAIN / RATION).
"""
import os, sys, json, math, re, collections, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe78_ckpt')
os.makedirs(CK, exist_ok=True)

PE_CNT = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}
PE_CAP = {'N39C': 1 / 120, 'N30D': 1 / 60, 'N30C': 1 / 30, 'N24': 1 / 10, 'N39B': 1 / 5, 'N01': 1, 'N14': 6,
          'N45': 60, 'N34': 180, 'N48': 1800}
PC_CNT = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000}
PC_CAP = {'N39A': 1 / 5, 'N39B': 1 / 5, 'N24': 1, 'N01': 5, 'N14': 30, 'N45': 300, 'N34': 1800}

# conventional PC referent classes (fixed before any run; used only as calibration labels)
PC_LIVING = {'UDU', 'U8', 'UDUNITA', 'MASZ', 'AB2', 'GU4', 'SZAH2', 'SAL', 'KUR', 'SAG', 'ERIM', 'GURUSZ',
             'KISZ', 'ANSZE', 'SILA4', 'AMAR', 'KU6', 'SUHUR', 'MUSZEN'}
PC_GOODS = {'SZE', 'GAR', 'KASZ', 'DUG', 'GA', 'I3', 'SIG2', 'TUG2', 'KU3', 'ZIZ2', 'NINDA', 'GU7', 'U2', 'GAN2',
            'BA', 'SZE3', 'DUG~a', 'DUG~b', 'GAR~a'}


def _val(nums, m):
    if not nums or not all(c in m for _, c in nums):
        return None
    return sum(n * m[c] for n, c in nums)


def pe_entries():
    """[(tablet, key, value, system)] for PE entries; key = last base sign of the line."""
    out = []
    for e in common.entries(common.load(), require_clean=True):
        sys_ = e['system']
        nums = [(n, c) for n, c in e['numerals'] if isinstance(n, int)]
        if sys_ == 'C':
            v = _val(nums, PE_CAP)
        elif sys_ == 'SDB':
            v = _val(nums, PE_CNT)
        else:
            v = None
        if v and v > 0:
            out.append((e['tablet'], e['signs'][-1], float(v), sys_))
    return out


def pc_entries():
    """[(tablet, key, value, class)] for proto-cuneiform lines containing exactly one calibration sign."""
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    out = []
    for t in T:
        for l in t['lines']:
            if l.get('lacuna'):
                continue
            sg = [common.base(s) for s in l['signs'] if s not in ('x', 'X')]
            nums = [(int(n), common.norm_code(c)) for n, c in l['numerals'] if isinstance(n, int)]
            if not sg or not nums:
                continue
            hit = [s for s in sg if s in PC_LIVING or s in PC_GOODS]
            key = sg[-1]
            cls = None
            if len(set(hit)) == 1:
                cls = 'LIVING' if hit[0] in PC_LIVING else 'GOODS'
                key = hit[0]
            codes = {c for _, c in nums}
            v = _val(nums, PC_CNT) if not codes & {'N39A', 'N39B', 'N24'} else _val(nums, PC_CAP)
            if v and v > 0:
                out.append((t['id'], key, float(v), cls))
    return out


def ur_entries():
    """[(tablet, key, value, class)] Ur III; key = class + first word after the number."""
    d = json.load(open(os.path.join(DATA, 'pe23_ckpt', 'ur3.json')))
    out = []
    for x in d:
        if x['cls'] not in ('LIVESTOCK', 'PEOPLE', 'GRAIN', 'RATION'):
            continue
        toks = x['raw'].split()
        w = [t for t in toks if not re.match(r'^\d|^\(', t)]
        key = x['cls'] + ':' + (w[0] if w else '?')
        out.append((x['tab'], key, float(x['val']), 'LIVING' if x['cls'] in ('LIVESTOCK', 'PEOPLE') else 'GOODS'))
    return out


# ---------------------------------------------------------------- Taylor signatures
def groups(entries, min_n=2):
    """{key: [(tablet, array of values)]} for tablets where the key occurs >= min_n times."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for t, k, v, *_ in entries:
        g[k][t].append(v)
    out = {}
    for k, d in g.items():
        L = [(t, np.array(vs)) for t, vs in d.items() if len(vs) >= min_n]
        if L:
            out[k] = L
    return out


FEAT = ['b', 'a', 'res', 'zero', 'cvall', 'bspan']


def signature(glist, min_g=5):
    """Taylor signature of one key from its tablet groups.  None if too few usable groups."""
    if len(glist) < min_g:
        return None
    m = np.array([g.mean() for _, g in glist])
    v = np.array([g.var(ddof=1) for _, g in glist])
    zero = float((v <= 1e-12).mean())
    ok = v > 1e-12
    if ok.sum() < 4 or np.ptp(np.log(m[ok])) < 0.3:
        return None
    x, y = np.log(m[ok]), np.log(v[ok])
    b, a = np.polyfit(x, y, 1)
    res = float(np.std(y - (a + b * x)))
    allv = np.concatenate([g for _, g in glist])
    return {'b': float(b), 'a': float(a + b * np.mean(x) - 2 * np.mean(x)),  # a: log CV^2 at the key's mean
            'res': res, 'zero': zero, 'cvall': float(np.log(allv.std() / allv.mean() + 1e-9)),
            'bspan': float(np.ptp(x)), 'ng': len(glist)}


def sig_table(entries, min_g=5):
    G = groups(entries)
    out = {}
    for k, L in G.items():
        s = signature(L, min_g)
        if s:
            out[k] = s
    return out


def vec(s):
    return np.array([s[f] for f in FEAT])


def shuffle_within_tablet(entries, rng):
    """Kill control: values permuted across keys within each tablet (keeps every tablet's value set)."""
    by = collections.defaultdict(list)
    for i, e in enumerate(entries):
        by[e[0]].append(i)
    vals = [e[2] for e in entries]
    out = list(entries)
    for t, idx in by.items():
        p = rng.permutation(len(idx))
        for j, i in enumerate(idx):
            e = entries[i]
            out[i] = (e[0], e[1], vals[idx[p[j]]]) + tuple(e[3:])
    return out


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- simulated worlds (random biology / rules)
MAXG = 60
KINDS = ['LIVING', 'ALLOC', 'MEASURED', 'POISSON']


def cap_groups(glist, rng, maxg=MAXG):
    if len(glist) <= maxg:
        return glist
    idx = rng.choice(len(glist), maxg, replace=False)
    return [glist[i] for i in idx]


def sim_living(sizes, rng, th=None):
    """Agent-free but individual-based stage herd (young J, adult females F, adult males M) per entry; an
    owner (tablet) shares wealth and a yearly environment.  Entry value = count of the recorded class.
    Zeros are not written (resampled up to 20 times, then dropped)."""
    if th is None:
        th = dict(f=rng.uniform(0.4, 1.3), m=rng.uniform(0.03, 0.3), mj=rng.uniform(0.1, 0.6),
                  o=rng.uniform(0, 0.35), se=rng.uniform(0, 0.6), Y=int(rng.integers(1, 16)),
                  sw=rng.uniform(0.2, 1.5), sh=rng.uniform(0, 1.0), mu0=math.exp(rng.uniform(math.log(2), math.log(300))),
                  cls=int(rng.integers(0, 4)))
    out = []
    for n in sizes:
        w = th['mu0'] * math.exp(rng.normal(0, th['sw']))
        vals = []
        for _ in range(20):
            k = n - len(vals)
            if k <= 0:
                break
            N0 = rng.poisson(w * np.exp(rng.normal(0, th['sh'], k)) + 0.5)
            F = rng.binomial(N0, 0.6); M = rng.binomial(N0 - F, 0.4); J = N0 - F - M
            for y in range(th['Y']):
                e = math.exp(rng.normal(0, th['se']))
                births = rng.binomial(F, min(1.0, th['f'] * e))
                sj = max(0.0, min(1.0, (1 - th['mj']) * e)); sa = max(0.0, min(1.0, (1 - th['m']) * e))
                Js = rng.binomial(J, sj); Fs = rng.binomial(F, sa); Ms = rng.binomial(M, sa)
                fem = rng.binomial(Js, 0.5)
                F = Fs + fem; M = Ms + Js - fem; J = births
                M = M - rng.binomial(M, th['o']); F = F - rng.binomial(F, th['o'] / 3)
            v = [J + F + M, F, J, M][th['cls']]
            v = v[v > 0]
            vals.extend(v.tolist())
        if len(vals) >= 2:
            out.append(np.array(vals[:n], float))
    return out, th


def sim_alloc(sizes, rng, th=None):
    if th is None:
        th = dict(mr=rng.uniform(-2, 3), sr=rng.uniform(0, 1.5), mh=math.exp(rng.uniform(0, math.log(60))),
                  sh=rng.uniform(0, 1.2), q=float(rng.choice([1, 1, 5, 10])), fix=rng.uniform(0, 0.7))
    out = []
    for n in sizes:
        r = math.exp(th['mr'] + rng.normal(0, th['sr']))
        h0 = th['mh'] * math.exp(rng.normal(0, 0.8))
        H = rng.poisson(h0 * np.exp(rng.normal(0, th['sh'], n)) + 0.3) + 1
        H = np.where(rng.random(n) < th['fix'], int(round(h0)) + 1, H)    # standard allotments
        v = np.round(r * H / th['q']) * th['q']
        v = v[v > 0]
        if len(v) >= 2:
            out.append(v.astype(float))
    return out, th


def sim_measured(sizes, rng, th=None):
    if th is None:
        th = dict(mu=rng.uniform(0, 6), sw=rng.uniform(0.2, 1.8), s=rng.uniform(0.1, 1.5),
                  q=float(rng.choice([1, 1, 5, 10, 30])))
    out = []
    for n in sizes:
        mu = th['mu'] + rng.normal(0, th['sw'])
        v = np.round(np.exp(mu + rng.normal(0, th['s'], n)) / th['q']) * th['q']
        v = v[v > 0]
        if len(v) >= 2:
            out.append(v.astype(float))
    return out, th


def sim_poisson(sizes, rng, th=None):
    if th is None:
        th = dict(mu=rng.uniform(0, 5), sw=rng.uniform(0.2, 1.8))
    out = []
    for n in sizes:
        lam = math.exp(th['mu'] + rng.normal(0, th['sw']))
        v = rng.poisson(lam, n)
        v = v[v > 0]
        if len(v) >= 2:
            out.append(v.astype(float))
    return out, th


SIMS = {'LIVING': sim_living, 'ALLOC': sim_alloc, 'MEASURED': sim_measured, 'POISSON': sim_poisson}


def sim_signature(kind, sizes, rng, th=None):
    vals, th = SIMS[kind](sizes, rng, th)
    s = signature([(i, v) for i, v in enumerate(vals)], min_g=5)
    return s, th
