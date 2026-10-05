#!/usr/bin/env python3
"""LA-58 SIMULATE MINOAN GOVERNMENT AND LET THE PAPERWORK CHOOSE: shared code.

An agent-based simulator (tools/la58_sim.c) builds random Bronze Age administrations (one palace with
dependents, a capital with satellite offices, independent peer centres, a temple centre, merchant houses,
or no institutions at all), lets them transact, writes their paperwork as opaque Linear A-like documents
(tablets, roundels, nodules, inscribed objects), thins it to the surviving document count of every real
site, and computes a fixed statistic panel. The same C code computes the panel for real corpora.
ABC (random-forest ABC + rejection on the forest's projections) then reads off which sites were centres.

Abstract document: site index, dtype (0 TAB, 1 ROU, 2 NOD, 3 OTH), word ids (logograms excluded; opaque),
numbers (in order). No sign values are used anywhere.
Corpora: LA (corpus.json), LB (DAMOS, all sites; W-series = sealing devices), UR3 (CDLI sample from la57).
"""
import os, re, sys, json, math, random, hashlib, collections, ctypes, unicodedata, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la58_ckpt')
os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(HERE, '..', 'loops')
sys.path.insert(0, HERE)

TYPES = ['T', 'S', 'P', 'I', 'M', 'N']   # SINGLE, CAPSAT, PEERS, TEMPLE, MERCH, NULL
TNAME = {'T': 'single palace + dependents', 'S': 'capital + satellite offices', 'P': 'independent peer centres',
         'I': 'temple centre', 'M': 'merchant houses', 'N': 'no institutions'}


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ corpora
LA_SITES = ['Haghia Triada', 'Khania', 'Phaistos', 'Knossos', 'Zakros', 'Palaikastro', 'Malia', 'Thera',
            'Iouktas', 'Arkhalkhori', 'Petras', 'Syme']
LA_ABBR = ['HT', 'KH', 'PH', 'KN', 'ZA', 'PK', 'MA', 'THE', 'IO', 'AR', 'PE', 'SY']


def _la_dtype(s):
    s = s.lower()
    if s.startswith('tablet') or s.startswith('lames') or 'bar' in s:
        return 0
    if s.startswith('roundel'):
        return 1
    if s.startswith('nodule') or s.startswith('sealing'):
        return 2
    return 3


def la_docs(sites=LA_SITES, horizon=None):
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for d in C:
        if d['site'] not in sites:
            continue
        if horizon and d['context'] not in horizon:
            continue
        words, nums = [], []
        for t in d['tokens']:
            if t['t'] == 'word':
                words.append('-'.join(t['s']))
            elif t['t'] == 'num':
                nums.append(float(t['v']))
        out.append(dict(id=d['id'], site=sites.index(d['site']), dtype=_la_dtype(d['support']),
                        words=words, nums=nums, scribe=d.get('scribe', '')))
    return out


def _strip(t):
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')


LB_SITES = ['KN', 'PY', 'TH', 'MY', 'TI', 'KH', 'MI']


def lb_docs_all():
    p = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(p):
        return json.load(open(p))
    from la6_common import LB_COM
    WORD = re.compile(r'^[a-z0-9*]+(-[a-z0-9*]+)*$')
    MEAS = set('TVZSMNPQ')
    out = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?\d?)', h)
        if not m or m.group(1) not in LB_SITES:
            continue
        words, nums = [], []
        for ln in (d.get('content') or '').split('\n'):
            for t in _strip(ln).split():
                s = re.sub(r'[\[\]⟦⟧?!,⌞⌟\'"]', '', t)
                if not s or s.startswith('.') or s in ('/', 'vac', 'vac.', 'v.', 'r.'):
                    continue
                b = s.split('+')[0]
                if b in LB_COM or re.fullmatch(r'\*1\d\d', b) or (b.isupper() and len(b) >= 3 and b not in MEAS):
                    continue
                if s.isdigit():
                    nums.append(float(int(s)))
                elif WORD.match(s) and '-' in s:
                    words.append(s)
        ser = m.group(2)
        dtype = 2 if ser.startswith('W') else 0
        out.append(dict(id=h, site=LB_SITES.index(m.group(1)), dtype=dtype, words=words, nums=nums, series=ser))
    json.dump(out, open(p, 'w'))
    return out


UR_SITES = ['Umma', 'Puzriš-Dagan', 'Girsu', 'Ur', 'Nippur', 'Garšana', 'Irisagrig']
UR_TRUTH_C = {'Puzriš-Dagan', 'Ur'}            # central depot and capital
UR_TRUTH_D = {'Umma', 'Girsu', 'Irisagrig', 'Garšana'}   # provinces and an estate


def ur_docs_all():
    src = json.load(open(os.path.join(D, 'la57_ckpt', 'ur3_docs.json')))
    out = []
    for d in src:
        s = d['site'].replace('Irisaĝrig', 'Irisagrig')
        if s not in UR_SITES:
            continue
        words, nums = [], []
        for t in d['toks']:
            if t[0] == 'T':
                words.append(t[1])
            elif t[0] == 'N' and t[1] is not None:
                nums.append(float(t[1]))
        out.append(dict(id=d['id'], site=UR_SITES.index(s), dtype=0, words=words, nums=nums))
    return out


def thin(docs, n_total, rng, K):
    """Proportional thinning to n_total documents (site proportions kept, at least 3 per site)."""
    by = collections.defaultdict(list)
    for d in docs:
        by[d['site']].append(d)
    N = len(docs)
    out = []
    for k in range(K):
        L = by[k]
        m = min(len(L), max(3, int(round(n_total * len(L) / N))))
        out += rng.sample(L, m)
    return out


# ------------------------------------------------------------------ C engine
SO = os.path.join(CK, 'la58_sim.so')


def build():
    src = os.path.join(HERE, 'la58_sim.c')
    if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(src):
        subprocess.check_call(['gcc', '-O2', '-shared', '-fPIC', '-o', SO, src, '-lm'])
    lib = ctypes.CDLL(SO)
    lib.la58_nstat.restype = ctypes.c_int
    lib.la58_ntheta.restype = ctypes.c_int
    return lib


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = build()
    return _LIB


def nstat(K):
    return lib().la58_nstat(K)


def ntheta(K):
    return lib().la58_ntheta(K)


def pack(docs, K):
    """Turn abstract docs into flat int/float arrays (word ids opaque, assigned by first appearance)."""
    vocab = {}
    site, dtype, wst, nst, W, X = [], [], [0], [0], [], []
    for d in docs:
        site.append(d['site']); dtype.append(d['dtype'])
        for w in d['words']:
            if w not in vocab:
                vocab[w] = len(vocab)
            W.append(vocab[w])
        for x in d['nums']:
            X.append(x)
        wst.append(len(W)); nst.append(len(X))
    I = lambda a: np.ascontiguousarray(np.array(a, dtype=np.int32))
    return dict(n=len(docs), site=I(site), dtype=I(dtype), wst=I(wst), nst=I(nst), W=I(W if W else [0]),
                X=np.ascontiguousarray(np.array(X if X else [0.0], dtype=np.float64)), nvocab=len(vocab),
                vocab=vocab)


def _p(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def stats(docs, K):
    P = pack(docs, K)
    out = np.zeros(nstat(K), dtype=np.float64)
    lib().la58_stats(ctypes.c_int(K), ctypes.c_int(P['n']), _p(P['site'], ctypes.c_int), _p(P['dtype'], ctypes.c_int),
                     _p(P['wst'], ctypes.c_int), _p(P['W'], ctypes.c_int), _p(P['nst'], ctypes.c_int),
                     _p(P['X'], ctypes.c_double), ctypes.c_int(P['nvocab']), _p(out, ctypes.c_double))
    return out


def simulate(K, nk, nworld, sd, force_type=-1, force_theta=None):
    """Run nworld random worlds; returns (stats [nworld, nstat], theta [nworld, ntheta])."""
    nk = np.ascontiguousarray(np.array(nk, dtype=np.int32))
    S = np.zeros((nworld, nstat(K)), dtype=np.float64)
    T = np.zeros((nworld, ntheta(K)), dtype=np.float64)
    ft = np.zeros(ntheta(K), dtype=np.float64) if force_theta is None else np.ascontiguousarray(force_theta, dtype=np.float64)
    lib().la58_batch(ctypes.c_int(K), _p(nk, ctypes.c_int), ctypes.c_int(nworld), ctypes.c_ulonglong(sd),
                     ctypes.c_int(force_type), ctypes.c_int(0 if force_theta is None else 1), _p(ft, ctypes.c_double),
                     _p(S, ctypes.c_double), _p(T, ctypes.c_double))
    return S, T


def sim_docs(K, nk, sd, force_type=-1, force_theta=None):
    """One world, returning its surviving documents (abstract form) plus word-class labels and theta."""
    nk = np.ascontiguousarray(np.array(nk, dtype=np.int32))
    cap = int(sum(nk)) + 10
    site = np.zeros(cap, np.int32); dt = np.zeros(cap, np.int32)
    wst = np.zeros(cap + 1, np.int32); nst = np.zeros(cap + 1, np.int32)
    W = np.zeros(cap * 64, np.int32); X = np.zeros(cap * 64, np.float64)
    WC = np.zeros(cap * 64, np.int32)
    th = np.zeros(ntheta(K), np.float64)
    ft = np.zeros(ntheta(K), dtype=np.float64) if force_theta is None else np.ascontiguousarray(force_theta, dtype=np.float64)
    n = lib().la58_world_docs(ctypes.c_int(K), _p(nk, ctypes.c_int), ctypes.c_ulonglong(sd), ctypes.c_int(force_type),
                              ctypes.c_int(0 if force_theta is None else 1), _p(ft, ctypes.c_double),
                              _p(site, ctypes.c_int), _p(dt, ctypes.c_int), _p(wst, ctypes.c_int), _p(W, ctypes.c_int),
                              _p(WC, ctypes.c_int), _p(nst, ctypes.c_int), _p(X, ctypes.c_double), _p(th, ctypes.c_double))
    docs = []
    for i in range(n):
        docs.append(dict(id='S%d' % i, site=int(site[i]), dtype=int(dt[i]),
                         words=['w%d' % x for x in W[wst[i]:wst[i + 1]]],
                         wclass=[int(x) for x in WC[wst[i]:wst[i + 1]]],
                         nums=[float(x) for x in X[nst[i]:nst[i + 1]]]))
    return docs, th


# theta layout (must match C): [type, ncentre, role_0..K-1, parent_0..K-1, nuisance...]
def theta_roles(th, K):
    return th[2:2 + K].astype(int), th[2 + K:2 + 2 * K].astype(int)


def wlog(path, row):
    with open(path, 'a') as f:
        f.write(row.rstrip('\n') + '\n')


def jdump(obj, name):
    json.dump(obj, open(os.path.join(CK, name), 'w'), indent=1, default=float)
