#!/usr/bin/env python3
"""PE-65 ONE STATE OR MANY OFFICES ACROSS IRAN? shared code (port of LA-58 to Proto-Elamite).

An agent-based simulator (tools/pe65_sim.c) builds random administrations spanning a hub city (Susa) and four
plateau outposts: one state with dependent outposts, a capital with satellite offices, independent offices
sharing a script, a temple estate centre, trading houses, mobile herding offices moving between sites, or no
institutions. Each world writes opaque Proto-Elamite-like tablets (entry strings, numbers, totals, sealing),
is thinned to the real surviving count of every unit (Susa dealt into 3 excavation batches), and the same C
code computes a fixed statistic panel for simulated and real corpora. ABC reads off the political structure.

Abstract document: unit index (0-2 hub batches, 3.. outposts), dtype (0 tablet, 1 short note <= 1 numbered
line, 2 sealed), words (one per line: the sign string of the line without numerals; opaque), numbers.
No sign values or readings are used.
Corpora: PE (CDLI), UR (Ur III, CDLI dump; hub Umma), LB (DAMOS; hub Knossos), all at PE shape.
"""
import os, re, sys, json, math, random, hashlib, collections, ctypes, subprocess, unicodedata
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
CK = os.path.join(DATA, 'pe65_ckpt')
os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(PEROOT, 'loops')
REPO = os.path.abspath(os.path.join(PEROOT, '..', '..'))
from common import load, base, is_sign  # noqa: E402

TYPES = ['T', 'S', 'P', 'I', 'M', 'H', 'N']
TNAME = {'T': 'one state, centre + dependents', 'S': 'capital + satellite offices', 'P': 'independent offices',
         'I': 'temple estate centre', 'M': 'trading houses', 'H': 'mobile herding offices', 'N': 'no institutions'}
K = 5                       # sites: hub + 4 outposts
NB = 3
UNITS = ['SusaA', 'SusaB', 'SusaC', 'Yahya', 'Malyan', 'Sialk', 'Sofalin']
SITES = ['Susa', 'Yahya', 'Malyan', 'Sialk', 'Sofalin']
NK = None                   # filled from the real PE corpus


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ numbers
SEX = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000, 'N51': 120, 'N54': 1200, 'N56': 7200,
       'N46': 120, 'N52': 1200, 'N02': .5, 'N03': .5, 'N04': .5, 'N05': .5, 'N08': .5, 'N08A': .5, 'N8A': .5, 'N8B': .5, 'N09': .5}
CAPV = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N28': 0.5, 'N29B': 0.5, 'N39A': 0.5}
CAPSET = set(CAPV) | {'N39N', 'N24A', 'N28C', 'N29A', 'N30A'}


def line_value(nums):
    codes = {c.split('@')[0] for _, c in nums}
    cap = bool(codes & CAPSET)
    v = 0.0
    for n, c in nums:
        c = c.split('@')[0]
        if cap:
            x = CAPV.get(c, SEX.get(c, 0) * 120)
        else:
            x = SEX.get(c, 0)
        try:
            v += float(n) * x
        except (TypeError, ValueError):
            pass
    return v if v > 0 else None


# ------------------------------------------------------------------ corpora
PE_SITE = {'Susa (mod. Shush)': 'Susa', 'Susa (mod. Shush) ?': 'Susa', 'uncertain (mod. Tepe Yahya)': 'Yahya',
           'Anšan (mod. Tell Malyan)': 'Malyan', 'uncertain (mod. Tepe Sialk)': 'Sialk',
           'uncertain (mod. Tepe Sofalin)': 'Sofalin'}


def susa_batch(desig):
    p = ' '.join(desig.split(',')[0].split()[:2])
    if p == 'MDP 17':
        return 0
    if p in ('MDP 26', 'MDP 26S'):
        return 1
    return 2


def seal_map():
    out, cur = {}, None
    for raw in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        if raw.startswith('&P'):
            cur = raw[1:8]; out[cur] = False; continue
        if cur and re.search(r'seal', raw, re.I) and not raw.startswith('@object') and not raw.startswith('&'):
            out[cur] = True
    return out


def pe_docs_all(batch_mode='pub', rs=None):
    """All PE tablets of the 5 panel sites as abstract docs. batch_mode 'pub' (publication batches) or 'rand'."""
    sm = seal_map()
    out = []
    for t in load():
        s = PE_SITE.get(t['provenience'])
        if not s or t['object_type'] != 'tablet':
            continue
        words, nums, nnl = [], [], 0
        for l in t['lines']:
            sg = [base(x) for x in l['signs'] if is_sign(x)]
            if sg:
                words.append('.'.join(sg))
            if l['numerals']:
                nnl += 1
                v = line_value(l['numerals'])
                if v:
                    nums.append(v)
        dt = 2 if sm.get(t['id']) else (1 if nnl <= 1 else 0)
        u = susa_batch(t['designation']) if s == 'Susa' else NB + SITES.index(s) - 1
        out.append(dict(id=t['id'], site=u, dtype=dt, words=words, nums=nums, siteName=s))
    if batch_mode == 'rand':
        rng = random.Random(rs or seed('pe65-randbatch'))
        hub = [d for d in out if d['siteName'] == 'Susa']
        lab = [d['site'] for d in hub]; rng.shuffle(lab)
        for d, q in zip(hub, lab):
            d['site'] = q
    return out


def nk_of(docs, Un=NB + K - 1):
    return [sum(1 for x in docs if x['site'] == u) for u in range(Un)]


def shuffle_units(docs, tag, hub_only=False):
    rng = random.Random(seed('pe65-shuf-' + tag))
    lab = [d['site'] for d in docs]; rng.shuffle(lab)
    return [dict(d, site=q) for d, q in zip(docs, lab)]


def ur_docs_all():
    fn = os.path.join(CK, 'ur_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    import pe38_common as P38
    src = P38.ur3_docs_all()
    out = []
    for d in src:
        words, nums, nnl = [], [], 0
        for l in d['lines']:
            tk = [x for x in l['toks'] if x]
            if tk:
                words.append(' '.join(tk))
            if l['sys']:
                nnl += 1
                if l['val'] is not None and float(l['val']) > 0:
                    nums.append(float(l['val']))
        out.append(dict(id=d['id'], site=d['site'], dtype=1 if nnl <= 1 else 0, words=words, nums=nums))
    json.dump(out, open(fn, 'w'))
    return out


def _strip(t):
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')


def lb_docs_all():
    fn = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    out = []
    for line in open(os.path.join(REPO, 'other-scripts', 'linear-a', 'data', 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?\d?)', h)
        if not m:
            continue
        words, nums, nnl = [], [], 0
        for ln in (d.get('content') or '').split('\n'):
            ln = _strip(ln)
            ln = re.sub(r'^\s*\.\S+', '', ln)
            ln = re.sub(r'[\[\]⟦⟧?!,⌞⌟\'"]', '', ln)
            toks = ln.split()
            ws, has = [], False
            for x in toks:
                if x.isdigit():
                    nums.append(float(x)); has = True
                elif re.fullmatch(r'[A-Za-z*0-9+\-]+', x) and not re.fullmatch(r'(vac|vacat|v|r|lat|inf|sup|mut|deest)\.?', x):
                    ws.append(x)
            if ws:
                words.append(' '.join(ws))
            nnl += has
        if not words and not nums:
            continue
        ser = m.group(2)
        dt = 2 if ser.startswith('W') else (1 if nnl <= 1 else 0)
        out.append(dict(id=h, site=m.group(1), dtype=dt, words=words, nums=nums))
    json.dump(out, open(fn, 'w'))
    return out


CONTROL_SITES = {'UR': ['Umma', 'Puzriš-Dagan', 'Girsu', 'Ur', 'Nippur'],
                 'LB': ['KN', 'PY', 'TH', 'MY', 'TI']}
# truth (for scoring only): role per site (1 centre, 2 dependent, 0 independent) and type class
CONTROL_TRUTH = {'UR': dict(type='T/S', centre=['Puzriš-Dagan', 'Ur'], dependent=['Umma', 'Girsu']),
                 'LB': dict(type='P/N', centre=['KN', 'PY', 'TH', 'MY'], dependent=['TI'])}


def control_docs(name, rep=0, nk=None):
    """Ur III or Linear B thinned to PE shape: hub of sum(nk[:3]) docs dealt into random batches of nk[0..2],
    then outposts of nk[3..]."""
    nk = nk or NK_PE()
    src = ur_docs_all() if name == 'UR' else lb_docs_all()
    sites = CONTROL_SITES[name]
    rng = random.Random(seed('pe65-%s-%d' % (name, rep)))
    by = collections.defaultdict(list)
    for d in src:
        by[d['site']].append(d)
    hub = rng.sample(by[sites[0]], sum(nk[:NB]))
    lab = sum([[b] * nk[b] for b in range(NB)], []); rng.shuffle(lab)
    out = [dict(d, site=q, siteName=sites[0]) for d, q in zip(hub, lab)]
    for i, s in enumerate(sites[1:]):
        out += [dict(d, site=NB + i, siteName=s) for d in rng.sample(by[s], min(nk[NB + i], len(by[s])))]
    return out


_NK = None


def NK_PE():
    global _NK
    if _NK is None:
        _NK = nk_of(pe_docs_all())
    return _NK


# ------------------------------------------------------------------ C engine
SO = os.path.join(CK, 'pe65_sim.so')


def build():
    src = os.path.join(HERE, 'pe65_sim.c')
    if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(src):
        subprocess.check_call(['gcc', '-O2', '-shared', '-fPIC', '-o', SO, src, '-lm'])
    lib = ctypes.CDLL(SO)
    for f in ('pe65_nstat', 'pe65_ntheta', 'pe65_nunit', 'pe65_world_docs'):
        getattr(lib, f).restype = ctypes.c_int
    return lib


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = build()
        nk = NK_PE()
        bw = np.ascontiguousarray(np.array(nk[:NB], float))
        _LIB.pe65_set_bw(_p(bw, ctypes.c_double))
    return _LIB


def nunit():
    return NB + K - 1


def nstat():
    return lib().pe65_nstat(nunit())


def ntheta():
    return lib().pe65_ntheta(K)


def _p(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def pack(docs):
    vocab = {}
    site, dtype, wst, nst, W, X = [], [], [0], [0], [], []
    for d in docs:
        site.append(d['site']); dtype.append(d['dtype'])
        for w in d['words']:
            if w not in vocab:
                vocab[w] = len(vocab)
            W.append(vocab[w])
        X += list(d['nums'])
        wst.append(len(W)); nst.append(len(X))
    I = lambda a: np.ascontiguousarray(np.array(a, dtype=np.int32))
    return dict(n=len(docs), site=I(site), dtype=I(dtype), wst=I(wst), nst=I(nst), W=I(W if W else [0]),
                X=np.ascontiguousarray(np.array(X if X else [0.0], dtype=np.float64)), nvocab=len(vocab), vocab=vocab)


def stats(docs):
    P = pack(docs)
    out = np.zeros(nstat(), dtype=np.float64)
    lib().pe65_stats(ctypes.c_int(nunit()), ctypes.c_int(P['n']), _p(P['site'], ctypes.c_int), _p(P['dtype'], ctypes.c_int),
                     _p(P['wst'], ctypes.c_int), _p(P['W'], ctypes.c_int), _p(P['nst'], ctypes.c_int),
                     _p(P['X'], ctypes.c_double), ctypes.c_int(P['nvocab']), _p(out, ctypes.c_double))
    return out


def simulate(nworld, sd, force_type=-1, force_theta=None, nk=None):
    nk = np.ascontiguousarray(np.array(nk or NK_PE(), dtype=np.int32))
    S = np.zeros((nworld, nstat()), dtype=np.float64)
    T = np.zeros((nworld, ntheta()), dtype=np.float64)
    ft = np.zeros(ntheta(), dtype=np.float64) if force_theta is None else np.ascontiguousarray(force_theta, dtype=np.float64)
    lib().pe65_batch(ctypes.c_int(K), _p(nk, ctypes.c_int), ctypes.c_int(nworld), ctypes.c_ulonglong(sd),
                     ctypes.c_int(force_type), ctypes.c_int(0 if force_theta is None else 1), _p(ft, ctypes.c_double),
                     _p(S, ctypes.c_double), _p(T, ctypes.c_double))
    return S, T


def sim_docs(sd, force_type=-1, force_theta=None, nk=None):
    nk = np.ascontiguousarray(np.array(nk or NK_PE(), dtype=np.int32))
    cap = int(sum(nk)) + 10
    site = np.zeros(cap, np.int32); dt = np.zeros(cap, np.int32)
    wst = np.zeros(cap + 1, np.int32); nst = np.zeros(cap + 1, np.int32)
    W = np.zeros(cap * 64, np.int32); X = np.zeros(cap * 64, np.float64); WC = np.zeros(cap * 64, np.int32)
    th = np.zeros(ntheta(), np.float64)
    ft = np.zeros(ntheta(), dtype=np.float64) if force_theta is None else np.ascontiguousarray(force_theta, dtype=np.float64)
    n = lib().pe65_world_docs(ctypes.c_int(K), _p(nk, ctypes.c_int), ctypes.c_ulonglong(sd), ctypes.c_int(force_type),
                              ctypes.c_int(0 if force_theta is None else 1), _p(ft, ctypes.c_double),
                              _p(site, ctypes.c_int), _p(dt, ctypes.c_int), _p(wst, ctypes.c_int), _p(W, ctypes.c_int),
                              _p(WC, ctypes.c_int), _p(nst, ctypes.c_int), _p(X, ctypes.c_double), _p(th, ctypes.c_double))
    docs = []
    for i in range(n):
        docs.append(dict(id='S%d' % i, site=int(site[i]), dtype=int(dt[i]),
                         words=['w%d' % x for x in W[wst[i]:wst[i + 1]]], wclass=[int(x) for x in WC[wst[i]:wst[i + 1]]],
                         nums=[float(x) for x in X[nst[i]:nst[i + 1]]]))
    return docs, th


def wlog(path, row):
    with open(path, 'a') as f:
        f.write(row.rstrip('\n') + '\n')


def jdump(obj, name):
    json.dump(obj, open(os.path.join(CK, name), 'w'), indent=1, default=float)
