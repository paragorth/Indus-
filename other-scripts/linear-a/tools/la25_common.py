#!/usr/bin/env python3
"""la25: THE FIRE SET THE CLOCK. Shared data extraction, simulator and ABC.

Each destruction archive is treated as a snapshot of one month. A random but agronomically
constrained crop calendar plus a destruction month per archive generate the commodity entry
profile; rejection ABC over a bank of simulations returns a posterior over destruction months.

Data used: logogram NAMES (GRA, VIN, OLE, OLIV ...) and numbers only. No readings.
Months: 0 = Jan ... 11 = Dec.
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la25_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)

MONTHS = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()

# core categories shared by both scripts (livestock, figs, wool folded into OTHER)
CORE = ['GRA', 'VIN', 'OLE', 'OLIV', 'OTHER']
# extended (Linear B only)
EXT = ['GRA', 'VIN', 'OLE', 'OLIV', 'FIC', 'LIV', 'LANA', 'OTHER']

# agronomic windows: month at which the commodity enters storage / is first recorded in bulk
# (CONTEXT.md sec. 4 + standard Cretan practice; windows deliberately wide)
WIN = {
    'GRA': [4, 5, 6],          # harvest/threshing May-Jul
    'VIN': [7, 8, 9],          # vintage + pressing Aug-Oct
    'OLIV': [9, 10, 11, 0],    # olive picking Oct-Jan
    'OLE': [10, 11, 0, 1],     # oil pressing Nov-Feb
    'FIC': [7, 8],             # figs Aug-Sep
    'LIV': [2, 3, 4],          # flock census at lambing end / shearing Mar-May
    'LANA': [3, 4, 5],         # shearing Apr-Jun
}


def cat_la(v):
    v = v.strip('*[]')
    if v.startswith('GRA') or v.startswith('QE+GRA'): return 'GRA'
    if v.startswith('VIN'): return 'VIN'
    if v.startswith('OLIV'): return 'OLIV'
    if v.startswith('OLE'): return 'OLE'
    if v.startswith('CAP') or v.startswith('OVIS') or v.startswith('SUS') or v.startswith('BOS'): return 'LIV'
    return 'OTHER'


def cat_lb(v):
    v = v.strip('*[]')
    if v.startswith(('GRA', 'HORD', 'FAR')): return 'GRA'
    if v.startswith('VIN'): return 'VIN'
    if v.startswith('OLIV'): return 'OLIV'
    if v.startswith('OLE'): return 'OLE'
    if v.startswith('NI') and (len(v) == 2 or not v[2].isalpha()): return 'FIC'
    if v.startswith(('OVIS', 'CAP', 'SUS', 'BOS')) and not v.startswith('CAPS'): return 'LIV'
    if v.startswith('LANA'): return 'LANA'
    return 'OTHER'


LA_SITES = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Tylissos': 'TY'}


def la_entries():
    """list of (archive, doc_id, scribe, category, amount or None) for LM IB destruction archives."""
    c = json.load(open(os.path.join(DATA, 'corpus.json')))
    out = []
    for d in c:
        a = LA_SITES.get(d['site'])
        if a is None or d['context'] not in ('LMIB', ''): continue
        toks = d['tokens']
        for i, t in enumerate(toks):
            if t['t'] != 'logo': continue
            amt = None
            for u in toks[i + 1:i + 3]:
                if u['t'] == 'num': amt = u['v']; break
                if u['t'] in ('nl', 'logo'): break
            out.append((a, d['id'], d.get('scribe', ''), cat_la(t['v']), amt))
    return out


def lb_entries():
    from la22_lb import parse_doc
    out = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        d = json.loads(l)
        if not d.get('heading') or not d.get('content'): continue
        h = d['heading'].split()
        s = h[0]
        if s not in ('KN', 'PY', 'TH', 'MY'): continue
        ser = re.sub(r'\(.*', '', h[1]) if len(h) > 1 else ''
        toks = parse_doc(d['content'])
        for i, t in enumerate(toks):
            if t['t'] != 'L': continue
            if len(t['c']) == 1: continue            # unit letters S V Z T etc.
            if t['c'] in ('DA', 'TA', 'PA', 'ME', 'RI', 'KE', 'WE', 'MO', 'KO', 'SA', 'ZE', 'PU', 'TE', 'RE', 'KU', 'PE', 'DE', 'MA', 'MI', 'DU', 'DI', 'PO', 'KA', 'RO', 'SE', 'TU', 'A2', 'RO2'):
                continue                             # adjunct syllabograms / unit marks
            amt = None
            for u in toks[i + 1:i + 4]:
                if u['t'] == 'N': amt = u['v']; break
                if u['t'] in ('nl', 'w'): break
            out.append((s, d['heading'], ser, cat_lb(t['c']), amt))
    return out


def profile(entries, archives, cats, unit='doc'):
    """counts[a, c]. unit='doc': each (document, category) counted once (entries on one tablet
    are not independent); unit='entry': every logogram occurrence."""
    M = np.zeros((len(archives), len(cats)), int)
    seen = set()
    for e in entries:
        if unit == 'doc':
            c0 = e[3] if e[3] in cats else 'OTHER'
            key = (e[0], e[1], c0)
            if key in seen: continue
            seen.add(key)
        if e[0] not in archives: continue
        c = e[3]
        if c not in cats: c = 'OTHER'
        M[archives.index(e[0]), cats.index(c)] += 1
    return M


# ---------------------------------------------------------------- simulator
def draw_calendar(rng, n, cats, mode='agro'):
    """returns h[n, C] (entry month), tau[n, C], floor[n, C]; OTHER flat."""
    C = len(cats)
    h = np.zeros((n, C), int); tau = np.ones((n, C)); fl = np.ones((n, C))
    for j, c in enumerate(cats):
        if c == 'OTHER': continue
        if mode == 'agro':
            h[:, j] = rng.choice(WIN[c], n)
        else:                                       # random calendar: any month
            h[:, j] = rng.integers(0, 12, n)
        tau[:, j] = np.exp(rng.uniform(np.log(0.5), np.log(8.0), n))
        fl[:, j] = rng.uniform(0.02, 0.6, n)
    return h, tau, fl


def activity(m, h, tau, fl, cats):
    """m[n, A] months -> log activity [n, A, C]."""
    d = (m[:, :, None] - h[:, None, :]) % 12
    a = fl[:, None, :] + (1 - fl[:, None, :]) * np.exp(-d / tau[:, None, :])
    la = np.log(a)
    oth = [j for j, c in enumerate(cats) if c == 'OTHER']
    la[:, :, oth] = 0.0
    return la


def simulate(rng, n, N, cats, mode='agro', same_frac=0.5, fixed_months=None, beta_sd=1.5,
             gamma_max=2.0, sigma_max=1.0, beta=None, h=None):
    """N: entries per archive (list). Returns dict of params and counts[n, A, C].
    fixed_months: list of (archive index, allowed months list) for archives whose month is drawn
    only from a given set (training archives with an argued season)."""
    A, C = len(N), len(cats)
    if h is None:
        h, tau, fl = draw_calendar(rng, n, cats, mode)
    else:
        h, tau, fl = h
    same = rng.random(n) < same_frac
    m = rng.integers(0, 12, (n, A))
    m[same] = m[same][:, :1]
    if fixed_months:
        for ai, allowed in fixed_months:
            m[:, ai] = rng.choice(allowed, n)
    gam = rng.uniform(0, gamma_max, n)
    sig = rng.uniform(0, sigma_max, n)
    if beta is None:
        beta = rng.normal(0, beta_sd, (n, C))
    oth = cats.index('OTHER')
    beta = beta.copy(); beta[:, oth] = 0.0
    u = rng.normal(0, 1, (n, A, C)) * sig[:, None, None]
    r = beta[:, None, :] + gam[:, None, None] * activity(m, h, tau, fl, cats) + u
    r -= r.max(-1, keepdims=True)
    p = np.exp(r); p /= p.sum(-1, keepdims=True)
    X = np.zeros((n, A, C), np.int32)
    for ai in range(A):
        X[:, ai, :] = rng.multinomial(N[ai], p[:, ai, :])
    return dict(m=m, same=same, gam=gam, sig=sig, h=h, tau=tau, fl=fl, beta=beta, X=X)


def summ(X):
    """centred log-ratio of (counts + 0.5) per archive, flattened."""
    L = np.log(X + 0.5)
    L = L - L.mean(-1, keepdims=True)
    return L.reshape(L.shape[0], -1)


def abc(S_bank, s_obs, k=500, scale=None):
    if scale is None:
        sub = S_bank[:: max(1, len(S_bank) // 50000)]
        scale = (np.median(np.abs(sub - np.median(sub, 0)), 0) + 1e-3).astype(np.float32)
    D = ((S_bank - s_obs[None].astype(np.float32)) / scale) ** 2
    D = D.sum(1)
    idx = np.argpartition(D, k)[:k]
    return idx, scale


def month_post(m_acc):
    """m_acc[k, A] -> posterior [A, 12] (Laplace +0.5)."""
    A = m_acc.shape[1]
    P = np.zeros((A, 12))
    for a in range(A):
        P[a] = np.bincount(m_acc[:, a], minlength=12) + 0.5
        P[a] /= P[a].sum()
    return P


def cred_set(p, level=0.8):
    o = np.argsort(-p); cs = np.cumsum(p[o])
    k = int(np.searchsorted(cs, level)) + 1
    return sorted(o[:k].tolist())


def kl_unif(p):
    return float((p * np.log(p * 12)).sum())


def fmt_months(ms):
    return ','.join(MONTHS[i] for i in ms)
