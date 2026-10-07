"""v90: WORDS AS WHEEL COORDINATES (Lullian combinatorial art read as an encoding surface).

A Voynich word is read as a set of coordinates on concept wheels: anchored glyph fields
(first, last, second, second-last, ... , middle rest) are partitioned into W wheels; the
value of a wheel is the tuple of its fields. Thousands of random decompositions are scored by
how compactly an independent-wheels product model describes held-out tokens (train folios).
Meaning test (test folios only): for each wheel of each surviving decomposition, how much the
page itself predicts the wheel value on held-out lines (odd lines -> even lines) beyond
  (a) stratum = section x hand x language x line role x word position   [g_page]
  (b) the neighbouring pages of the same section (slow habit drift)     [g_self = g_page - g_neigh]
A content wheel = one wheel with high g_self per bit of entropy while the others are flat.
"""
import os, sys, json, math, hashlib, random, re
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v90_ckpt')
LOOPS = os.path.join(ROOT, 'loops')
os.makedirs(CK, exist_ok=True)

FIELDS = ['f1', 'l1', 'f2', 'l2', 'f3', 'l3', 'mid']
NF = len(FIELDS)


def fields_of(u, scheme=0):
    """glyph-unit list -> 7 field strings ('_' = absent). Lossless.
    scheme 0: alternate from both ends (f1 l1 f2 l2 f3 l3, rest = mid)
    scheme 1: start-heavy (f1 f2 f3 f4->'l3' slot, l1 l2, rest mid)
    scheme 2: end-heavy (f1, l1 l2 l3 l4->'f3' slot, f2, rest mid)"""
    n = len(u)
    out = ['_'] * NF
    if scheme == 0:
        order = [(0, 0), (1, -1), (2, 1), (3, -2), (4, 2), (5, -3)]
    elif scheme == 1:
        order = [(0, 0), (2, 1), (4, 2), (1, -1), (5, 3), (3, -2)]
    else:
        order = [(0, 0), (1, -1), (3, -2), (5, -3), (2, 1), (4, -4)]
    taken = [False] * n
    for k, (fi, pos) in enumerate(order):
        if k >= n:
            break
        idx = pos if pos >= 0 else n + pos
        if idx < 0 or idx >= n or taken[idx]:
            continue
        out[fi] = u[idx]; taken[idx] = True
    rest = ''.join(u[i] for i in range(n) if not taken[i])
    out[6] = rest if rest else '_'
    return out


# ------------------------------------------------------------------ corpora
def voynich_tokens(name='ZL3b'):
    """-> list of token dicts with page, para, stratum, parity, line idx, units."""
    L = vlib.load_voynich(name, ltypes=('P',))
    toks = []
    page_order = []
    para = -1
    lineno = defaultdict(int)
    for r in L:
        f = r['folio']
        if f not in page_order:
            page_order.append(f)
        if r['para_start'] or para < 0:
            para += 1
        ws = [vlib.glyphs(w) for w in r['words']]
        ws = [w for w in ws if w and '?' not in w and '*' not in w]
        if not ws:
            continue
        role = 'F' if r['para_start'] else ('L' if r['para_end'] else 'B')
        li = lineno[f]; lineno[f] += 1
        for i, u in enumerate(ws):
            wp = 'a' if i == 0 else ('z' if i == len(ws) - 1 else 'm')
            toks.append({'page': f, 'para': para, 'sec': r['illus'], 'hand': r['hand'], 'lang': r['lang'] or '-',
                         'role': role, 'wp': wp, 'line': li, 'u': u})
    return toks, page_order


def split_pages(page_order, salt='v90'):
    tr, te = [], []
    for p in page_order:
        h = int(hashlib.md5((salt + p).encode()).hexdigest(), 16)
        (tr if h % 2 == 0 else te).append(p)
    return set(tr), set(te)


# ------------------------------------------------------------------ encoding to arrays
class Enc:
    def __init__(self, toks, page_order, scheme=0):
        self.toks = toks
        self.n = len(toks)
        F = [fields_of(t['u'], scheme) for t in toks]
        self.fvocab = []
        self.F = np.zeros((self.n, NF), dtype=np.int64)
        for j in range(NF):
            voc = {}
            for i, f in enumerate(F):
                self.F[i, j] = voc.setdefault(f[j], len(voc))
            self.fvocab.append(voc)
        pidx = {p: i for i, p in enumerate(page_order)}
        self.page = np.array([pidx[t['page']] for t in toks])
        self.npage = len(page_order)
        self.page_order = page_order
        sk = {}
        self.strat = np.array([sk.setdefault((t['sec'], t['hand'], t['lang'], t['role'], t['wp']), len(sk)) for t in toks])
        self.nstrat = len(sk)
        ck = {}
        self.coarse = np.array([ck.setdefault((t['sec'], t['hand'], t['lang']), len(ck)) for t in toks])
        self.par = np.array([t['line'] % 2 for t in toks])
        self.para = np.array([t['para'] for t in toks])
        # page section for neighbours (same coarse stratum, nearest in order)
        psec = {}
        for i in range(self.n):
            psec.setdefault(self.page[i], self.coarse[i])
        self.psec = psec
        self.neigh = {}
        for p in range(self.npage):
            if p not in psec:
                continue
            same = [q for q in range(self.npage) if q != p and psec.get(q) == psec[p]]
            same.sort(key=lambda q: abs(q - p))
            self.neigh[p] = same[:2]

    def wheel_codes(self, cols):
        if len(cols) == 1:
            return self.F[:, cols[0]].copy()
        _, inv = np.unique(self.F[:, cols], axis=0, return_inverse=True)
        return inv.ravel()


# ------------------------------------------------------------------ compactness
def field_cost(enc, cols):
    """bits to spell an unseen wheel value field by field (uniform per field)."""
    return sum(math.log2(len(enc.fvocab[c]) + 1) for c in cols)


def compactness(enc, decomp, fit_mask, ev_mask):
    """held-out bits/token of the independent-wheel product model. decomp = list of col lists."""
    tot = 0.0
    nev = ev_mask.sum()
    for cols in decomp:
        v = enc.wheel_codes(cols)
        K = v.max() + 1
        c = np.bincount(v[fit_mask], minlength=K).astype(float)
        N = c.sum(); seen = (c > 0).sum()
        a = 0.5
        denom = N + a * (seen + 1)
        pv = np.where(c > 0, (c + a) / denom, 0.0)
        esc = a / denom
        ve = v[ev_mask]
        p = pv[ve]
        unseen = p == 0
        bits = -np.log2(np.where(unseen, 1.0, p)).sum()
        bits += unseen.sum() * (-math.log2(esc) + field_cost(enc, cols))
        tot += bits
    return tot / nev


def random_decomp(rng):
    W = int(rng.integers(2, 5))
    lab = rng.integers(0, W, size=NF)
    groups = [sorted(np.where(lab == w)[0].tolist()) for w in range(W)]
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return random_decomp(rng)
    return groups


def canon(decomp):
    return tuple(sorted(tuple(g) for g in decomp))


# ------------------------------------------------------------------ meaning test
def _smooth_strat(v, strat, mask, K, nstrat, a=1.0):
    g = np.bincount(v[mask], minlength=K).astype(float) + 0.1
    g /= g.sum()
    S = np.zeros((nstrat, K))
    np.add.at(S, (strat[mask], v[mask]), 1.0)
    S = (S + a * 20 * g) / (S.sum(1, keepdims=True) + a * 20)
    return S


ALPHAS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000, 10000]


def _logps(enc, v, pages_set, alphas):
    """log2 probabilities of even-line tokens of pages_set under stratum, own-page, neighbour-page,
    same-paragraph and other-paragraph(same page) models for each alpha."""
    K = v.max() + 1
    inset = np.isin(enc.page, list(pages_set))
    ev = inset & (enc.par == 1)
    fit = ~ev
    S = _smooth_strat(v, enc.strat, fit, K, enc.nstrat)
    Cp = np.zeros((enc.npage, K)); np.add.at(Cp, (enc.page[fit], v[fit]), 1.0)
    npg = Cp.sum(1)
    Cn = np.zeros((enc.npage, K))
    for p in range(enc.npage):
        for q in enc.neigh.get(p, []):
            Cn[p] += Cp[q]
    nn = Cn.sum(1); scale = np.where(nn > 0, npg / np.maximum(nn, 1), 0.0)
    Cq = {}
    for i in np.where(fit & inset)[0]:
        q = Cq.get(enc.para[i])
        if q is None:
            q = Cq[enc.para[i]] = np.zeros(K)
        q[v[i]] += 1
    vv = v[ev]; pg = enc.page[ev]; pa = enc.para[ev]
    ps = S[enc.strat[ev], vv]
    cown = Cp[pg, vv]; nown = npg[pg]
    cn = Cn[pg, vv] * scale[pg]; nnn = npg[pg] * (nn[pg] > 0)
    cs = np.zeros(len(vv)); ns = np.zeros(len(vv)); co = np.zeros(len(vv)); no = np.zeros(len(vv))
    valid = np.zeros(len(vv), bool)
    for j in range(len(vv)):
        q = Cq.get(pa[j])
        if q is None:
            continue
        nq = q.sum()
        oth = Cp[pg[j]] - q
        no_ = oth.sum()
        if nq < 5 or no_ < 5:
            continue
        valid[j] = True
        cs[j] = q[vv[j]]; ns[j] = nq
        co[j] = oth[vv[j]] * nq / no_; no[j] = nq
    out = {'ls': np.log2(ps), 'pg': pg, 'valid': valid}
    for a in alphas:
        out[('own', a)] = np.log2((cown + a * ps) / (nown + a))
        out[('nb', a)] = np.log2((cn + a * ps) / (nnn + a))
        out[('same', a)] = np.log2((cs + a * ps) / (ns + a))
        out[('oth', a)] = np.log2((co + a * ps) / (no + a))
    return out


def page_gains(enc, v, pages_set, tune_set):
    """alpha tuned on tune_set (train folios); gains (bits/token) reported on pages_set."""
    T = _logps(enc, v, tune_set, ALPHAS)
    a = max(ALPHAS, key=lambda a: T[('own', a)].mean())
    ap = max(ALPHAS, key=lambda a: T[('same', a)][T['valid']].mean() if T['valid'].any() else 0)
    E = _logps(enc, v, pages_set, sorted({a, ap}))
    Hs = -E['ls'].mean()
    gp = (E[('own', a)] - E['ls']).mean()
    gn = (E[('nb', a)] - E['ls']).mean()
    va = E['valid']
    gq = (E[('same', ap)][va] - E[('oth', ap)][va]).mean() if va.any() else 0.0
    return {'H': Hs, 'g_page': gp, 'g_neigh': gn, 'g_self': gp - max(gn, 0.0), 'g_para': gq,
            'alpha': a, 'alpha_p': ap, 'n': int(len(E['ls'])), 'n_para': int(va.sum()),
            'pp_own': {int(p): float((E[('own', a)] - E['ls'])[E['pg'] == p].sum()) for p in np.unique(E['pg'])}}


def wheel_profile(enc, decomp, pages_set, tune_set):
    out = []
    for cols in decomp:
        v = enc.wheel_codes(cols)
        r = page_gains(enc, v, pages_set, tune_set)
        r['cols'] = [FIELDS[c] for c in cols]
        r['r_self'] = r['g_self'] / max(r['H'], 1e-9)
        r['r_page'] = r['g_page'] / max(r['H'], 1e-9)
        r['r_para'] = r['g_para'] / max(r['H'], 1e-9)
        rl = recur_lift(enc, v, pages_set)
        r['lift'] = rl['lift']; r['lift_z'] = rl['z']; r['lift_n'] = rl['n']
        comp = [c for c in range(NF) if c not in cols]
        if comp:
            rc = recur_lift(enc, enc.wheel_codes(comp), pages_set)
            r['clift'] = rc['lift']; r['clift_z'] = rc['z']; r['clift_n'] = rc['n']
        out.append(r)
    return out


def specificity(prof, key='r_self'):
    vals = sorted([p[key] for p in prof], reverse=True)
    return vals[0] - float(np.mean(vals[1:])) if len(vals) > 1 else 0.0


# ------------------------------------------------------------------ controls
def shuffle_within(toks, key=('sec', 'hand'), seed=0):
    """shuffle word units across positions within hand+section (destroys page/para signal)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, t in enumerate(toks):
        groups[tuple(t[k] for k in key)].append(i)
    out = [dict(t) for t in toks]
    for g, idx in groups.items():
        us = [toks[i]['u'] for i in idx]
        rng.shuffle(us)
        for i, u in zip(idx, us):
            out[i]['u'] = u
    return out


def markov_text(toks, order=2, seed=0):
    """per-(section,hand) glyph Markov-`order` words, word lengths natural; positions kept."""
    rng = random.Random(seed)
    models = defaultdict(lambda: defaultdict(Counter))
    for t in toks:
        g = (t['sec'], t['hand'])
        seq = ['^'] * order + t['u'] + ['$']
        for i in range(order, len(seq)):
            models[g][tuple(seq[i - order:i])][seq[i]] += 1
    cache = {}
    out = []
    for t in toks:
        g = (t['sec'], t['hand'])
        ctx = ('^',) * order
        u = []
        while len(u) < 15:
            key = (g, ctx)
            if key not in cache:
                c = models[g][ctx]
                cache[key] = (list(c.keys()), np.cumsum(list(c.values())))
            ks, cs = cache[key]
            x = ks[int(np.searchsorted(cs, rng.random() * cs[-1], side='right'))]
            if x == '$':
                break
            u.append(x); ctx = ctx[1:] + (x,)
        if not u:
            u = ['o']
        tt = dict(t); tt['u'] = u
        out.append(tt)
    return out


# ------------------------------------------------------------------ plants
LULL = ['o', 'e', 'a', 'i', 'C', 'k', 't', 'd', 'l']   # 9 concept letters (B C D E F G H I K)
LULL18 = LULL + ['y', 'r', 'S', 'p', 'f', 'n', 's', 'm', 'q']


def plaintext_stream(text_id, need):
    T = json.load(open(os.path.join(DATA, 'v89_ckpt', 'texts.json')))[text_id]
    units = [u['tok'] for u in T['units'] if u['tok']]
    return units


def concept_of(w, salt, A=LULL):
    h = int(hashlib.md5((salt + w).encode()).hexdigest(), 16)
    k = len(A)
    return A[h % k], A[(h // k) % k]


HOMO = [chr(0x3b1 + i) for i in range(18)]   # 18 synthetic glyphs (homophone partners of LULL18)


def plant(toks, page_order, mode, text_id=None, seed=0, frac=1.0, drift=0.0, unit='para', A=None, hstr=0.35, homo=False):
    """Carrier = Voynich words shuffled within hand+section (no page signal), lengths >= 4.
    mode 'concept': f2,l2 <- Lullian concept pair of the plaintext token; plaintext laid out entry
                    by entry, one entry starting at each paragraph start (entries continue if short).
    mode 'habit'  : f2,l2 <- per-page distribution drifting smoothly along the page order (AR(1)).
    frac: share of tokens that carry the plant (others keep carrier glyphs).
    drift: strength of an additional smooth habit drift on f1 (both modes)."""
    rng = np.random.default_rng(seed)
    base = shuffle_within(toks, seed=seed + 101)
    # pool of long carriers per (sec,hand)
    pool = defaultdict(list)
    for t in base:
        if len(t['u']) >= 4:
            pool[(t['sec'], t['hand'])].append(t['u'])
    allpool = [t['u'] for t in base if len(t['u']) >= 4]
    pidx = {p: i for i, p in enumerate(page_order)}
    # smooth habit drifts
    P = len(page_order)
    def ar_drift(dim, strength, rho=0.9):
        z = np.zeros((P, dim)); x = rng.normal(size=dim)
        for i in range(P):
            x = rho * x + math.sqrt(1 - rho ** 2) * rng.normal(size=dim)
            z[i] = x
        w = np.exp(strength * z)
        return w / w.sum(1, keepdims=True)
    AA = A or LULL
    hab1 = ar_drift(len(AA), hstr)
    hab2 = ar_drift(len(AA), hstr)
    f1_letters = sorted({t['u'][0] for t in base})
    f1_base = Counter(t['u'][0] for t in base)
    fl = [x for x, _ in f1_base.most_common(8)]
    hdr = ar_drift(len(fl), drift) if drift > 0 else None
    units = plaintext_stream(text_id, len(toks)) if mode == 'concept' else None
    func = set()
    if units:
        func = {w for w, _ in Counter(w for u in units for w in u).most_common(60)}
    salt = 'lull%d' % seed
    ui, wi = 0, 0
    out = []
    last_para = None
    for t in base:
        tt = dict(t)
        u = list(t['u'])
        if len(u) < 4:
            pl = pool.get((t['sec'], t['hand'])) or allpool
            u = list(pl[int(rng.integers(len(pl)))])
        p = pidx[t['page']]
        if mode == 'concept':
            key = t['para'] if unit == 'para' else t['page']
            if key != last_para:
                last_para = key
                ui = (ui + 1) % len(units); wi = 0
            w = units[ui][wi]
            wi += 1
            if wi >= len(units[ui]):
                ui = (ui + 1) % len(units); wi = 0
            a, b = concept_of(w, salt, A or LULL) if w not in func else (u[1], u[-2])
        else:
            a = AA[int(rng.choice(len(AA), p=hab1[p]))]
            b = AA[int(rng.choice(len(AA), p=hab2[p]))]
        if homo:
            if rng.random() < 0.5 and a in LULL18:
                a = HOMO[LULL18.index(a)]
            if rng.random() < 0.5 and b in LULL18:
                b = HOMO[LULL18.index(b)]
        if rng.random() < frac:
            u[1] = a; u[-2] = b
        if hdr is not None:
            u[0] = fl[int(rng.choice(len(fl), p=hdr[p]))]
        tt['u'] = u
        out.append(tt)
    return out


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def recur_lift(enc, v, pages_set, lo=2, hi=20):
    """rare wheel values (corpus count lo..hi) on even lines of pages_set: is the value present in the
    odd lines of its own page more often than in the odd lines of the other pages of the same
    section x hand x language stratum? -> lift, z, n. (vectorised)"""
    v = np.asarray(v)
    K = int(v.max()) + 1
    tot = np.bincount(v, minlength=K)
    odd = enc.par == 0
    keys = np.unique(enc.page[odd].astype(np.int64) * K + v[odd])
    kp = keys // K; kv = keys % K
    if not hasattr(enc, '_pc'):
        pc = np.full(enc.npage, -1)
        for p, c in enc.psec.items():
            pc[p] = c
        enc._pc = pc
        enc._npc = np.bincount(pc[pc >= 0])
        enc._pset = np.zeros(enc.npage, bool)
    pc = enc._pc
    ckeys = pc[kp].astype(np.int64) * K + kv
    cu, cc = np.unique(ckeys, return_counts=True)
    inset = np.isin(enc.page, list(pages_set))
    ev = np.where(inset & (enc.par == 1) & (tot[v] >= lo) & (tot[v] <= hi))[0]
    if len(ev) == 0:
        return {'lift': 0.0, 'z': 0.0, 'n': 0, 'hits': 0.0, 'exp': 0.0}
    p = enc.page[ev]; x = v[ev]; c = pc[p]
    m = enc._npc[c] - 1
    ok = m >= 2
    p, x, c, m = p[ok], x[ok], c[ok], m[ok]
    own = np.isin(p.astype(np.int64) * K + x, keys).astype(float)
    ck = c.astype(np.int64) * K + x
    idx = np.searchsorted(cu, ck)
    idx = np.minimum(idx, len(cu) - 1)
    cnt = np.where(cu[idx] == ck, cc[idx], 0)
    e = (cnt - own) / m
    H, E, Var = own.sum(), e.sum(), (e * (1 - e)).sum()
    return {'lift': float(H / max(E, 1e-9)), 'z': float((H - E) / math.sqrt(max(Var, 1e-9))), 'n': int(len(e)),
            'hits': float(H), 'exp': float(E)}
