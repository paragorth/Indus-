"""v81 THE MESSAGE IS IN THE FIRST LETTERS OF WORDS: shared helpers.

Idea: if word beginnings carry the topic (v59) and the rest is surface, the real stream may be the sequence of word
ONSETS alone (an initials-only / abbreviation script). Extract onset streams under thousands of random onset
definitions (preprocessing variant x onset length 1-3 x random glyph-class merge) and score each stream as a text:
order information beyond the line bag (lag 1, skip 1, trigram), onset information beyond the previous word's ending
(the junction), page and section information, neighbour avoidance, recurrence of onset n-grams across pages.

Controls (through the v72 surface machinery): real texts in which ONLY the initials carry content
  INIT_LA  Caesar, each word replaced by (its initial, merge-coded) + a random Voynich tail
  INIT_IT  Manzoni, same
  ABBR_LA  Isidore-type Latin (v21 LA), suspension abbreviation: first two letters kept + random tail
  INIT_SY  Syon catalogue list (acrostic-style list of entries), initials + random tail
  INIT_SH  negative: Caesar initials shuffled over the whole text (no order) + random tail
full-text references LA_ISID, IT_BRUM (v78 corpora), generators SELFCIT SC10 MK2 JUNC STACK fitted to ZL3b.
"""
import os, sys, json, math, random, re, pickle, zlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v81_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V

E1C = V.RULES['E1c_keepd']
assert V.rule_hash('E1c_keepd') == '2803cbeb0deb51bf'


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def psave(name, obj): pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    p = os.path.join(CK, name)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def leaf(pid): return V.leaf_half(pid)


def fold(pid):
    import hashlib
    return int(hashlib.md5(('v81' + pid).encode()).hexdigest(), 16) % 2


# ------------------------------------------------------------------ control corpora
def _tails(seed):
    """tails: E1c forms of real Voynich words with the first payload symbol removed (iid, independent of message)"""
    Z = V.voynich('ZL3b')
    T = [V.extract_word(w, False, False, E1C) for p in Z for l in p['lines'] for w in l['w']]
    T = [t[1:] for t in T if len(t) >= 2]
    return T


def initials_corpus(entries, k, seed, shuffle=False, prefix='in'):
    """entries: list of (sec, words). Message = first k letters of each word; each letter merge-coded to a payload
    symbol; cover word = coded message symbols + random tail; then the planted v72 surface."""
    rng = random.Random(seed)
    letters = [w[:k] for s, ws in entries for w in ws]
    if shuffle:
        rng.shuffle(letters)
    it = iter(letters)
    ents = [(s, [next(it) for _ in ws]) for s, ws in entries]
    code = V.payload_code([w for s, ws in ents for w in ws], seed=seed, mode='merge')
    tails = _tails(seed)
    cov = [(s, [''.join(code[c] for c in m) + rng.choice(tails) for m in ws]) for s, ws in ents]
    pages = V._pages_from_entries(cov, None, line_w=9, page_tok=160, cap=36000, prefix=prefix)
    S = V.surface(pages, seed=seed + 1)
    for p in S:
        for l in p['lines']: l.pop('orig', None)
    return S


def v75(sid):
    S = json.load(open(os.path.join(ROOT, 'data', 'v75_ckpt', 'systems.json')))
    out = []
    for s, ws in S[sid]['entries']:
        ws = [re.sub(r'[^a-z]', '', w.lower()) for w in ws]; ws = [w for w in ws if w]
        if ws: out.append((s, ws))
    return out


def sectioned(ents, nsec=6):
    n = len(ents)
    return [('b%d' % min(nsec - 1, i * nsec // n), ws) for i, (s, ws) in enumerate(ents)]


def build():
    C = {}
    C['INIT_LA'] = initials_corpus(sectioned(v75('LANG_caesar')), 1, 8101, prefix='ila')
    C['INIT_IT'] = initials_corpus(sectioned(v75('LANG_manzoni')), 1, 8102, prefix='iit')
    C['ABBR_LA'] = initials_corpus(sectioned(v75('LANG_v21_LA')), 2, 8103, prefix='abl')
    C['INIT_SY'] = initials_corpus(sectioned(v75('CAT_SYON')), 1, 8104, prefix='isy')
    C['INIT_SH'] = initials_corpus(sectioned(v75('LANG_caesar')), 1, 8101, shuffle=True, prefix='ish')
    old = pickle.load(open(os.path.join(ROOT, 'data', 'v78_ckpt', 'corpora.pkl'), 'rb'))
    C['FULL_LA'] = old['LA_ISID']['pages']; C['FULL_IT'] = old['IT_BRUM']['pages']
    import v77_lib as G
    for g in ('SELFCIT', 'SC10', 'MK2', 'JUNC', 'STACK'):
        C['GEN_' + g] = G.generate('ZL3b', g, 8111)
    C['VOY_ZL'] = V.voynich('ZL3b'); C['VOY_IT'] = V.voynich('IT2a')
    psave('corpora.pkl', C)
    return C


KIND = dict(INIT_LA='init', INIT_IT='init', ABBR_LA='init', INIT_SY='init', INIT_SH='null', FULL_LA='full',
            FULL_IT='full', GEN_SELFCIT='gen', GEN_SC10='gen', GEN_MK2='gen', GEN_JUNC='gen', GEN_STACK='gen',
            VOY_ZL='voy', VOY_IT='voy')

# ------------------------------------------------------------------ preprocessing variants (word -> unit string)
ALPHA = list('abcdefghijklmnopqrstuvwxyzCSTKPFGH$_')
AIX = {c: i for i, c in enumerate(ALPHA)}


def _strip_line(w, li0, para0):
    if para0 and len(w) > 1 and w[0] in 'pf': w = w[1:]
    if li0 and len(w) > 1 and w[0] in 'ysdt': w = w[1:]
    return w


def variant(w, li0, para0, v):
    if v == 'raw': return w
    if v == 'noq': return w[1:] if len(w) > 1 and w[0] == 'q' else w
    if v == 'noq_line':
        w = w[1:] if len(w) > 1 and w[0] == 'q' else w
        return _strip_line(w, li0, para0)
    if v == 'noe': return (w[1:] if len(w) > 1 and w[0] == 'q' else w).replace('e', '') or w
    if v == 'e1c': return V.extract_word(w, False, False, E1C)
    if v == 'e1c_line': return V.extract_word(_strip_line(w, li0, para0), False, False, E1C)
    if v == 'e1_line': return V.extract_word(w, li0, para0, V.RULES['E2_line'])
    if v == 'e1c_nogal':
        x = V.extract_word(_strip_line(w, li0, para0), False, False, E1C).replace('G', '').replace('H', '')
        return x or '_'
    if v == 'e1c_noC':
        x = V.extract_word(_strip_line(w, li0, para0), False, False, E1C).replace('C', '')
        return x or '_'
    raise KeyError(v)


VARIANTS = ['raw', 'noq', 'noq_line', 'noe', 'e1c', 'e1c_line', 'e1_line', 'e1c_nogal', 'e1c_noC']


# ------------------------------------------------------------------ compiled corpus (numpy)
class Comp:
    """tokens of one corpus with page/line structure, glyph arrays for every variant, pair/triple indices and
    within-line shuffles."""

    def __init__(self, pages, nshuf=4, seed=81):
        rng = np.random.default_rng(seed)
        self.pids = [p['id'] for p in pages]
        secs = sorted({p['sec'] for p in pages}); sx = {s: i for i, s in enumerate(secs)}
        W, pg, ln, pos, lih, ps0 = [], [], [], [], [], []
        L = 0
        for pi, p in enumerate(pages):
            for l in p['lines']:
                for i, w in enumerate(l['w']):
                    W.append(w); pg.append(pi); ln.append(L); pos.append(i); ps0.append(bool(l['ps']) and i == 0)
                L += 1
        self.W = W; self.pg = np.array(pg); self.ln = np.array(ln); self.pos = np.array(pos)
        self.sec = np.array([sx[pages[i]['sec']] for i in self.pg]); self.nsec = len(secs)
        self.leaf = np.array([leaf(pages[i]['id']) for i in self.pg])
        self.fold = np.array([fold(pages[i]['id']) for i in self.pg])
        n = len(W); self.n = n
        li0 = self.pos == 0
        self.G = {}
        for v in VARIANTS:
            A = np.full((n, 3), AIX['$'], np.int16)
            cache = {}
            for t, w in enumerate(W):
                key = (w, bool(li0[t]), ps0[t])
                x = cache.get(key)
                if x is None:
                    x = variant(w, li0[t], ps0[t], v); cache[key] = x
                for j, c in enumerate(x[:3]): A[t, j] = AIX.get(c, AIX['_'])
            self.G[v] = A
        self.end = np.array([AIX.get(w[-1], AIX['_']) for w in W])
        # pairs within lines
        # line-interior only: the line-first word (markers, p/f) is excluded from every pair and from shuffles
        same1 = (self.ln[1:] == self.ln[:-1]) & (self.pos[:-1] >= 1)
        self.p1 = np.nonzero(same1)[0]
        same2 = (self.ln[2:] == self.ln[:-2]) & (self.pos[:-2] >= 1)
        self.p2 = np.nonzero(same2)[0]
        # within-line shuffles (permutation of token positions)
        self.perm = []
        for _ in range(nshuf):
            perm = np.arange(n)
            starts = np.r_[0, np.nonzero(np.diff(self.ln))[0] + 1, n]
            for a, b in zip(starts[:-1], starts[1:]):
                if b - a > 2: perm[a + 1:b] = a + 1 + rng.permutation(b - a - 1)
            self.perm.append(perm)
        # column shuffles: a line-interior token swaps with tokens at the same word position on other lines of the
        # same page (keeps the position-in-line law and page vocabulary; breaks line mode and adjacency)
        self.cperm = []
        for _ in range(nshuf):
            perm = np.arange(n)
            key = self.pg * 1000 + np.minimum(self.pos, 999)
            for kk in np.unique(key[self.pos >= 1]):
                ix = np.nonzero(key == kk)[0]
                if len(ix) > 1: perm[ix] = ix[rng.permutation(len(ix))]
            self.cperm.append(perm)
        # cross-page shuffle within section (for page information null)
        pp = np.arange(n)
        for s in range(self.nsec):
            ix = np.nonzero(self.sec == s)[0]; pp[ix] = ix[rng.permutation(len(ix))]
        self.pperm = pp


# ------------------------------------------------------------------ onset definitions
def random_def(rng):
    v = rng.choice(VARIANTS)
    k = rng.choice([1, 1, 2, 2, 3])
    if rng.random() < 0.3:
        part = None; m = None
    else:
        m = rng.randint(2, 12)
        part = [rng.randrange(m) for _ in ALPHA]
        part[AIX['$']] = m  # end-of-word stays its own class
    return dict(v=v, k=k, part=part, m=m)


def onset(C, d, perm=None):
    A = C.G[d['v']][:, :d['k']].astype(np.int64)
    if d['part'] is not None:
        A = np.array(d['part'], np.int64)[A]; M = d['m'] + 1
    else:
        M = len(ALPHA)
    o = np.zeros(C.n, np.int64)
    for j in range(d['k']): o = o * M + A[:, j]
    _, o = np.unique(o, return_inverse=True)
    return o


# ------------------------------------------------------------------ scoring
def _cnt(keys):
    u, c = np.unique(keys, return_counts=True); return u, c


def _look(u, c, keys):
    if len(u) == 0: return np.zeros(len(keys))
    i = np.searchsorted(u, keys); i = np.minimum(i, len(u) - 1)
    return np.where(u[i] == keys, c[i], 0).astype(float)


BETA = 5.0


def _ce_cond(ctx_tr, y_tr, ctx_te, y_te, base_te, K):
    """held-out bits of p(y|ctx) = (C[ctx,y] + B*base)/(C[ctx]+B); base_te = backoff prob of the test y."""
    u2, c2 = _cnt(ctx_tr * K + y_tr); u1, c1 = _cnt(ctx_tr)
    p = (_look(u2, c2, ctx_te * K + y_te) + BETA * base_te) / (_look(u1, c1, ctx_te) + BETA)
    return -np.log2(p), p


def stream_feats(C, o, tr, te, K=None):
    """tr, te: boolean token masks (page-level). Returns dict of features (bits/token or logs)."""
    K = int(o.max()) + 2
    out = {}
    uc = np.bincount(o[tr], minlength=K).astype(float) + 0.5
    pu = uc / uc.sum()
    res = {}
    for tag, perm in [('r', None)] + [('s%d' % i, P) for i, P in enumerate(C.perm)] + [('c%d' % i, P) for i, P in enumerate(C.cperm)]:
        s = o if perm is None else o[perm]
        e = C.end if perm is None else C.end[perm]
        i1 = C.p1; a, b = s[i1], s[i1 + 1]; ea = e[i1]
        m_tr = tr[i1]; m_te = te[i1]
        hu = -np.log2(pu[b[m_te]])
        hb, pb = _ce_cond(a[m_tr], b[m_tr], a[m_te], b[m_te], pu[b[m_te]], K)
        # junction: ending -> next onset, then add own onset
        he, pe = _ce_cond(ea[m_tr], b[m_tr], ea[m_te], b[m_te], pu[b[m_te]], K)
        ctx_tr = ea[m_tr] * K + a[m_tr]; ctx_te = ea[m_te] * K + a[m_te]
        hea, _ = _ce_cond(ctx_tr, b[m_tr], ctx_te, b[m_te], pe, K)
        # trigram (needs i, i+1, i+2 in line)
        i2 = C.p2; a2, b2, c2 = s[i2], s[i2 + 1], s[i2 + 2]; m2tr = tr[i2]; m2te = te[i2]
        hs, _ = _ce_cond(a2[m2tr], c2[m2tr], a2[m2te], c2[m2te], pu[c2[m2te]], K)
        hu2 = -np.log2(pu[c2[m2te]])
        _, pbc = _ce_cond(b2[m2tr], c2[m2tr], b2[m2te], c2[m2te], pu[c2[m2te]], K)
        hbc = -np.log2(pbc)
        htri, _ = _ce_cond(a2[m2tr] * K + b2[m2tr], c2[m2tr], a2[m2te] * K + b2[m2te], c2[m2te], pbc, K)
        same = float(np.mean(a[m_te] == b[m_te])) if m_te.any() else 0.0
        res[tag] = dict(lag1=hu.mean() - hb.mean(), skip=hu2.mean() - hs.mean(), beyond_end=he.mean() - hea.mean(),
                        tri=hbc.mean() - htri.mean(), junc=hu.mean() - he.mean(), same=same)
    sh = [res[k] for k in res if k[0] == 's']; ch = [res[k] for k in res if k[0] == 'c']
    for f in ('lag1', 'skip', 'beyond_end', 'tri', 'junc'):
        out[f] = res['r'][f] - max(np.mean([x[f] for x in sh]), np.mean([x[f] for x in ch]))
        out[f + '_w'] = res['r'][f] - np.mean([x[f] for x in sh])
    out['nbr'] = math.log((res['r']['same'] + 1e-3) / (np.mean([x['same'] for x in sh]) + 1e-3))
    # section information (held out)
    sec_te = C.sec[te]; y = o[te]
    sc = np.zeros((C.nsec, K)); np.add.at(sc, (C.sec[tr], o[tr]), 1)
    psec = (sc[sec_te, y] + 50 * pu[y]) / (sc.sum(1)[sec_te] + 50)
    out['sec'] = float(np.mean(-np.log2(pu[y])) - np.mean(-np.log2(psec)))
    # page information (test half): plug-in MI(page; onset) minus same after cross-page shuffle within section
    out['page'] = _mi(C.pg[te], o[te]) - _mi(C.pg[te], o[C.pperm][te])
    out['K'] = int(len(np.unique(o)))
    out['H'] = float(-(pu * np.log2(pu)).sum())
    out['J'] = out['lag1'] + out['skip'] + out['beyond_end']
    return out


def _mi(x, y):
    _, x = np.unique(x, return_inverse=True); _, y = np.unique(y, return_inverse=True)
    nx, ny = x.max() + 1, y.max() + 1
    j = np.bincount(x * ny + y, minlength=nx * ny).astype(float).reshape(nx, ny); j /= j.sum()
    px = j.sum(1, keepdims=True); py = j.sum(0, keepdims=True)
    m = j > 0
    return float((j[m] * np.log2(j[m] / (px @ py)[m])).sum())


def masks(C, stage):
    """stage 'sel': train fold0, test fold1 inside leaf half 0; 'hold': train leaf 0, test leaf 1."""
    if stage == 'sel':
        return (C.leaf == 0) & (C.fold == 0), (C.leaf == 0) & (C.fold == 1)
    return C.leaf == 0, C.leaf == 1


FEATS = ['J', 'lag1', 'skip', 'beyond_end', 'tri', 'junc', 'nbr', 'sec', 'page', 'K', 'H']
