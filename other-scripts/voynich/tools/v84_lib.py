"""v84 THE LAST THREAD: CAN ANY MEANINGLESS PROCESS FAKE THE PAGE VOCABULARY?  Shared helpers.

v81 found that once a word's onset is fixed, the rest of the word still carries page information (z 43-90). v82 showed
that the onset link itself is reproduced by a line-mood plus glyph-harmony generator. What no generator has matched
so far is page-level vocabulary carried by whole words beyond line and onset effects.

FROZEN STATISTIC (PB, page information per word beyond onset, line and section; frozen 7 Oct 2026 before any
generator or control was scored):
  tokens   : line-interior words (word position >= 1) of every paragraph line.
  onset    : o(w) = first glyph unit of w after one leading 'q' is removed.
  baseline : p0(w | o, section), trained on leaf half 0 only:
               p0 = (C_sec(o,w) + A p_g(w|o)) / (C_sec(o) + A),  p_g = (C(o,w) + B p_ch(w|o)) / (C(o) + B),
               p_ch = q-flag x glyph-bigram chain started at the onset (add-0.5), A = 20, B = 2.
  page     : scored on leaf half 1. For token t on page P, cache = words with the same onset on the OTHER lines of P
             (own line excluded, so within-line repeats and line mood do not count):
               p1 = (1 - lam) p0 + lam c(w)/n,  lam = n / (n + KAPPA); KAPPA chosen from {2,5,10,20,50,100,200,500,1000,
               2000,5000} to maximise G on the swapped halves (baseline trained on leaf 1, scored on leaf 0), so the test
               half never tunes it.
  gain     : G = mean(log2 p1 - log2 p0) over test tokens.
  null     : the test-half lines are dealt at random to the page slots of the same section (page line counts kept;
             line content, onset, section and p0 untouched), 10 deals.  PB = G - mean(G_null), z = PB / sd(G_null).
  variants : PB_far (cache only from lines >= 3 lines away on the page), PB_word (no onset conditioning: cache over
             all words of the other lines, baseline p0(w|sec) built the same way with o = '*').
"""
import os, sys, math, random, pickle, json, re, zlib, hashlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v84_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V

A_SEC, B_GLOB, KAPPA, NNULL = 20.0, 2.0, 10.0, 10
FROZEN_HASH_SRC = 'PB v84: onset=first unit after q; p0 A20 B2 bigram chain; cache other lines same onset KAPPA grid fit on swap; ' \
                  'null deal test lines to page slots within section x10; train leaf0 test leaf1; pos>=1'
FROZEN_HASH = hashlib.sha256(FROZEN_HASH_SRC.encode()).hexdigest()[:16]


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def psave(name, obj): pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    p = os.path.join(CK, name)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def onset(w):
    if len(w) > 1 and w[0] == 'q': w = w[1:]
    return w[0]


def leaf(pid): return V.leaf_half(pid)


# ------------------------------------------------------------------ compiled token table
class Tok:
    def __init__(self, pages, swap=False):
        secs = sorted({p['sec'] for p in pages}); sx = {s: i for i, s in enumerate(secs)}
        W, pg, ln, lip, pos, lf = [], [], [], [], [], []
        L = 0; self.page_lines = []
        for pi, p in enumerate(pages):
            fl = leaf(p['id']) ^ (1 if swap else 0); pl = []
            for li, l in enumerate(p['lines']):
                for i, w in enumerate(l['w']):
                    W.append(w); pg.append(pi); ln.append(L); lip.append(li); pos.append(i); lf.append(fl)
                pl.append(L); L += 1
            self.page_lines.append(pl)
        self.W = W; self.n = len(W); self.nline = L
        self.pg = np.array(pg); self.ln = np.array(ln); self.lip = np.array(lip); self.pos = np.array(pos)
        self.leaf = np.array(lf); self.sec = np.array([sx[pages[i]['sec']] for i in self.pg]); self.nsec = len(secs)
        self.secs = secs; self.pages_sec = np.array([sx[p['sec']] for p in pages])
        self.page_leaf = np.array([leaf(p['id']) ^ (1 if swap else 0) for p in pages])
        self.line_page = np.zeros(L, int); self.line_lip = np.zeros(L, int)
        for pi, pl in enumerate(self.page_lines):
            for li, x in enumerate(pl): self.line_page[x] = pi; self.line_lip[x] = li
        _, self.wid = np.unique(W, return_inverse=True)
        self.ons = [onset(w) for w in W]
        _, self.oid = np.unique(self.ons, return_inverse=True)


def _chain_model(words_on):
    """glyph bigram chain p(w | onset): q-flag given onset, then bigrams from the onset to end."""
    qc = Counter(); oc = Counter(); bg = Counter(); ctx = Counter(); alpha = set('$')
    for w, o in words_on:
        q = len(w) > 1 and w[0] == 'q'; s = w[1:] if q else w
        qc[(o, q)] += 1; oc[o] += 1
        for a, b in zip(s, s[1:] + '$'): bg[(a, b)] += 1; ctx[a] += 1; alpha.add(b)
    K = len(alpha) + 1

    def logp(w, o):
        q = len(w) > 1 and w[0] == 'q'; s = w[1:] if q else w
        lp = math.log((qc[(o, q)] + 0.5) / (oc[o] + 1.0))
        for a, b in zip(s, s[1:] + '$'): lp += math.log((bg[(a, b)] + 0.5) / (ctx[a] + 0.5 * K))
        return lp
    return logp


def baseline(T, use_onset=True, idx=None):
    """log p0 for every token (trained on leaf 0 tokens with pos >= 1)."""
    tr = (T.leaf == 0) & (T.pos >= 1)
    O = T.ons if use_onset else ['*'] * T.n
    cs = Counter(); cso = Counter(); cg = Counter(); cgo = Counter()
    for t in np.nonzero(tr)[0]:
        w, o, s = T.W[t], O[t], T.sec[t]
        cs[(s, o, w)] += 1; cso[(s, o)] += 1; cg[(o, w)] += 1; cgo[o] += 1
    chain = _chain_model([(T.W[t], O[t] if use_onset else onset(T.W[t])) for t in np.nonzero(tr)[0]])
    pon = Counter(onset(T.W[t]) for t in np.nonzero(tr)[0]); npon = sum(pon.values()) + 0.5 * (len(pon) + 1)
    out = np.zeros(T.n); memo = {}
    for t in (np.nonzero((T.leaf == 1) & (T.pos >= 1))[0] if idx is None else idx):
        w, o, s = T.W[t], O[t], T.sec[t]; key = (w, o, s)
        v = memo.get(key)
        if v is None:
            if use_onset: pc = math.exp(chain(w, o))
            else: pc = (pon[onset(w)] + 0.5) / npon * math.exp(chain(w, onset(w)))
            pgl = (cg[(o, w)] + B_GLOB * pc) / (cgo[o] + B_GLOB)
            v = math.log2((cs[(s, o, w)] + A_SEC * pgl) / (cso[(s, o)] + A_SEC)); memo[key] = v
        out[t] = v
    return out


KGRID = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def _gain(T, te, lp0, page_of_line, use_onset=True, far=0, kappa=None):
    """mean log2 p1 - log2 p0 on test tokens te, pages given by page_of_line (array over lines)."""
    pg = page_of_line[T.ln[te]]
    o = T.oid[te] if use_onset else np.zeros(len(te), int)
    w = T.wid[te]; l = T.ln[te]
    if far == 0:
        def cnt(keys):
            u, inv, c = np.unique(keys, return_inverse=True, return_counts=True); return c[inv]
        M = int(T.wid.max()) + 1; MO = int(T.oid.max()) + 1; ML = T.nline + 1
        c_pw = cnt(pg * M + w) - cnt(l * M + w)
        n_po = cnt(pg * MO + o) - cnt(l * MO + o)
    else:
        # cache from lines >= far lines away on the same page (line order = slot order on the page)
        slot = slot_of_line(T, page_of_line)[l]
        c_pw = np.zeros(len(te)); n_po = np.zeros(len(te))
        byp = defaultdict(list)
        for j in range(len(te)): byp[pg[j]].append(j)
        for p, js in byp.items():
            js = np.array(js); sl = slot[js]
            d = np.abs(sl[:, None] - sl[None, :]) >= far
            c_pw[js] = (d & (w[js][:, None] == w[js][None, :])).sum(1)
            n_po[js] = (d & (o[js][:, None] == o[js][None, :])).sum(1)
    lam = n_po / (n_po + (KAPPA if kappa is None else kappa))
    p0 = np.exp2(lp0[te])
    p1 = (1 - lam) * p0 + np.where(n_po > 0, lam * c_pw / np.maximum(n_po, 1), 0.0)
    return float(np.mean(np.log2(p1) - lp0[te]))


_SLOT = {}


def slot_of_line(T, page_of_line):
    """position of each line inside its (possibly re-dealt) page: lines keep their dealt order."""
    key = id(page_of_line)
    if key in _SLOT: return _SLOT[key]
    slot = np.zeros(T.nline, int); seen = Counter()
    for L in range(T.nline):
        p = page_of_line[L]; slot[L] = seen[p]; seen[p] += 1
    _SLOT.clear(); _SLOT[key] = slot
    return slot


def deal(T, rng):
    """test-half lines dealt at random to the page slots of the same section (line counts per page kept)."""
    pol = T.line_page.copy()
    test_lines = np.nonzero(T.page_leaf[T.line_page] == 1)[0]
    for s in range(T.nsec):
        ix = test_lines[T.pages_sec[T.line_page[test_lines]] == s]
        if len(ix) > 1: pol[ix] = T.line_page[ix][rng.permutation(len(ix))]
    # keep dealt lines ordered randomly within the page (slot order = order of line ids is irrelevant for far=0)
    return pol


def PB(pages, nnull=NNULL, seed=841, variants=('PB',), swap=False, T=None):
    """the frozen statistic and its variants. Returns dict name -> (excess bits/word, z, G real, null mean)."""
    T = T or Tok(pages, swap=swap)
    TS = Tok(pages, swap=not swap)
    te = np.nonzero((T.leaf == 1) & (T.pos >= 1))[0]; tes = np.nonzero((TS.leaf == 1) & (TS.pos >= 1))[0]
    out = {}
    for v in variants:
        use_on = v != 'PB_word'; far = 3 if v == 'PB_far' else 0
        lps = baseline(TS, use_on)
        kap = max(KGRID, key=lambda k: _gain(TS, tes, lps, TS.line_page, use_on, 0, k))
        lp0 = baseline(T, use_on)
        rng = np.random.default_rng(seed)
        g = _gain(T, te, lp0, T.line_page, use_on, far, kap)
        nul = []
        for _ in range(nnull):
            pol = deal(T, rng)
            if far:   # random slot order inside the dealt page
                order = rng.permutation(T.nline); pol2 = pol  # slot = order of appearance in permuted scan
                slot = np.zeros(T.nline, int); seen = Counter()
                for L in order: slot[L] = seen[pol[L]]; seen[pol[L]] += 1
                _SLOT.clear(); _SLOT[id(pol)] = slot
            nul.append(_gain(T, te, lp0, pol, use_on, far, kap))
        m, s = float(np.mean(nul)), float(np.std(nul))
        out[v] = (g - m, (g - m) / (s + 1e-9), g, m, kap)
    out['ntest'] = int(len(te))
    return out


# ------------------------------------------------------------------ control corpora through the v72 surface
def sectioned(ents, nsec=6):
    n = len(ents)
    return [('b%d' % min(nsec - 1, i * nsec // n), ws) for i, (s, ws) in enumerate(ents)]


def surfaced(entries, seed, prefix, cap=36000):
    words = [w for s, ws in entries for w in ws]
    code = V.payload_code(words, seed=seed, mode='merge')
    pages = V._pages_from_entries(entries, None, line_w=8, page_tok=160, cap=cap, prefix=prefix)
    S = V.surface(V.encode_payload(pages, code), seed=seed + 1)
    for p in S:
        for l in p['lines']: l.pop('orig', None)
    return S


def _v75(sid):
    S = json.load(open(os.path.join(ROOT, 'data', 'v75_ckpt', 'systems.json')))
    out = []
    for s, ws in S[sid]['entries']:
        ws = [re.sub(r'[^a-z]', '', w.lower()) for w in ws]; ws = [w for w in ws if w]
        if ws: out.append((s, ws))
    return out


def controls():
    """real herbals and recipe collections (one entry = one plant / one recipe, entries in book order, 6 contiguous
    sections), each written through a merge payload code and the planted v72 surface; plus a page-word-shuffled twin
    of each (words dealt at random to pages within section: no page vocabulary, negative control)."""
    C = pload('controls.pkl')
    if C: return C
    import v37_lib as L37
    def recs(lst): return [('x', [re.sub(r'[^a-z]', '', w.lower()) for w in r if re.sub(r'[^a-z]', '', w.lower())]) for r in lst]
    src = {
        'HERB_KONRAD': _v75('LANG_konrad_plants'),        # Konrad von Megenberg, plant book (German)
        'HERB_MACER': _v75('LANG_macer'),                 # Macer Floridus (Latin verse herbal)
        'HERB_CULP': recs(L37.culpeper()),                # Culpeper, English herbal
        'RECI_APIC': recs(L37.apicius()),                 # Apicius (Latin recipes)
        'RECI_CURY': recs(L37.cury()),                    # Forme of Cury (Middle English recipes)
    }
    C = {}
    for i, (k, E) in enumerate(src.items()):
        E = [e for e in E if len(e[1]) >= 3]
        C[k] = surfaced(sectioned(E), 8400 + 7 * i, prefix=k[:2].lower() + k[5:7].lower())
    for k in list(C):
        C[k + '_SHUF'] = page_shuffle(C[k], 8490)
    psave('controls.pkl', C)
    return C


def page_shuffle(pages, seed):
    """negative control: line-interior words dealt across pages of the same section (line lengths, line-first words
    kept). Removes page vocabulary, keeps section vocabulary."""
    rng = random.Random(seed)
    bys = defaultdict(list)
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']):
            for i in range(1, len(l['w'])): bys[p['sec']].append(l['w'][i])
    for s in bys: rng.shuffle(bys[s])
    it = {s: iter(v) for s, v in bys.items()}
    out = []
    for p in pages:
        nl = [dict(l, w=[l['w'][0]] + [next(it[p['sec']]) for _ in l['w'][1:]]) for l in p['lines']]
        out.append(dict(p, lines=nl))
    return out


def voynich(name):
    return V.voynich(name)


# ------------------------------------------------------------------ cycle 2: the page-vocabulary kill generator family
import v82_lib as K82


def sample_params(rng, center=None):
    """v82 line-mood + harmony generator (onset layer) extended with page-level mechanisms over word BODIES:
    gm_*  glyph page mood: per-page random weights over body glyph (or glyph-bigram) features, AR(1) across pages
          (gm_rho; near 1 = section-level drift) and drifting line by line inside the page (gm_line);
    lx_*  lexical page mood: per-type random log-weights, AR(1) across pages, drifting by line; lx_K > 0 = K discrete
          moods that recur (each page picks one);
    sd_*  per-page seed vocabulary: sd_n words drawn at page start from the section pool (token frequency ^ sd_alpha),
          used for an interior word with prob sd_p, varied by one glyph with prob sd_mod (the variant replaces the seed
          with prob sd_keep): copy-and-vary restarted each page;
    cite  the v82 within-page self-citation (p_cite, window cite_win, scope page or book)."""
    def lu(a, b): return math.exp(rng.uniform(math.log(a), math.log(b)))
    P = K82.sample_params(rng)
    P['base'] = rng.choice(['stack', 'stack', 'selfcit', 'junc'])
    if P['base'] != 'stack': P['p_vert'] = 0.0
    on = rng.random()
    # which page mechanisms are switched on (each alone or in combination)
    mech = rng.choice(['copy', 'gm', 'lx', 'seed', 'gm+seed', 'lx+seed', 'all', 'none', 'gm+lx'])
    P['mech'] = mech
    if 'copy' not in mech and mech != 'all':
        P['p_cite'] = 0.0 if rng.random() < 0.7 else P['p_cite'] * 0.2
        P['p_vert'] = P['p_vert'] * (0.0 if rng.random() < 0.5 else 0.3)
    P['cite_win'] = rng.choice([20, 60, 200])
    P['cite_scope'] = rng.choice(['page', 'page', 'book'])
    P['gm_beta'] = rng.uniform(0.2, 3.0) if ('gm' in mech or mech == 'all') else 0.0
    P['gm_feat'] = rng.choice(['uni', 'bi'])
    P['gm_rho'] = rng.choice([0.0, rng.uniform(0, 0.9), rng.uniform(0.9, 0.995)])
    P['gm_line'] = rng.choice([0.0, lu(0.02, 0.5)])
    P['lx_beta'] = rng.uniform(0.2, 3.0) if ('lx' in mech or mech == 'all') else 0.0
    P['lx_rho'] = rng.choice([0.0, rng.uniform(0, 0.9), rng.uniform(0.9, 0.995)])
    P['lx_line'] = rng.choice([0.0, lu(0.02, 0.5)])
    P['lx_K'] = rng.choice([0, 0, 4, 12, 40])
    P['sd_n'] = rng.randint(1, 15) if ('seed' in mech or mech == 'all') else 0
    P['sd_p'] = rng.uniform(0.02, 0.4)
    P['sd_mod'] = rng.uniform(0, 0.8)
    P['sd_keep'] = rng.uniform(0, 1)
    P['sd_alpha'] = rng.uniform(0, 1)
    return P


def _feats(types, kind, nb=48):
    F = np.zeros((len(types), 32 if kind == 'uni' else nb))
    gi = {c: i for i, c in enumerate(K82.GLYPHS)}
    for t, w in enumerate(types):
        b = w[1:] if len(w) > 1 and w[0] == 'q' else w
        b = b[1:]                                     # body = word minus its onset unit
        if kind == 'uni':
            for c in b:
                if c in gi: F[t, gi[c]] += 1
        else:
            for x, y in zip('^' + b, b + '$'): F[t, zlib.crc32((x + y).encode()) % nb] += 1
    return F


class _Pool2:
    __slots__ = ('ons', 'cnt', 'ids', 'wc')

    def __init__(self, ws, k, tix):
        d = defaultdict(Counter)
        for w in ws: d[K82.front(w, k)][tix[w]] += 1
        self.ons = list(d); self.cnt = np.array([sum(d[o].values()) for o in self.ons], float)
        self.ids = [np.array(list(d[o].keys())) for o in self.ons]
        self.wc = [np.array(list(d[o].values()), float) for o in self.ons]


def gen_page(pages, P, seed, moodcache=None):
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    k = P['k']
    gk = lambda p: '%s|%s' % (p['sec'], p.get('lang', '-'))
    POOLW = defaultdict(list); FOLW = defaultdict(list); GL = defaultdict(Counter); SECW = defaultdict(list)
    for p in pages:
        g = gk(p)
        for l in p['lines']:
            n = len(l['w'])
            for i, w in enumerate(l['w']):
                pc = 'F' if i == 0 else ('L' if i == n - 1 else 'M')
                POOLW[(g, l['ps'], pc)].append(w); GL[g].update(w); SECW[g].append(w)
                if i > 0: FOLW[(g, l['ps'], pc, l['w'][i - 1][-1])].append(w)
    types = sorted({w for ws in SECW.values() for w in ws}); tix = {w: i for i, w in enumerate(types)}; NT = len(types)
    pools = {}

    def pool(key, src):
        if key not in pools: pools[key] = _Pool2(src[key], k, tix) if src.get(key) else None
        return pools[key]
    GLc = {g: (list(c), np.cumsum([c[x] for x in c]) / sum(c.values())) for g, c in GL.items()}
    SECc = {}
    for g, ws in SECW.items():
        c = Counter(ws); SECc[g] = (list(c), np.array(list(c.values()), float))
    # onset moods (v82)
    if P['M'] > 1:
        if P['mood_src'] == 'fit':
            key = ('mood', k, P['M'])
            if moodcache is not None and key in moodcache: voc, prof = moodcache[key]
            else:
                voc, prof = K82.fit_moods(pages, k, P['M'], 82)
                if moodcache is not None: moodcache[key] = (voc, prof)
        else:
            ons = sorted({K82.front(w, k) for ws in SECW.values() for w in ws}); voc = {o: i for i, o in enumerate(ons)}
            prof = nrng.normal(0, 1, size=(P['M'], len(voc)))
        prof = prof * P['mood_beta']
    else:
        voc, prof = {}, None
    hr = random.Random(P['hseed']); hcl = {c: hr.randrange(P['H']) for c in K82.GLYPHS}
    # page-level body moods
    gm_on = P.get('gm_beta', 0) > 0; lx_on = P.get('lx_beta', 0) > 0
    if gm_on:
        fk = ('feat', P['gm_feat'])
        if moodcache is not None and fk in moodcache and moodcache[fk][0] == NT: F = moodcache[fk][1]
        else:
            F = _feats(types, P['gm_feat']); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
            if moodcache is not None: moodcache[fk] = (NT, F)
        gm = nrng.normal(0, 1, F.shape[1])
    if lx_on:
        if P['lx_K'] > 0: LXK = nrng.normal(0, 1, (P['lx_K'], NT))
        lx = nrng.normal(0, 1, NT)
    S = np.zeros(NT)

    def ar(x, rho):
        return rho * x + math.sqrt(max(0.0, 1 - rho * rho)) * nrng.normal(0, 1, x.shape)

    def recompute():
        s = np.zeros(NT)
        if gm_on: s += P['gm_beta'] * (F @ gm) / math.sqrt(F.shape[1]) * 2.0
        if lx_on: s += P['lx_beta'] * lx
        return s

    def agree_ok(o, prevw):
        if P['agr'] in ('none', 'pmi') or prevw is None: return 0.0
        pw = K82.front(prevw, 2)
        if len(pw) <= P['agr_src']: return 0.0
        src = pw[P['agr_src']]
        if P['agr'] == 'copy': return float(o[:1] == src)
        if P['agr'] == 'copy2': return float(o[:2] == pw[:2])
        return float(bool(o) and hcl.get(o[0], -2) == hcl.get(src, -1))

    def draw(pl, mood, prevw):
        lw = np.zeros(len(pl.ons))
        if prof is not None: lw += np.array([prof[mood, voc[o]] if o in voc else 0.0 for o in pl.ons])
        if P['agr'] not in ('none', 'pmi') and prevw is not None:
            lw += P['agr_lam'] * np.array([agree_ok(o, prevw) for o in pl.ons])
        wt = pl.cnt * np.exp(lw - lw.max()); wt /= wt.sum()
        j = min(int(np.searchsorted(np.cumsum(wt), rng.random())), len(wt) - 1)
        ids, wc = pl.ids[j], pl.wc[j]
        if gm_on or lx_on:
            x = S[ids]; ww = wc * np.exp(x - x.max())
        else:
            ww = wc
        c = np.cumsum(ww)
        return types[ids[min(int(np.searchsorted(c, rng.random() * c[-1])), len(ids) - 1)]]

    def mutate(w, g):
        ks, cp = GLc[g]
        j = rng.randrange(1, len(w)) if len(w) > 1 else 0          # vary the body, keep the onset
        c = ks[min(int(np.searchsorted(cp, rng.random())), len(ks) - 1)]
        return w[:j] + c + w[j + 1:]

    M = max(1, P['M']); out = []; mood = rng.randrange(M); bookhist = []
    for p in pages:
        g = gk(p); hist = []; prev = None
        if P['mood_scope'] != 'passage': mood = rng.randrange(M)
        if gm_on: gm = ar(gm, P['gm_rho'])
        if lx_on:
            lx = LXK[rng.randrange(P['lx_K'])].copy() if P['lx_K'] > 0 else ar(lx, P['lx_rho'])
        if gm_on or lx_on: S = recompute()
        seeds = []
        if P.get('sd_n', 0) > 0:
            ty, ct = SECc[g]; wt = ct ** P['sd_alpha']; c = np.cumsum(wt)
            seeds = [ty[min(int(np.searchsorted(c, rng.random() * c[-1])), len(ty) - 1)] for _ in range(P['sd_n'])]
        nl = []
        for li, l in enumerate(p['lines']):
            if li > 0 and (P.get('gm_line', 0) > 0 or P.get('lx_line', 0) > 0):
                if gm_on and P['gm_line'] > 0: gm = gm + P['gm_line'] * nrng.normal(0, 1, gm.shape)
                if lx_on and P['lx_line'] > 0 and P['lx_K'] == 0: lx = lx + P['lx_line'] * nrng.normal(0, 1, NT)
                if gm_on or lx_on: S = recompute()
            if P['mood_scope'] in ('line', 'passage') and rng.random() < P['mood_rate']: mood = rng.randrange(M)
            n = len(l['w']); ws = []
            for i in range(n):
                if P['mood_scope'] == 'word' and rng.random() < P['mood_rate']: mood = rng.randrange(M)
                pc = 'F' if i == 0 else ('L' if i == n - 1 else 'M')
                prevw = ws[-1] if ws else None
                u = rng.random(); w = None
                H = hist if P.get('cite_scope', 'page') == 'page' else bookhist
                if seeds and i > 0 and rng.random() < P['sd_p']:
                    j = rng.randrange(len(seeds)); w = seeds[j]
                    if rng.random() < P['sd_mod']:
                        w = mutate(w, g)
                        if rng.random() < P['sd_keep']: seeds[j] = w
                elif prev is not None and u < P['p_vert'] and prev:
                    j = min(i, len(prev) - 1) if pc != 'L' else len(prev) - 1
                    w = prev[j]
                    if rng.random() < P['p_mod']: w = mutate(w, g)
                elif len(H) >= 3 and u < P['p_vert'] + P['p_cite']:
                    w = rng.choice(H[-P.get('cite_win', 60):])
                    if rng.random() < P['p_mod']: w = mutate(w, g)
                if w is None:
                    pl = None
                    if P['base'] == 'selfcit':
                        pl = pool(g, SECW)
                    else:
                        if i > 0: pl = pool((g, l['ps'], pc, ws[-1][-1]), FOLW)
                        if pl is None: pl = pool((g, l['ps'], pc), POOLW) or pool((g, False, 'M'), POOLW) or pool((g, True, 'M'), POOLW)
                    w = draw(pl, mood, prevw if i > 0 else None)
                ws.append(w); hist.append(w); bookhist.append(w)
            nl.append(dict(l, w=ws)); prev = ws
        out.append(dict(p, lines=nl))
    return out
