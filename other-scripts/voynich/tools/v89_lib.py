"""v89: find the source text by its word-RECURRENCE structure (encoding-invariant), not by spelling or length.

A text = sequence of units (entries / pages / paragraphs), each a bag of word types.
Feature = residual sharing matrix: cosine of binary type sets between units, double-centred and with the
mean of each lag removed (so generic topic locality and long/short units cancel). What is left is WHICH
non-adjacent units share vocabulary -- a property a word-level encoding (or a translation) keeps.

Alignment (random): a Voynich section of N units is laid onto a span of a candidate source's token stream,
cut into N pieces (modes: length-proportional cuts, the same snapped to source entry boundaries, random cuts).
Score = Pearson between the Voynich residual matrix and the source-piece residual matrix over pairs of units
in one half (odd units = A, selection) ; held-out = the same alignment scored on even-unit pairs (B).
Null = the same full search against rotated / reversed Voynich unit orders (keeps block structure).
"""
import os, sys, json, math, re
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.dirname(HERE)
CK = os.path.join(VD, 'data', 'v89_ckpt')
sys.path.insert(0, HERE)


def fnum(f):
    m = re.match(r'f(\d+)([rv])(\d*)', f)
    return (int(m.group(1)), 0 if m.group(2) == 'r' else 1, int(m.group(3) or 0)) if m else (9999, 0, 0)


# ---------------------------------------------------------------- Voynich units
def voynich_paras(name='ZL3b'):
    import vlib
    os.environ.setdefault('VOY_MODE', 'glyph')
    recs = vlib.load_voynich(name)
    paras, cur, fol = [], None, None
    for r in recs:
        if r['para_start'] or r['folio'] != fol or cur is None:
            if cur: paras.append(cur)
            cur = {'folio': r['folio'], 'illus': r['illus'], 'lang': r['lang'], 'quire': r['quire'], 'tok': []}
        fol = r['folio']; cur['tok'] += r['words']
        if r.get('para_end'):
            paras.append(cur); cur = None
    if cur: paras.append(cur)
    paras = [p for p in paras if p['tok']]
    paras.sort(key=lambda p: fnum(p['folio']))
    return paras


def voynich_sections(name='ZL3b'):
    P = voynich_paras(name)
    def pages(ps):
        d, order = defaultdict(list), []
        for p in ps:
            if p['folio'] not in d: order.append(p['folio'])
            d[p['folio']] += p['tok']
        return [{'t': f, 'tok': d[f]} for f in order]
    S = {}
    S['herbalA_pages'] = pages([p for p in P if p['illus'] == 'H' and p['lang'] == 'A'])
    S['herbalB_pages'] = pages([p for p in P if p['illus'] == 'H' and p['lang'] == 'B'])
    S['pharma_paras'] = [{'t': p['folio'], 'tok': p['tok']} for p in P if p['illus'] == 'P']
    S['bio_paras'] = [{'t': p['folio'], 'tok': p['tok']} for p in P if p['illus'] == 'B']
    S['stars_paras'] = [{'t': p['folio'], 'tok': p['tok']} for p in P if p['illus'] == 'S']
    S['bio_pages'] = pages([p for p in P if p['illus'] == 'B'])
    S['stars_pages'] = pages([p for p in P if p['illus'] == 'S'])
    return S


def view(tok, v):
    if v == 'w': return tok
    if v == 'nf': return [t[:-1] if len(t) > 2 else t for t in tok]
    if v == 'p4': return [t[:4] for t in tok]
    raise ValueError(v)


# ---------------------------------------------------------------- residual sharing
def resid_from_sets(X):
    """X: binary csr (n x V). Returns residual cosine matrix (diag nan)."""
    S = (X @ X.T).toarray().astype(float)
    k = np.diag(S).copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        C = S / np.sqrt(np.outer(k, k))
    C[~np.isfinite(C)] = 0.0
    return resid(C, k)


LOWRANK = int(os.environ.get('V89_LOWRANK', '3'))
_LAG = {}
def _lagmat(n):
    if n not in _LAG:
        i = np.arange(n); _LAG[n] = np.abs(i[:, None] - i[None, :])
    return _LAG[n]


def resid(C, k=None):
    n = C.shape[0]
    C = C.copy()
    np.fill_diagonal(C, np.nan)
    rm = np.nanmean(C, axis=1)
    gm = np.nanmean(C)
    C = C - rm[:, None] - rm[None, :] + gm
    # remove lag means (vectorised)
    D = _lagmat(n)
    Cz = np.where(np.isnan(C), 0.0, C)
    ok = ~np.isnan(C)
    sm = np.bincount(D.ravel(), Cz.ravel(), minlength=n)
    ct = np.bincount(D.ravel(), ok.ravel().astype(float), minlength=n)
    m = sm / np.maximum(ct, 1)
    C = C - m[D]
    if k is not None:   # regress out unit-size structure (log size sum, product, squared difference)
        lk = np.log1p(k); lk = lk - lk.mean()
        I, J = np.triu_indices(n, 1)
        F = np.stack([np.ones(len(I)), lk[I] + lk[J], lk[I] * lk[J], (lk[I] - lk[J]) ** 2, np.abs(lk[I] - lk[J])], 1)
        y = C[I, J]
        beta, *_ = np.linalg.lstsq(F, y, rcond=None)
        e = y - F @ beta
        C = np.full((n, n), np.nan); C[I, J] = e; C[J, I] = e
    if LOWRANK:   # strip the strongest r components (book / block structure shared by any two long texts)
        Z = np.nan_to_num(C); w, V = np.linalg.eigh(Z)
        idx = np.argsort(-np.abs(w))[:LOWRANK]
        C = Z - (V[:, idx] * w[idx]) @ V[:, idx].T
    np.fill_diagonal(C, np.nan)
    return C


def sets_matrix(units_tok, vocab=None, drop_top=0, min_df=2):
    """list of token lists -> binary csr over types with df>=min_df, excluding the drop_top most frequent."""
    cnt = Counter(t for u in units_tok for t in u)
    df = Counter(t for u in units_tok for t in set(u))
    top = set(w for w, _ in cnt.most_common(drop_top))
    keep = sorted(w for w in df if df[w] >= min_df and w not in top)
    ix = {w: i for i, w in enumerate(keep)}
    rows, cols = [], []
    for r, u in enumerate(units_tok):
        for w in set(u):
            j = ix.get(w)
            if j is not None: rows.append(r); cols.append(j)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(units_tok), max(1, len(keep))))
    return X


def pair_index(n, half):
    """half 0 = even positions, 1 = odd positions; pairs i<j both in half."""
    ids = np.arange(half, n, 2)
    I, J = np.triu_indices(len(ids), 1)
    return ids[I], ids[J]


def zvec(R, P):
    v = R[P[0], P[1]]
    v = v - v.mean(); s = v.std()
    return v / s if s > 0 else v


# ---------------------------------------------------------------- target with null variants
class Target:
    def __init__(self, units_tok, n_null=20, seed=0, drop_top=0, null_kind='rot'):
        self.n = n = len(units_tok)
        self.lens = np.array([len(u) for u in units_tok], float)
        rng = np.random.default_rng(seed)
        perms = [np.arange(n)]
        for k in range(n_null):
            if null_kind == 'rot':
                s = rng.integers(n // 10, n - n // 10)
                p = np.roll(np.arange(n), s)
                if k % 2: p = p[::-1]
            else:
                p = rng.permutation(n)
            perms.append(p)
        self.perms = perms
        X = sets_matrix(units_tok, drop_top=drop_top)
        self.PA = pair_index(n, 1); self.PB = pair_index(n, 0)
        ZA, ZB = [], []
        for p in perms:
            R = resid_from_sets(X[p])
            ZA.append(zvec(R, self.PA)); ZB.append(zvec(R, self.PB))
        self.ZA = np.array(ZA); self.ZB = np.array(ZB)
        # length profile of each variant (for proportional cuts the alignment uses the variant's own order)
        self.vlens = [self.lens[p] for p in perms]


# ---------------------------------------------------------------- source
class Source:
    DROPS = (0, 30, 100)

    def __init__(self, units):
        toks = [u['tok'] for u in units]
        self.M = len(toks)
        bounds = np.cumsum([0] + [len(t) for t in toks])
        self.bounds = bounds
        self.T = int(bounds[-1])
        cnt = Counter(t for u in toks for t in u)
        order = [w for w, _ in cnt.most_common()]
        rank = {w: i for i, w in enumerate(order)}
        stream = np.array([rank[t] for u in toks for t in u], dtype=np.int64)
        self.V = len(order)
        self.kept = {}
        for d in self.DROPS:
            pos = np.nonzero(stream >= d)[0]
            self.kept[d] = (pos, stream[pos])

    def pieces(self, cuts, drop):
        """cuts: N+1 increasing token positions -> binary csr (N x V)."""
        pos, ids = self.kept[drop]
        a = np.searchsorted(pos, cuts)
        n = len(cuts) - 1
        cnts = np.diff(a)
        rows = np.repeat(np.arange(n, dtype=np.int64), cnts)
        cols = ids[a[0]:a[-1]]
        key = np.sort(rows * self.V + cols)
        if len(key): key = key[np.r_[True, key[1:] != key[:-1]]]
        r = key // self.V; c = key % self.V
        indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=n))])
        X = sp.csr_matrix((np.ones(len(c)), c, indptr), shape=(n, self.V))
        X.has_sorted_indices = True
        return X


def draw_align(rng, src, n, base=None, step=None):
    """alignment params dict. base+step -> local perturbation."""
    if base is None:
        spanU = float(np.exp(rng.uniform(np.log(max(1.0, n / 5)), np.log(n * 3))))
        spanU = min(spanU, src.M * 0.98)
        off = float(rng.uniform(0, src.M - spanU))
        a = {'off': off, 'span': spanU, 'mode': ['prop', 'snap', 'rand', 'unif', 'unif'][rng.integers(5)],
             'jit': float(rng.uniform(0, 0.5)), 'drop': int(rng.choice(Source.DROPS)), 'seed': int(rng.integers(1 << 30))}
    else:
        a = dict(base)
        a['span'] = float(min(src.M * 0.98, max(n / 6, a['span'] * np.exp(rng.normal(0, step)))))
        a['off'] = float(np.clip(a['off'] + rng.normal(0, step * a['span'] / 4), 0, src.M - a['span']))
        if rng.random() < 0.3: a['jit'] = float(np.clip(a['jit'] + rng.normal(0, 0.1), 0, 0.6))
        if rng.random() < 0.2: a['seed'] = int(rng.integers(1 << 30))
    return a


def unit_to_tok(src, u):
    i = np.minimum(np.floor(u).astype(int), src.M - 1); f = u - i
    return src.bounds[i] + f * (src.bounds[i + 1] - src.bounds[i])


def make_cuts(src, a, lens):
    n = len(lens)
    t0 = unit_to_tok(src, a['off']); t1 = unit_to_tok(src, min(src.M - 1e-6, a['off'] + a['span']))
    rng = np.random.default_rng(a['seed'])
    if a['mode'] == 'rand':
        inner = np.sort(rng.uniform(t0, t1, n - 1))
    elif a['mode'] == 'unif':
        us = a['off'] + np.arange(1, n) * a['span'] / n + rng.normal(0, a['jit'] * 0.2, n - 1)
        us = np.clip(np.sort(us), a['off'], a['off'] + a['span'])
        inner = unit_to_tok(src, us)
    else:
        cl = np.cumsum(lens)[:-1] / lens.sum()
        mean = (t1 - t0) / n
        inner = t0 + cl * (t1 - t0) + rng.normal(0, a['jit'] * mean, n - 1)
        inner = np.clip(np.sort(inner), t0, t1)
        if a['mode'] == 'snap':
            b = src.bounds
            j = np.searchsorted(b, inner)
            j = np.clip(j, 1, len(b) - 1)
            left, right = b[j - 1], b[j]
            inner = np.where(inner - left < right - inner, left, right).astype(float)
            inner = np.clip(inner, t0, t1)
    return np.concatenate([[t0], inner, [t1]]).astype(np.int64)


def score_align(src, tgt, a, variants=None):
    """returns (scoreA[v], scoreB[v]) for all target variants (each with its own length profile)."""
    V = range(len(tgt.perms)) if variants is None else variants
    sa, sb = [], []
    cache = {}
    for v in V:
        key = 0 if a['mode'] in ('rand', 'unif') else v   # random cuts ignore the length profile
        if key not in cache:
            cuts = make_cuts(src, a, tgt.vlens[v])
            R = resid_from_sets(src.pieces(cuts, a['drop']))
            cache[key] = (zvec(R, tgt.PA), zvec(R, tgt.PB))
        za, zb = cache[key]
        sa.append(float(za @ tgt.ZA[v]) / len(za)); sb.append(float(zb @ tgt.ZB[v]) / len(zb))
    return np.array(sa), np.array(sb)


def search(src, tgt, n_rand=1500, top=5, refine=30, seed=0):
    """random search + local refinement, run separately for every target variant (real = 0, nulls 1..K).
    Returns per variant: best A score and its held-out B score."""
    rng = np.random.default_rng(seed)
    nv = len(tgt.perms)
    A = []
    SA = np.zeros((n_rand, nv)); SB = np.zeros((n_rand, nv))
    for i in range(n_rand):
        a = draw_align(rng, src, tgt.n)
        A.append(a)
        SA[i], SB[i] = score_align(src, tgt, a)
    out = []
    for v in range(nv):
        best = []
        for i in np.argsort(-SA[:, v])[:top]:
            a, sa, sb = A[i], SA[i, v], SB[i, v]
            for r in range(refine):
                b = draw_align(rng, src, tgt.n, base=a, step=0.15 if r < refine // 2 else 0.05)
                x, y = score_align(src, tgt, b, variants=[v])
                if x[0] > sa: a, sa, sb = b, x[0], y[0]
            best.append((sa, sb, a))
        best.sort(key=lambda z: -z[0])
        out.append({'A': best[0][0], 'B': best[0][1], 'a': best[0][2], 'Btop': float(np.mean([z[1] for z in best]))})
    return out


def summarize(res):
    """res = search() output -> real vs null stats."""
    realB = res[0]['B']; nullB = np.array([r['B'] for r in res[1:]])
    realA = res[0]['A']; nullA = np.array([r['A'] for r in res[1:]])
    zB = (realB - nullB.mean()) / (nullB.std() + 1e-9)
    zA = (realA - nullA.mean()) / (nullA.std() + 1e-9)
    return {'A': realA, 'B': realB, 'zA': float(zA), 'zB': float(zB), 'nullB_mu': float(nullB.mean()), 'nullB_sd': float(nullB.std()),
            'Btop': res[0]['Btop'], 'a': res[0]['a']}


def load_texts():
    return json.load(open(os.path.join(CK, 'texts.json')))


def load_extra():
    """Proverbs and Ben Sira (Sefaria) he/en pairs, chapter units."""
    out = {}
    WRE = re.compile(r"[^\W\d_]+", re.UNICODE)
    import unicodedata
    for b in ('Proverbs', 'Ecclesiasticus'):
        for v, lg in (('hebrew', 'he'), ('english', 'en')):
            f = os.path.join(CK, 'src', '%s_%s.json' % (b, v))
            if not os.path.exists(f): continue
            d = json.load(open(f))['versions'][0]['text']
            us = []
            for i, ch in enumerate(d):
                txt = ' '.join(x if isinstance(x, str) else ' '.join(x) for x in ch)
                txt = re.sub(r'<[^>]+>', ' ', txt).replace('־', ' ')
                txt = ''.join(c for c in txt if unicodedata.category(c) != 'Mn')
                tok = [w.lower() for w in WRE.findall(txt)]
                if tok: us.append({'t': 'ch%d' % (i + 1), 'tok': tok})
            out['%s_%s' % (b.lower()[:8], lg)] = {'lang': lg, 'genre': 'wisdom', 'units': us}
    return out


def all_texts():
    T = load_texts()
    T.update(load_extra())
    return T


# ---------------------------------------------------------------- Voynich-like encoding of a known text
def encode_text(units_tok, rng, homo=3, split=0.3, filler=0.2, drop=0.05):
    """word-level opaque code: each plaintext type -> 1..homo codewords (token picks one at random),
    a fraction of types written as two codewords, filler words inserted, some tokens dropped."""
    types = sorted(set(t for u in units_tok for t in u))
    code = {}
    nid = [0]
    def new():
        nid[0] += 1; return 'c%d' % nid[0]
    for t in types:
        k = int(rng.integers(1, homo + 1))
        if rng.random() < split:
            code[t] = [(new(), new()) for _ in range(k)]
        else:
            code[t] = [(new(),) for _ in range(k)]
    fill = ['f%d' % i for i in range(40)]
    out = []
    for u in units_tok:
        o = []
        for t in u:
            if rng.random() < drop: continue
            o += list(code[t][rng.integers(len(code[t]))])
            if rng.random() < filler: o.append(fill[int(rng.zipf(1.5)) % 40])
        out.append(o)
    return out
