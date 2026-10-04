"""v25 THE GLYPHS ARE BUILT FROM FEATURES.  Shared library.

Corpora are lists of words; a word is a tuple of glyph units.  Behaviour of a glyph = its contexts
(L1, R1, L2, R2 glyphs, word edges as '<' '>').  Three behaviour models:
  ppmi : positive PMI rows of glyph x context counts (cosine similarity)
  svd  : 8-dim SVD embedding of the PPMI matrix (cosine)
  potts: pseudolikelihood Potts / maximum-entropy model -- multinomial logistic regression of the
         centre glyph on one-hot L1 R1 L2 R2 contexts; a glyph's behaviour = its coupling row (cosine)
Shape similarity = weighted Jaccard of stroke-primitive count vectors (v25_shapes).
Main statistic = Mantel (Spearman over glyph pairs) between shape and behaviour similarity;
null = random glyph-to-shape assignment (permutation of glyph labels on the shape matrix).
"""
import os, sys, json, re, random, unicodedata
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v18_lib import glyphs as vglyphs
import v25_shapes as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'v25_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = os.environ.get('V25_SCRATCH', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad')
TOK = 120000  # glyph tokens per corpus (matched)


# ------------------------------------------------------------------ corpora
def voynich(src='ZL3b', lang=None, fold=None):
    d = json.load(open(os.path.join(DATA, 'derived', f'{src}_lines.json')))
    keep = set(S.VOYNICH)
    out = []
    for L in d:
        if L['ltype'] != 'P':
            continue
        if lang and L['lang'] != lang:
            continue
        if fold is not None:
            m = re.match(r'f(\d+)', L['folio']);
            if int(m.group(1)) % 2 != fold:
                continue
        for w, u in zip(L['words'], L['uncertain']):
            if u or '?' in w:
                continue
            g = tuple(vglyphs(w))
            if g and all(x in keep for x in g):
                out.append(g)
    return out


def _truncate(words, tok):
    out, n = [], 0
    for w in words:
        out.append(w); n += len(w)
        if n >= tok:
            break
    return out


def latin(tok=TOK):
    t = open(os.path.join(DATA, 'plain', 'la.txt'), encoding='utf-8').read().lower()
    t = t.replace('j', 'i').replace('v', 'u')
    keep = set(S.LATIN)
    ws = [tuple(w) for w in re.findall(r'[a-z]+', t) if all(c in keep for c in w)]
    return _truncate(ws, tok)


def greek(tok=TOK):
    t = ''
    for i in (27760, 28658, 39536):
        f = os.path.join(SCR, f'gr_{i}.txt')
        x = open(f, encoding='utf-8').read()
        a = x.find('*** START'); b = x.find('*** END')
        t += x[a:b] + '\n'
    t = unicodedata.normalize('NFD', t.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    keep = set(S.GREEK)
    ws = [tuple(w) for w in re.findall(r'[α-ως]+', t) if all(c in keep for c in w)]
    return _truncate(ws, tok)


_CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
_JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
_JONG = ['', 'ㄱ', 'ㄲ', 'ㄱㅅ', 'ㄴ', 'ㄴㅈ', 'ㄴㅎ', 'ㄷ', 'ㄹ', 'ㄹㄱ', 'ㄹㅁ', 'ㄹㅂ', 'ㄹㅅ', 'ㄹㅌ', 'ㄹㅍ', 'ㄹㅎ', 'ㅁ', 'ㅂ', 'ㅂㅅ',
         'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def hangul_decomp(syl):
    o = ord(syl) - 0xAC00
    c, r = divmod(o, 588); j, f = divmod(r, 28)
    return [_CHO[c], _JUNG[j]] + list(_JONG[f])


def hangul(tok=TOK):
    out = []
    for line in open(os.path.join(SCR, 'ko_nsmc.txt'), encoding='utf-8').read().split('\n')[1:]:
        p = line.split('\t')
        if len(p) < 2:
            continue
        for w in p[1].split():
            if w and all(0xAC00 <= ord(c) <= 0xD7A3 for c in w):
                out.append(tuple(j for c in w for j in hangul_decomp(c)))
    return _truncate(out, tok)


def corpus(name, tok=TOK):
    if name == 'voynich':
        return _truncate(voynich('ZL3b'), tok)
    if name == 'voynich_it':
        return _truncate(voynich('IT2a'), tok)
    return dict(latin=latin, greek=greek, hangul=hangul)[name](tok)


def halves(words, block=400):
    a, b = [], []
    for i in range(0, len(words), block):
        (a if (i // block) % 2 == 0 else b).extend(words[i:i + block])
    return a, b


# ------------------------------------------------------------------ behaviour
def alphabet(words, shapes, minc=20):
    c = Counter(g for w in words for g in w)
    return [g for g in shapes if c[g] >= minc]


def contexts(words, alph):
    idx = {g: i for i, g in enumerate(alph)}
    ctx = alph + ['<', '>']
    cidx = {g: i for i, g in enumerate(ctx)}
    nC = len(ctx)
    rows, cols = [], []
    for w in words:
        ww = ['<', '<'] + list(w) + ['>', '>']
        for k in range(2, len(ww) - 2):
            g = ww[k]
            if g not in idx:
                continue
            feats = []
            for off, slot in ((-1, 0), (1, 1), (-2, 2), (2, 3)):
                x = ww[k + off]
                if x in cidx:
                    feats.append(slot * nC + cidx[x])
            rows.append((idx[g], feats))
    C = np.zeros((len(alph), 4 * nC))
    for g, fs in rows:
        for f in fs:
            C[g, f] += 1
    return C, rows, 4 * nC


def ppmi(C):
    tot = C.sum(); r = C.sum(1, keepdims=True); c = C.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.log((C * tot) / (r * c))
    p[~np.isfinite(p)] = 0
    return np.maximum(p, 0)


def cos_sim(X):
    X = X - 0  # no centering
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    Y = X / n
    return Y @ Y.T


def potts(rows, nG, nF, C=0.5, maxn=120000, seed=0):
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import csr_matrix
    rng = random.Random(seed)
    rr = rows if len(rows) <= maxn else rng.sample(rows, maxn)
    y = np.array([g for g, _ in rr])
    ind, ptr = [], [0]
    for _, fs in rr:
        ind.extend(fs); ptr.append(len(ind))
    X = csr_matrix((np.ones(len(ind)), ind, ptr), shape=(len(rr), nF))
    m = LogisticRegression(C=C, max_iter=400)
    m.fit(X, y)
    W = np.zeros((nG, nF)); W[m.classes_] = m.coef_
    return W


def behaviour(words, alph, models=('ppmi', 'svd', 'potts')):
    C, rows, nF = contexts(words, alph)
    out = {}
    P = ppmi(C)
    if 'ppmi' in models:
        out['ppmi'] = cos_sim(P)
    if 'svd' in models or 'emb' in models:
        U, s, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
        E = U[:, :8] * s[:8]
        out['svd'] = cos_sim(E); out['_emb'] = E
    if 'potts' in models:
        W = potts(rows, len(alph), nF)
        out['potts'] = cos_sim(W - W.mean(0)); out['_W'] = W
    out['_freq'] = C.sum(1) / 4.0
    return out


# ------------------------------------------------------------------ shapes
def shape_matrix(shapes, alph, prims=None):
    if prims is None:
        prims = sorted({p for g in alph for p in shapes[g]})
    return np.array([[shapes[g].get(p, 0) for p in prims] for g in alph], float), prims


def jaccard_sim(F):
    n = len(F)
    S_ = np.zeros((n, n))
    for i in range(n):
        mn = np.minimum(F[i], F).sum(1); mx = np.maximum(F[i], F).sum(1)
        S_[i] = np.where(mx > 0, mn / np.maximum(mx, 1e-9), 0)
    return S_


def l1(F):
    return np.abs(F[:, None, :] - F[None, :, :]).sum(2)


# ------------------------------------------------------------------ Mantel
def triu(M):
    i, j = np.triu_indices(len(M), 1)
    return M[i, j]


def spearman_vec(a, b):
    ra = rankdata(a); rb = rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def mantel(Ssh, B, nperm=2000, rng=None, strata=None):
    """Spearman(shape sim, behaviour sim) over pairs; null permutes glyph labels of the shape matrix
    (optionally within strata).  Returns r, p (one-sided, greater), z."""
    rng = rng or np.random.default_rng(0)
    n = len(B)
    iu = np.triu_indices(n, 1)
    rb = rankdata(B[iu]); rb -= rb.mean(); nb = np.sqrt((rb ** 2).sum())

    def stat(perm):
        a = rankdata(Ssh[perm][:, perm][iu]); a -= a.mean()
        d = np.sqrt((a ** 2).sum()) * nb
        return (a * rb).sum() / d if d > 0 else 0.0
    r0 = stat(np.arange(n))
    null = np.empty(nperm)
    for k in range(nperm):
        if strata is None:
            perm = rng.permutation(n)
        else:
            perm = np.arange(n)
            for s in set(strata):
                ix = np.where(np.array(strata) == s)[0]
                perm[ix] = rng.permutation(ix)
        null[k] = stat(perm)
    p = (1 + (null >= r0).sum()) / (nperm + 1)
    z = (r0 - null.mean()) / (null.std() + 1e-12)
    return float(r0), float(p), float(z), null


def partial_spearman(x, y, covs):
    """Spearman partial correlation of x and y given covariates (rank-residualised)."""
    R = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in covs])
    def res(v):
        v = rankdata(v); b, *_ = np.linalg.lstsq(R, v, rcond=None); return v - R @ b
    a, b = res(x), res(y)
    return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))


def freq_strata(freq, k=3):
    q = np.quantile(np.log(freq), np.linspace(0, 1, k + 1)[1:-1])
    return list(np.searchsorted(q, np.log(freq)))


# ------------------------------------------------------------------ swaps (Hamming-1 neighbours at lag 1)
def swap_counts(words, block=None):
    c = Counter()
    for a, b in zip(words[:-1], words[1:]):
        if len(a) != len(b) or a == b:
            continue
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        if len(diff) == 1:
            c[tuple(sorted(diff[0]))] += 1
    return c


def swap_excess(lines, nshuf=50, seed=0):
    """lines = list of lists of words; within-line shuffle null."""
    rng = random.Random(seed)
    def cnt(ls):
        c = Counter()
        for L in ls:
            c.update(swap_counts(L))
        return c
    obs = cnt(lines)
    sims = []
    for _ in range(nshuf):
        sh = []
        for L in lines:
            L2 = L[:]; rng.shuffle(L2); sh.append(L2)
        sims.append(cnt(sh))
    keys = set(obs) | {k for s in sims for k in s}
    out = {}
    for k in keys:
        v = np.array([s.get(k, 0) for s in sims], float)
        out[k] = (obs.get(k, 0), v.mean(), (obs.get(k, 0) - v.mean()) / np.sqrt(v.var() + v.mean() + 1))
    return out


def voynich_lines(src='ZL3b'):
    d = json.load(open(os.path.join(DATA, 'derived', f'{src}_lines.json')))
    keep = set(S.VOYNICH); out = []
    for L in d:
        if L['ltype'] != 'P':
            continue
        ws = []
        for w, u in zip(L['words'], L['uncertain']):
            g = tuple(vglyphs(w))
            if u or '?' in w or not all(x in keep for x in g):
                ws.append(None)
            else:
                ws.append(g)
        out.append(ws)
    return out


def pseudo_lines(words, n=10):
    return [words[i:i + n] for i in range(0, len(words), n)]


# ------------------------------------------------------------------ analogies
def analogy_score(E, alph, fam):
    idx = {g: i for i, g in enumerate(alph)}
    pr = [(a, b) for a, b in fam if a in idx and b in idx]
    if len(pr) < 2:
        return None
    O = np.array([E[idx[b]] - E[idx[a]] for a, b in pr])
    O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-12)
    G = O @ O.T
    return float(G[np.triu_indices(len(O), 1)].mean())


def analogy_test(E, alph, fam, nperm=5000, rng=None):
    rng = rng or np.random.default_rng(1)
    s0 = analogy_score(E, alph, fam)
    if s0 is None:
        return None
    null = []
    for _ in range(nperm):
        perm = dict(zip(alph, rng.permutation(alph)))
        null.append(analogy_score(E, alph, [(perm[a], perm[b]) for a, b in fam if a in perm and b in perm]))
    null = np.array(null)
    return s0, float((1 + (null >= s0).sum()) / (nperm + 1)), float(null.mean())


def write_rows(path, rows, header=None):
    with open(path, 'w') as f:
        if header:
            f.write(header.rstrip() + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(x) for x in r) + ' |\n')
