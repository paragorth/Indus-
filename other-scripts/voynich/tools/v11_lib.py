"""v11 shared helpers: 'the text is a walk through a hidden map'.

Corpora: every corpus is poured into the exact ZL3b paragraph-line layout (same folios, line lengths),
so the page-level train/test split is identical across corpora.
  pages(): list of lines {folio, fi (folio index), lang, hand, para_start, words}
Models (numpy, full batch, Adam) on within-line bigram counts restricted to a top-V vocabulary:
  uni      logit_ab = beta_b + g*[a==b]
  dist d   logit_ab = -|x_a - x_b|^2 + beta_b + g*[a==b]       (a walk on a d-dim map, popularity beta)
  bilin d  logit_ab = u_a . v_b + beta_b + g*[a==b]            (non-geometric, asymmetric low-rank)
"""
import os, sys, re, math, random, json
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, gen

CK = os.path.join(vlib.DATA, 'results', 'v11'); os.makedirs(CK, exist_ok=True)
OUT = os.path.join(vlib.ROOT, 'loops')


def fnum(f):
    m = re.match(r'f(\d+)([rv])(\d*)', f)
    return (int(m.group(1)), 0 if m.group(2) == 'r' else 1, int(m.group(3) or 0)) if m else (9999, 0, 0)


def voy(name='ZL3b'):
    L = vlib.load_voynich(name, drop_uncertain=True)
    fols = sorted({l['folio'] for l in L}, key=fnum); fi = {f: i for i, f in enumerate(fols)}
    out = []
    for l in L:
        out.append({'folio': l['folio'], 'fi': fi[l['folio']], 'lang': l.get('lang'), 'hand': l.get('hand'),
                    'illus': l.get('illus'), 'para_start': l['para_start'], 'words': list(l['words'])})
    return out


TEMPLATE = None
def template():
    global TEMPLATE
    if TEMPLATE is None: TEMPLATE = voy('ZL3b')
    return TEMPLATE


def pour(words, tpl=None, offset=0):
    tpl = tpl or template(); out, k = [], offset
    for l in tpl:
        q = dict(l); q['words'] = words[k:k + len(l['words'])]; k += len(l['words']); out.append(q)
    return out


def latin_words():
    t = open(os.path.join(vlib.DATA, 'plain', 'la.txt'), encoding='utf-8', errors='replace').read().lower()
    t = t.replace('æ', 'ae').replace('œ', 'oe').replace('j', 'i').replace('v', 'u')
    return re.findall(r'[a-z]+', t)


def ref_words(key):
    return [w for l in vlib.load_ref(key, skip_frac=0.05) for w in l['words']]


def ntok():
    return sum(len(l['words']) for l in template())


def planted(kind='grid', eps=0.3, side=30, seed=1, start_region=False):
    """Random walk on a hidden map. Cells carry Voynich word types (top side^2), popularity = their Voynich
    frequency (placed at random). kind 'grid': neighbourhood = Chebyshev radius 2 (24 cells + self);
    kind 'graph': 24 fixed random cells + self (same degree, no geometry). With prob eps a token is noise
    drawn from the full Voynich unigram. start_region: lines start in the 6x6 corner block chosen by the
    first glyph cycle (control for cycle 3)."""
    rng = random.Random(seed)
    tpl = template(); vw = [w for l in tpl for w in l['words']]; c = Counter(vw)
    types = [w for w, _ in c.most_common(side * side)]; rng.shuffle(types)
    W = [c[t] for t in types]; n = side * side
    coord = [(i // side, i % side) for i in range(n)]
    if kind == 'grid':
        nb = [[j for j in range(n) if max(abs(coord[i][0] - coord[j][0]), abs(coord[i][1] - coord[j][1])) <= 2]
              for i in range(n)]
    else:
        nb = [[i] + rng.sample([j for j in range(n) if j != i], 24) for i in range(n)]
    nbw = [[W[j] for j in nb[i]] for i in range(n)]
    allw = list(c.keys()); allc = [c[w] for w in allw]
    out = []
    for l in tpl:
        cur = rng.choices(range(n), W)[0]
        ws = []
        for k in range(len(l['words'])):
            if k > 0:
                cur = rng.choices(nb[cur], nbw[cur])[0]
            ws.append(types[cur] if rng.random() >= eps else rng.choices(allw, allc)[0])
        q = dict(l); q['words'] = ws; out.append(q)
    truth = {types[i]: coord[i] for i in range(n)}
    return out, truth


def corpus(name, seed=1):
    """name -> (lines, truth or None)."""
    if name == 'Voynich-ZL': return template(), None
    if name == 'Voynich-IT': return voy('IT2a'), None
    if name == 'Voynich-A': return [l for l in template() if l['lang'] == 'A'], None
    if name == 'Voynich-B': return [l for l in template() if l['lang'] == 'B'], None
    if name == 'Latin-Isidore':
        w = latin_words(); return pour(w, offset=len(w) // 10), None
    if name in ('Italian-Manzoni', 'Spanish-Cervantes', 'Italian-Dante'):
        w = ref_words(name); assert len(w) >= ntok(), (name, len(w)); return pour(w), None
    if name == 'Shuffle-line': return gen.within_line_shuffle(template(), seed), None
    if name == 'Shuffle-global': return gen.word_shuffle(template(), seed), None
    if name == 'SelfCitation': return gen.self_citation(template(), seed), None
    if name == 'Latin-shufline': return gen.within_line_shuffle(corpus('Latin-Isidore')[0], seed), None
    m = re.match(r'Planted-(grid|graph)-(\d+)', name)
    if m: return planted(m.group(1), int(m.group(2)) / 100, seed=seed)
    raise KeyError(name)


# ---------------- counts ----------------
def vocab(lines, V=400):
    c = Counter(w for l in lines for w in l['words'])
    return [w for w, _ in c.most_common(V)]


def bigram_counts(lines, idx, lag=1):
    V = len(idx); C = np.zeros((V, V))
    for l in lines:
        ws = l['words']
        for a, b in zip(ws, ws[lag:]):
            if a in idx and b in idx: C[idx[a], idx[b]] += 1
    return C


def split(lines, fold):
    """page-level 2-fold split (alternate folios)."""
    tr = [l for l in lines if l['fi'] % 2 != fold]; te = [l for l in lines if l['fi'] % 2 == fold]
    return tr, te


# ---------------- models ----------------
def _softmax_rows(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def ll(logits, C):
    Z = logits - logits.max(1, keepdims=True)
    lp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
    return float((C * lp).sum())


def fit(C, kind='dist', d=2, iters=1500, lr=0.05, lam=0.1, seed=0, init_scale=None, init=None, mix=True):
    """P(b|a) = (1-rho) softmax_b(logit_ab) + rho u_b  (u = train successor unigram; rho = teleport rate).
    Loss = -LL/N + lam*|embedding params|^2/V. Returns dict with probs (V x V), rho, params."""
    rng = np.random.default_rng(seed); V = C.shape[0]; N = C.sum()
    u = C.sum(0) + 0.5; u = u / u.sum()
    beta = np.log(u); beta -= beta.mean(); g = np.zeros(1); th = np.array([-1.0 if mix else -30.0])
    if kind == 'dist':
        X = rng.normal(0, init_scale or 1.0, (V, d)) if init is None else init.copy(); P = [X, beta, g, th]
    elif kind == 'bilin':
        U = rng.normal(0, 0.1, (V, d)); W = rng.normal(0, 0.1, (V, d)); P = [U, W, beta, g, th]
    else:
        P = [beta, g, th]
    m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
    b1, b2, eps = 0.9, 0.999, 1e-8
    I = np.eye(V)
    def logits(P):
        if kind == 'dist':
            X = P[0]; sq = (X * X).sum(1); D = sq[:, None] + sq[None, :] - 2 * X @ X.T
            return -D + P[1][None, :] + P[2][0] * I
        if kind == 'bilin':
            return P[0] @ P[1].T + P[2][None, :] + P[3][0] * I
        return np.repeat(P[0][None, :], V, 0) + P[1][0] * I
    def probs(P):
        S = _softmax_rows(logits(P)); rho = 1 / (1 + np.exp(-P[-1][0]))
        return S, rho, (1 - rho) * S + rho * u[None, :]
    for t in range(1, iters + 1):
        S, rho, Pr = probs(P)
        H = C / Pr                                    # dLL/dPr
        HS = (1 - rho) * H
        G = -(S * (HS - (S * HS).sum(1, keepdims=True))) / N      # d(-LL/N)/dlogit
        grho = -((H * (u[None, :] - S)).sum() / N) * rho * (1 - rho) if mix else 0.0
        if kind == 'dist':
            X = P[0]
            gX = 2 * (G @ X - G.sum(1)[:, None] * X) + 2 * (G.T @ X - G.sum(0)[:, None] * X)
            grads = [gX + 2 * lam * X / V, G.sum(0), np.array([np.trace(G)]), np.array([grho])]
        elif kind == 'bilin':
            U, W = P[0], P[1]
            grads = [G @ W + 2 * lam * U / V, G.T @ U + 2 * lam * W / V, G.sum(0), np.array([np.trace(G)]), np.array([grho])]
        else:
            grads = [G.sum(0), np.array([np.trace(G)]), np.array([grho])]
        for i, (p, gr) in enumerate(zip(P, grads)):
            m[i] = b1 * m[i] + (1 - b1) * gr; v[i] = b2 * v[i] + (1 - b2) * gr * gr
            p -= lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + eps)
    S, rho, Pr = probs(P)
    return {'P': P, 'kind': kind, 'd': d, 'train_ll': float((C * np.log(Pr)).sum()), 'probs': Pr, 'rho': float(rho)}


def test_ll(f, C):
    return float((C * np.log(f['probs'])).sum())


def spectral_init(C, d, scale=3.0):
    A = C + C.T + 1e-3; np.fill_diagonal(A, 0)
    r = A.sum(1); Ex = np.outer(r, r) / r.sum()
    A = np.maximum(np.log(A / Ex), 0) * (A > 1)          # positive PMI on observed pairs
    deg = A.sum(1) + 1e-9; Dm = 1 / np.sqrt(deg)
    M = Dm[:, None] * A * Dm[None, :]
    w, U = np.linalg.eigh(M); U = U[:, ::-1][:, 1:d + 1] * Dm[:, None]
    U = U / (U.std(0) + 1e-12) * scale
    return U


def best_fit(C, kind, d, restarts, spectral=True, **kw):
    best = None
    inits = [None] * restarts
    if kind == 'dist' and spectral:
        inits[0] = spectral_init(C, d)
    for r in range(restarts):
        f = fit(C, kind, d, seed=r, init=inits[r], **kw)
        if best is None or f['train_ll'] > best['train_ll']: best = f
    return best


def procrustes_r2(X, Y):
    """similarity-transform fit of X onto Y (both n x d, d=2); returns R^2 (allows reflection)."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(X.T @ Y); R = U @ Vt
    sc = s.sum() / (X * X).sum(); Yh = sc * X @ R
    return 1 - ((Y - Yh) ** 2).sum() / (Y * Y).sum()


def knn_recovery(X, Y, k=5, radius=2):
    """fraction of each point's k nearest learned neighbours that lie within true Chebyshev radius; and chance rate."""
    X = np.asarray(X); Y = np.asarray(Y)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
    T = np.abs(Y[:, None, :] - Y[None, :, :]).max(-1) <= radius; np.fill_diagonal(T, False)
    nn = np.argsort(D, 1)[:, :k]
    hit = T[np.arange(len(X))[:, None], nn].mean()
    return float(hit), float(T.sum() / (len(X) * (len(X) - 1)))


def jdump(obj, name):
    json.dump(obj, open(os.path.join(CK, name), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def jload(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None
