"""v43 FIND THE ISLAND.

If part of the manuscript is real language and the rest filler or generated, the language-like part should stand
out.  The text (reading order) is cut into 100-token CHUNKS (lines kept, cut at chunk edges).  Each chunk gets the
v31 alphabet-free feature vector and the v31 multinomial classifier posterior (LANG, INVENT, GEN, GIBB, MAGIC).
A WINDOW is W consecutive tokens (W/100 consecutive chunks): its classifier score is the mean chunk posterior; the
window also gets the v33 gap ratio (own trigram resynthesis), a line-split v23 frequency arrow, a page topic
re-use ratio, and (group level only) the v21 forgery AUC.

Score used for islands:  s = P(LANG) + P(INVENT)   ("language or conlang"), mean over the window's chunks.
Language band: tau = 5th percentile of s over size-matched windows of real languages (classifier trained without
the corpus the window comes from).  Scan null: chunk order permuted (keeps the chunk multiset, destroys place).
"""
import os, sys, json, math, random, re
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(ROOT, 'data', 'v43_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
N = 100

# real-language calibration corpora (v31 corpora.json) -> classifier exclusion
LANGS = ['L_Isidore', 'L_msI_Lat', 'L_msG_Alem', 'L_msG_Bav1', 'L_msI_Ita', 'L_msC_Old', 'L_pgDante', 'L_pgCaesar',
         'L_AngloSaxon_Lite', 'L_Turkish_Lite']


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ corpora
def voynich(name='ZL3b'):
    """v21 pages (paragraph text, v21 units) with metadata: sec, illus, lang (Currier), hand, quire, bifolio."""
    import v21_lib as V, v8_lib as E
    P = V.voynich_pages(name)
    meta = {p['id']: p for p in E.voynich_pages(min_tokens=0, name='ZL3b')}
    recs = json.load(open(os.path.join(ROOT, 'data', 'derived', name + '_lines.json')))
    lm = {}
    for r in recs: lm.setdefault(r['folio'], (r.get('illus'), r.get('lang'), r.get('hand'), r.get('quire')))
    for p in P:
        il, lg, hd, q = lm.get(p['id'], (None,) * 4)
        m = meta.get(p['id'], {})
        p['illus'] = il or 'x'; p['lang'] = lg or 'x'; p['hand'] = hd or 'x'; p['quire'] = q or m.get('quire') or 'x'
        p['bifolio'] = f"{p['quire']}{m.get('bifolio')}" if m.get('bifolio') else 'x'
    return P


_C = None


def corpus_lines(name, maxtok=None):
    global _C
    if _C is None: _C = json.load(open(os.path.join(SCR, 'v31', 'corpora.json')))
    out, n = [], 0
    for d in _C[name]['docs']:
        for l in d:
            l = [w for w in (re.sub(r'[^\w]', '', x.lower()) for x in l) if w and not any(c.isdigit() for c in w)]
            if l: out.append(l); n += len(l)
            if maxtok and n >= maxtok: return out
    return out


def stream_of_pages(P):
    """[(word, page_index, line_key)] in reading order."""
    S = []
    for pi, p in enumerate(P):
        li = 0
        for pa in p['paras']:
            for l in pa:
                for w in l: S.append((w, pi, (pi, li)))
                li += 1
    return S


def stream_of_lines(lines, lines_per_page=20):
    S = []
    for li, l in enumerate(lines):
        for w in l: S.append((w, li // lines_per_page, (li // lines_per_page, li)))
    return S


def chunks(S, n=N):
    """non-overlapping n-token chunks; each chunk = list of lines (word lists)."""
    out = []
    for s in range(0, len(S) - n + 1, n):
        seg = S[s:s + n]; lines = []; last = None
        for w, _, lk in seg:
            if lk != last: lines.append([]); last = lk
            lines[-1].append(w)
        out.append(lines)
    return out


# ------------------------------------------------------------------ features and classifier
def chunk_feats(chs, seed=0):
    import v31_lib as L
    rng = random.Random(seed)
    return [L.features(c, rng) for c in chs]


def feats_pool(chs, nw=2, seed=0):
    from multiprocessing import Pool
    if nw <= 1 or len(chs) < 20: return chunk_feats(chs, seed)
    k = math.ceil(len(chs) / nw)
    parts = [chs[i:i + k] for i in range(0, len(chs), k)]
    with Pool(nw) as P:
        res = P.starmap(chunk_feats, [(p, seed + i) for i, p in enumerate(parts)])
    return [f for r in res for f in r]


_CLF = {}


def classifier(exclude=()):
    key = tuple(sorted(exclude))
    if key in _CLF: return _CLF[key]
    import v31_lib as L
    from sklearn.linear_model import LogisticRegression
    R = L.load('feats_N100.json')
    keys = sorted(R[0]['F'].keys())
    R = [r for r in R if r['cls'] in L.CLASSES and r['corpus'] not in exclude]
    X = np.array([[r['F'][k] for k in keys] for r in R], float); X[~np.isfinite(X)] = 0
    y = np.array([r['cls'] for r in R])
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    m = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit((X - mu) / sd, y)
    _CLF[key] = (m, mu, sd, keys)
    return _CLF[key]


def probs(F, exclude=()):
    """F: list of feature dicts -> array (n, 5) in order LANG INVENT GEN GIBB MAGIC."""
    m, mu, sd, keys = classifier(exclude)
    X = np.array([[f.get(k, 0.0) for k in keys] for f in F], float); X[~np.isfinite(X)] = 0
    pr = m.predict_proba((X - mu) / sd)
    idx = [list(m.classes_).index(c) for c in ('LANG', 'INVENT', 'GEN', 'GIBB', 'MAGIC')]
    return pr[:, idx]


def lscore(pr):
    return pr[:, 0] + pr[:, 1]


# ------------------------------------------------------------------ window battery (token based)
def gap_ratio_tokens(lines, seed=0):
    import v33_lib as G
    rules = [G.Rule('fix', 1), G.Rule('fix', 2), G.Rule('pos', 0.5), G.Rule('freq', 25)]
    toks = [w for l in lines for w in l]
    tri = G.Trigram(lines)
    rs = []
    for b in range(2):
        gen = tri.gen(len(toks), random.Random(seed * 10 + b))
        for r in rules:
            mx = G.matrix(toks, G.Rule(r.kind, r.param).fit(toks)).mean()
            mt = G.matrix(gen, G.Rule(r.kind, r.param).fit(gen)).mean()
            if mt > 0: rs.append(mx / mt)
    return float(np.mean(rs)) if rs else float('nan')


def arrow_lines(lines, nprobe=200, seed=0):
    """v23 word-frequency-class arrow with lines (odd / even) as the replication split; frequencies from the window."""
    rng = random.Random(seed)
    gf = Counter(w for l in lines for w in l)
    pairs = {k: ([], [], []) for k in range(1, 4)}
    for li, l in enumerate(lines):
        lf = [math.log2(gf[w] + 1) for w in l]
        for k in range(1, 4):
            for i in range(len(l) - k):
                pairs[k][0].append(lf[i]); pairs[k][1].append(lf[i + k]); pairs[k][2].append(li)
    pairs = {k: (np.array(a), np.array(b), np.array(c, int)) for k, (a, b, c) in pairs.items()}
    nL = len(lines); half = np.arange(nL) % 2 == 0
    surv = 0
    for _ in range(nprobe):
        m = rng.choice([2, 3]); th = sorted(rng.sample([1, 1.6, 2.3, 3, 4], m - 1))
        k = rng.randrange(1, 4); a, b = rng.sample(range(m), 2)
        x, y, pg = pairs[k]
        if len(x) < 20: continue
        cx = np.digitize(x, th, right=True); cy = np.digitize(y, th, right=True)
        d = ((cx == a) & (cy == b)).astype(float) - ((cx == b) & (cy == a))
        D = np.bincount(pg, weights=d, minlength=nL)
        zs = []
        for h in (half, ~half):
            v = D[h]; sd = v.std(ddof=1)
            zs.append(0.0 if sd == 0 else v.mean() / (sd / math.sqrt(len(v))))
        if abs(zs[0]) >= 2.5 and zs[1] * np.sign(zs[0]) >= 1.5: surv += 1
    return surv / nprobe


def topic_reuse(S_window):
    """pages in the window: share of second-half types (of a page) seen in the same page's first half, over the
    share seen in other pages' first halves (size matched by truncation)."""
    pg = defaultdict(list)
    for w, pi, _ in S_window: pg[pi].append(w)
    pg = {k: v for k, v in pg.items() if len(v) >= 30}
    if len(pg) < 2: return float('nan')
    own, oth = [], []
    ks = list(pg)
    for k in ks:
        v = pg[k]; h = len(v) // 2; A1, B2 = v[:h], set(v[h:])
        own.append(len(B2 & set(A1)) / len(B2))
        o = [len(B2 & set(pg[j][:h])) / len(B2) for j in ks if j != k and len(pg[j]) >= h]
        if o: oth.append(np.mean(o))
    return float(np.mean(own) / max(1e-6, np.mean(oth))) if oth else float('nan')


def window_lines(S_window):
    lines, last = [], None
    for w, _, lk in S_window:
        if lk != last: lines.append([]); last = lk
        lines[-1].append(w)
    return lines


# ------------------------------------------------------------------ scan
def window_scores(cs, W):
    """cs: per-chunk scores; W in chunks -> sliding means (step 1 chunk)."""
    c = np.concatenate([[0], np.cumsum(cs)])
    return (c[W:] - c[:-W]) / W


def scan_null(cs, W, nperm=2000, seed=0):
    rng = np.random.default_rng(seed)
    mx = np.empty(nperm)
    for i in range(nperm):
        mx[i] = window_scores(rng.permutation(cs), W).max()
    return mx


def runs_above(ws, tau):
    out, s = [], None
    for i, v in enumerate(list(ws) + [-1e9]):
        if v >= tau and s is None: s = i
        if v < tau and s is not None: out.append((s, i - 1)); s = None
    return out


# ------------------------------------------------------------------ plants
def translit_map(lang_words, voy_words):
    """letters of the language -> Voynich units by frequency rank (a Voynich-like glyph scheme)."""
    cl = [c for c, _ in Counter(c for w in lang_words for c in w).most_common()]
    cv = [c for c, _ in Counter(c for w in voy_words for c in w).most_common()]
    return {a: cv[i % len(cv)] for i, a in enumerate(cl)}


def plant(S, lang_lines, start, size, vmap):
    """replace tokens S[start:start+size] by the language words (transliterated), keeping page and line slots."""
    lw = [''.join(vmap.get(c, c) for c in w) for l in lang_lines for w in l]
    S2 = list(S)
    for i in range(size):
        w, pi, lk = S[start + i]
        S2[start + i] = (lw[i], pi, lk)
    return S2
