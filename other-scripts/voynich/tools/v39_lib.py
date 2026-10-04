"""v39 COLLAPSE THE TWINS AND LOOK AGAIN.

v35 found that the Voynich look-alike glyph pairs (gallows k~t, p~f, benched gallows ckh~cth, cph~cfh, and ch~sh)
are near-interchangeable in context, like a planted homophone cipher.  If they are one unit spread over two glyphs,
the generator-like signatures found earlier (v23 no word-frequency arrow, v31 classifier votes 'generator',
v33 no lexical gaps, v21 forgery battery) could be artefacts of the split.  Here the twins are merged and the
battery is rerun, against random merges of equally frequent pairs (null), planted homophone splits of real
Latin / German / Italian (positive controls: must move back when merged), random merges in real languages (must not
make them more language-like), and the Copiale cipher merged by its published key (real homophone cipher).

Corpus format: pages = [{'id', 'sec', 'paras': [[[word, ...], ...line], ...para]}], word = string of single-char units
(Voynich: v21 units, benched gallows and ch/sh as single chars T K P F C S).

Battery (battery()):
  h1, h2, h3      unit entropy and conditional entropies (order 1, 2; word boundary is a unit), h2/h1
  wl_mu, wl_sd    word length in units;  ttr (types / 10k tokens), hapax share, zipf slope (ranks 1-500)
  arrow           v23-style word FREQUENCY-class arrow survivors out of NPROBE random probes (|zA|>=3, zB>=2)
  gap             v33 gap ratio: connectance(text) / connectance(own glyph-trigram resynthesis), 6 rules x 2 blocks
  pL, pG, pB      v31 multinomial classifier (trained on the v31 feature table without the source corpus):
                  mean P(LANG), P(GEN), P(GIBB) over 100-token samples
  auc             v21 forgery battery: ridge AUC real pages vs F3 junction+section forger, 5-fold, 2 seeds
"""
import os, sys, json, math, random, re, zlib
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(ROOT, 'data', 'v39_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
NPROBE = 400

# the v35 twin pairs (v21 unit alphabet)
TWINS = {'kt': ('k', 't'), 'pf': ('p', 'f'), 'KT': ('K', 'T'), 'PF': ('P', 'F'), 'CS': ('C', 'S')}


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ corpora
def lines_of(P):
    return [l for p in P for pa in p['paras'] for l in pa]


def tokens(P):
    return [w for l in lines_of(P) for w in l]


def voynich(name='ZL3b'):
    import v21_lib as V
    return V.voynich_pages(name)


def _docs_to_pages(docs, pfx, lines_per_page=20):
    P = []
    for i, d in enumerate(docs):
        for j in range(0, len(d), lines_per_page):
            ch = [l for l in d[j:j + lines_per_page] if l]
            if sum(map(len, ch)) >= 40:
                P.append({'id': f'{pfx}{i}_{j}', 'sec': 'x', 'paras': [ch]})
    return P


_V31 = None


def v31_corpus(name):
    global _V31
    if _V31 is None:
        _V31 = json.load(open(os.path.join(SCR, 'v31', 'corpora.json')))
    return _V31[name]['docs']


def lang(code):
    """real-language controls (letters only, lower case).  la: Isidore herbal/lapidary (v21);
    it: Italian herbal (v21); de: Alemannic German MSS 1350-1450 (ReF, v31)."""
    import v21_lib as V
    if code == 'la':
        P = V.latin_herbal()
    elif code == 'it':
        P = V.italian_herbal()
    elif code == 'de':
        docs = [[[re.sub(r'[^a-zäöüß]', '', w.lower()) for w in l] for l in d] for d in v31_corpus('L_msG_Alem')]
        docs = [[[w for w in l if w] for l in d] for d in docs]
        P = _docs_to_pages(docs, 'de')
    for p in P:
        p['paras'] = [[[re.sub(r'[^a-zäöüßàèéìíòóùú]', '', w.lower()) for w in l] for l in pa] for pa in p['paras']]
        p['paras'] = [[[w for w in l if w] for l in pa] for pa in p['paras']]
        p['paras'] = [[l for l in pa if l] for pa in p['paras']]
    return P


# Copiale: keyboard char of the Copiale font -> plaintext (Knight, Megyesi & Schaefer 2011, Figure 6)
_COP_PLAIN = {'P': 'a', 'N': 'a', 'H': 'a', '0': 'a', '|': 'UML', 'Q': 'b', '?': 'c', '>': 'd', 'z': 'd',
              'A': 'e', 'E': 'e', 'I': 'e', 'O': 'e', 'U': 'e', ')': 'e', 'Z': 'e', '~': 'f', '6': 'g', 'X': 'g',
              '-': 'h', '5': 'h', 'y': 'i', 'Y': 'i', '!': 'i', '4': 'j', 'C': 'l', '+': 'm', 'B': 'n', 'F': 'n',
              'D': 'n', 'g': 'n', '<': 'o', '&': 'o', 'W': 'ö', 'd': 'p', 'R': 'r', '3': 'r', 'j': 'r', '[': 'ss',
              '^': 't', '=': 'u', '"': 'u', ']': 'ü', '1': 'v', 'M': 'w', '8': 'y', 'S': 'z', 'T': 'sch',
              '7': 'st', '/': 'ch', ':': 'REP', 'G': 'en'}
_COP_SPACE = set('abcLef\\Khiklmnopqrs`tuvwx(J')


def copiale():
    """returns (pages_cipher, pages_plainunit): words split at the cipher's space symbols; units are cipher symbols
    (names) or, merged by the key, plaintext units (logograms and unknown symbols stay themselves)."""
    import v35_lib as V
    d = json.load(open(os.path.join(SCR, 'copiale', 'ann.json')))
    def key(k):
        m = re.match(r'(\d+)(B?)_(\d+)', k)
        return (int(m.group(1)), m.group(2), int(m.group(3)))
    pages = defaultdict(list)
    for k in sorted(d, key=key):
        pg = key(k)[:2]
        pages[pg].append(d[k]['label'].split())
    names = sorted({g for ls in pages.values() for l in ls for g in l})
    enc = {}
    def code(u):
        if u not in enc: enc[u] = chr(0x100 + len(enc))
        return enc[u]
    Pc, Pp = [], []
    for pg in sorted(pages):
        lc, lp = [], []
        for l in pages[pg]:
            wc, wp, cur_c, cur_p = [], [], '', ''
            for g in l:
                kb = V.COP_NAME.get(g)
                if kb is not None and kb in _COP_SPACE:
                    if cur_c: wc.append(cur_c); wp.append(cur_p)
                    cur_c, cur_p = '', ''
                    continue
                cur_c += code('c:' + g)
                pl = _COP_PLAIN.get(kb) if kb is not None else None
                cur_p += code('p:' + (pl if pl else g))
            if cur_c: wc.append(cur_c); wp.append(cur_p)
            if wc: lc.append(wc); lp.append(wp)
        if lc:
            Pc.append({'id': f'cop{pg}', 'sec': 'x', 'paras': [lc]})
            Pp.append({'id': f'cop{pg}', 'sec': 'x', 'paras': [lp]})
    return Pc, Pp, enc


# ------------------------------------------------------------------ transforms
def apply_map(P, mp):
    if not mp: return P
    tr = str.maketrans(mp)
    out = []
    for p in P:
        q = dict(p)
        q['paras'] = [[[w.translate(tr) for w in l] for l in pa] for pa in p['paras']]
        out.append(q)
    return out


def merge_map(pairs):
    """pairs [(a,b), ...] -> union-find map unit -> class representative (the most frequent kept by caller order)."""
    par = {}
    def f(x):
        while par.get(x, x) != x: x = par[x]
        return x
    for a, b in pairs:
        ra, rb = f(a), f(b)
        if ra != rb: par[rb] = ra
    return {x: f(x) for x in list(par) if f(x) != x}


def unit_freq(P):
    return Counter(c for w in tokens(P) for c in w)


def random_pairs(P, real_pairs, rng, window=2, tries=200):
    """for each real pair (a,b) draw (a',b') with a' within +-window frequency ranks of a and b' of b; all units
    distinct; the set must not equal the real set."""
    fr = [u for u, _ in unit_freq(P).most_common()]
    rk = {u: i for i, u in enumerate(fr)}
    real = {frozenset(x) for x in real_pairs}
    for _ in range(tries):
        used, out = set(), []
        ok = True
        for a, b in real_pairs:
            ca = [u for u in fr[max(0, rk[a] - window): rk[a] + window + 1] if u not in used]
            if not ca: ok = False; break
            a2 = rng.choice(ca); used.add(a2)
            cb = [u for u in fr[max(0, rk[b] - window): rk[b] + window + 1] if u not in used]
            if not cb: ok = False; break
            b2 = rng.choice(cb); used.add(b2)
            out.append((a2, b2))
        if ok and {frozenset(x) for x in out} != real:
            return out
    return None


def plant_split(P, letters, ratios, seed=0, mode='free'):
    """homophone plant: each letter in `letters` is written as one of two glyphs (new private-use chars).
    mode 'free': variant chosen at random with P(minor)=ratio.  mode 'pos': variant chosen by the previous unit
    (deterministic hash, 15% noise) = positional allograph.  Returns pages and the merge map that undoes it."""
    rng = random.Random(seed)
    twin = {c: chr(0xE000 + i) for i, c in enumerate(letters)}
    rat = dict(zip(letters, ratios))
    out = []
    for p in P:
        q = dict(p); paras = []
        for pa in p['paras']:
            ls = []
            for l in pa:
                ws = []
                for w in l:
                    s, prev = '', '^'
                    for c in w:
                        if c in twin:
                            if mode == 'free':
                                minor = rng.random() < rat[c]
                            else:
                                minor = (zlib.crc32((prev + c).encode()) % 1000) / 1000 < rat[c]
                                if rng.random() < 0.15: minor = not minor
                            s += twin[c] if minor else c
                        else:
                            s += c
                        prev = c
                    ws.append(s)
                ls.append(ws)
            paras.append(ls)
        q['paras'] = paras
        out.append(q)
    return out, {v: k for k, v in twin.items()}


# ------------------------------------------------------------------ cheap measures
def _H(c):
    n = sum(c.values()); p = np.array(list(c.values()), float) / n
    return float(-(p * np.log2(p)).sum())


def entropies(P):
    s = ''.join(' ' + ' '.join(l) for l in lines_of(P)) + ' '
    s = re.sub(' +', ' ', s)
    c1 = Counter(s); c2 = Counter(s[i:i + 2] for i in range(len(s) - 1)); c3 = Counter(s[i:i + 3] for i in range(len(s) - 2))
    h1 = _H(c1); H2 = _H(c2); H3 = _H(c3)
    return dict(h1=h1, h2=H2 - h1, h3=H3 - H2, h21=(H2 - h1) / h1, nunit=len(c1) - 1)


def lexstats(P, n=10000):
    T = tokens(P)
    L = np.array([len(w) for w in T], float)
    t = T[:n] if len(T) >= n else T
    c = Counter(T)
    f = np.array(sorted(c.values(), reverse=True), float)[:500]
    r = np.arange(1, len(f) + 1)
    slope = float(np.polyfit(np.log(r), np.log(f), 1)[0])
    return dict(wl_mu=float(L.mean()), wl_sd=float(L.std()), ttr=len(set(t)) / len(t),
                hapax=sum(1 for v in c.values() if v == 1) / len(c), zipf=slope)


# ------------------------------------------------------------------ v23-style frequency-class arrow
def freq_arrow(P, nprobe=NPROBE, seed=0):
    rng = random.Random(seed)
    T = tokens(P)
    gf = Counter(T)
    pairs = {k: ([], [], []) for k in range(1, 5)}
    for pi, p in enumerate(P):
        for pa in p['paras']:
            for l in pa:
                lf = [math.log2(gf[w] + 1) for w in l]
                for k in range(1, 5):
                    for i in range(len(l) - k):
                        pairs[k][0].append(lf[i]); pairs[k][1].append(lf[i + k]); pairs[k][2].append(pi)
    pairs = {k: (np.array(a), np.array(b), np.array(c)) for k, (a, b, c) in pairs.items()}
    nP = len(P)
    half = np.array([rng.random() < 0.5 for _ in range(nP)])
    surv = 0
    for _ in range(nprobe):
        m = rng.choice([2, 3, 4]); th = sorted(rng.sample(range(1, 7), m - 1))
        k = rng.randrange(1, 5)
        a, b = rng.sample(range(m), 2)
        x, y, pg = pairs[k]
        cx = np.digitize(x, th, right=True); cy = np.digitize(y, th, right=True)
        d = ((cx == a) & (cy == b)).astype(float) - ((cx == b) & (cy == a))
        D = np.bincount(pg, weights=d, minlength=nP)
        zs = []
        for h in (half, ~half):
            v = D[h]
            sd = v.std(ddof=1)
            zs.append(0.0 if sd == 0 else v.mean() / (sd / math.sqrt(len(v))))
        if abs(zs[0]) >= 3 and zs[1] * np.sign(zs[0]) >= 2: surv += 1
    return surv


# ------------------------------------------------------------------ v33 gap ratio
def gap_ratio(P, seed=34):
    import v33_lib as G
    lines = lines_of(P)
    rules = [G.Rule('fix', 1), G.Rule('fix', 2), G.Rule('pos', 0.5), G.Rule('pos', 0.7), G.Rule('freq', 25),
             G.Rule('harris', 0)]
    tri = G.Trigram(lines)
    rs = []
    for b, toks in enumerate(G.blocks(lines, seed=seed)):
        rng = random.Random(1000 + b)
        gen = tri.gen(len(toks), rng)
        for r in rules:
            mx = G.matrix(toks, G.Rule(r.kind, r.param).fit(toks)).mean()
            mt = G.matrix(gen, G.Rule(r.kind, r.param).fit(gen)).mean()
            rs.append(mx / mt if mt > 0 else np.nan)
    return float(np.nanmean(rs))


# ------------------------------------------------------------------ v31 classifier
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


def v31_probs(P, exclude=(), maxs=40, seed=0):
    import v31_lib as L
    m, mu, sd, keys = classifier(exclude)
    docs = {}
    for p in P: docs.setdefault(p['sec'], []).extend(l for pa in p['paras'] for l in pa)
    S = L.samples(list(docs.values()), N=100, maxs=maxs, seed=seed)
    rng = random.Random(seed)
    X = []
    for s in S:
        F = L.features(s, rng)
        X.append([F.get(k, 0.0) for k in keys])
    X = np.array(X, float); X[~np.isfinite(X)] = 0
    pr = m.predict_proba((X - mu) / sd).mean(0)
    d = dict(zip(m.classes_, pr))
    return dict(pL=float(d['LANG']), pG=float(d['GEN']), pB=float(d['GIBB']), pM=float(d['MAGIC']), pI=float(d['INVENT']))


# ------------------------------------------------------------------ v21 forgery battery (F3)
def forge_auc(P, seeds=(0, 1)):
    import v21_lib as V
    FZ = V.Featurizer(P)
    Xr, keys = FZ.matrix(P)
    out = []
    for s in seeds:
        F = V.Forger(P, scope='sec', pos=True, name='F3')
        Q = F.forge(P, random.Random(s))
        Xf, _ = FZ.matrix(Q, keys)
        ok = np.isfinite(Xr).all(0) & np.isfinite(Xf).all(0)
        out.append(V.cv_auc(Xr[:, ok], Xf[:, ok], 'ridge', seed=s))
    return float(np.mean(out))


# ------------------------------------------------------------------ battery
CHEAP = ('h1', 'h2', 'h3', 'h21', 'wl_mu', 'wl_sd', 'ttr', 'hapax', 'zipf', 'arrow')


def battery(P, exclude=(), full=True, seed=0):
    r = {}
    r.update(entropies(P)); r.update(lexstats(P))
    r['arrow'] = freq_arrow(P, seed=seed)
    r.update(v31_probs(P, exclude, seed=seed))
    if full:
        r['gap'] = gap_ratio(P)
        r['auc'] = forge_auc(P)
    return r


def interchange_index(P, minc=50):
    """v35 index I for all unit pairs (alternating 400-word halves, PPMI contexts L1 R1 L2 R2)."""
    import v25_lib as L
    words = [tuple(w) for w in tokens(P)]
    c = Counter(g for w in words for g in w)
    A = sorted([g for g, n in c.items() if n >= minc], key=lambda g: -c[g])
    a, b = L.halves(words)
    Ps = []
    for h in (a, b):
        C, _, _ = L.contexts(h, A)
        X = L.ppmi(C)
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
        Ps.append(X / n)
    Xc = Ps[0] @ Ps[1].T
    d = np.sqrt(np.maximum(np.outer(np.diag(Xc), np.diag(Xc)), 1e-12))
    return A, 0.5 * (Xc + Xc.T) / d
