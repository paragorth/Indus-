"""v87 INFER THE PLAINTEXT, NOT THE KEY.

Inverted problem: do not decode. Draw thousands of random meaning-preserving encoders (letter-verbose, syllable /
chunk codebooks, whole-word nomenclators, mixed nomenclator + spelled remainder, positional line-first tables,
abbreviation, homophones, nulls) and random plaintext spans from real texts of known genre (prose, verse, recipe,
herbal, chronicle, law, glossary, index, catalogue, drug labels, star designations, nomenclator lists, sign lists).
Every simulation outputs a 2,000-token text; alphabet-free statistics are computed on it. Rejection ABC (with a
local-linear adjustment for continuous properties) against Voynich 2,000-token chunks gives a posterior over the
PLAINTEXT's properties: running text vs list, plaintext vocabulary richness, plaintext word length, entry length.

Only data and symbols are used from the sources, never anyone's reading of the Voynich.
"""
import os, sys, json, math, random, re, hashlib
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
VD = os.path.dirname(HERE)
CK = os.path.join(VD, 'data', 'v87_ckpt'); os.makedirs(CK, exist_ok=True)
NTOK = 2000

LISTK = {'GLOSS', 'INDEX', 'CATAL', 'APOTH', 'ASTRO', 'NOMEN', 'ALCH'}


# ---------------------------------------------------------------- plaintext sources
def load_sources():
    """sid -> dict(kind, genre, is_list, entries=[[words]]). Entries: list entries / paragraphs / 9-word lines."""
    f = os.path.join(CK, 'sources.json')
    if os.path.exists(f):
        return json.load(open(f))
    S = {}
    sy = json.load(open(os.path.join(VD, 'data', 'v75_ckpt', 'systems.json')))
    merge = {'AST': ['AST_NORTH', 'AST_ZODIA', 'AST_SOUTH'], 'ALC': ['ALC_GESS', 'ALC_UNIC'],
             'APO': ['APO_SIMP', 'APO_PECH', 'APO_COMP'], 'ING': ['ING_ANTID', 'ING_APIC'],
             'SIG': ['SIG_CLUNY', 'SIG_HIRS']}
    used = set()
    for m, ks in merge.items():
        ents = [e[1] for k in ks for e in sy[k]['entries'] if e[1]]
        kind = sy[ks[0]]['kind']; used.update(ks)
        S[m] = dict(kind=kind, genre=kind.lower(), entries=ents)
    for k, v in sy.items():
        if k in used or v['kind'] == 'MODERN':
            continue
        ents = [e[1] for e in v['entries'] if e[1]]
        genre = v['kind'].lower()
        if v['kind'] == 'LANG':
            genre = {'LANG_macer': 'verse_herbal', 'LANG_circa_fr': 'herbal', 'LANG_konrad_plants': 'herbal',
                     'LANG_hildegard': 'herbal', 'LANG_v21_IT': 'herbal', 'LANG_v21_LA': 'herbal'}.get(k, 'prose')
        S[k] = dict(kind=v['kind'], genre=genre, entries=ents)
    c53 = json.load(open(os.path.join(VD, 'data', 'v53_ckpt', 'corpora.json')))
    for k, (lang, genre, lines) in c53.items():
        if k == 'IT_herb':  # same text as LANG_v21_IT
            continue
        S['C53_' + k] = dict(kind='LANG', genre=genre, entries=[l for l in lines if l])
    for k, v in S.items():
        v['is_list'] = int(v['kind'] in LISTK)
        v['nwords'] = sum(len(e) for e in v['entries'])
    json.dump(S, open(f, 'w'))
    return S


def plain_span(src, nwords, rng):
    """random contiguous span of entries giving ~nwords words (wraps if the source is short)."""
    ents = src['entries']; n = len(ents)
    i = rng.randrange(n); out = []; t = 0; guard = 0
    while t < nwords and guard < 100000:
        e = ents[i % n]
        if len(e) > 60:  # long paragraphs: cut into 9-12 word pseudo-lines (keeps entries for lists only)
            for j in range(0, len(e), 10):
                out.append(e[j:j + 10]); t += len(e[j:j + 10])
        else:
            out.append(list(e)); t += len(e)
        i += 1; guard += 1
    return out


def plain_props(ents, src):
    ws = [w for e in ents for w in e]
    w1 = ws[:1000]
    return dict(is_list=src['is_list'],
                ptt=math.log(len(set(w1)) / max(1, len(w1))),   # log type/token at 1,000 plaintext words
                pwl=float(np.mean([len(w) for w in ws])))        # mean plaintext word length (letters)


# ---------------------------------------------------------------- random encoders
def rand_markov(rng, A, conc):
    P = np.array([[rng.gammavariate(conc, 1.0) for _ in range(A + 1)] for _ in range(A + 1)])
    P /= P.sum(1, keepdims=True)
    return P  # row A = start state; column A = end


def gen_word(P, rng, A, minl, maxl):
    s = A; w = []
    while True:
        row = P[s].copy()
        if len(w) < minl: row[A] = 0
        if len(w) >= maxl: return w
        row /= row.sum()
        x = rng.random(); c = 0; k = 0
        for k in range(A + 1):
            c += row[k]
            if x < c: break
        if k == A: return w
        w.append(k); s = k


def make_codebook(units_by_freq, rng, P, A, lmin, lmax, freq_len, used):
    """unit -> codeword (tuple of glyph ints); frequent units shorter when freq_len."""
    cb = {}
    n = len(units_by_freq)
    for r, u in enumerate(units_by_freq):
        if freq_len:
            target = lmin + int((lmax - lmin) * (math.log(r + 1) / math.log(n + 1)))
            lo, hi = max(1, target - 1), target + 1
        else:
            lo, hi = lmin, lmax
        for _ in range(50):
            w = tuple(gen_word(P, rng, A, lo, hi))
            if w and w not in used:
                break
        used.add(w); cb[u] = w
    return cb


def random_scheme(rng):
    sc = dict(
        unit=rng.choice(['letter', 'letter', 'chunk', 'chunk', 'word', 'nomen']),
        chunk=rng.choice([2, 2, 3]),
        trunc=rng.choice([None, None, None, 3, 4, 5, 6]),
        A=rng.randint(12, 26),
        conc=rng.choice([0.1, 0.2, 0.4, 0.8, 2.0]),
        homo=rng.choice([1, 1, 2, 3]),
        freq_len=rng.random() < 0.6,
        lmin=rng.choice([1, 1, 2]), lmax=rng.choice([3, 4, 5, 6, 8]),
        group=rng.choice(['pword', 'pword', 'fixed', 'random']),  # output token = plaintext word / g units / random cut
        g=rng.choice([1, 2, 3]),
        nomen_k=rng.choice([20, 50, 150, 400]),
        linefirst=rng.random() < 0.35,
        nulls=rng.choice([0.0, 0.0, 0.05, 0.15]),
        lines=rng.choice(['entry', 'wrap', 'wrap']),
        width=rng.randint(6, 12),
        seed=rng.randrange(1 << 30),
    )
    return sc


def chunks_of(w, k):
    return [w[i:i + k] for i in range(0, len(w), k)] or ['']


def encode(ents, sc, ntok=NTOK):
    """plaintext entries -> list of output lines (each a list of tuples of glyph ints)."""
    rng = random.Random(sc['seed'])
    A = sc['A']; P = rand_markov(rng, A, sc['conc'])
    if sc['trunc']:
        ents = [[w[:sc['trunc']] for w in e] for e in ents]
    # units per plaintext word
    def units(w):
        u = sc['unit']
        if u == 'letter': return list(w)
        if u == 'chunk': return chunks_of(w, sc['chunk'])
        if u == 'word': return [w]
        return [w] if w in topk else list(w)
    wc = Counter(w for e in ents for w in e)
    topk = set(w for w, _ in wc.most_common(sc['nomen_k']))
    uc = Counter(x for e in ents for w in e for x in units(w))
    order = [u for u, _ in uc.most_common()]
    used = set(); books = []
    nbooks = 2 if sc['linefirst'] else 1
    for b in range(nbooks):
        hb = []
        for h in range(sc['homo']):
            lmin, lmax = sc['lmin'], sc['lmax']
            if sc['unit'] in ('letter',) or (sc['unit'] == 'chunk'):
                lmax = min(lmax, 4)
            hb.append(make_codebook(order, rng, P, A, lmin, max(lmin, lmax), sc['freq_len'], used))
        books.append(hb)
    nullset = [tuple(gen_word(P, rng, A, 1, 4)) or (0,) for _ in range(5)]
    # stream of plaintext words with entry breaks
    out_lines = []; cur = []; ntot = 0; first = True
    def emit_line():
        nonlocal cur, first, ntot
        if cur:
            out_lines.append(cur); ntot += len(cur)
        cur = []; first = True
    pend = []  # pending glyph units for random grouping
    for e in ents:
        for w in e:
            us = units(w)
            book = books[1] if (sc['linefirst'] and first) else books[0]
            codes = [book[rng.randrange(len(book))].get(x, (0,)) for x in us]
            if sc['group'] == 'pword':
                toks = [tuple(g for c in codes for g in c)]
            elif sc['group'] == 'fixed':
                toks = [tuple(g for c in codes[i:i + sc['g']] for g in c) for i in range(0, len(codes), sc['g'])]
            else:
                flat = [g for c in codes for g in c]; toks = []
                pend.extend(flat)
                while len(pend) >= 6:
                    k = rng.randint(2, 7); toks.append(tuple(pend[:k])); pend = pend[k:]
            for t in toks:
                if not t: continue
                if sc['nulls'] and rng.random() < sc['nulls']:
                    cur.append(nullset[rng.randrange(5)])
                cur.append(t); first = False
                if sc['lines'] == 'wrap' and len(cur) >= sc['width']:
                    emit_line()
            if ntot + len(cur) >= ntok: break
        if sc['lines'] == 'entry' or len(cur) >= 3 * sc['width']:
            emit_line()
        if ntot + len(cur) >= ntok: break
    emit_line()
    # trim to ntok
    res = []; t = 0
    for L in out_lines:
        if t >= ntok: break
        L = L[:ntok - t]; res.append(L); t += len(L)
    return res, t


def expansion_guess(sc):
    """plaintext words needed for ntok output tokens (rough; we over-supply)."""
    if sc['group'] == 'pword' or sc['unit'] == 'word':
        return 1.2
    return 0.6


def simulate(src, sc, rng, ntok=NTOK):
    need = int(ntok * 1.3)
    for _ in range(4):
        ents = plain_span(src, need, rng)
        lines, t = encode(ents, sc, ntok)
        if t >= ntok: break
        need *= 2
    return ents, lines, t


# ---------------------------------------------------------------- statistics (alphabet-free)
TRAIN = ['wl_mean', 'wl_sd', 'h1', 'h2', 'ttr', 'hapax', 'top10', 'zipf', 'bmi', 'hfirst', 'heaps']
# held-out statistics (never used in fitting): posterior predictive checks
HELD = ['rep', 'hlast', 'long7', 'lencorr', 'nb1', 'lfjsd', 'bigr_fill']


def H(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values() if v)


def feats(lines):
    toks = [tuple(t) for L in lines for t in L]
    N = len(toks)
    wl = np.array([len(t) for t in toks], float)
    g1 = Counter(g for t in toks for g in t)
    g2 = Counter((t[i], t[i + 1]) for t in toks for i in range(len(t) - 1))
    gp = Counter(t[i] for t in toks for i in range(len(t) - 1))
    h2 = H(g2) - H(gp) if g2 else 0.0
    tc = Counter(toks)
    hap = sum(1 for v in tc.values() if v == 1)
    top = sum(v for _, v in tc.most_common(10)) / N
    fr = np.array(sorted(tc.values(), reverse=True), float)
    r = np.arange(1, len(fr) + 1); m = (r <= 100)
    zipf = np.polyfit(np.log(r[m]), np.log(fr[m]), 1)[0] if m.sum() > 5 else -1.0
    pairs = [(L[i], L[i + 1]) for L in lines for i in range(len(L) - 1)]
    pc = Counter(pairs); a = Counter(p[0] for p in pairs); b = Counter(p[1] for p in pairs); n = len(pairs)
    def mi(pairs):
        pc = Counter(pairs); a = Counter(p[0] for p in pairs); b = Counter(p[1] for p in pairs); n = len(pairs)
        return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in pc.items()) if n else 0
    sh = list(toks); random.Random(7).shuffle(sh)
    bmi = mi(pairs) - mi(list(zip(sh[:-1], sh[1:])))  # word-pair MI above a token-shuffle of the same chunk
    rep = sum(1 for x, y in pairs if x == y) / max(1, n)
    hfirst = H(Counter(t[0] for t in toks if t)); hlast = H(Counter(t[-1] for t in toks if t))
    # held-out
    heaps = len(set(toks[:500])) / max(1, len(tc))
    long7 = float((wl >= 7).mean())
    lp = np.array([(len(x), len(y)) for x, y in pairs], float)
    lencorr = float(np.corrcoef(lp[:, 0], lp[:, 1])[0, 1]) if len(lp) > 3 and lp.std(0).min() > 0 else 0.0
    types = list(tc)
    tset = set(types); nb = 0
    for t in types:
        hit = False
        for i in range(len(t)):
            if t[:i] + t[i + 1:] in tset: hit = True; break
        nb += hit
    nb1 = nb / len(types)
    lf = Counter(L[0][0] for L in lines if L and L[0]); ot = Counter(t[0] for L in lines for t in L[1:] if t)
    keys = set(lf) | set(ot); sl = sum(lf.values()); so = sum(ot.values())
    def kl(p, q):
        return sum(p[k] * math.log2(p[k] / q[k]) for k in p if p[k] > 0)
    pa = {k: lf[k] / sl for k in keys}; pb = {k: ot[k] / so for k in keys}; mm = {k: (pa[k] + pb[k]) / 2 for k in keys}
    lfjsd = 0.5 * kl(pa, mm) + 0.5 * kl(pb, mm) if sl and so else 0.0
    bigr_fill = len(g2) / max(1, len(g1) ** 2)
    return dict(wl_mean=wl.mean(), wl_sd=wl.std(), h1=H(g1), h2=h2, ttr=len(tc) / N, hapax=hap / len(tc), top10=top,
                zipf=zipf, bmi=bmi, rep=rep, hfirst=hfirst, hlast=hlast, heaps=heaps, long7=long7, lencorr=lencorr,
                nb1=nb1, lfjsd=lfjsd, bigr_fill=bigr_fill)


# ---------------------------------------------------------------- Voynich
def voy_lines(name='ZL3b'):
    import vlib
    L = vlib.load_voynich(name)
    out = []
    for l in L:
        out.append(dict(sec=(l['illus'] or '?') + (l['lang'] or '?'), folio=l['folio'],
                        words=[tuple(vlib.glyphs(w)) for w in l['words']]))
    return out


def voy_chunks(vl, ntok=NTOK, key=None):
    """contiguous chunks of ntok tokens (lines kept whole, last line cut); key(line) -> group to chunk within."""
    groups = {}
    for l in vl:
        groups.setdefault(key(l) if key else 'all', []).append(l)
    res = {}
    for g, ls in groups.items():
        cs = []; cur = []; t = 0
        for l in ls:
            w = l['words']
            if t + len(w) >= ntok:
                cur.append(w[:ntok - t]); cs.append(cur); cur = []; t = 0
                continue
            cur.append(w); t += len(w)
        res[g] = cs
    return res


# ---------------------------------------------------------------- ABC
PROPS = ['is_list', 'ptt', 'pwl']


def scale_of(F):
    med = np.median(F, 0); mad = np.median(np.abs(F - med), 0) * 1.4826
    mad[mad == 0] = 1.0
    return med, mad


def abc(Fbank, Pbank, f_obs, scale, q=0.02, mask=None):
    """rejection ABC + local-linear adjustment. returns dict prop -> posterior summary."""
    med, mad = scale
    Z = (Fbank - med) / mad; z = (f_obs - med) / mad
    d = np.sqrt(((Z - z) ** 2).sum(1))
    if mask is not None:
        d = np.where(mask, d, np.inf)
    k = max(20, int(q * np.isfinite(d).sum()))
    idx = np.argsort(d)[:k]; h = d[idx[-1]] + 1e-9
    w = 1 - (d[idx] / h) ** 2
    out = {'dist': float(np.median(d[idx])), 'idx': idx, 'w': w}
    X = Z[idx] - z
    for j, p in enumerate(PROPS):
        y = Pbank[idx, j]
        mean_raw = float(np.sum(w * y) / np.sum(w))
        if p == 'is_list':
            out[p] = mean_raw; continue
        Xd = np.hstack([np.ones((k, 1)), X]); W = np.diag(w)
        try:
            beta = np.linalg.lstsq(Xd.T @ W @ Xd + 1e-3 * np.eye(Xd.shape[1]), Xd.T @ W @ y, rcond=None)[0]
            adj = y - X @ beta[1:]
        except Exception:
            adj = y
        out[p] = float(np.sum(w * adj) / np.sum(w))
        o = np.argsort(adj); cw = np.cumsum(w[o]) / w.sum()
        out[p + '_lo'] = float(adj[o][np.searchsorted(cw, 0.1)]); out[p + '_hi'] = float(adj[o][min(len(o) - 1, np.searchsorted(cw, 0.9))])
    return out


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()
