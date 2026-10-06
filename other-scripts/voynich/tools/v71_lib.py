"""v71: order-dependence fingerprints ("shuffle the sentences to learn the genre").

A chunk is a list of pages; page = list of paragraphs; paragraph = list of lines;
line = list of words; word = string, one character per glyph.

For each chunk, ~800 random model types (target representation x context atoms x
smoothing, plus cache models and glyph models) are trained on 3/4 of the pages
(original order) and scored on the held-out quarter, in the original order and
after shuffling at six scales:
  S0 glyphs within word, S1 words within line, S1p words within paragraph
  (line lengths kept), S2 lines within paragraph, S3 paragraphs within page,
  S4 pages within chunk.
Relative damage RD = (G_orig - G_shuf) / G_orig, where G = held-out bits/token
the context adds over the unigram of the same representation.  The fingerprint is
RD per (scale, model family), ratio of sums over models.
"""
import os, sys, json, random, zlib, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.dirname(HERE)
CK = os.path.join(VD, 'data', 'v71_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)

SCALES = ['S0', 'S1', 'S1p', 'S2', 'S3', 'S4']

# ------------------------------------------------------------------ Voynich
GLYPH_MULTI = [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]
def vglyphs(w):
    for a, b in GLYPH_MULTI: w = w.replace(a, b)
    return w

SECTIONS = {'herbalA': ('H', 'A'), 'herbalB': ('H', 'B'), 'stars': ('S', 'B'), 'bio': ('B', 'B'),
            'pharma': ('P', 'A'), 'text': ('T', 'B')}

def voynich_sections(name='ZL3b'):
    d = json.load(open(os.path.join(VD, 'data', 'derived', name + '_lines.json')))
    out = {}
    for sec, (il, lg) in SECTIONS.items():
        pages, cur_fol = [], None
        for r in d:
            if r['ltype'] != 'P' or r['illus'] != il or r['lang'] != lg: continue
            ws = [vglyphs(w) for w, u in zip(r['words'], r['uncertain']) if not u and '?' not in w]
            if not ws: continue
            if r['folio'] != cur_fol:
                pages.append([]); cur_fol = r['folio']
            if r['para_start'] or not pages[-1]:
                pages[-1].append([])
            pages[-1][-1].append(ws)
        out[sec] = pages
    return out

def nwords(pages):
    return sum(len(l) for pg in pages for pa in pg for l in pa)

def split_chunks(pages, lo=2000, hi=3000):
    """consecutive pages into chunks of ~equal size within [lo, hi] words."""
    n = nwords(pages)
    k = max(1, int(round(n / ((lo + hi) / 2))))
    if n < lo: return [pages] if n >= 1200 else []
    target = n / k
    out, cur, c = [], [], 0
    for pg in pages:
        cur.append(pg); c += sum(len(l) for pa in pg for l in pa)
        if c >= target and len(out) < k - 1:
            out.append(cur); cur, c = [], 0
    if cur:
        if out and c < 0.5 * target: out[-1].extend(cur)
        else: out.append(cur)
    return out

# ------------------------------------------------------------------ encoding + layout of references
CODE_GLYPHS = 'abcdefghijklmnop'
PAD_I, PAD_F = 'q', 'y'

def make_code(entries, rng):
    from collections import Counter
    fr = Counter(ch for e in entries for p in e for w in p for ch in w)
    letters = [c for c, _ in fr.most_common()]
    used, code = set(), {}
    for i, c in enumerate(letters):
        L = 1 if i < 8 else (2 if i < 26 else 3)
        if rng.random() < 0.25: L = min(3, L + 1)
        for _ in range(1000):
            s = ''.join(rng.choice(CODE_GLYPHS) for _ in range(L))
            if s not in used: break
            L = min(4, L + 1) if _ % 50 == 49 else L
        used.add(s); code[c] = s
    return code

def encode_word(w, code, rng, pi, pf):
    s = ''.join(code.get(ch, '') for ch in w)
    if rng.random() < pi: s = PAD_I + s
    if rng.random() < pf: s = s + PAD_F
    return s or PAD_F

def layout(entries, rng, width=None, page_words=None):
    """entries -> pages (list of paras of lines). Paragraph = entry paragraph; lines filled to a
    glyph width; a new page when the next entry would overflow; long paragraphs carry over."""
    if width is None:
        ws = [len(w) for e in entries for p in e for w in p]
        width = rng.uniform(7, 11) * (sum(ws) / max(1, len(ws)))
    page_words = page_words or rng.uniform(60, 350)
    pages, page, pw = [], [], 0
    def flush():
        nonlocal page, pw
        if page: pages.append(page)
        page, pw = [], 0
    for e in entries:
        ew = sum(len(p) for p in e)
        if page and pw + ew > page_words: flush()
        for p in e:
            lines, cur, cw = [], [], 0
            for w in p:
                if cur and cw + len(w) > width:
                    lines.append(cur); cur, cw = [], 0
                cur.append(w); cw += len(w)
            if cur: lines.append(cur)
            para = []
            for l in lines:
                if para and pw >= page_words:
                    page.append(para); flush(); para = []
                para.append(l); pw += len(l)
            if para: page.append(para)
            if pw >= page_words: flush()
    flush()
    return pages

def ref_chunks(R, key, rng, nmax=6, lo=2000, hi=3000):
    ents = []
    for e in R[key]['entries']:          # very long entries (chapters of prose) -> one entry per paragraph
        e = [p[i:i + 800] for p in e for i in range(0, len(p), 800)]
        if sum(len(p) for p in e) > 1000: ents.extend([[p] for p in e])
        else: ents.append(e)
    tot = sum(len(p) for e in ents for p in e)
    out = []
    starts = []
    ncht = min(nmax, max(1 if tot >= 1500 else 0, tot // 2600))
    if ncht == 0: return out
    seg = len(ents) / ncht
    for j in range(ncht):
        size = rng.uniform(lo, hi)
        a = int(j * seg); sel, c = [], 0
        i = a
        while i < len(ents) and c < size:
            sel.append(ents[i]); c += sum(len(p) for p in ents[i]); i += 1
        if c < 1200: continue
        code = make_code(sel, rng)
        pi, pf = rng.uniform(0.1, 0.35), rng.uniform(0.15, 0.45)
        enc = [[[encode_word(w, code, rng, pi, pf) for w in p] for p in e] for e in sel]
        out.append(layout(enc, rng))
    return out

# ------------------------------------------------------------------ generators (keep the input layout)
def relayout(pages, words):
    it = iter(words)
    return [[[[next(it) for _ in l] for l in pa] for pa in pg] for pg in pages]

def flat(pages):
    return [w for pg in pages for pa in pg for l in pa for w in l]

def gen_selfcit(pages, seed):
    import gen
    lines = [{'words': l} for pg in pages for pa in pg for l in pa]
    out = gen.self_citation(lines, seed=seed)
    return relayout(pages, [w for L in out for w in L['words']])

def gen_grille(pages, seed):
    import gen
    lines = [{'words': l} for pg in pages for pa in pg for l in pa]
    out = gen.table_grille(lines, seed=seed)
    return relayout(pages, [w for L in out for w in L['words']])

def gen_markov(pages, seed):
    """word-bigram Markov resynthesis of the whole chunk, poured into the same layout."""
    from collections import defaultdict
    rng = random.Random(seed)
    ws = flat(pages)
    nxt = defaultdict(list)
    for a, b in zip(ws, ws[1:]): nxt[a].append(b)
    out = [rng.choice(ws)]
    while len(out) < len(ws):
        c = nxt.get(out[-1])
        out.append(rng.choice(c) if c and rng.random() < 0.97 else rng.choice(ws))
    return relayout(pages, out)

# ------------------------------------------------------------------ shuffles
def shuffle(pages, scale, rng):
    P = [[[list(l) for l in pa] for pa in pg] for pg in pages]
    ids = list(range(len(P)))
    if scale == 'S0':
        P = [[[[''.join(rng.sample(w, len(w))) for w in l] for l in pa] for pa in pg] for pg in P]
    elif scale == 'S1':
        for pg in P:
            for pa in pg:
                for l in pa: rng.shuffle(l)
    elif scale == 'S1p':
        for pg in P:
            for k, pa in enumerate(pg):
                ws = [w for l in pa for w in l]; rng.shuffle(ws); it = iter(ws)
                pg[k] = [[next(it) for _ in l] for l in pa]
    elif scale == 'S2':
        for pg in P:
            for pa in pg: rng.shuffle(pa)
    elif scale == 'S3':
        for pg in P: rng.shuffle(pg)
    elif scale == 'S4':
        rng.shuffle(ids); P = [P[i] for i in ids]
    return P, ids

# ------------------------------------------------------------------ token tables
def bucket_pos(i, n, cap):
    if n == 1: return cap + 1          # single
    if i == n - 1: return cap         # last
    return min(i, cap - 1)

def token_table(pages, page_ids):
    """columns of per-token attributes; words returned as list."""
    W, cols = [], {k: [] for k in ['fold', 'pg', 'pa', 'li', 'wi', 'prev', 'prevx', 'next', 'fl', 'fp', 'fg',
                                    'lineid', 'paraid', 'pageid', 'posinpage']}
    npg = len(pages); lid = pid = 0
    for gi, pg in enumerate(pages):
        fold = page_ids[gi] % 4
        pgb = min(4, int(5 * gi / npg))
        fg_word = pg[0][0][0] if pg and pg[0] and pg[0][0] else None
        pos_in_page = 0
        for ai, pa in enumerate(pg):
            fp_word = pa[0][0]
            prevx = None
            for li, l in enumerate(pa):
                for wi, w in enumerate(l):
                    W.append(w)
                    cols['fold'].append(fold); cols['pg'].append(pgb)
                    cols['pa'].append(bucket_pos(ai, len(pg), 2)); cols['li'].append(bucket_pos(li, len(pa), 3))
                    cols['wi'].append(bucket_pos(wi, len(l), 4))
                    cols['prev'].append(l[wi - 1] if wi else None)
                    cols['prevx'].append(prevx)
                    cols['next'].append(l[wi + 1] if wi + 1 < len(l) else None)
                    cols['fl'].append(None if wi == 0 else l[0])
                    cols['fp'].append(None if (li == 0 and wi == 0) else fp_word)
                    cols['fg'].append(None if (ai == 0 and li == 0 and wi == 0) else fg_word)
                    cols['lineid'].append(lid); cols['paraid'].append(pid); cols['pageid'].append(gi)
                    cols['posinpage'].append(pos_in_page); pos_in_page += 1
                    prevx = w
                lid += 1
            pid += 1
    return W, cols

# ------------------------------------------------------------------ representations
def make_reps(orig_words, all_words):
    from collections import Counter
    fr = Counter(orig_words)
    rank = {w: i for i, (w, _) in enumerate(fr.most_common())}
    vocab = [None] + sorted(set(w for w in all_words if w is not None))
    reps = {}
    for V in (20, 60, 200, 600):
        reps['id%d' % V] = {w: (rank[w] + 1 if (w in rank and rank[w] < V) else V + 1) if w is not None else 0 for w in vocab}
    for k in (4, 8, 16, 32):
        for s in (1, 2):
            reps['h%d_%d' % (k, s)] = {w: (zlib.crc32(('%d|%s' % (s, w)).encode()) % k) + 1 if w is not None else 0 for w in vocab}
    def strmap(f):
        m, out = {}, {}
        for w in vocab:
            if w is None: out[w] = 0; continue
            key = f(w); out[w] = m.setdefault(key, len(m) + 1)
        return out
    reps['pre1'] = strmap(lambda w: w[:1]); reps['pre2'] = strmap(lambda w: w[:2])
    reps['suf1'] = strmap(lambda w: w[-1:]); reps['suf2'] = strmap(lambda w: w[-2:])
    reps['len'] = strmap(lambda w: min(len(w), 9))
    return reps

REPNAMES = ['id20', 'id60', 'id200', 'id600', 'h4_1', 'h4_2', 'h8_1', 'h8_2', 'h16_1', 'h16_2', 'h32_1', 'h32_2',
            'pre1', 'pre2', 'suf1', 'suf2', 'len']
ATOMS_REP = ['prev', 'prevx', 'next', 'fl', 'fp', 'fg']
ATOMS_POS = ['wi', 'li', 'pa', 'pg']
CTX_REPS = ['id20', 'id60', 'h4_1', 'h8_1', 'h16_2', 'pre1', 'pre2', 'suf1', 'suf2', 'len']
CACHES = ['prevline', 'prevpara', 'prevpage', 'pagesofar', 'parasofar']

def model_types(n=800, seed=71):
    rng = random.Random(seed)
    out = []
    atoms = ATOMS_REP + ATOMS_POS
    for _ in range(int(n * 0.8)):
        tgt = rng.choice(REPNAMES)
        k = 1 if rng.random() < 0.6 else 2
        A = rng.sample(atoms, k)
        ctx = [(a, rng.choice(CTX_REPS) if a in ATOMS_REP else None) for a in A]
        out.append({'kind': 'ctx', 'tgt': tgt, 'ctx': ctx, 'a': rng.choice([1.0, 4.0, 16.0]),
                    'fam': A[0] if k == 1 else '+'.join(sorted(A))})
    for _ in range(int(n * 0.17)):
        out.append({'kind': 'cache', 'tgt': rng.choice(REPNAMES), 'win': rng.choice(CACHES), 'fam': None})
        out[-1]['fam'] = 'cache_' + out[-1]['win']
    for o in range(1, 3):
        for pw in (0, 1):
            out.append({'kind': 'glyph', 'order': o, 'posw': pw, 'a': 2.0, 'fam': 'glyph'})
    return out

# ------------------------------------------------------------------ scoring
def _lookup(train_keys, test_keys):
    u, c = np.unique(train_keys, return_counts=True)
    idx = np.searchsorted(u, test_keys)
    idx[idx >= len(u)] = 0
    hit = (len(u) > 0) & (u[idx] == test_keys) if len(u) else np.zeros(len(test_keys), bool)
    return np.where(hit, c[idx], 0)

def ctx_logp(ctx_tr, t_tr, ctx_te, t_te, R, a):
    """log2 p(t|c) and log2 pu(t) for test tokens; Dirichlet-smoothed toward the unigram."""
    cu = np.bincount(t_tr, minlength=R + 1).astype(float) + 0.5
    pu = cu / cu.sum()
    put = pu[t_te]
    nct = _lookup(ctx_tr * (R + 1) + t_tr, ctx_te * (R + 1) + t_te)
    nc = _lookup(ctx_tr, ctx_te)
    p = (nct + a * put) / (nc + a)
    return np.log2(p), np.log2(put)

def cache_probs(cols, T, win):
    """p_cache(T[i]) = share of T[i] in the cache window; has = window non-empty."""
    from collections import Counter
    n = len(T)
    line, para, page = [cols[k] for k in ('lineid', 'paraid', 'pageid')]
    pc = np.zeros(n); has = np.zeros(n, bool)
    if win in ('prevline', 'prevpara', 'prevpage'):
        g = {'prevline': line, 'prevpara': para, 'prevpage': page}[win]
        groups = {}
        for i in range(n): groups.setdefault(g[i], []).append(i)
        order = sorted(groups)
        for j in range(1, len(order)):
            gp, gid = groups[order[j - 1]], groups[order[j]]
            if win != 'prevpage' and page[gp[0]] != page[gid[0]]: continue
            c = Counter(T[i] for i in gp); tot = len(gp)
            for i in gid: pc[i] = c.get(T[i], 0) / tot; has[i] = True
    else:
        g = page if win == 'pagesofar' else para
        cur, cg, tot = Counter(), None, 0
        for i in range(n):
            if g[i] != cg: cur, cg, tot = Counter(), g[i], 0
            if tot: pc[i] = cur.get(T[i], 0) / tot; has[i] = True
            cur[T[i]] += 1; tot += 1
    return pc, has

def fingerprint(pages, models, seeds=(1, 2), detail=False):
    versions = {'orig': [(pages, list(range(len(pages))))]}
    for s in SCALES:
        versions[s] = [shuffle(pages, s, random.Random(1000 * sd + SCALES.index(s))) for sd in seeds]
    tabs = {}
    allw = []
    for v, lst in versions.items():
        tabs[v] = [token_table(P, ids) for P, ids in lst]
        for W, _ in tabs[v]: allw.extend(W)
    W0, C0 = tabs['orig'][0]
    reps = make_reps(W0, allw)
    # precompute integer columns per version
    def colarr(W, C, name, rep):
        m = reps[rep]
        if name == 'w': return np.array([m[w] for w in W])
        return np.array([m[x] for x in C[name]])
    cache = {}
    def get(vi, k, name, rep):
        key = (vi, k, name, rep)
        if key not in cache:
            W, C = tabs[vi][k]
            if rep is None: cache[key] = np.asarray(C[name])
            else: cache[key] = colarr(W, C, name, rep)
        return cache[key]
    vkeys = ['orig'] + SCALES
    res = []
    for m in models:
        if m['kind'] == 'glyph':
            res.append(glyph_model(tabs, m, vkeys)); continue
        Rn = max(reps[m['tgt']].values()) + 1
        G = {v: [] for v in vkeys}
        for f in range(4):
            outs = {}
            for v in vkeys:
                for k in range(len(tabs[v])):
                    fold = get(v, k, 'fold', None)
                    t = get(v, k, 'w', m['tgt'])
                    if m['kind'] == 'ctx':
                        ctx = np.zeros(len(t), dtype=np.int64)
                        for a, rp in m['ctx']:
                            col = get(v, k, a, rp)
                            ctx = ctx * 1009 + col
                        if v == 'orig' and k == 0:
                            ftr = fold != f
                            tr = (ctx[ftr], t[ftr])
                        te = fold == f
                        lp, lu = ctx_logp(tr[0], tr[1], ctx[te], t[te], Rn, m['a'])
                        G[v].append((lp - lu))
                    else:
                        W, C = tabs[v][k]
                        T = t.tolist()
                        cvkey = (v, k, 'cv', m['win'], m['tgt'])
                        if cvkey not in cache: cache[cvkey] = cache_probs(C, T, m['win'])
                        pc, has = cache[cvkey]
                        if v == 'orig' and k == 0:
                            ftr = fold != f
                            cu = np.bincount(t[ftr], minlength=Rn + 1).astype(float) + 0.5; pu = cu / cu.sum()
                            best, bl = -1e18, 0.0
                            for lam in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5):
                                sc = np.log2(lam * pc[ftr] + (1 - lam) * pu[t[ftr]]).sum()
                                if sc > best: best, bl = sc, lam
                            tr = (pu, bl)
                        pu, bl = tr
                        te = fold == f
                        lp = np.log2(bl * pc[te] + (1 - bl) * pu[t[te]]); lu = np.log2(pu[t[te]])
                        G[v].append(lp - lu)
        g0 = np.concatenate(G['orig']).mean()
        rec = {'G': float(g0)}
        for s in SCALES:
            gs = np.concatenate(G[s]).mean()
            rec[s] = float(g0 - gs)
        res.append(rec)
    return res

def glyph_model(tabs, m, vkeys):
    """glyph-level n-gram inside words (with optional position-in-word context)."""
    def seqs(W, fold):
        ctx, tgt, fd = [], [], []
        for w, f0 in zip(W, fold):
            s = '^' * m['order'] + w + '$'
            for i in range(m['order'], len(s)):
                c = s[i - m['order']:i] + (('%d' % min(i, 5)) if m['posw'] else '')
                ctx.append(zlib.crc32(c.encode()) % 1000003); tgt.append(ord(s[i]) % 4096); fd.append(f0)
        return np.array(ctx, dtype=np.int64), np.array(tgt, dtype=np.int64), np.array(fd)
    G = {v: [] for v in vkeys}
    S = {v: [seqs(W, C['fold']) for W, C in tabs[v]] for v in vkeys}
    for f in range(4):
        c0, t0, f0 = S['orig'][0]
        ctr, ttr = c0[f0 != f], t0[f0 != f]
        for v in vkeys:
            for c1, t1, f1 in S[v]:
                lp, lu = ctx_logp(ctr, ttr, c1[f1 == f], t1[f1 == f], 4096, m['a'])
                G[v].append(lp - lu)
    g0 = np.concatenate(G['orig']).mean()
    rec = {'G': float(g0)}
    for s in SCALES: rec[s] = float(g0 - np.concatenate(G[s]).mean())
    return rec

def summarise(res, models):
    """fingerprint {scale|family: mean held-out damage in bits/token over the family's models};
    scale|ALLrel = sum of damage / sum of gain over models with positive gain."""
    from collections import defaultdict
    num, cnt = defaultdict(float), defaultdict(int)
    rn, rd = defaultdict(float), 0.0
    for r, m in zip(res, models):
        fams = m['fam'].split('+') if m['kind'] == 'ctx' else [m['fam']]
        for fam in fams + ['ALL']:
            cnt[fam] += 1
            for s in SCALES: num[(s, fam)] += r[s]
        if r['G'] > 0:
            rd += r['G']
            for s in SCALES: rn[s] += r[s]
    out = {'%s|%s' % k: num[k] / cnt[k[1]] for k in num}
    for s in SCALES: out['%s|ALLrel' % s] = rn[s] / rd if rd > 0 else 0.0
    return out

# ------------------------------------------------------------------ partial shuffles (cycle 2: simulate a scrambling writer)
def partial_shuffle(pages, fr, rng):
    """fr: {scale: fraction of units whose children are permuted}. Applied top-down
    (S4 pages, S3 paras, S2 lines, S1 words); S0 not used."""
    P = [[[list(l) for l in pa] for pa in pg] for pg in pages]
    if fr.get('S4', 0) > 0:
        idx = [i for i in range(len(P)) if rng.random() < fr['S4']]
        sub = idx[:]; rng.shuffle(sub)
        Q = list(P)
        for a, b in zip(idx, sub): Q[a] = P[b]
        P = Q
    for pg in P:
        if rng.random() < fr.get('S3', 0): rng.shuffle(pg)
        for pa in pg:
            if rng.random() < fr.get('S2', 0): rng.shuffle(pa)
            for l in pa:
                if rng.random() < fr.get('S1', 0): rng.shuffle(l)
    return P
