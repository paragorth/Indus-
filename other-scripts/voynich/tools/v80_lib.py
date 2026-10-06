"""v80 THE PAGES ARE TABLES WRITTEN AS LINES: shared helpers.

Idea: if Voynich tokens are values (v78, v75), a page may be a table written out row by row: each line a row, each
word position (or x-position) a column, and the values of one column drawn from a column-specific stock or stepping
down the column. Search thousands of random table hypotheses (column count, alignment by word index / from the right /
modulo / physical x / two-line rows, table unit = page / paragraph / 8-line block / half page, token representation)
and score how much the PAGE-LOCAL column identity predicts held-out tokens beyond: the global unigram, the global
position-in-line law (word index and the hypothesis' own column, both global), the page's own vocabulary and the
line's own vocabulary (line mode). Extra terms of the table model: page-local column law (Pcol) and the token
directly above in the same column (Pvert, a learned stepping rule).

Controls: real tables (Julian kalendar with computus columns and feast names from the OCR of Hampson's Medii aevi
kalendarium, Alfonsine mean-motion tables with the Alfonsine parameters, planetary-hour and unequal-hour tables,
an index/concordance of Isidore, a recipe dosage table on Antidotarium ingredients) written row by row through the v72
surface machinery (merge code + planted surface); prose (Isidore, Brumati, German) through the same machinery;
generators fitted to ZL3b. Everything is scored after the frozen v72 E1c extraction.

Corpus format as v72: list of pages; page = dict(id, sec, lines=[dict(w=[words], ps=bool)]).
"""
import os, sys, json, math, random, re, pickle, zlib, unicodedata
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v80_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V72

E1C = V72.RULES['E1c_keepd']


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def psave(name, obj):
    pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    p = os.path.join(CK, name)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def half_of(pid):
    m = re.match(r'f(\d+)', pid)
    return int(m.group(1)) % 2 if m else zlib.crc32(pid.encode()) % 2


# ------------------------------------------------------------------ real tables (plain, one row = one line)
ROMAN = [(1000, 'm'), (900, 'cm'), (500, 'd'), (400, 'cd'), (100, 'c'), (90, 'xc'), (50, 'l'), (40, 'xl'),
         (10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iiii'), (1, 'i')]


def rom(n):
    if n == 0: return 'n'
    s = ''
    for v, r in ROMAN:
        while n >= v: s += r; n -= v
    return s


def _pages(rows, prefix, per_page, sec_fn=lambda i: 'T', para_every=None):
    pages = []
    for i in range(0, len(rows), per_page):
        rr = rows[i:i + per_page]
        lines = [dict(w=list(r), ps=(j == 0 or (para_every and j % para_every == 0))) for j, r in enumerate(rr) if r]
        pages.append(dict(id='%s%03d' % (prefix, len(pages)), sec=sec_fn(i), lang='-', hand='-', quire='-', lines=lines))
    return pages


def _norm(s):
    s = unicodedata.normalize('NFD', s.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.findall(r'[a-z]+', s)


def hampson_verses():
    fn = os.path.join(CK, 'dl', 'mediiaevikalend01hampgoog.txt')
    out = []
    for raw in open(fn, encoding='utf-8', errors='ignore'):
        s = raw.strip()
        toks = s.split()
        if not (4 <= len(toks) <= 10): continue
        if not s[:1].isupper(): continue
        al = [t for t in toks if re.fullmatch(r'[A-Za-z]+[.,;:]?', t)]
        if len(al) < 0.8 * len(toks): continue
        ws = _norm(s)
        if len(ws) >= 4: out.append(ws)
    return out


MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
NONES = [5, 5, 7, 5, 7, 5, 7, 5, 5, 7, 5, 5]
MNAME = ['ian', 'feb', 'mar', 'apr', 'mai', 'iun', 'iul', 'aug', 'sep', 'oct', 'nou', 'dec']


def roman_date(m, d):
    """Roman date of day d (1-based) of month m: [count, 'non'/'id'/'kl'] or ['kl'] etc."""
    n, idus = NONES[m], NONES[m] + 8
    if d == 1: return ['kl', MNAME[m]]
    if d < n: return ([rom(n - d + 1), 'non'] if n - d + 1 > 2 else ['pridie', 'non'])
    if d == n: return ['nonas']
    if d < idus: return ([rom(idus - d + 1), 'id'] if idus - d + 1 > 2 else ['pridie', 'id'])
    if d == idus: return ['idus']
    nd = MONTHS[m]; k = nd - d + 2
    return [rom(k), 'kl'] if k > 2 else ['pridie', 'kl']


def kalendar_rows(years=4):
    import v79_lib
    gn = v79_lib.golden_numbers()
    verses = hampson_verses()
    rows, vi = [], 0
    for y in range(years):
        day = 0
        for m, nd in enumerate(MONTHS):
            for d in range(1, nd + 1):
                r = []
                if gn[day] is not None: r.append(rom(gn[day]))
                r.append('abcdefg'[day % 7])
                r += roman_date(m, d)
                v = verses[vi % len(verses)]; vi += 1
                r += v[:2 + (vi % 2)]        # feast name: 2-3 words of a real kalendar verse
                rows.append(r); day += 1
    return rows


def kalendar_pages():
    rows = kalendar_rows()
    return _pages(rows, 'kal', 16, sec_fn=lambda i: 'K%d' % ((i // 91) % 4))


def _sexa(x):
    """degrees -> [signs of 30, deg, min, sec]"""
    x = x % 360.0
    s = int(x // 30); r = x - 30 * s
    d = int(r); r = (r - d) * 60; mi = int(r); r = (r - mi) * 60; se = int(r)
    return [rom(s), rom(d), rom(mi), rom(se)]


SUN_D = 0.98564734      # Alfonsine mean motion of the sun per day (0;59,8,19,37,19,13,56 deg)
MOON_D = 13.17639653    # mean moon per day
ANOM_D = 13.06497701    # mean lunar anomaly per day


def alfonsine_pages():
    rows = []
    for unit, ns in (('dies', list(range(1, 61))), ('menses', list(range(1, 13))), ('anni', list(range(1, 61))),
                     ('anni', [20 * k for k in range(1, 61)])):
        mult = {'dies': 1, 'menses': 30.4375, 'anni': 365.25}[unit]
        for n in ns:
            t = n * mult
            rows.append([rom(n)] + _sexa(SUN_D * t) + _sexa(MOON_D * t) + _sexa(ANOM_D * t))
    return _pages(rows, 'alf', 20, sec_fn=lambda i: 'A%d' % min(3, i // 60))


CHALD = ['saturnus', 'iupiter', 'mars', 'sol', 'uenus', 'mercurius', 'luna']
DAYSTART = [3, 6, 2, 5, 1, 4, 0]
SIGNS = ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'uirgo', 'libra', 'scorpio', 'sagittarius', 'capricornus',
         'aquarius', 'pisces']


def hours_pages():
    rows = []
    for clime in range(7):          # one copy per clime, as in tables of planetary hours with hour lengths
        for h in range(24):
            rows.append([rom(h % 12 + 1), 'diei' if h < 12 else 'noctis'] +
                        [CHALD[(DAYSTART[d] + h) % 7] for d in range(7)])
        phi = math.radians(16 + 6 * clime)
        for s in range(12):
            for dg in range(0, 30, 3):
                lam = math.radians(s * 30 + dg)
                dec = math.asin(math.sin(math.radians(23.5)) * math.sin(lam))
                x = -math.tan(phi) * math.tan(dec); x = max(-1, min(1, x))
                dayh = 2 * math.degrees(math.acos(x)) / 12.0     # unequal day hour in degrees of equator
                nh = 30 - dayh
                rows.append([SIGNS[s], rom(dg)] + [rom(int(dayh)), rom(int((dayh % 1) * 60))] +
                            [rom(int(nh)), rom(int((nh % 1) * 60))])
    return _pages(rows, 'hor', 24, sec_fn=lambda i: 'H%d' % min(6, i // 144))


def concordance_pages():
    P = V72.isidore_plain()
    occ = defaultdict(list)
    for pi, p in enumerate(P):
        for li, l in enumerate(p['lines']):
            for w in l['w']: occ[w].append((pi // 10 + 1, pi % 10 + 1, li + 1))
    rows = []
    for w in sorted(occ):
        o = occ[w]
        if len(o) < 3 or len(w) < 4: continue
        rng = random.Random(zlib.crc32(w.encode()))
        refs = sorted(rng.sample(o, min(len(o), rng.choice([1, 2, 3, 4]))))
        r = [w]
        for b, c, l in refs: r += [rom(b), rom(c)]
        rows.append(r)
    return _pages(rows, 'con', 20, sec_fn=lambda i: 'C%d' % min(5, i // 400))


def dosage_pages():
    import v78_corpora as C78
    ents = C78.v75_entries('ING_ANTID') + C78.v75_entries('ING_APIC')
    rng = random.Random(801)
    rows = []
    for _, ws in ents:
        ing = [w for w in ws if len(w) > 3]
        for i in range(0, len(ing) - 1, 2):
            r = ing[i:i + 2]
            u = rng.choices(['unc', 'dr', 'scr', 'ana', 'lb'], [0.3, 0.35, 0.15, 0.15, 0.05])[0]
            q = rng.choices([1, 2, 3, 4, 5, 6, 8, 10, 12], [0.25, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04])[0]
            r += [u, rom(q)]
            if rng.random() < 0.3: r += ['et', 'semis']
            rows.append(r)
    return _pages(rows, 'dos', 20, sec_fn=lambda i: 'D%d' % min(3, i // 500))


def degrade(pages, seed, p_drop=0.15):
    """table copied carelessly: each cell dropped with p_drop (blank cells vanish), as when empty cells are not written."""
    rng = random.Random(seed)
    out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            ws = [w for w in l['w'] if rng.random() >= p_drop] or l['w'][:1]
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


def through_surface(pages, seed):
    words = [w for p in pages for l in p['lines'] for w in l['w']]
    code = V72.payload_code(words, seed=seed, mode='merge')
    pay = V72.encode_payload(pages, code)
    surf = V72.surface(pay, seed=seed + 1)
    for p in surf:
        for l in p['lines']: l.pop('orig', None)
    return surf


def shuffle_interior(pages, seed):
    """null: words at interior positions shuffled inside each line; first and last word kept in place."""
    rng = random.Random(seed)
    out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            ws = list(l['w'])
            if len(ws) > 3:
                mid = ws[1:-1]; rng.shuffle(mid); ws = [ws[0]] + mid + [ws[-1]]
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


def shuffle_lines(pages, seed):
    rng = random.Random(seed)
    out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            ws = list(l['w']); rng.shuffle(ws); nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


# ------------------------------------------------------------------ flattening
REPRS = ('full', 'pre2', 'suf')


def flatten(pages):
    """pages (surface) -> token table after E1c extraction. Returns dict of numpy arrays."""
    X = V72.extract(pages, E1C)
    rec = defaultdict(list)
    line_id = 0; para_id = -1
    for pi, (p, q) in enumerate(zip(pages, X)):
        h = half_of(p['id'])
        nlines = len(q['lines'])
        for li, (l0, l) in enumerate(zip(p['lines'], q['lines'])):
            if l['ps'] or li == 0: para_id += 1
            ws = l['w']; n = len(ws)
            x = 0
            for i, w in enumerate(ws):
                rec['page'].append(pi); rec['line'].append(line_id); rec['para'].append(para_id)
                rec['lip'].append(li); rec['nlp'].append(nlines)
                rec['i'].append(i); rec['n'].append(n); rec['x'].append(x)
                rec['tok'].append(w); rec['half'].append(h)
                x += len(l0['w'][i]) + 1
            rec['_xl'] += [x] * n
            line_id += 1
    T = {k: np.array(v) for k, v in rec.items() if k != 'tok'}
    T['xr'] = T['_xl'] - T['x']
    toks = rec['tok']
    for r in REPRS:
        if r == 'full': f = toks
        elif r == 'pre2': f = [t[:2] for t in toks]
        else: f = [t[-1:] for t in toks]
        u = {}
        T['r_' + r] = np.array([u.setdefault(t, len(u)) for t in f])
    return T


# ------------------------------------------------------------------ scoring
def _cnt(keys):
    """count of each element's key among all elements."""
    u, inv, c = np.unique(keys, return_inverse=True, return_counts=True)
    return c[inv].astype(float)


def em_weights(P, iters=40):
    k = P.shape[1]; w = np.full(k, 1.0 / k)
    for _ in range(iters):
        R = P * w; s = R.sum(1, keepdims=True); R /= s
        w = R.mean(0)
    return w


def columns(T, align, C, w=None, off=0):
    i, n = T['i'], T['n']
    if align == 'L': return np.minimum(i, C - 1)
    if align == 'R': return np.minimum(n - 1 - i, C - 1)
    if align == 'M': return i % C
    if align == 'X': return np.minimum((T['x'] + off) // w, C - 1)
    if align == 'XR': return np.minimum((T['xr'] + off) // w, C - 1)
    if align == 'P2':
        return (T['lip'] % 2) * C + np.minimum(i, C - 1)
    raise ValueError(align)


def units(T, unit):
    if unit == 'page': return T['page']
    if unit == 'para': return T['para']
    if unit == 'blk8': return T['page'] * 100 + T['lip'] // 8
    if unit == 'half': return T['page'] * 2 + (2 * T['lip'] >= T['nlp']).astype(int)
    raise ValueError(unit)


def base_terms(T, rep):
    """hypothesis-independent terms: global unigram (LOO, smoothed), global word-index law, page vocab, line vocab."""
    t = T['r_' + rep]; N = len(t); V = t.max() + 1
    nt = _cnt(t)
    pglob = (nt - 1 + 0.5) / (N - 1 + 0.5 * V)
    ipos = np.where(T['i'] == T['n'] - 1, 3, np.minimum(T['i'], 2))   # edge law: first, second, last, interior
    k = ipos * V + t
    pip = np.maximum(_cnt(k) - 1, 0) / np.maximum(_cnt(ipos) - 1, 1)
    # page vocabulary without the own line
    pg = T['page'].astype(np.int64)
    ln = T['line'].astype(np.int64)
    n_pt = _cnt(pg * V + t); n_lt = _cnt(ln * V + t)
    n_p = _cnt(pg); n_l = _cnt(ln)
    ppage = np.where(n_p - n_l > 0, (n_pt - n_lt) / np.maximum(n_p - n_l, 1), 0)
    pline = np.where(n_l > 1, (n_lt - 1) / np.maximum(n_l - 1, 1), 0)
    return np.stack([pglob, pip, ppage, pline], 1)


def hyp_terms(T, rep, col, unit_ids):
    t = T['r_' + rep].astype(np.int64); V = int(t.max() + 1)
    col = col.astype(np.int64); u = unit_ids.astype(np.int64); ln = T['line'].astype(np.int64)
    NC = int(col.max() + 1)
    # global column law (LOO)
    pgc = np.maximum(_cnt(col * V + t) - 1, 0) / np.maximum(_cnt(col) - 1, 1)
    # page-local (unit-local) column law without the own line
    uc = u * NC + col
    n_uct = _cnt(uc * V + t); n_uc = _cnt(uc)
    lc = ln * NC + col
    n_lct = _cnt(lc * V + t); n_lc = _cnt(lc)
    den = n_uc - n_lc
    pcol = np.where(den > 0, (n_uct - n_lct) / np.maximum(den, 1), 0)
    # token above in the same column (previous line of the same unit): first token of (line-1, col)
    key = ln * NC + col
    order = np.lexsort((np.arange(len(t)), key))
    ks = key[order]
    first = np.ones(len(ks), bool); first[1:] = ks[1:] != ks[:-1]
    fk = ks[first]; ft = t[order][first]; fu = u[order][first]
    q = (ln - 1) * NC + col
    pos = np.searchsorted(fk, q)
    pos_c = np.minimum(pos, len(fk) - 1)
    has = (fk[pos_c] == q) & (fu[pos_c] == u)
    above = np.where(has, ft[pos_c], -1)
    pair = np.where(has, (above + 1) * V + t, -1)
    n_pair = _cnt(pair); n_ab = _cnt(np.where(has, above, -1))
    pvert = np.where(has & (n_ab > 1), np.maximum(n_pair - 1, 0) / np.maximum(n_ab - 1, 1), 0)
    return np.stack([pgc, pcol, pvert], 1)


def score(T, rep, col, unit_ids, base=None, masks=None, full=False):
    """Nested held-out ladder (weights fitted by EM on the discovery half, bits/token on the holdout half):
    A0 = global unigram + edge law (first/second/last/interior) + page vocab + line vocab (line mode)
    A1 = A0 + global column law of the hypothesis      -> g_glob (column identity beyond line edges)
    A2 = A1 + page(unit)-local column law               -> g_loc  (column-specific value sets per table)
    A3 = A2 + token above in the same column            -> g_vert (stepping / vertical rule)
    total = A3 - A0. Also the same on the discovery half (in-sample, for ranking)."""
    if base is None: base = base_terms(T, rep)
    H = hyp_terms(T, rep, col, unit_ids)
    d = T['half'] == 0; h = ~d
    M = np.concatenate([base, H], 1)       # cols: glob, edge, page, line, gcol, pcol, pvert
    L = []
    for k in (4, 5, 6, 7):
        X = M[:, :k]
        w = em_weights(X[d], 25)
        L.append(np.log2(X @ w))
    res = dict(total=float((L[3] - L[0])[h].mean()), g_glob=float((L[1] - L[0])[h].mean()),
               g_loc=float((L[2] - L[1])[h].mean()), g_vert=float((L[3] - L[2])[h].mean()),
               disc=float((L[3] - L[0])[d].mean()))
    if masks:
        for mn, m in masks.items():
            res['total_' + mn] = float((L[3] - L[0])[h & m].mean())
            res['loc_' + mn] = float((L[3] - L[1])[h & m].mean())
    return res


def interior_mask(T):
    return (T['i'] > 0) & (T['i'] < T['n'] - 1)


ALIGNS = ('L', 'R', 'M', 'X', 'XR', 'P2')
UNITS = ('page', 'para', 'blk8', 'half')


def random_hyps(n, seed):
    rng = random.Random(seed)
    H = []
    for _ in range(n):
        a = rng.choice(ALIGNS)
        h = dict(align=a, C=rng.randint(2, 14), unit=rng.choice(UNITS), rep=rng.choice(REPRS))
        if a in ('X', 'XR'):
            h['w'] = rng.randint(3, 14); h['off'] = rng.randint(0, h['w'] - 1)
        if a == 'P2': h['C'] = rng.randint(2, 10)
        H.append(h)
    return H


def hyp_key(h):
    return '%s/C%d/%s/%s/w%s/o%s' % (h['align'], h['C'], h['unit'], h['rep'], h.get('w', '-'), h.get('off', '-'))


def run_hyps(T, H, masks=True, full=False):
    bases = {r: base_terms(T, r) for r in REPRS}
    mk = dict(int=interior_mask(T)) if masks else None
    out = []
    for h in H:
        col = columns(T, h['align'], h['C'], h.get('w'), h.get('off', 0))
        u = units(T, h['unit'])
        r = score(T, h['rep'], col, u, base=bases[h['rep']], masks=mk, full=full)
        r.update(h); out.append(r)
    return out


def corpora(which='all'):
    """build (or load) all corpora used by v80. Returns dict name -> (kind, pages)."""
    C = pload('corpora.pkl')
    if C is not None: return C
    import v77_lib as V77
    C = {}
    Z = V72.voynich('ZL3b')
    C['VOY_ZL'] = ('VOY', Z)
    C['VOY_IT'] = ('VOY', V72.voynich('IT2a'))
    tabs = dict(T_KAL=kalendar_pages(), T_ALF=alfonsine_pages(), T_HOR=hours_pages(), T_CON=concordance_pages(),
                T_DOS=dosage_pages())
    for i, (k, P) in enumerate(tabs.items()):
        C[k] = ('TABLE', through_surface(P, 8000 + 10 * i))
        C[k + 'd'] = ('TABLE_DEG', through_surface(degrade(P, 8100 + i), 8001 + 10 * i))
    C['P_ISID'] = ('PROSE', through_surface(V72.isidore_plain(), 8201))
    C['P_BRUM'] = ('PROSE', through_surface(V72.brumati_plain(), 8202))
    C['P_GERM'] = ('PROSE', through_surface(V72.german_plain(), 8203))
    for g, f in V77.GENS.items():
        C['G_' + g] = ('GEN', V77._ungroup(f(V77._grouped(Z, V77.gkey), seed=8301)))
    C['G_WSHUF'] = ('GEN', V72.gen_wshuf(Z, seed=8302))
    psave('corpora.pkl', C)
    return C


TRUTH = {'T_KAL': 'variable 5-7 cells (golden no. optional, letter, Roman date 1-2, feast 2-3)',
         'T_ALF': '13 fixed cells', 'T_HOR': '9 cells (planetary) / 6 cells (hour lengths)',
         'T_CON': 'headword + 1-4 (book, chapter) pairs', 'T_DOS': '4-6 cells (2 ingredients, unit, qty, [et semis])'}


# ------------------------------------------------------------------ cycle 2: the row as a monotone column HMM
def line_batches(T, rep, mask_lines=None):
    """padded (L, maxlen) token matrix per line, plus half and page per line."""
    t = T['r_' + rep]; ln = T['line']
    nL = ln.max() + 1
    lens = np.bincount(ln, minlength=nL)
    ml = lens.max()
    M = np.full((nL, ml), -1, np.int64)
    pos = T['i']
    M[ln, pos] = t
    half = np.zeros(nL, int); half[ln] = T['half']
    page = np.zeros(nL, int); page[ln] = T['page']
    return M, lens, half, page


class RowHMM:
    """K column states, left-to-right: from column j the next token is in column j (cell continues, prob s_j) or in a
    later column j' > j (skipped cells vanish), with a learned jump law; emissions per column, smoothed to the
    global unigram. K=1 is the unigram. Fitted by EM (scaled forward-backward, batched over lines)."""

    def __init__(self, K, V, pglob, seed=0, lam=0.7):
        rng = np.random.default_rng(seed)
        self.K, self.V = K, V
        self.pg = pglob
        self.lam = lam
        self.pi = np.ones(K) / K
        A = np.triu(rng.random((K, K)) + 0.5)
        if K > 1: A[np.arange(K), np.arange(K)] += 1.0
        self.A = A / A.sum(1, keepdims=True)
        E = rng.random((K, V)) + 0.5
        self.E = E / E.sum(1, keepdims=True)

    def emis(self, M):
        Mc = np.where(M < 0, 0, M)
        P = self.lam * self.E[:, Mc] + (1 - self.lam) * self.pg[Mc][None]   # K, L, T
        P = np.transpose(P, (1, 2, 0))
        P[M < 0] = 1.0
        return P

    def fb(self, M, lens):
        nL, Tm = M.shape; K = self.K
        B = self.emis(M)
        valid = M >= 0
        al = np.zeros((nL, Tm, K)); c = np.ones((nL, Tm))
        a = self.pi[None] * B[:, 0]
        c[:, 0] = a.sum(1); al[:, 0] = a / c[:, 0:1]
        for t in range(1, Tm):
            a = (al[:, t - 1] @ self.A) * B[:, t]
            s = a.sum(1)
            s = np.where(valid[:, t], s, 1.0)
            a = np.where(valid[:, t][:, None], a / s[:, None], al[:, t - 1])
            c[:, t] = s; al[:, t] = a
        ll = np.log2(np.where(valid, c, 1.0))
        return al, c, B, valid, ll

    def loglik(self, M, lens):
        return self.fb(M, lens)[4]

    def fit(self, M, lens, iters=25):
        nL, Tm = M.shape; K = self.K
        for _ in range(iters):
            al, c, B, valid, ll = self.fb(M, lens)
            be = np.ones((nL, Tm, K))
            for t in range(Tm - 2, -1, -1):
                b = (self.A @ (B[:, t + 1] * be[:, t + 1]).T).T / c[:, t + 1:t + 2]
                be[:, t] = np.where(valid[:, t + 1][:, None], b, be[:, t + 1] if t + 1 < Tm else 1)
            g = al * be
            g /= g.sum(2, keepdims=True)
            g[~valid] = 0
            # transitions
            xi = np.zeros((K, K))
            for t in range(Tm - 1):
                m = valid[:, t + 1]
                if not m.any(): continue
                x = al[m, t][:, :, None] * self.A[None] * (B[m, t + 1] * be[m, t + 1])[:, None, :] / c[m, t + 1][:, None, None]
                xi += x.sum(0)
            A = np.triu(xi) + 1e-3 * np.triu(np.ones((K, K)))
            self.A = A / A.sum(1, keepdims=True)
            pi = g[:, 0].sum(0) + 1e-3; self.pi = pi / pi.sum()
            # emissions: responsibility of the column-specific part
            Mc = np.where(M < 0, 0, M)
            Ek = self.lam * self.E[:, Mc].transpose(1, 2, 0)
            r = g * Ek / np.maximum(B, 1e-300)
            E = np.zeros((K, self.V))
            for k in range(K):
                E[k] = np.bincount(Mc[valid], weights=r[..., k][valid], minlength=self.V)
            E += 1e-3
            self.E = E / E.sum(1, keepdims=True)
        return self


def edge_law_ll(T, rep, train_mask, test_mask):
    """held-out bits/token of the edge law (first/second/last/interior), add-0.5 smoothed, trained on train_mask."""
    t = T['r_' + rep]; V = t.max() + 1
    e = np.where(T['i'] == T['n'] - 1, 3, np.minimum(T['i'], 2))
    C = np.zeros((4, V)) + 0.5
    np.add.at(C, (e[train_mask], t[train_mask]), 1)
    P = C / C.sum(1, keepdims=True)
    return np.log2(P[e[test_mask], t[test_mask]])


# ------------------------------------------------------------------ cycle 3: the table written column by column
def transposed_pages(plain_pages, block=8, line_w=8, prefix='tr'):
    """take a real table (one row per line) and write it the other way round: blocks of `block` rows; for each
    column c the block's values of that column are written as one line (wrapping at line_w words). A page holds a
    few blocks. The line-type sequence then repeats with period = number of lines per block."""
    rows = [l['w'] for p in plain_pages for l in p['lines']]
    pages, cur = [], []
    for b0 in range(0, len(rows), block):
        B = rows[b0:b0 + block]
        nc = max(len(r) for r in B)
        for c in range(nc):
            vals = [r[c] for r in B if len(r) > c]
            for i in range(0, len(vals), line_w):
                cur.append(dict(w=vals[i:i + line_w], ps=(c == 0 and i == 0)))
        if len(cur) >= 24:
            pages.append(dict(id='%s%03d' % (prefix, len(pages)), sec='T', lang='-', hand='-', quire='-', lines=cur)); cur = []
    if cur: pages.append(dict(id='%s%03d' % (prefix, len(pages)), sec='T', lang='-', hand='-', quire='-', lines=cur))
    return pages


def line_vectors(T, rep, idf=True, skip_first=False):
    """sparse line x type matrix (scipy) of counts (tf-idf if idf); returns matrix, line->page, line->para, line index in page, half."""
    from scipy import sparse
    t = T['r_' + rep]; ln = T['line']
    m = np.ones(len(t), bool)
    if skip_first: m &= T['i'] > 0
    nL = ln.max() + 1; V = t.max() + 1
    X = sparse.csr_matrix((np.ones(m.sum()), (ln[m], t[m])), shape=(nL, V))
    if idf:
        df = np.bincount(t[m], minlength=V) if False else np.asarray((X > 0).sum(0)).ravel()
        w = np.log((nL + 1) / (df + 1)) + 0.0
        X = X @ sparse.diags(w)
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); nrm[nrm == 0] = 1
    X = sparse.diags(1 / nrm) @ X
    page = np.zeros(nL, int); page[ln] = T['page']
    para = np.zeros(nL, int); para[ln] = T['para']
    lip = np.zeros(nL, int); lip[ln] = T['lip']
    half = np.zeros(nL, int); half[ln] = T['half']
    return X.tocsr(), page, para, lip, half


def lag_profile(S, groups, maxlag=14, order=None):
    """mean cosine similarity of line pairs at lag g inside groups (lines in written order or in `order`)."""
    sums = np.zeros(maxlag + 1); cnts = np.zeros(maxlag + 1)
    for g in groups:
        idx = g if order is None else order[g]
        if len(idx) < 3: continue
        sub = S[idx][:, idx]
        n = len(idx)
        for k in range(1, min(maxlag, n - 1) + 1):
            dgn = np.diagonal(sub, k)
            sums[k] += dgn.sum(); cnts[k] += len(dgn)
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan), cnts
