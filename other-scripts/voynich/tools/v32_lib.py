"""v32 THE CALENDAR BEAT: shared library.

Question: do the entries of the starred-paragraph section (quire 20, f103r-f116r) repeat any feature at a calendar
period (7 planets/days, 12 signs, 27/28 lunar mansions, 29.5/30 lunar or civil month, 36 decans, 365-folds)?

Units
  star units  : a new unit starts at every text line preceded by a star comment in ZL3b, or at a ZL paragraph start
  para units  : ZL paragraph starts only
Every unit carries its star attributes (points, dark, dotted, tail) when one is attached.

Statistics, for a sequence x_0..x_{N-1} and a trial period P (non-integer allowed):
  FOLD  epoch folding: phase bin b_i = floor(K * frac(i/P)), K = round(P) bins; one-way ANOVA F of x over bins
        (numeric), or G statistic of the bin x category table (categorical). Equivalent to a rigid circular HMM.
  COMB  autocorrelation comb: mean ACF at lags P,2P,3P (rounded) minus mean ACF at lags kP +- 1, 2 (drift-matched)
        (numeric), or same-value rate at the comb lags minus at the neighbour lags (categorical), or mean cosine
        similarity of vocabulary vectors (multivariate).
  CHMM  circular HMM with slips (advance 1 / stay / skip 1), categorical emissions, Baum-Welch; gain in log-lik
        per unit over the iid model (cycle 2).
Numeric series are detrended with a degree-4 Legendre fit (removes the slow v19-type drift, keeps periods < ~60).
Family-wise correction: each (feature, statistic, P) value is z-scored against its own null distribution, then the
maximum z over the family is compared with the maxima of the null replicates themselves.
"""
import os, re, sys, math, json, random
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'v32_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = os.environ.get('V32_SCR', CK)

PERIODS = np.array(sorted(set([round(x, 2) for x in np.arange(2, 40.01, 0.5)] + [27.32, 29.53, 30.44, 36.5])))
CAL = {7: 'planets/days', 12: 'signs/months', 27.32: 'sidereal month', 28: 'lunar mansions', 29.53: 'synodic month',
       30: 'civil month', 30.44: 'mean month (365/12)', 36: 'decans', 36.5: '365/10'}

GALL = set('ktpf')


# ------------------------------------------------------------------ parsing
def _clean(txt):
    txt = re.sub(r'<[^>]*>', '', txt)
    txt = re.sub(r'\{[^}]*\}', 'x', txt)
    txt = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', txt)
    ws = re.split(r'[.,\s]+', txt)
    return [w for w in ws if w and '?' not in w and re.fullmatch(r'[a-z\']+', w)]


def _star_attr(c):
    c = c.lower()
    m = re.search(r'(\d+)\??\s*points', c)
    pts = int(m.group(1)) if m else None
    dark = 1 if re.search(r'\bdark', c) else 0
    return {'pts': pts, 'dark': dark, 'dotted': int('dotted' in c), 'tail': int('tail' in c and 'tail?' not in c)}


def load_q20(name='ZL3b', mode='star', folios=None):
    """Units of quire 20 in manuscript order. mode 'star' or 'para'. Star info from ZL3b only."""
    path = os.path.join(DATA, name + '-n.txt')
    L = open(path, encoding='utf-8', errors='replace').read().split('\n')
    inq, pending, units, f = False, None, [], None
    for ln in L:
        m = re.match(r'<(f\d+[rv]\d?)>', ln)
        if m:
            f = m.group(1)
            inq = ('$I=S' in ln) and not f.startswith('f58')
            pending = None
            continue
        if not inq: continue
        if ln.startswith('#'):
            if re.search(r'star', ln, re.I) and 'text only' not in ln: pending = ln
            continue
        m = re.match(r'<(f\d+[rv]\d?)\.(\d+[a-z]?),([@+=*&~]?)(\w+)>\s*(.*)', ln)
        if not m: continue
        txt = m.group(5)
        ps = m.group(3) == '@' or '<%>' in txt
        new = ps or (mode == 'star' and pending is not None)
        ws = _clean(txt)
        if new or not units:
            u = {'f': m.group(1), 'line0': m.group(2), 'zlpara': ps, 'star': _star_attr(pending) if pending else None,
                 'lines': []}
            units.append(u)
        if ws: units[-1]['lines'].append(ws)
        pending = None
    units = [u for u in units if sum(len(l) for l in u['lines']) >= 3]
    if folios: units = [u for u in units if u['f'] in folios]
    return units


def load_q20_it(mode='para'):
    """IT2a has no star comments: paragraph units only."""
    return load_q20('IT2a', mode='para')


def load_section(illus, lang=None, name='ZL3b', minw=8):
    """Non-starred comparison sections from the derived line file: ZL paragraphs in order."""
    d = json.load(open(os.path.join(DATA, 'derived', name + '_lines.json')))
    units, cur = [], None
    for r in d:
        if r['ltype'] != 'P' or r['illus'] != illus or (lang and r['lang'] != lang):
            cur = None if r['ltype'] == 'P' else cur
            continue
        ws = [w for w, u in zip(r['words'], r['uncertain']) if not u and '?' not in w]
        if r['para_start'] or cur is None:
            cur = {'f': r['folio'], 'lines': [], 'star': None, 'zlpara': True}; units.append(cur)
        if ws: cur['lines'].append(ws)
        if r['para_end']: cur = None
    return [u for u in units if sum(len(l) for l in u['lines']) >= minw]


# ------------------------------------------------------------------ features
def _gl(w):
    for a, b in (('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')):
        w = w.replace(a, b)
    return w


def first_class(w):
    g = _gl(w)
    c = g[0]
    if c in 'kt': return 'kt'
    if c in 'pf': return 'pf'
    if c in 'KTPF': return 'bench-g'
    if c == 'q': return 'q'
    if c in 'CS': return 'ch/sh'
    if c == 'o': return 'o'
    if c in 'dsy': return c
    return 'other'


def features(units, star=True):
    """-> num: dict name -> float array; cat: dict name -> list of labels; vec: (N, V) vocab count matrix."""
    num, cat = {}, {}
    W = [[w for l in u['lines'] for w in l] for u in units]
    N = len(units)
    num['log_nwords'] = np.log([len(w) for w in W])
    num['nlines'] = np.array([len(u['lines']) for u in units], float)
    num['mean_wlen'] = np.array([np.mean([len(_gl(x)) for x in w]) for w in W])
    num['gallows_pw'] = np.array([np.mean([sum(c in GALL for c in x) for x in w]) for w in W])
    num['pf_pw'] = np.array([np.mean([sum(c in 'pf' for c in x) for x in w]) for w in W])
    num['q_frac'] = np.array([np.mean([x.startswith('q') for x in w]) for w in W])
    num['ch_frac'] = np.array([np.mean(['ch' in x for x in w]) for w in W])
    num['sh_frac'] = np.array([np.mean(['sh' in x for x in w]) for w in W])
    num['dy_frac'] = np.array([np.mean([x.endswith('dy') for x in w]) for w in W])
    num['aiin_frac'] = np.array([np.mean(['aiin' in x or 'ain' in x for x in w]) for w in W])
    num['ol_frac'] = np.array([np.mean([x.startswith('ol') or x.startswith('al') for x in w]) for w in W])
    num['ttr'] = np.array([len(set(w[:20])) / len(w[:20]) for w in W])
    num['w1_len'] = np.array([len(_gl(w[0])) for w in W], float)
    num['w1_gall'] = np.array([sum(c in GALL for c in w[0]) for w in W], float)
    num['line1_len'] = np.array([len(u['lines'][0]) for u in units], float)
    num['last_line_len'] = np.array([len(u['lines'][-1]) for u in units], float)
    cat['w1_first'] = [first_class(w[0]) for w in W]
    cat['w1_last'] = [_gl(w[0])[-1] for w in W]
    cat['w2_first'] = [first_class(w[1]) if len(w) > 1 else 'none' for w in W]
    cat['end_last'] = [_gl(w[-1])[-1] for w in W]
    top = Counter(x for w in W for x in w[:1])
    cat['w1_word'] = [w[0] if top[w[0]] >= 3 else 'rare' for w in W]
    if star and any(u.get('star') for u in units):
        S = [u.get('star') or {} for u in units]
        cat['star_pts'] = [str(s.get('pts')) for s in S]
        cat['star_dark'] = [str(s.get('dark')) for s in S]
        cat['star_dotted'] = [str(s.get('dotted')) for s in S]
        cat['star_tail'] = [str(s.get('tail')) for s in S]
    vocab = [w for w, c in Counter(x for w in W for x in w).most_common(150)]
    vi = {w: i for i, w in enumerate(vocab)}
    vec = np.zeros((N, len(vocab)))
    for i, w in enumerate(W):
        for x in w:
            if x in vi: vec[i, vi[x]] += 1
    # glyph-unigram profile as a second multivariate feature
    gl = sorted(set(c for w in W for x in w for c in _gl(x)))
    gi = {c: i for i, c in enumerate(gl)}
    gvec = np.zeros((N, len(gl)))
    for i, w in enumerate(W):
        for x in w:
            for c in _gl(x): gvec[i, gi[c]] += 1
    return num, cat, {'vocab': vec, 'glyphs': gvec}


# ------------------------------------------------------------------ statistics
def detrend(x, deg=4):
    x = np.asarray(x, float)
    t = np.linspace(-1, 1, len(x))
    c = np.polynomial.legendre.legfit(t, x, deg)
    r = x - np.polynomial.legendre.legval(t, c)
    s = r.std()
    return r / s if s > 0 else r


def _bins(N, P):
    K = int(round(P))
    i = np.arange(N)
    return np.minimum((K * ((i / P) % 1.0)).astype(int), K - 1), K


_BIN_CACHE = {}
def bins(N, P):
    k = (N, P)
    if k not in _BIN_CACHE: _BIN_CACHE[k] = _bins(N, P)
    return _BIN_CACHE[k]


def fold_num(x, P):
    b, K = bins(len(x), P)
    n = np.bincount(b, minlength=K); s = np.bincount(b, weights=x, minlength=K)
    ok = n > 0
    m = s[ok] / n[ok]
    gm = x.mean()
    ssb = (n[ok] * (m - gm) ** 2).sum()
    ssw = ((x - (s / np.maximum(n, 1))[b]) ** 2).sum()
    k = ok.sum()
    return (ssb / (k - 1)) / (ssw / (len(x) - k) + 1e-12)


def fold_cat(codes, ncat, P):
    b, K = bins(len(codes), P)
    T = np.zeros((K, ncat)); np.add.at(T, (b, codes), 1)
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    nz = T > 0
    G = 2 * (T[nz] * np.log(T[nz] / E[nz])).sum()
    dof = max((K - 1) * (np.count_nonzero(T.sum(0)) - 1), 1)
    return (G - dof) / math.sqrt(2 * dof)   # rough normalisation; real calibration is by the null


def _comb_lags(P, N):
    on, off = [], []
    for k in (1, 2, 3):
        L = int(round(k * P))
        if L + 2 >= N - 10: break
        on.append(L); off += [L - 2, L - 1, L + 1, L + 2]
    off = [l for l in off if l >= 1 and l not in on]
    return on, off


def acf(x, maxlag):
    x = x - x.mean(); d = (x * x).sum() + 1e-12
    return np.array([1.0] + [(x[:-l] * x[l:]).sum() / d for l in range(1, maxlag + 1)])


def comb_from_profile(prof, P, N):
    on, off = _comb_lags(P, N)
    if not on or not off: return 0.0
    return prof[on].mean() - prof[off].mean()


def same_profile(codes, maxlag):
    c = np.asarray(codes)
    return np.array([1.0] + [(c[:-l] == c[l:]).mean() for l in range(1, maxlag + 1)])


def cos_profile(V, maxlag):
    X = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    X = X - X.mean(0)
    S = X @ X.T
    N = len(X)
    return np.array([1.0] + [np.diag(S, l).mean() for l in range(1, maxlag + 1)])


def encode(labels):
    u = sorted(set(labels)); m = {x: i for i, x in enumerate(u)}
    return np.array([m[x] for x in labels]), len(u)


def scan(num, cat, vec, periods=PERIODS, perm=None):
    """All statistics for one ordering. perm: index array applied to the units first. -> dict key -> array over P."""
    out = {}
    N = len(next(iter(num.values())))
    idx = np.arange(N) if perm is None else perm
    maxlag = int(3 * periods.max()) + 3
    maxlag = min(maxlag, N - 1)
    for k, x in num.items():
        y = detrend(np.asarray(x)[idx])
        out[('FOLD', k)] = np.array([fold_num(y, P) for P in periods])
        pr = acf(y, maxlag)
        out[('COMB', k)] = np.array([comb_from_profile(pr, P, N) for P in periods])
    for k, lab in cat.items():
        c, nc = encode([lab[i] for i in idx])
        if nc < 2: continue
        out[('FOLD', k)] = np.array([fold_cat(c, nc, P) for P in periods])
        pr = same_profile(c, maxlag)
        out[('COMB', k)] = np.array([comb_from_profile(pr, P, N) for P in periods])
    for k, V in vec.items():
        pr = cos_profile(V[idx], maxlag)
        out[('COMB', k)] = np.array([comb_from_profile(pr, P, N) for P in periods])
    return out


def family_test(obs, nulls):
    """obs: dict key -> array; nulls: list of such dicts. Returns per-key z arrays, observed max z,
    null max-z distribution (leave-one-out standardised), family-wise p, and the top cells."""
    keys = list(obs.keys())
    O = np.stack([obs[k] for k in keys])                    # F x P
    Nn = np.stack([np.stack([n[k] for k in keys]) for n in nulls])   # R x F x P
    mu = Nn.mean(0); sd = Nn.std(0)
    # sd floor: a quarter of the feature's median null sd over periods, so that near-degenerate cells (rare
    # categories, few comb lags) cannot produce huge z in the observed data or in the null replicates
    floor = 0.25 * np.median(sd, axis=1, keepdims=True) + 1e-9
    Z = (O - mu) / np.maximum(sd, floor)
    R = len(nulls)
    # leave-one-out for the null maxima
    S1 = Nn.sum(0); S2 = (Nn ** 2).sum(0)
    nullmax = []
    for r in range(R):
        m = (S1 - Nn[r]) / (R - 1); v = np.sqrt(np.maximum((S2 - Nn[r] ** 2) / (R - 1) - m ** 2, 0))
        nullmax.append(((Nn[r] - m) / np.maximum(v, floor)).max())
    nullmax = np.array(nullmax)
    zmax = Z.max()
    p = (1 + (nullmax >= zmax).sum()) / (R + 1)
    flat = [(Z[i, j], keys[i], j) for i in range(len(keys)) for j in range(Z.shape[1])]
    flat.sort(key=lambda t: -t[0])
    # per-cell empirical p for the top cells
    tops = []
    for z, k, j in flat[:10]:
        i = keys.index(k)
        pc = (1 + (Nn[:, i, j] >= O[i, j]).sum()) / (R + 1)
        tops.append((k, j, float(z), float(pc)))
    return {'keys': keys, 'Z': Z, 'zmax': float(zmax), 'nullmax': nullmax, 'p_fw': float(p), 'tops': tops, 'O': O, 'Nn': Nn}


def perm_nulls(num, cat, vec, R, seed, periods=PERIODS, kind='shuffle', groups=None):
    rng = np.random.default_rng(seed)
    N = len(next(iter(num.values())))
    outs = []
    for r in range(R):
        if kind == 'shuffle':
            p = rng.permutation(N)
        elif kind == 'within':          # shuffle inside page groups (keeps page-scale drift, kills within-page order)
            p = np.arange(N)
            for g in np.unique(groups):
                ix = np.where(groups == g)[0]; p[ix] = rng.permutation(ix)
        elif kind == 'blocks':          # shuffle whole pages (keeps within-page order and layout, kills cross-page phase)
            gs = list(dict.fromkeys(groups.tolist())); rng.shuffle(gs)
            p = np.concatenate([np.where(groups == g)[0] for g in gs])
        elif kind == 'rotate':          # circular shift + reversal options (keeps all local structure, breaks nothing periodic)
            p = np.roll(np.arange(N), rng.integers(1, N))
        outs.append(scan(num, cat, vec, periods, perm=p))
    return outs


def fmt_top(res, periods=PERIODS, n=5):
    s = []
    for (stat, feat), j, z, pc in res['tops'][:n]:
        s.append(f'{stat}:{feat}@P{periods[j]:g} z{z:.1f} (cell p {pc:.3f})')
    return '; '.join(s)


def write_rows(path, rows):
    with open(path, 'a') as fh:
        for r in rows: fh.write('| ' + ' | '.join(str(x) for x in r) + ' |\n')


# ------------------------------------------------------------------ controls
def markov_resynth(units, seed):
    """Word-bigram resynthesis of the section, keeping each unit's line lengths; first word of each unit drawn from
    the real unit-first words, first word of each line from real line-first words."""
    rng = random.Random(seed)
    succ = {}
    first_u, first_l = [], []
    for u in units:
        first_u.append(u['lines'][0][0])
        for l in u['lines']:
            first_l.append(l[0])
            for a, b in zip(l, l[1:]): succ.setdefault(a, []).append(b)
    allw = [w for u in units for l in u['lines'] for w in l]
    out = []
    for u in units:
        ls = []
        for li, l in enumerate(u['lines']):
            w = rng.choice(first_u) if li == 0 else rng.choice(first_l)
            nl = [w]
            while len(nl) < len(l):
                w = rng.choice(succ.get(w) or allw); nl.append(w)
            ls.append(nl)
        out.append({'f': u['f'], 'lines': ls, 'star': u.get('star'), 'zlpara': True})
    return out


def plant_calendar(units, period, strength, seed, mode='opening'):
    """Plant a calendar in real Voynich units: unit i has day d = i mod period (a random phase).
    opening : with prob strength the first word is replaced by the day's own word (a Voynich word chosen per day)
    vocab   : with prob strength each body word is replaced by a word from the day's own 5-word list
    length  : unit truncated / padded so that its length has a day-specific multiplier (1 +- strength)."""
    rng = random.Random(seed)
    allw = [w for u in units for l in u['lines'] for w in l]
    pool = [w for w, c in Counter(allw).most_common(400)]
    day_w = [rng.choice(pool) for _ in range(int(math.ceil(period)))]
    day_v = [[rng.choice(pool) for _ in range(5)] for _ in range(int(math.ceil(period)))]
    day_m = [1 + strength * rng.uniform(-1, 1) for _ in range(int(math.ceil(period)))]
    ph = rng.random() * period
    out = []
    for i, u in enumerate(units):
        d = int(((i + ph) % period))
        ls = [list(l) for l in u['lines']]
        if mode == 'opening' and rng.random() < strength:
            ls[0][0] = day_w[d]
        elif mode == 'vocab':
            ls = [[(rng.choice(day_v[d]) if rng.random() < strength else w) for w in l] for l in ls]
        elif mode == 'length':
            flat = [w for l in ls for w in l]
            n = max(3, int(round(len(flat) * day_m[d])))
            while len(flat) < n: flat.append(rng.choice(allw))
            flat = flat[:n]
            ls = [flat[j:j + 10] for j in range(0, len(flat), 10)]
        out.append({'f': u['f'], 'lines': ls, 'star': u.get('star'), 'zlpara': True})
    return out


# ------------------------------------------------------------------ Latin calendar control (Usuard)
def usuard_units(path, header=True):
    """Usuardus, Martyrologium (la.wikisource): one unit per day. header=True keeps the Roman date words
    ('IV Nonas', 'Kalendis', 'XVIII Kal.') but drops 'Die N'; header=False keeps only the entry text."""
    t = open(path, encoding='utf-8').read()
    t = re.sub(r'\d{3}\.\d{4}[A-D]?\|', ' ', t)
    t = re.sub(r'\[[^\]]*\]', ' ', t)
    parts = re.split(r'\n:([^\n]*?Die \d+\.)\s*\n', t)
    units = []
    for i in range(1, len(parts) - 1, 2):
        h = re.sub(r'Die \d+\.', '', parts[i]); body = parts[i + 1]
        body = re.split(r'\n:MENSIS', body)[0]
        ws = re.findall(r'[A-Za-z]+', ((h + ' ') if header else '') + body)
        ws = [w.lower() for w in ws if not re.fullmatch(r'[IVXLC]+', w) or header]
        # chop into 'lines' of 10 words so line features are defined
        if len(ws) >= 3:
            units.append({'f': 'u%03d' % len(units), 'lines': [ws[j:j + 10] for j in range(0, len(ws), 10)], 'star': None})
    return units


def latin_features(units):
    """Same feature set, letters instead of glyphs (first_class falls back to the first letter class)."""
    return features(units, star=False)


# Ado of Vienne, Martyrologium (la.wikisource, PL 123): a full-year calendar, 359 day entries. Each header carries the
# dominical letter (A-G, an exact period-7 cycle) and the Roman date (Kalends/Nones/Ides countdown, ~30.4-day cycle).
HDR = re.compile(r'^([A-G])\.\s*((?:[IVXL]+|PRIDIE)?\s*(?:KAL|NON|ID)[A-Z]*\.?\s*[A-Z]*\.?)(.*)$')

def ado_units(path=os.path.join(CK, 'ado.wiki'), letter=True, date=True, body=True, maxw=None):
    t = open(path, encoding='utf-8').read()
    t = re.sub(r'\d{3}\.\d{4}[A-D]?\|', ' ', t)
    t = re.sub(r'\([^)]*\)', ' ', t)
    units, cur = [], None
    for ln in t.split('\n'):
        m = HDR.match(ln.strip())
        if m:
            cur = {'f': 'a%03d' % len(units), 'hdr': [m.group(1), m.group(2)], 'body': [m.group(3)], 'star': None}
            units.append(cur)
        elif cur is not None and not ln.startswith('LITANIAE'):
            cur['body'].append(ln)
    out = []
    for u in units:
        ws = []
        if letter: ws.append(u['hdr'][0].lower() + 'x')       # 'ax'..'gx' so the letter is a word
        if date: ws += [w.lower() for w in re.findall(r'[A-Za-z]+', u['hdr'][1])]
        if body: ws += [w.lower() for w in re.findall(r'[A-Za-z]+', ' '.join(u['body']))]
        if maxw: ws = ws[:maxw]
        if len(ws) >= 3:
            out.append({'f': u['f'], 'lines': [ws[j:j + 10] for j in range(0, len(ws), 10)], 'star': None,
                        'letter': u['hdr'][0]})
    return out
