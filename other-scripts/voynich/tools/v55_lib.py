"""v55 THE SKY IS THE ANSWER KEY -- shared engine.

Zodiac nymph labels (10 signs, 299 labels) are laid on the real sky of a
year: nymph i of sign s = the day the Sun stands at degree i (+ rotation o,
optional reversal) of that sign. Each day carries sky classes (Moon sign,
phase, weekday, events ...). Random word classes (no sound values: first or
last glyph, length, substrings, random merges) are crossed with sky classes
by a G-test, for every year 1290-1611, every rotation and both directions.
Halves A/B (5 signs each) give held-out replication.
"""
import os, sys, json, math
import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v55_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
from vlib import glyphs  # noqa
import v55_sky as sky

SIGNS = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius',
         'Capricorn', 'Aquarius', 'Pisces']
PAGE_ORDER = ['f70v2', 'f70v1', 'f71r', 'f71v', 'f72r1', 'f72r2', 'f72r3', 'f72v3', 'f72v2', 'f72v1', 'f73r', 'f73v']
HALF_A = {'Pisces', 'Taurus', 'Cancer', 'Virgo', 'Scorpio'}  # alternate signs
Y0, Y1 = 1290, 1611
WINDOW = (1404, 1438)


def load_labels():
    d = json.load(open(os.path.join(DATA, 'derived', 'v4_zodiac_labels.json')))
    d.sort(key=lambda x: (SIGNS.index(x['month']), PAGE_ORDER.index(x['page']), x['ring'], x['n']))
    out = []
    for s in SIGNS:
        rows = [x for x in d if x['month'] == s]
        n = len(rows)
        for i, x in enumerate(rows):
            out.append(dict(sign=SIGNS.index(s), name=s, i=i, n=n, label=x['label'].replace('?', ''),
                            words=x['words'], page=x['page'], ring=x['ring']))
    return out


# ---------------- sky ----------------
_SKY_CACHE = {}


def get_sky(kind='real'):
    """kind 'real' or 'fake<k>' (periods rescaled / permuted, seeded)."""
    if kind in _SKY_CACHE:
        return _SKY_CACHE[kind]
    f = os.path.join(CK, f'sky_{kind}.npz')
    if os.path.exists(f):
        t = dict(np.load(f))
    else:
        rate = None
        if kind.startswith('fake'):
            r = np.random.default_rng(1000 + int(kind[4:]))
            rate = {'moon': 1 + r.choice([-1, 1]) * r.uniform(0.04, 0.09)}
            for p in ['mercury', 'venus', 'mars', 'jupiter', 'saturn']:
                rate[p] = r.uniform(0.6, 1.6)
        t = sky.build_daily(Y0 - 1, Y1 + 2, rate=rate)
        np.savez_compressed(f, **t)
    _SKY_CACHE[kind] = t
    return t


def degree_days(t):
    """dd[y - Y0, deg] = table index of the day the Sun crosses deg + 0.5 in civil year y."""
    lon = np.unwrap(t['sun'] * np.pi / 180) * 180 / np.pi
    nY = Y1 - Y0 + 1
    dd = np.full((nY, 360), -1, np.int64)
    for deg in range(360):
        tgt = deg + 0.5
        k = np.floor((lon - tgt) / 360.0)
        idx = np.where(k[1:] > k[:-1])[0] + 1  # first day at/after crossing
        yrs = t['year'][idx]
        ok = (yrs >= Y0) & (yrs <= Y1)
        dd[yrs[ok] - Y0, deg] = idx[ok]
    return dd


def day_features(t):
    """Categorical sky classes per day: name -> (int array, K)."""
    el = (t['moon'] - t['sun']) % 360
    ms = (t['moon'] // 30).astype(int)
    f = {}
    f['moon_sign'] = (ms, 12)
    f['moon_elem'] = (ms % 4, 4)
    f['moon_mode'] = (ms % 3, 3)
    f['phase8'] = ((el // 45).astype(int), 8)
    f['phase4'] = ((((el + 45) % 360) // 90).astype(int), 4)
    f['weekday'] = (t['weekday'].astype(int), 7)
    f['moon_rel'] = ((((t['moon'] - t['sun']) % 360) // 30).astype(int), 12)
    f['moonlat'] = ((t['moonlat'] > 0).astype(int), 2)
    return f


def day_events(t):
    """Binary events per day (Bool arrays)."""
    n = len(t['jdn'])
    el = (t['moon'] - t['sun']) % 360
    nxt = lambda a: np.r_[a[1:], a[-1:]]
    e = {}
    def cross(target):
        x = (el - target + 180) % 360 - 180
        return (x < 0) & (nxt(x) >= 0)
    e['new'] = cross(0)
    e['first_q'] = cross(90)
    e['full'] = cross(180)
    e['last_q'] = cross(270)
    e['moon_ingress'] = (t['moon'] // 30) != (nxt(t['moon']) // 30)
    lat = np.abs(t['moonlat'])
    e['eclipse'] = (e['new'] & ((lat < 1.0) | (nxt(lat) < 1.0))) | (e['full'] & ((lat < 0.5) | (nxt(lat) < 0.5)))
    for p in ['saturn', 'jupiter', 'mars', 'venus', 'mercury']:
        e['moon_conj_' + p] = np.abs((t['moon'] - t[p] + 180) % 360 - 180) < 6.5
        e['ingress_' + p] = (t[p] // 30) != (nxt(t[p]) // 30)
    mv = (np.r_[t['mercury'][1:], t['mercury'][-1]] - t['mercury'] + 180) % 360 - 180
    e['mercury_retro'] = mv < 0
    st = np.sign(mv); e['mercury_station'] = st != np.r_[st[1:], st[-1]]
    e['sunday'] = t['weekday'] == 0
    # movable feasts from Julian computus Easter
    ea = np.zeros(n, bool); fe = np.zeros(n, bool)
    for y in range(Y0 - 1, Y1 + 2):
        m, d = sky.easter_julian(y)
        j = sky.julian_cal_to_jd(y, m, d)
        k = j - t['jdn'][0]
        if 0 <= k < n:
            ea[k] = True
            for off in (-46, 39, 49, 60):  # Ash Wed, Ascension, Pentecost, Corpus Christi
                if 0 <= k + off < n:
                    fe[k + off] = True
    e['easter'] = ea
    e['movable_feast'] = fe | ea
    for p in ['saturn', 'jupiter', 'mars', 'venus', 'mercury', 'sun']:
        pass
    return e


def alignments(labels, dd):
    """Return idx[y, a, L] (table index per label) for a in 0..59 = (dir, rotation)."""
    nY = dd.shape[0]
    L = len(labels)
    idx = np.zeros((nY, 60, L), np.int64)
    for li, x in enumerate(labels):
        s, i, n = x['sign'], x['i'], x['n']
        for di in range(2):
            ii = i if di == 0 else n - 1 - i
            for o in range(30):
                deg = 30 * s + int(math.floor(ii * 30.0 / n + o)) % 30
                idx[:, di * 30 + o, li] = dd[:, deg]
    return idx


# ---------------- word classifiers ----------------
SUBS = ['al', 'ar', 'ot', 'ok', 'ee', 'eo', 'ol', 'dy', 'ly', 'ry', 'am', 'ch', 'aiin', 'te', 'ke', 'y', 'd', 'l', 'r', 's']


def word_feats(lab):
    g = glyphs(lab) or ['?']
    f = {}
    f['first'] = g[0]
    f['second'] = g[1] if len(g) > 1 else '#'
    f['last'] = g[-1]
    f['penult'] = g[-2] if len(g) > 1 else '#'
    f['len'] = str(min(len(g), 12) // 2)
    f['n_o'] = str(min(lab.count('o'), 3))
    f['n_gal'] = str(min(sum(lab.count(c) for c in 'ktpf'), 2))
    f['nwords'] = str(min(len(lab.split('.')), 2))
    for s in SUBS:
        f['has_' + s] = str(int(s in lab))
    return f


def random_classifiers(labels, n, seed=0, kmax=6):
    """Return list of (desc, class array) with 2..kmax classes, no sound values."""
    rng = np.random.default_rng(seed)
    F = [word_feats(x['label']) for x in labels]
    keys = list(F[0].keys())
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        mode = rng.random()
        if mode < 0.6:
            k = keys[rng.integers(len(keys))]
            vals = sorted(set(f[k] for f in F))
            if len(vals) < 2:
                continue
            K = int(rng.integers(2, min(kmax, len(vals)) + 1))
            grp = {v: int(rng.integers(K)) for v in vals}
            c = np.array([grp[f[k]] for f in F])
            desc = f'{k}:' + ','.join(f'{v}>{grp[v]}' for v in vals)
        elif mode < 0.85:
            k1, k2 = rng.choice([k for k in keys if k.startswith('has_')], 2, replace=False)
            c = np.array([int(f[k1]) * 2 + int(f[k2]) for f in F])
            desc = f'{k1}x{k2}'
        else:  # random hash partition of label types
            K = int(rng.integers(2, kmax + 1))
            types = sorted(set(x['label'] for x in labels))
            grp = {w: int(rng.integers(K)) for w in types}
            c = np.array([grp[x['label']] for x in labels])
            desc = f'hash{K}:{int(rng.integers(1 << 30))}'
        _, c = np.unique(c, return_inverse=True)
        if c.max() < 1 or np.bincount(c).min() < 3:
            continue
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        out.append((desc, c))
    return out


# ---------------- G-test engine ----------------
def g_scores(C1h, rows, S1h, Kf):
    """C1h: (R, L) one-hot rows of all classifiers stacked; rows: list of (r0, r1).
    S1h: (L, A*Kf). Returns -log10 p array (ncls, A)."""
    n = C1h @ S1h  # (R, A*Kf)
    A = S1h.shape[1] // Kf
    n = n.reshape(n.shape[0], A, Kf)
    colsum = S1h.sum(0).reshape(A, Kf)
    N = colsum.sum(1)  # (A,)
    nzc = (colsum > 0).sum(1)
    out = np.zeros((len(rows), A), np.float32)
    for ci, (r0, r1) in enumerate(rows):
        nb = n[r0:r1]
        rs = nb.sum(2)  # (Kc, A)
        exp = rs[:, :, None] * colsum[None] / np.maximum(N, 1)[None, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(nb > 0, nb * np.log(nb / exp), 0.0)
        G = 2 * term.sum((0, 2))
        nzr = (rs > 0).sum(0)
        df = np.maximum((nzr - 1) * (nzc - 1), 1)
        out[ci] = -chi2.logsf(G, df) / np.log(10)
    return out


def _G(n):
    """n: (Kc, Aall, Kf) counts -> G statistic per alignment."""
    rs = n.sum(2); cs = n.sum(0); N = cs.sum(1)
    exp = rs[:, :, None] * cs[None] / np.maximum(N, 1)[None, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(n > 0, n * np.log(n / np.where(exp > 0, exp, 1)), 0.0)
    return 2 * term.sum((0, 2))


def _lp(G, df):
    return (-chi2.logsf(G, df) / np.log(10)).astype(np.float32)


def run_search(labels, classes, feats, idx, maskA, chunk=40):
    """For each classifier x feature x year: best alignment by A (score A,B), by B (score B,A),
    best full score and its alignment. Scores are -log10 p of a G-test with df (Kc-1)(Kf-1)."""
    L = len(labels)
    nY = idx.shape[0]
    nc, nf = len(classes), len(feats)
    res = {k: np.zeros((nc, nf, nY), np.float32) for k in ['full', 'A', 'AB', 'B', 'BA']}
    res['full_al'] = np.zeros((nc, nf, nY), np.int16)
    Aall = nY * 60
    flat = idx.reshape(Aall, L)
    tk = lambda g, a: np.take_along_axis(g, a[:, None], 1)[:, 0]
    Ks = [int(c.max()) + 1 for _, c in classes]
    for fi, (fname, (farr, Kf)) in enumerate(feats):
        cls = farr[flat]  # (Aall, L)
        O = np.zeros((L, Aall, Kf), np.float32)
        O[np.arange(L)[None, :], np.arange(Aall)[:, None], cls] = 1
        O = O.reshape(L, Aall * Kf)
        OA, OB = O[maskA], O[~maskA]
        for c0 in range(0, nc, chunk):
            sub = list(range(c0, min(nc, c0 + chunk)))
            rows, R = [], []
            r = 0
            for ci in sub:
                oh = np.zeros((Ks[ci], L), np.float32); oh[classes[ci][1], np.arange(L)] = 1
                R.append(oh); rows.append((r, r + Ks[ci])); r += Ks[ci]
            C = np.vstack(R)
            NA = (C[:, maskA] @ OA).reshape(-1, Aall, Kf)
            NB = (C[:, ~maskA] @ OB).reshape(-1, Aall, Kf)
            for ci, (r0, r1) in zip(sub, rows):
                df = (Ks[ci] - 1) * (Kf - 1)
                nA, nB = NA[r0:r1].astype(np.float64), NB[r0:r1].astype(np.float64)
                gA = _G(nA).reshape(nY, 60); gB = _G(nB).reshape(nY, 60); gF = _G(nA + nB).reshape(nY, 60)
                aA, aB, aF = gA.argmax(1), gB.argmax(1), gF.argmax(1)
                res['A'][ci, fi] = _lp(tk(gA, aA), df); res['AB'][ci, fi] = _lp(tk(gB, aA), df)
                res['B'][ci, fi] = _lp(tk(gB, aB), df); res['BA'][ci, fi] = _lp(tk(gA, aB), df)
                res['full'][ci, fi] = _lp(tk(gF, aF), df); res['full_al'][ci, fi] = aF
    return res


def summarize(res, thr=2.0):
    """Statistics of one search."""
    full = res['full']
    nY = full.shape[2]
    years = np.arange(Y0, Y0 + nY)
    win = (years >= WINDOW[0]) & (years <= WINDOW[1])
    ymax = full.max((0, 1))
    ymean_best = full.max(1).mean(0)  # mean over classifiers of best feature
    rep = ((res['AB'] >= thr) & (res['A'] >= thr)).sum() + ((res['BA'] >= thr) & (res['B'] >= thr)).sum()
    # replication strength: mean held-out score at the in-sample best alignment
    hold = 0.5 * (res['AB'].mean() + res['BA'].mean())
    i = np.unravel_index(full.argmax(), full.shape)
    return dict(max_full=float(full.max()), argmax=[int(v) for v in i], peak_year=int(years[ymax.argmax()]),
                peak_year_mean=int(years[ymean_best.argmax()]),
                win_z=float((ymax[win].mean() - ymax[~win].mean()) / (ymax[~win].std() + 1e-9)),
                win_z_mean=float((ymean_best[win].mean() - ymean_best[~win].mean()) / (ymean_best[~win].std() + 1e-9)),
                n_rep=int(rep), hold=float(hold),
                y1300=float(ymax[years == 1300][0]), y1420=float(ymax[years == 1420][0]),
                y1550=float(ymax[years == 1550][0]))


def make_condition(cond, labels, idx, T, plant_year=1421):
    """Labels for a condition: real, shuf<k> (within-sign permutation), fake<k> (labels real),
    plant* (synthetic almanac for plant_year in opaque code glyphs), gen<k> (Markov generator ring)."""
    labs = [dict(x) for x in labels]
    if cond.startswith('shuf'):
        r = np.random.default_rng(int(cond[4:]))
        for s in set(x['sign'] for x in labs):
            ii = [k for k, x in enumerate(labs) if x['sign'] == s]
            perm = r.permutation(ii)
            for k, p in zip(ii, perm):
                labs[k]['label'] = labels[p]['label']
    elif cond.startswith('plant'):
        r = np.random.default_rng(7)
        yi = plant_year - Y0
        F = day_features(T); E = day_events(T)
        q = {'plantP1': 0.5, 'plantP2': 0.3, 'plantW': 0.5, 'plantE': 1.0, 'plantH': 1.0}.get(cond, 0.5)
        for k, x in enumerate(labs):
            day = idx[yi, 0, k]
            if cond == 'plantE':
                # event-marker almanac: the label at each full/new moon day becomes a fixed marker word
                win = slice(max(day - 1, 0), day + 2)
                if E['full'][win].any():
                    x['label'] = 'otolal'
                elif E['new'][win].any():
                    x['label'] = 'okolal'
            elif r.random() < q:
                if cond == 'plantW':
                    x['label'] = x['label'] + 'ydlrsmg'[F['weekday'][0][day]]
                else:
                    x['label'] = ['q', 'y', 'd', 's'][F['moon_elem'][0][day]] + x['label']
    elif cond.startswith('gen'):
        r = np.random.default_rng(100 + int(cond[3:]))
        from collections import defaultdict, Counter
        tr = defaultdict(Counter)
        for x in labels:
            w = '^' + x['label'] + '$'
            for a, b in zip(w, w[1:]):
                tr[a][b] += 1
        for x in labs:
            w, c = '', '^'
            while True:
                ks = list(tr[c]); p = np.array([tr[c][k] for k in ks], float)
                c = ks[r.choice(len(ks), p=p / p.sum())]
                if c == '$' or len(w) > 14:
                    break
                w += c
            x['label'] = w or 'o'
    return labs


def marker_classifiers(labs, min_type=2, min_affix=4):
    """Binary classes: recurrent label types, recurrent 2/3-glyph prefixes and suffixes, substrings."""
    from collections import Counter
    G = [glyphs(x['label']) for x in labs]
    out = []
    ct = Counter(x['label'] for x in labs)
    for w, v in sorted(ct.items()):
        if v >= min_type:
            out.append(('type=' + w, np.array([int(x['label'] == w) for x in labs])))
    for k in (2, 3):
        for side in ('pre', 'suf'):
            aff = [''.join(g[:k]) if side == 'pre' else ''.join(g[-k:]) for g in G]
            for a, v in sorted(Counter(aff).items()):
                if min_affix <= v <= len(labs) - min_affix:
                    out.append((f'{side}{k}={a}', np.array([int(z == a) for z in aff])))
    for s in SUBS:
        c = np.array([int(s in x['label']) for x in labs])
        if 3 <= c.sum() <= len(labs) - 3:
            out.append(('has=' + s, c))
    return out


def event_features(T, dil=1):
    E = day_events(T)
    out = []
    for k, v in E.items():
        if dil and k in ('new', 'full', 'first_q', 'last_q', 'eclipse', 'mercury_station', 'easter'):
            v = v | np.r_[v[1:], False] | np.r_[False, v[:-1]]
        out.append((k, (v.astype(int), 2)))
    return out


def load_ring_words():
    """Words of the circular texts (@Cc lines) of the zodiac pages, per sign in page order
    (outer ring first as transcribed); each word gets (sign, i, n) like a label."""
    import re
    txt = open(os.path.join(DATA, 'ZL3b-n.txt'), encoding='utf-8', errors='replace').read().splitlines()
    page, per = None, {}
    for ln in txt:
        m = re.match(r'^<(f\d+[rv]\d?)>', ln)
        if m:
            page = m.group(1); continue
        m = re.match(r'^<(f\d+[rv]\d?)\.\d+,[@+*=&]C', ln)
        if m and page in PAGE_ORDER:
            body = ln.split('>', 1)[1].strip()
            body = re.sub(r'<[^>]*>', '', body)
            body = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', body)
            body = re.sub(r'@\d+;', '', body)
            body = body.replace('{', '').replace('}', '').replace("'", '').replace('?', '').replace(',', '.')
            per.setdefault(page, []).extend([w for w in body.split('.') if w])
    month = {}
    for x in json.load(open(os.path.join(DATA, 'derived', 'v4_zodiac_labels.json'))):
        month[x['page']] = x['month']
    out = []
    for s in SIGNS:
        ws = []
        for p in PAGE_ORDER:
            if month.get(p) == s:
                ws += per.get(p, [])
        for i, w in enumerate(ws):
            out.append(dict(sign=SIGNS.index(s), name=s, i=i, n=len(ws), label=w, words=[w], page=None, ring=0))
    return out
