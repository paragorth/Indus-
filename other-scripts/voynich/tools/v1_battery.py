"""V1 battery: word-level and line-level statistics of a line corpus, in glyph units.

A corpus is a list of lines: {'words': [glyph-unit strings], 'para_start': bool, ...metadata}.
Glyph units: one character per glyph (ch=C, sh=S, cth=T, ckh=K, cph=P, cfh=F), as vlib.glyphs.
All statistics are computed on corpora of the same layout (same line/paragraph word counts), so size-dependent
biases cancel between the real text and the generators. Effect statistics are observed minus a null that
destroys the effect (within-line shuffle, permutation of pairs, or re-dealing words to lines in a paragraph).

Battery (keys):
  word level : zipf, ttr, hapax, h2, wlen_m, wlen_sd, rep (adjacent identical / within-line shuffle),
               wlen_ac (lag-1 word-length autocorrelation inside lines, minus shuffle)
  line level : J_in  junction MI (last glyph -> next first glyph) inside lines, excess over pair permutation
               J_x   the same across a line break (same paragraph), same n          -> the 'reset'
               F1/L1 first-glyph / last-glyph excess MI with line-initial / line-final position
               W1/W2 word-identity excess MI with line-initial / line-final position (opener / closer vocab)
               PG    paragraph-first line starts with gallows minus other lines
               Q_mg  m/g per-line variance ratio vs re-dealing (<1 = quota)
               Q_gal, Q_e  gallows / e variance ratio (>1 = line make-up clumps)
               C_qa, C_che, C_qd  per-line residual correlation, observed minus re-dealing null (two line modes)
               lag1  lag-1 persistence of line-mode score inside paragraphs, minus line-order shuffle
               endq  share of line-final words starting with q / share of all words starting with q
"""
import sys, os, math, random, json
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

GALLOWS = set('ktpfTKPF')
WORD_KEYS = ['zipf', 'ttr', 'hapax', 'h2', 'wlen_m', 'wlen_sd', 'rep', 'wlen_ac']
LINE_KEYS = ['J_in', 'J_x', 'F1', 'L1', 'W1', 'W2', 'PG', 'Q_mg', 'Q_gal', 'Q_e', 'C_qa', 'C_che', 'C_qd', 'lag1', 'endq']
KEYS = WORD_KEYS + LINE_KEYS
# the line phenomena named in FINDINGS.md, grouped
GROUPS = {'reset': ['J_in', 'J_x'], 'edges': ['F1', 'L1', 'W1', 'W2', 'endq'], 'quota': ['Q_mg'],
          'modes': ['Q_gal', 'Q_e', 'C_qa', 'C_che', 'C_qd', 'lag1'], 'para': ['PG'], 'word': WORD_KEYS}


def load_real(name='ZL3b'):
    vl = vlib.load_voynich(name, drop_uncertain=True)
    out = []
    for r in vl:
        ws = [''.join(vlib.glyphs(w)) for w in r['words']]
        ws = [w for w in ws if w and '?' not in w]
        if not ws:
            continue
        out.append({'words': ws, 'para_start': bool(r.get('para_start')), 'folio': r.get('folio'),
                    'lang': r.get('lang'), 'illus': r.get('illus'), 'hand': r.get('hand')})
    return out


def relayout(lines, words):
    out, i = [], 0
    for L in lines:
        k = len(L['words']); nl = dict(L); nl['words'] = words[i:i + k]; i += k; out.append(nl)
    return out


# ---------------- helpers ----------------
def _codes(seq):
    d = {}; return np.array([d.setdefault(x, len(d)) for x in seq], dtype=np.int64), len(d)


def mi_codes(a, b, na, nb):
    n = len(a)
    if n == 0:
        return 0.0
    j = np.bincount(a * nb + b, minlength=na * nb).astype(float)
    pa = np.bincount(a, minlength=na).astype(float); pb = np.bincount(b, minlength=nb).astype(float)
    nz = j > 0
    jj = j[nz]; ia = np.nonzero(nz)[0] // nb; ib = np.nonzero(nz)[0] % nb
    return float(np.sum(jj / n * np.log2(jj * n / (pa[ia] * pb[ib]))))


def mi_excess_perm(xs, ys, rng, reps=10):
    a, na = _codes(xs); b, nb = _codes(ys)
    o = mi_codes(a, b, na, nb); s = []
    for _ in range(reps):
        s.append(mi_codes(a, rng.permutation(b), na, nb))
    return o - float(np.mean(s))


def paragraphs(lines, minlen=3):
    P, cur = [], []
    for L in lines:
        if L.get('para_start') and cur:
            P.append(cur); cur = []
        cur.append(L)
    if cur:
        P.append(cur)
    return [p for p in P if len(p) >= minlen]


def zipf_slope(words):
    c = np.array(sorted(Counter(words).values(), reverse=True)[:1000], dtype=float)
    x = np.log(np.arange(1, len(c) + 1)); y = np.log(c)
    return float(np.polyfit(x, y, 1)[0])


def h2_of(words):
    seq = []
    for w in words:
        seq.extend(w); seq.append('_')
    a, na = _codes(seq[:-1]); b, nb = _codes(seq[1:])
    j = np.bincount(a * nb + b).astype(float); j = j[j > 0]; n = j.sum()
    hj = -np.sum(j / n * np.log2(j / n))
    pa = np.bincount(a).astype(float); pa = pa[pa > 0]
    ha = -np.sum(pa / n * np.log2(pa / n))
    return float(hj - ha)


# ---------------- the battery ----------------
def battery(lines, seed=0, reps=10, keys=None):
    rng = np.random.default_rng(seed)
    R = {}
    words = [w for L in lines for w in L['words']]
    c = Counter(words); n = len(words)
    R['zipf'] = zipf_slope(words); R['ttr'] = len(c) / n
    R['hapax'] = sum(1 for v in c.values() if v == 1) / len(c)
    R['h2'] = h2_of(words)
    lens = np.array([len(w) for w in words], float); R['wlen_m'] = float(lens.mean()); R['wlen_sd'] = float(lens.std())

    # within-line adjacent pairs, and within-line shuffles
    def adj_stats(LL):
        same = tot = 0; la, lb = [], []
        for ws in LL:
            for x, y in zip(ws, ws[1:]):
                tot += 1; same += (x == y); la.append(len(x)); lb.append(len(y))
        cc = np.corrcoef(la, lb)[0, 1] if len(la) > 2 else 0.0
        return same / max(tot, 1), cc
    LW = [L['words'] for L in lines]
    o_rep, o_ac = adj_stats(LW)
    srep, sac = [], []
    prng = random.Random(seed)
    for _ in range(3):
        sh = [prng.sample(ws, len(ws)) for ws in LW]
        r_, a_ = adj_stats(sh); srep.append(r_); sac.append(a_)
    R['rep'] = o_rep / max(np.mean(srep), 1e-9); R['wlen_ac'] = float(o_ac - np.mean(sac))

    # junction reset: within-line vs across-line (same paragraph), equal n
    inner = [(ws[i][-1], ws[i + 1][0]) for ws in LW for i in range(len(ws) - 1)]
    cross = [(lines[i]['words'][-1][-1], lines[i + 1]['words'][0][0]) for i in range(len(lines) - 1)
             if not lines[i + 1].get('para_start')]
    m = min(len(cross), len(inner))
    jin, jx = [], []
    for k in range(3):
        idx = rng.choice(len(inner), m, replace=False); P = [inner[i] for i in idx]
        jin.append(mi_excess_perm([p[0] for p in P], [p[1] for p in P], rng, reps))
        idx = rng.choice(len(cross), m, replace=False) if m < len(cross) else np.arange(len(cross))
        P = [cross[i] for i in idx]
        jx.append(mi_excess_perm([p[0] for p in P], [p[1] for p in P], rng, reps))
    R['J_in'] = float(np.mean(jin)); R['J_x'] = float(np.mean(jx))

    # edge vocabularies: position indicator vs glyph / word, excess over within-line shuffle
    top = {w for w, _ in c.most_common(200)}
    def edge_mi(LL, para_flags):
        fg, fi, lg, li, fw, lw_ = [], [], [], [], [], []
        for ws, ps in zip(LL, para_flags):
            if len(ws) < 3:
                continue
            k = len(ws)
            for j, w in enumerate(ws):
                tw = w if w in top else '#'
                if not ps:
                    fg.append(w[0]); fi.append(j == 0); fw.append(tw)
                lg.append(w[-1]); li.append(j == k - 1); lw_.append(tw)
        def m_(x, y):
            a, na = _codes(x); b, nb = _codes(y); return mi_codes(a, b, na, nb)
        return np.array([m_(fg, fi), m_(lg, li), m_(fw, fi), m_(lw_, li)])
    pf = [L.get('para_start', False) for L in lines]
    o = edge_mi(LW, pf); s = []
    for _ in range(3):
        s.append(edge_mi([prng.sample(ws, len(ws)) for ws in LW], pf))
    e = o - np.mean(s, axis=0)
    R['F1'], R['L1'], R['W1'], R['W2'] = [float(x) for x in e]

    # paragraph-initial gallows
    pg = [L['words'][0][0] in GALLOWS for L in lines if L.get('para_start')]
    og = [L['words'][0][0] in GALLOWS for L in lines if not L.get('para_start')]
    R['PG'] = (np.mean(pg) if pg else 0) - (np.mean(og) if og else 0)

    # line-final q
    fq = np.mean([L['words'][-1][0] == 'q' for L in lines]); aq = np.mean([w[0] == 'q' for w in words])
    R['endq'] = float(fq / max(aq, 1e-9))

    # quota / clumping / modes: per-word class counts, re-dealt within paragraphs
    P = paragraphs(lines)
    cls = {'mg': set('mg'), 'gal': GALLOWS, 'e': {'e'}, 'q': {'q'}, 'a': {'a'}, 'ch': {'C', 'S'}, 'd': {'d'}, 'len': None}
    pw, lid, pid, linelen = [], [], [], []
    li_ = 0
    for pi, p in enumerate(P):
        for L in p:
            for w in L['words']:
                pw.append(w); lid.append(li_); pid.append(pi)
            linelen.append(len(L['words'])); li_ += 1
    pid = np.array(pid); lid = np.array(lid)
    CNT = {k: np.array([len(w) if v is None else sum(ch in v for ch in w) for w in pw], float) for k, v in cls.items()}
    starts = np.concatenate([[0], np.cumsum(linelen)[:-1]])
    line_para = np.array([pi for pi, p in enumerate(P) for _ in p])
    np_ = len(P)

    def line_sums(perm):
        return {k: np.add.reduceat(v[perm], starts) for k, v in CNT.items()}

    def within_var(x):
        pm = np.bincount(line_para, weights=x, minlength=np_) / np.bincount(line_para, minlength=np_)
        r = x - pm[line_para]; return float(np.mean(r * r))

    def resid(S, k):
        t = S['len']; tp = np.bincount(line_para, weights=t, minlength=np_)
        kp = np.bincount(line_para, weights=S[k], minlength=np_)
        rate = kp / np.maximum(tp, 1); ex = t * rate[line_para]
        return (S[k] - ex) / np.sqrt(np.maximum(ex, 1e-9))

    def corrs(S):
        out = []
        for a, b in (('q', 'a'), ('ch', 'e'), ('q', 'd')):
            ra, rb = resid(S, a), resid(S, b)
            out.append(float(np.corrcoef(ra, rb)[0, 1]))
        return out

    ident = np.arange(len(pw))
    So = line_sums(ident)
    ov = [within_var(So[k]) for k in ('mg', 'gal', 'e')]; oc = corrs(So)
    nv, nc = [], []
    for _ in range(reps):
        perm = np.lexsort((rng.random(len(pw)), pid))
        Sn = line_sums(perm)
        nv.append([within_var(Sn[k]) for k in ('mg', 'gal', 'e')]); nc.append(corrs(Sn))
    nv = np.mean(nv, axis=0); nc = np.mean(nc, axis=0)
    R['Q_mg'], R['Q_gal'], R['Q_e'] = [float(a / b) for a, b in zip(ov, nv)]
    R['C_qa'], R['C_che'], R['C_qd'] = [float(a - b) for a, b in zip(oc, nc)]

    # line-mode persistence (N4 score), demeaned per paragraph (paragraphs with >=4 lines)
    def score(ws):
        return (sum(w[0] == 'q' for w in ws) - sum(('ai' in w) or ('ar' in w) for w in ws)) / len(ws)
    PP = [[score(L['words']) for L in p] for p in paragraphs(lines, 4)]
    PP = [[x - np.mean(p) for x in p] for p in PP]
    def lag(PP_):
        xs = [p[i] for p in PP_ for i in range(len(p) - 1)]; ys = [p[i + 1] for p in PP_ for i in range(len(p) - 1)]
        return float(np.corrcoef(xs, ys)[0, 1])
    o = lag(PP); s = [lag([prng.sample(p, len(p)) for p in PP]) for _ in range(30)]
    R['lag1'] = o - float(np.mean(s))
    return {k: float(R[k]) for k in KEYS}


# ---------------- controls ----------------
def redeal_lines(lines, seed=1):
    """Shuffled-line control: the words of each paragraph are re-dealt into its lines at random (line word counts kept)."""
    rng = random.Random(seed); out = []
    P, cur = [], []
    for L in lines:
        if L.get('para_start') and cur:
            P.append(cur); cur = []
        cur.append(L)
    if cur:
        P.append(cur)
    for p in P:
        ws = [w for L in p for w in L['words']]; rng.shuffle(ws)
        out += relayout(p, ws)
    return out


def rewrap_lines(lines, seed=1):
    """Re-wrapped control: word order kept, but each paragraph's line breaks moved by a random offset (continuous
    text wrapped into lines, as in prose)."""
    rng = random.Random(seed); out = []
    P, cur = [], []
    for L in lines:
        if L.get('para_start') and cur:
            P.append(cur); cur = []
        cur.append(L)
    if cur:
        P.append(cur)
    allw = [w for L in lines for w in L['words']]
    k = 0
    for p in P:
        n = sum(len(L['words']) for L in p); off = rng.randrange(1, 4)
        ws = allw[k + off:k + off + n]
        if len(ws) < n:
            ws = ws + allw[:n - len(ws)]
        out += relayout(p, ws); k += n
    return out


def line_order_shuffle(lines, seed=1):
    """Lines kept intact, their order shuffled within each paragraph (paragraph-first line kept first)."""
    rng = random.Random(seed); out = []
    P, cur = [], []
    for L in lines:
        if L.get('para_start') and cur:
            P.append(cur); cur = []
        cur.append(L)
    if cur:
        P.append(cur)
    for p in P:
        rest = p[1:]; rng.shuffle(rest)
        out += [p[0]] + rest
    return out


# ---------------- tolerances and scoring ----------------
BASE_TOL = {'zipf': 0.08, 'ttr': 0.03, 'hapax': 0.05, 'h2': 0.15, 'wlen_m': 0.25, 'wlen_sd': 0.2, 'rep': 0.6,
            'wlen_ac': 0.03, 'J_x': 0.03, 'endq': 0.15}


def tolerances(real, null, se):
    """Pass band per statistic. Effect stats: one third of the real effect (real minus the shuffled-line
    control), word stats: fixed bands; never below 2 sampling SE of the real corpus."""
    tol = {}
    for k in KEYS:
        t = BASE_TOL.get(k)
        if t is None:
            t = abs(real[k] - null[k]) / 3
        tol[k] = max(t, 2 * se.get(k, 0), 1e-3)
    return tol


def distance(sim, real, tol, keys=KEYS, cap=25.0):
    return float(sum(min(((sim[k] - real[k]) / tol[k]) ** 2, cap) for k in keys))


def passes(sim, real, tol):
    return {k: abs(sim[k] - real[k]) <= tol[k] for k in KEYS}


def paragraph_half_se(lines, n=6, seed=3):
    """Sampling SE of each statistic, from random half-splits of paragraphs: SD(half) / sqrt(2)."""
    rng = random.Random(seed)
    P, cur = [], []
    for L in lines:
        if L.get('para_start') and cur:
            P.append(cur); cur = []
        cur.append(L)
    if cur:
        P.append(cur)
    vals = []
    for i in range(n):
        idx = list(range(len(P))); rng.shuffle(idx); h = sorted(idx[:len(P) // 2])
        vals.append(battery([L for j in h for L in P[j]], seed=i))
    return {k: float(np.std([v[k] for v in vals]) / math.sqrt(2)) for k in KEYS}


if __name__ == '__main__':
    import time
    t = time.time(); Z = load_real('ZL3b'); print('lines', len(Z), 'words', sum(len(L['words']) for L in Z))
    b = battery(Z); print('ZL', {k: round(v, 3) for k, v in b.items()}, round(time.time() - t, 1), 's')
    for nm, f in (('redeal', redeal_lines), ('rewrap', rewrap_lines), ('lineorder', line_order_shuffle)):
        print(nm, {k: round(v, 3) for k, v in battery(f(Z)).items()})
