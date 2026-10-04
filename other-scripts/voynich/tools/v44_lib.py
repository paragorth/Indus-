"""v44 THE HAND'S COMFORT WRITES THE TEXT: shared library.

Arrow in the dark: if the Voynich is long meaningless writing, its glyph order may be shaped by what is easy for
the hand (short pen travel, few pen lifts, smooth turns), not by a language.

Motor signature of a glyph (all fixed from the drawn shapes BEFORE any statistic was run):
  w            width (x-height units)
  entry (x,y)  where the pen first touches, measured from the glyph's left edge; y: 0 baseline, 1 x-height,
               2 ascender top, -1 descender bottom
  edir         direction of the first stroke (degrees; 0 right, 90 up, -90 down, 180 left)
  exit (x,y)   where the pen leaves the glyph, from the glyph's left edge
  xdir         direction of the last stroke at the pen exit
Joins are derived from geometry, not set by hand: a glyph can be joined OUT if its exit lies in the right 40% and
inside the body zone (-0.25 <= y <= 1.25); joined IN if its entry lies in the left 35% and inside the body zone.

Transition cost a -> b (next glyph starts GAP to the right of a's right edge):
  joined:  dist(exit_a, entry_b) + KAPPA * (turn(xdir_a, travel) + turn(travel, edir_b)) / pi
  lifted:  LIFT + dist(exit_a, entry_b) + KAPPA_AIR * turn(xdir_a... ) ... (air travel, small turn term)
Intra-glyph cost is a constant of the glyph, so it cancels in every reordering test (glyph multisets are kept).

Statistics are linear in the cost matrix c:  C_real = <B, c>, C_shuf = <Q, c>, C_edge = <Q2, c>, where B counts
adjacent glyph pairs inside words, Q the expected pair counts under a uniform within-word shuffle (exact, analytic)
and Q2 under a shuffle that keeps each word's first and last glyph. Saving S = 1 - C_real / C_null.
"""
import os, sys, json, re, math, random
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v44_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

GAP = 0.25
LIFT = 1.5
KAPPA = 0.5
KAPPA_AIR = 0.15


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ motor signatures (hand ductus)
# (w, entry_x, entry_y, edir, exit_x, exit_y, xdir)
VOYNICH_HAND = {
    'o':   (0.8, 0.60, 0.95, 180, 0.65, 0.95, 30),
    'e':   (0.6, 0.50, 0.90, 180, 0.55, 0.10, 0),
    'ch':  (1.2, 0.40, 0.90, 180, 1.15, 1.00, 0),     # two c's, bench bar drawn last to the right
    'sh':  (1.2, 0.40, 0.90, 180, 0.90, 1.50, 45),    # plume last, above the bench
    'a':   (0.8, 0.50, 0.90, 180, 0.75, 0.05, 0),     # c then minim with foot
    'i':   (0.3, 0.10, 1.00, -90, 0.25, 0.05, 30),
    'n':   (0.6, 0.10, 1.00, -90, 0.55, 0.90, 70),
    'r':   (0.5, 0.10, 0.90, -90, 0.50, 0.05, 0),
    'm':   (0.8, 0.10, 1.00, -90, 0.70, 0.90, 90),
    'y':   (0.8, 0.60, 1.00, 180, 0.05, -0.90, 200),
    'd':   (0.8, 0.60, 0.90, 180, 0.10, 1.90, 160),
    'l':   (0.7, 0.30, 1.80, -90, 0.70, 0.30, 30),
    'k':   (1.1, 0.20, 2.00, -90, 0.90, 0.00, -90),
    't':   (1.3, 0.20, 2.00, -90, 1.10, 0.00, -90),
    'f':   (1.4, 0.20, 2.00, -90, 1.30, 1.90, 0),
    'p':   (1.5, 0.20, 2.00, -90, 1.40, 1.90, 0),
    'ckh': (2.0, 0.40, 0.90, 180, 1.90, 1.00, 0),
    'cth': (2.2, 0.40, 0.90, 180, 2.10, 1.00, 0),
    'cfh': (2.2, 0.40, 0.90, 180, 2.10, 1.00, 0),
    'cph': (2.3, 0.40, 0.90, 180, 2.20, 1.00, 0),
    'q':   (0.7, 0.40, 1.00, -135, 0.60, 0.90, 0),
    's':   (0.6, 0.50, 0.90, 180, 0.60, 1.30, 60),
    'g':   (0.8, 0.60, 1.00, 180, 0.60, -0.60, 30),
}

# 15th-century Latin/German cursive (cursiva / bastarda), letters a-z (u=v, i=j); looped ascenders entered by an
# upstroke from the baseline; c-based letters (a c d g o q) begin at the top right of the bowl.
CURSIVA = {
    'a': (0.8, 0.60, 0.90, 180, 0.80, 0.05, 30),
    'b': (0.8, 0.00, 0.10, 70, 0.70, 0.80, 0),
    'c': (0.6, 0.50, 0.90, 180, 0.55, 0.10, 20),
    'd': (0.9, 0.60, 0.90, 180, 0.80, 1.50, -30),
    'e': (0.6, 0.00, 0.40, 30, 0.55, 0.10, 20),
    'f': (0.6, 0.00, 0.20, 70, 0.60, 1.00, 0),
    'g': (0.8, 0.60, 0.90, 180, 0.40, 0.00, 45),
    'h': (0.9, 0.00, 0.10, 70, 0.85, -0.30, -60),
    'i': (0.3, 0.00, 0.60, 60, 0.30, 0.05, 30),
    'k': (0.8, 0.00, 0.10, 70, 0.75, 0.05, 30),
    'l': (0.4, 0.00, 0.10, 70, 0.40, 0.05, 30),
    'm': (1.2, 0.00, 0.80, 60, 1.15, 0.05, 30),
    'n': (0.9, 0.00, 0.80, 60, 0.85, 0.05, 30),
    'o': (0.8, 0.60, 0.90, 180, 0.65, 0.90, 0),
    'p': (0.8, 0.00, 0.80, 60, 0.30, 0.10, 180),
    'q': (0.8, 0.60, 0.90, 180, 0.60, -0.90, -90),
    'r': (0.5, 0.00, 0.70, 60, 0.50, 0.90, 0),
    's': (0.6, 0.00, 0.10, 70, 0.60, 1.60, 0),        # long s, joins at the top
    't': (0.5, 0.10, 1.20, -90, 0.50, 0.05, 30),
    'u': (0.8, 0.00, 0.80, 60, 0.75, 0.05, 30),
    'w': (1.1, 0.00, 0.80, 60, 1.00, 0.80, 0),
    'x': (0.7, 0.00, 0.90, -45, 0.65, 0.00, 30),
    'y': (0.8, 0.00, 0.80, 60, 0.20, -0.80, 200),
    'z': (0.7, 0.00, 0.90, 0, 0.60, -0.50, -45),
}

# Codex Seraphinianus, Ponzi's units, from labelled word images in his training set (connected cursive)
CS_HAND = {
    'i': (0.35, 0.00, 0.90, -90, 0.35, 0.10, 45),
    'r': (0.40, 0.00, 0.10, 70, 0.40, 0.10, -60),
    'g': (0.60, 0.00, 0.90, 0, 0.60, 0.00, 30),
    'n': (0.60, 0.00, 0.80, -90, 0.50, 0.90, 80),
    'f': (0.45, 0.00, 0.10, 75, 0.40, 0.10, 30),
    'e': (0.45, 0.00, 0.30, 30, 0.45, 0.10, 20),
    'B': (0.60, 0.00, 0.90, 0, 0.20, -0.70, 160),
    'j': (0.50, 0.00, 0.80, -90, 0.10, -0.80, 200),
    'l': (0.40, 0.00, 0.10, 70, 0.35, 0.05, 30),
    'L': (0.50, 0.00, 0.10, 70, 0.50, 0.50, 30),
    'O': (1.00, 0.70, 1.00, 180, 0.75, 1.00, 0),
    'o': (0.70, 0.50, 0.90, 180, 0.55, 0.90, 0),
    'd': (0.70, 0.00, 0.30, 30, 0.60, 0.90, 100),
    'u': (0.60, 0.00, 0.80, -90, 0.55, 0.80, 60),
    't': (0.70, 0.00, 0.10, 70, 0.65, 0.10, 30),
    'c': (0.50, 0.40, 0.90, 180, 0.45, 0.10, 0),
    'y': (0.60, 0.00, 0.80, -90, 0.15, -0.80, 200),
    's': (0.60, 0.40, 1.60, 180, 0.55, 0.30, 0),
    'p': (0.70, 0.00, 0.80, -90, 0.30, 0.10, 180),
    'b': (0.60, 0.00, 0.90, 0, 0.60, -0.50, 30),
    'E': (1.00, 0.60, 1.40, 180, 1.00, 0.20, 0),
    'W': (1.00, 0.00, 0.80, -90, 1.00, 0.80, 60),
    'S': (0.80, 0.60, 1.80, 180, 0.70, 0.30, 0),
    'X': (1.00, 0.60, 1.40, 180, 0.90, 0.20, 0),
    'C': (0.90, 0.60, 1.30, 180, 0.80, 0.20, 0),
    'A': (1.00, 0.00, 0.20, 60, 0.90, 0.30, 0),
    'R': (0.90, 0.10, 1.30, -90, 0.85, 0.20, 0),
    'P': (0.90, 0.20, 1.60, -90, 0.80, 0.20, 0),
    'M': (0.90, 0.10, 1.20, 0, 0.80, 0.10, 30),
    'D': (1.00, 0.70, 1.00, 180, 0.90, 0.80, 0),
    'G': (0.80, 0.00, 0.90, 0, 0.60, 0.00, 30),
}


def sig_array(sig, alph):
    return np.array([sig[a] for a in alph], float)


def joinable(S):
    """S: (n,7) signature array -> (join_in, join_out) boolean arrays (geometry-derived)."""
    w, ex, ey, xx, xy = S[:, 0], S[:, 1], S[:, 2], S[:, 4], S[:, 5]
    jin = (ex <= 0.35 * w + 1e-9) & (ey >= -0.25) & (ey <= 1.25)
    jout = (xx >= 0.6 * w - 1e-9) & (xy >= -0.25) & (xy <= 1.25)
    return jin, jout


def _turn(a, b):
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return np.deg2rad(d) / np.pi


def cost_matrix(S, lift=LIFT, kappa=KAPPA, kappa_air=KAPPA_AIR, comp=None):
    """S: (n,7) -> (n,n) transition costs. comp: optional dict of component weights for decomposition."""
    w = S[:, 0]
    exx = S[:, 4][:, None]; exy = S[:, 5][:, None]
    enx = (w[:, None] + GAP + S[:, 1][None, :]); eny = S[:, 2][None, :]
    dx = enx - exx; dy = eny - exy
    dist = np.sqrt(dx ** 2 + dy ** 2)
    trav = np.rad2deg(np.arctan2(dy, dx))
    t1 = _turn(S[:, 6][:, None], trav); t2 = _turn(trav, S[:, 3][None, :])
    jin, jout = joinable(S)
    J = jout[:, None] & jin[None, :]
    cj = dist + kappa * (t1 + t2)
    cl = lift + dist + kappa_air * (t1 + t2)
    if comp is not None:
        return dict(lift=np.where(J, 0.0, 1.0), dist=dist, turn=np.where(J, t1 + t2, 0.0), join=J.astype(float))
    return np.where(J, cj, cl)


def random_sigs(n, rng):
    """random motor signatures in the same ranges as the hand models."""
    w = rng.uniform(0.3, 1.5, n)
    S = np.stack([w, rng.uniform(0, 1, n) * w, rng.uniform(-0.9, 2.0, n), rng.uniform(-180, 180, n),
                  rng.uniform(0, 1, n) * w, rng.uniform(-0.9, 2.0, n), rng.uniform(-180, 180, n)], 1)
    return S


# ------------------------------------------------------------------ corpora (lists of lines of glyph tuples)
def _vlines(src, lang=None, hand=None, fold=None):
    from v18_lib import glyphs as vg
    d = json.load(open(os.path.join(ROOT, 'data', 'derived', f'{src}_lines.json')))
    keep = set(VOYNICH_HAND)
    out = []
    for L in d:
        if L['ltype'] != 'P': continue
        if lang and L['lang'] != lang: continue
        if hand and L.get('hand') != hand: continue
        if fold is not None:
            m = re.match(r'f(\d+)', L['folio'])
            if int(m.group(1)) % 2 != fold: continue
        ln = []
        for w, u in zip(L['words'], L['uncertain']):
            if u or '?' in w: ln.append(None); continue
            g = tuple(vg(w))
            ln.append(g if g and all(x in keep for x in g) else None)
        out.append(ln)
    return out


def voynich(src='ZL3b', **kw):
    return _vlines(src, **kw)


def _latin_word(w, umlaut=True):
    w = w.lower()
    if umlaut:
        for a, b in (('ä', 'a'), ('ö', 'o'), ('ü', 'u'), ('ß', 's'), ('æ', 'ae'), ('œ', 'oe'), ('ſ', 's')):
            w = w.replace(a, b)
    w = w.replace('j', 'i').replace('v', 'u')
    return tuple(w) if w and all(c in CURSIVA for c in w) else None


def v31_corpus(name, maxtok=150000):
    import v31_lib as L31
    C = _v31()
    docs = C[name]['docs']
    out, n = [], 0
    for d in docs:
        for l in d:
            ln = [_latin_word(w) for w in l]
            out.append(ln); n += sum(len(x) for x in ln if x)
            if n >= maxtok: return out
    return out


_V31 = None


def _v31():
    global _V31
    if _V31 is None:
        import v31_lib as L31
        _V31 = json.load(open(L31.CORPORA))
    return _V31


def gibberish_names():
    return sorted(k for k, v in _v31().items() if v['cls'] == 'GIBB')


def cs_lines(collapse=False):
    import v42_lib as L42
    out = []
    for p in L42.cs_pages():
        for pa in p['paras']:
            for l in pa:
                ln = []
                for w in l:
                    if collapse: w = L42.collapse(w)
                    ln.append(tuple(w) if all(c in CS_HAND for c in w) else None)
                out.append(ln)
    return out


def words_of(lines):
    return [w for l in lines for w in l if w]


# ------------------------------------------------------------------ pair-count tensors
def pair_mats(words, alph):
    """-> B (real adjacent pairs), Q (uniform within-word shuffle expectation), Q2 (shuffle keeping first & last).
    Exact expectations; words of length 1 contribute nothing."""
    ix = {a: i for i, a in enumerate(alph)}; n = len(alph)
    B = np.zeros((n, n)); Q = np.zeros((n, n)); Q2 = np.zeros((n, n))
    for w, k in Counter(words).items():
        L = len(w)
        if L < 2: continue
        v = [ix[g] for g in w]
        for a, b in zip(v, v[1:]): B[a, b] += k
        cnt = np.bincount(v, minlength=n).astype(float)
        P = np.outer(cnt, cnt) - np.diag(cnt)        # ordered pairs of distinct positions
        Q += k * P / L                                 # (L-1) transitions * P / (L(L-1))
        if L <= 3:
            for a, b in zip(v, v[1:]): Q2[a, b] += k
        else:
            f, l = v[0], v[-1]; mid = v[1:-1]; m = len(mid)
            c = np.bincount(mid, minlength=n).astype(float)
            Q2[f, :] += k * c / m; Q2[:, l] += k * c / m
            Q2 += k * (np.outer(c, c) - np.diag(c)) / m
    return B, Q, Q2


def saving(B, Qn, c):
    return 1.0 - (B * c).sum() / (Qn * c).sum()


def perm_null(B, Qn, S, n=1000, seed=0, **kw):
    """saving under n random glyph->signature reassignments of the same signature set."""
    rng = np.random.default_rng(seed); out = np.empty(n)
    for i in range(n):
        p = rng.permutation(len(S)); out[i] = saving(B, Qn, cost_matrix(S[p], **kw))
    return out


def boot_ci(words, alph, S, which='Q', nb=200, seed=0, **kw):
    """word-token bootstrap CI of the saving."""
    rng = random.Random(seed); c = cost_matrix(S, **kw)
    vals = []
    ws = list(words)
    for _ in range(nb):
        s = [ws[rng.randrange(len(ws))] for _ in range(len(ws))]
        B, Q, Q2 = pair_mats(s, alph)
        vals.append(saving(B, Q if which == 'Q' else Q2, c))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def alphabet(words, sig):
    a = sorted({g for w in words for g in w})
    assert all(x in sig for x in a), [x for x in a if x not in sig]
    return a


def summarize(words, sig, nperm=1000, seed=0, **kw):
    alph = alphabet(words, sig)
    S = sig_array(sig, alph)
    B, Q, Q2 = pair_mats(words, alph)
    c = cost_matrix(S, **kw)
    out = {}
    for nm, Qn in (('shuf', Q), ('edge', Q2)):
        s = saving(B, Qn, c); nl = perm_null(B, Qn, S, nperm, seed, **kw)
        out[nm] = dict(S=s, null_mean=float(nl.mean()), null_sd=float(nl.std()),
                       z=float((s - nl.mean()) / (nl.std() + 1e-12)), pct=float((nl < s).mean()))
    out['ntrans'] = float(B.sum()); out['nglyph'] = len(alph)
    return out


# ------------------------------------------------------------------ planted effort-minimiser
def plant(words, alph, c, beta, K=24, seed=0, mix=1.0):
    """each word token replaced by one of K random orderings of its glyphs (incl. the original), sampled with
    probability ~ exp(-beta * mean transition cost). mix: fraction of tokens treated (the rest keep their real
    order, so the real sequential structure survives)."""
    rng = random.Random(seed); ix = {a: i for i, a in enumerate(alph)}
    out = []
    for w in words:
        if len(w) < 2 or beta == 0 or rng.random() >= mix:
            out.append(w); continue
        cands = {w}
        for _ in range(K - 1):
            l = list(w); rng.shuffle(l); cands.add(tuple(l))
        cands = list(cands)
        e = np.array([np.mean([c[ix[a], ix[b]] for a, b in zip(x, x[1:])]) for x in cands])
        p = np.exp(-beta * (e - e.min())); p /= p.sum()
        out.append(cands[int(np.searchsorted(np.cumsum(p), rng.random() * 0.999999))])
    return out
