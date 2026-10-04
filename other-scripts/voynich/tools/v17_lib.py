"""v17 shared helpers: THE GHOST OF THE EXEMPLAR.

Hypothesis: the text was copied from an exemplar with different line breaks. The lost exemplar
lines would leave ghosts in mid-line positions of the copy: words that look line-initial (INIT),
words that look line-final (FIN) and repeated stretches (REP, dittography). If so, these ghosts
should line up on a periodic grid of hypothetical exemplar line widths, anchored at paragraph
starts (an exemplar paragraph also starts a new line).

Representation: a 'para' is {'lines': [[word,...],...], 'sec': str, 'fold': 0/1, 'folio': str}.
Gaps: the space before each word except the paragraph's first word. Gaps that are copy line breaks
get weight 0 (they are real line starts); all others carry residual ghost scores.
Positions: x = cumulative width before the gap, in glyph units + 1 per word (space), or in words.
"""
import sys, os, math, random, json, re
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from numba import njit

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'results', 'v17'); os.makedirs(CK, exist_ok=True)
G = vlib.glyphs
MAIN_SECS = ['HA', 'SB', 'BB', 'HB', 'PA', 'TB']

def fnum(f):
    m = re.match(r'f(\d+)([rv])(\d*)', f)
    if not m: return (9999, 0, 0)
    return (int(m.group(1)), 0 if m.group(2) == 'r' else 1, int(m.group(3) or 0))

def voynich_paras(name='ZL3b', minlines=1):
    lines = vlib.load_voynich(name, drop_uncertain=True)
    paras, cur, fol = [], None, None
    def close():
        if cur and len(cur['lines']) >= minlines: paras.append(cur)
    for L in lines:
        if L['para_start'] or L['folio'] != fol or cur is None:
            close()
            sec = (L.get('illus') or 'x') + (L.get('lang') or 'x')
            if sec not in MAIN_SECS: sec = 'other'
            cur = {'folio': L['folio'], 'sec': sec, 'hand': L.get('hand'), 'quire': L.get('quire'), 'lines': []}
        fol = L['folio']; cur['lines'].append(list(L['words']))
        if L.get('para_end'):
            close(); cur = None
    close()
    fols = sorted({p['folio'] for p in paras}, key=fnum)
    fi = {f: i for i, f in enumerate(fols)}
    for p in paras: p['fold'] = fi[p['folio']] % 2
    return paras

# ---------------- marker tables ----------------
def tables(paras, unit=G, alpha=1.0):
    """LLR of first unit (line-initial vs mid-line) and of last unit (line-final vs mid-line).
    Paragraph-first lines are excluded from the 'initial' class (gallows openers)."""
    ini, mid_f, fin, mid_l = Counter(), Counter(), Counter(), Counter()
    for p in paras:
        Ls = p['lines']
        for li, ws in enumerate(Ls):
            for k, w in enumerate(ws):
                u = unit(w)
                if not u: continue
                if k == 0:
                    if li > 0: ini[u[0]] += 1
                else:
                    mid_f[u[0]] += 1
                if k == len(ws) - 1:
                    if li < len(Ls) - 1: fin[u[-1]] += 1
                else:
                    mid_l[u[-1]] += 1
    def llr(a, b):
        keys = set(a) | set(b); na = sum(a.values()) + alpha * len(keys); nb = sum(b.values()) + alpha * len(keys)
        return {k: math.log((a[k] + alpha) / na) - math.log((b[k] + alpha) / nb) for k in keys}
    return {'init': llr(ini, mid_f), 'fin': llr(fin, mid_l)}

def near(a, b):
    if a == b: return True
    if abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 3: return False
    # edit distance <= 1
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b): a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]: i += 1
    return a[i:] == b[i + 1:]

def crossfit_tables(paras, unit=G):
    """tables for fold f fitted on the other fold (no self-reference)."""
    return {f: tables([p for p in paras if p['fold'] != f], unit) for f in (0, 1)}

def gap_table(paras, tabsf, unit=G):
    """Return dict of numpy arrays, one entry per gap. tabsf: {fold: tables}."""
    rows = defaultdict(list)
    secs = sorted({p['sec'] for p in paras}); si = {s: i for i, s in enumerate(secs)}
    for pi, p in enumerate(paras):
        flat, lid, pos, rem = [], [], [], []
        for li, ws in enumerate(p['lines']):
            for k, w in enumerate(ws):
                flat.append(w); lid.append(li); pos.append(k); rem.append(len(ws) - 1 - k)
        n = len(flat)
        if n < 2: continue
        tabs = tabsf[p['fold']]
        widths = [len(unit(w)) + 1 for w in flat]
        cum = np.cumsum([0] + widths)
        for g in range(1, n):
            a, b = flat[g - 1], flat[g]
            ua, ub = unit(a), unit(b)
            brk = lid[g] != lid[g - 1]
            rep = 1.0 if (a == b or (g >= 2 and g + 1 < n and near(flat[g - 2], b) and near(a, flat[g + 1]))) else 0.0
            rows['para'].append(pi); rows['sec'].append(si[p['sec']]); rows['fold'].append(p['fold'])
            rows['xg'].append(cum[g]); rows['xw'].append(g); rows['brk'].append(brk)
            rows['init'].append(tabs['init'].get(ub[0], 0.0) if ub else 0.0)
            rows['fin'].append(tabs['fin'].get(ua[-1], 0.0) if ua else 0.0)
            rows['rep'].append(rep)
            rows['bin'].append(min(pos[g], 4) * 4 + min(rem[g], 3))
        rows['pend_g'].append((pi, cum[-1], n))
    out = {k: np.array(v) for k, v in rows.items() if k != 'pend_g'}
    out['pend'] = rows['pend_g']; out['secs'] = secs
    return out

def residual(gt, chans=('init', 'fin', 'rep')):
    """Residualise each channel by (section, copy-position bin) on mid-line gaps, z-score, sum.
    Copy line-break gaps get weight 0."""
    mid = ~gt['brk']
    W = {}
    for c in chans:
        v = gt[c].astype(float).copy(); r = np.zeros_like(v)
        key = gt['sec'] * 100 + gt['bin']
        for k in np.unique(key[mid]):
            m = mid & (key == k)
            r[m] = v[m] - v[m].mean()
        sd = r[mid].std()
        W[c] = np.where(mid, r / sd if sd > 0 else 0.0, 0.0)
    W['ghost'] = sum(W[c] for c in chans) / math.sqrt(len(chans))
    return W

# ---------------- periodic search ----------------
LG = np.arange(15.0, 100.01, 0.5)   # glyph+space units
LW = np.arange(3.0, 14.01, 0.1)     # words

def periodogram(x, w, grid):
    """anchored (phase 0 at paragraph start) cosine score and free-phase Rayleigh power."""
    ph = 2 * np.pi * np.outer(1.0 / grid, x)       # (L, gaps)
    c = (np.cos(ph) * w).sum(1); s = (np.sin(ph) * w).sum(1)
    q = (w ** 2).sum()
    if q <= 0: return np.zeros(len(grid)), np.zeros(len(grid))
    return c / math.sqrt(q / 2), (c ** 2 + s ** 2) / q

def scan(gt, W, chan='ghost', unit='g', secs=None):
    grid = LG if unit == 'g' else LW
    x = gt['xg'] if unit == 'g' else gt['xw'].astype(float)
    w = W[chan]
    res = {}
    allA = np.zeros(len(grid)); q = 0.0
    for si, s in enumerate(gt['secs']):
        if secs and s not in secs: continue
        m = (gt['sec'] == si) & (w != 0)
        if m.sum() < 50: continue
        A, R = periodogram(x[m], w[m], grid)
        res[s] = (A, R)
        allA += A * math.sqrt((w[m] ** 2).sum()); q += (w[m] ** 2).sum()
    res['ALL'] = (allA / math.sqrt(q) if q else allA, None)
    return grid, res

def summarize(grid, res):
    """max anchored A and max free R per section (search over L)."""
    out = {}
    for s, (A, R) in res.items():
        i = int(np.argmax(A)); d = {'Amax': float(A[i]), 'LA': float(grid[i])}
        if R is not None:
            j = int(np.argmax(R)); d.update({'Rmax': float(R[j]), 'LR': float(grid[j])})
        out[s] = d
    return out

# ---------------- flexible-grid DP ----------------
@njit(cache=True)
def _dp_para(x, s, xend, L, delta):
    n = len(x); lo = L * (1 - delta); hi = L * (1 + delta)
    dp = np.full(n, -1e18)
    for i in range(n):
        xi = x[i]; cand = -1e18
        if xi >= lo and xi <= hi: cand = 0.0
        bestd = 1e18; fb = -1e18
        if abs(xi - L) < bestd: bestd = abs(xi - L); fb = 0.0
        for j in range(i - 1, -1, -1):
            d = xi - x[j]
            if d > 2 * hi: break
            if dp[j] < -1e17: continue
            if d >= lo and d <= hi:
                if dp[j] > cand: cand = dp[j]
            elif d > 0 and abs(d - L) < bestd:
                bestd = abs(d - L); fb = dp[j]
        if cand < -1e17 and xi > lo:
            cand = fb   # fallback: nearest-to-L predecessor (no gap fits the window)
        if cand > -1e17:
            dp[i] = cand + s[i]
    best = 0.0
    for i in range(n):
        if dp[i] > -1e17 and xend - x[i] <= hi and dp[i] > best: best = dp[i]
    if best == 0.0:
        for i in range(n):
            if dp[i] > best: best = dp[i]
    return best

@njit(cache=True)
def dp_section(x, s, starts, xends, grid, delta):
    out = np.zeros(len(grid))
    for k in range(len(grid)):
        tot = 0.0
        for p in range(len(starts) - 1):
            a = starts[p]; b = starts[p + 1]
            if b - a < 1: continue
            tot += _dp_para(x[a:b], s[a:b], xends[p], grid[k], delta)
        out[k] = tot
    return out

def dp_scan(gt, W, chan='ghost', unit='g', delta=0.15, grid=None):
    if grid is None: grid = np.arange(16.0, 90.01, 1.0) if unit == 'g' else np.arange(3.0, 14.01, 0.25)
    x = (gt['xg'] if unit == 'g' else gt['xw']).astype(float)
    pend = {pi: (xe if unit == 'g' else n) for pi, xe, n in gt['pend']}
    res = {}
    for si, sname in enumerate(gt['secs']):
        m = gt['sec'] == si
        if m.sum() < 50: continue
        idx = np.where(m)[0]; paras = gt['para'][idx]
        starts = np.concatenate([[0], np.where(np.diff(paras) != 0)[0] + 1, [len(idx)]]).astype(np.int64)
        xends = np.array([float(pend[paras[a]]) for a in starts[:-1]])
        res[sname] = dp_section(x[idx], W[chan][idx].astype(float), starts, xends, grid.astype(float), delta)
    return grid, res

# ---------------- nulls / resyntheses ----------------
def null_wshuf(paras, rng):
    out = []
    for p in paras:
        q = dict(p); q['lines'] = [rng.sample(ws, len(ws)) for ws in p['lines']]; out.append(q)
    return out

def null_lshuf(paras, rng):
    out = []
    for p in paras:
        Ls = p['lines']; rest = Ls[1:]; rest = rng.sample(rest, len(rest))
        q = dict(p); q['lines'] = [Ls[0]] + rest; out.append(q)
    return out

class Markov:
    """Composed-directly generator: per section, line-initial unigram (para-first and other lines
    separately) + word bigram for the rest, with unigram backoff. Same words-per-line as the input."""
    def __init__(self, paras):
        self.pf = defaultdict(Counter); self.li = defaultdict(Counter); self.bi = defaultdict(lambda: defaultdict(Counter))
        self.uni = defaultdict(Counter)
        for p in paras:
            s = p['sec']
            for li, ws in enumerate(p['lines']):
                (self.pf if li == 0 else self.li)[s][ws[0]] += 1
                for a, b in zip(ws, ws[1:]):
                    self.bi[s][a][b] += 1; self.uni[s][b] += 1
        self.cache = {}
    def _draw(self, rng, c, key):
        if key not in self.cache: self.cache[key] = (list(c.keys()), list(np.cumsum(list(c.values()))))
        ks, cs = self.cache[key]
        import bisect
        return ks[bisect.bisect_right(cs, rng.random() * cs[-1])]
    def gen(self, paras, rng):
        out = []
        for p in paras:
            s = p['sec']; Ls = []
            for li, ws in enumerate(p['lines']):
                src = self.pf[s] if li == 0 else self.li[s]
                w = self._draw(rng, src, ('pf' if li == 0 else 'li', s)); line = [w]
                for _ in range(len(ws) - 1):
                    c = self.bi[s].get(w)
                    if c and sum(c.values()) >= 2 and rng.random() < 0.9:
                        w = self._draw(rng, c, ('bi', s, w))
                    else:
                        w = self._draw(rng, self.uni[s], ('u', s))
                    line.append(w)
                Ls.append(line)
            q = dict(p); q['lines'] = Ls; out.append(q)
        return out

# ---------------- planted exemplar copies (positive controls) ----------------
def copy_from_exemplar(paras, rng, width_factor=1.3, wsd=0.08, p_ditto=0.0, p_skip=0.0, unit=G,
                       fixed_width=None, plant_init=None, plant_fin=None):
    """Treat each para's lines as the exemplar layout (or re-lay it at fixed_width units first,
    optionally planting a line-initial word drawn from plant_init and a line-final from plant_fin).
    Copy with dittography (repeat last 1-2 words of an exemplar line at the start of the next) and
    eye-skip (at an exemplar break, drop 1..4 words) and re-lineate to copy widths ~ N(W, wsd*W),
    W = width_factor * mean exemplar width. Returns (copied paras, mean exemplar width)."""
    ex = []
    for p in paras:
        if fixed_width:
            flat = [w for ws in p['lines'] for w in ws]; Ls, cur, cw = [], [], 0
            tw = fixed_width * (1 + rng.gauss(0, 0.04))
            for w in flat:
                wd = len(unit(w)) + 1
                if cur and cw + wd > tw:
                    Ls.append(cur); cur, cw = [], 0; tw = fixed_width * (1 + rng.gauss(0, 0.04))
                cur.append(w); cw += wd
            if cur: Ls.append(cur)
            if plant_init:
                for L in Ls[1:]:
                    L[0] = plant_init[rng.randrange(len(plant_init))]
            if plant_fin:
                for L in Ls[:-1]:
                    L[-1] = plant_fin[rng.randrange(len(plant_fin))]
            ex.append(Ls)
        else:
            ex.append([list(ws) for ws in p['lines']])
    mw = np.mean([sum(len(unit(w)) + 1 for w in L) for Ls in ex for L in Ls])
    W = width_factor * mw
    out = []
    for p, Ls in zip(paras, ex):
        stream = []
        for li, L in enumerate(Ls):
            L = list(L)
            if li > 0 and p_skip and rng.random() < p_skip:
                k = rng.randint(1, 4); L = L[k:] if len(L) > k else L[-1:]
            if li > 0 and p_ditto and rng.random() < p_ditto:
                prev = Ls[li - 1]; k = rng.randint(1, 2); L = prev[-k:] + L
            stream.extend(L)
        lines, cur, cw = [], [], 0; tw = W * (1 + rng.gauss(0, wsd))
        for w in stream:
            wd = len(unit(w)) + 1
            if cur and cw + wd > tw:
                lines.append(cur); cur, cw = [], 0; tw = W * (1 + rng.gauss(0, wsd))
            cur.append(w); cw += wd
        if cur: lines.append(cur)
        q = dict(p); q['lines'] = lines; out.append(q)
    return out, mw

def save(name, obj):
    with open(os.path.join(CK, name), 'w') as f: json.dump(obj, f, indent=1, default=float)

def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None
