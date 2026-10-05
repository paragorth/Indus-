"""v58 shared helpers: Voynich length fingerprints, C alignment wrapper, encoding simulator, nulls."""
import os, re, json, math, random, ctypes, subprocess
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.dirname(HERE)
CK = os.path.join(VD, 'data', 'v58_ckpt')
BIN = os.path.join(CK, 'bin'); os.makedirs(BIN, exist_ok=True)
SO = os.path.join(BIN, 'v58_align.so')
if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(os.path.join(HERE, 'v58_align.c')):
    subprocess.check_call(['gcc', '-O3', '-shared', '-fPIC', '-o', SO, os.path.join(HERE, 'v58_align.c'), '-lm'])
_lib = ctypes.CDLL(SO)
_D = ctypes.POINTER(ctypes.c_double); _I = ctypes.POINTER(ctypes.c_int)
_lib.align_best.restype = ctypes.c_double
_lib.align_best.argtypes = [_D, ctypes.c_int, _D, ctypes.c_int, _D, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, _I]
_lib.align_path.restype = ctypes.c_int
_lib.align_path.argtypes = [_D, ctypes.c_int, _D, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, _I, _D]

PAR = dict(lam=1.0, s=0.30, pm=0.30, g=0.5)
CGRID = np.round(np.arange(-1.6, 1.61, 0.1), 2)

def _arr(x):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return a, a.ctypes.data_as(_D)

def cgrid_for(x, y, half=0.8, step=0.1):
    c0 = float(np.median(np.log(x)) - np.median(np.log(y)))
    return np.round(np.arange(c0 - half, c0 + half + 1e-9, step), 3)

def align(x, y, par=PAR, cgrid=None):
    """best local alignment score over the c grid (default: median-centred +-0.8); returns (score, c)."""
    if cgrid is None: cgrid = cgrid_for(x, y)
    ax, px = _arr(x); ay, py = _arr(y); ac, pc = _arr(cgrid)
    bc = ctypes.c_int(0)
    v = _lib.align_best(px, len(ax), py, len(ay), pc, len(ac), par['lam'], par['s'], par['pm'], par['g'], ctypes.byref(bc))
    return v, float(cgrid[bc.value])

def path(x, y, c, par=PAR):
    ax, px = _arr(x); ay, py = _arr(y)
    out = (ctypes.c_int * (4 * (len(ax) + len(ay) + 4)))()
    sc = ctypes.c_double(0)
    n = _lib.align_path(px, len(ax), py, len(ay), c, par['lam'], par['s'], par['pm'], par['g'], out, ctypes.byref(sc))
    steps = [(out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]) for i in range(n)][::-1]
    return sc.value, steps

# ---------------- Voynich ----------------
def fnum(f):
    m = re.match(r'f(\d+)([rv])(\d*)', f)
    return (int(m.group(1)), 0 if m.group(2) == 'r' else 1, int(m.group(3) or 0)) if m else (9999, 0, 0)

def voynich_units(name='ZL3b'):
    """paragraphs (P lines, all words) with folio, illus, lang, quire; folio order."""
    recs = json.load(open(os.path.join(VD, 'data', 'derived', name + '_lines.json')))
    paras, cur, fol = [], None, None
    for r in recs:
        if r['ltype'] != 'P': continue
        if r['para_start'] or r['folio'] != fol or cur is None:
            if cur: paras.append(cur)
            cur = {'folio': r['folio'], 'illus': r['illus'], 'lang': r['lang'], 'quire': r['quire'], 'hand': r['hand'], 'w': 0, 'lines': 0}
        fol = r['folio']; cur['w'] += len(r['words']); cur['lines'] += 1
        if r.get('para_end'):
            paras.append(cur); cur = None
    if cur: paras.append(cur)
    paras = [p for p in paras if p['w'] > 0]
    paras.sort(key=lambda p: fnum(p['folio']))   # stable: keeps order within page
    return paras

def voynich_sequences(name='ZL3b'):
    P = voynich_units(name)
    seqs = {}
    def pages(ps):
        d = defaultdict(int); order = []
        for p in ps:
            if p['folio'] not in d: order.append(p['folio'])
            d[p['folio']] += p['w']
        return [d[f] for f in order], order
    H = [p for p in P if p['illus'] == 'H']
    seqs['herbal_pages'] = pages(H)
    seqs['herbalA_pages'] = pages([p for p in H if p['lang'] == 'A'])
    seqs['herbalB_pages'] = pages([p for p in H if p['lang'] == 'B'])
    seqs['herbal_paras'] = ([p['w'] for p in H], [p['folio'] for p in H])
    for code, nm in [('S', 'stars_paras'), ('B', 'bio_paras'), ('P', 'pharma_paras'), ('T', 'text_paras')]:
        ps = [p for p in P if p['illus'] == code]
        seqs[nm] = ([p['w'] for p in ps], [p['folio'] for p in ps])
    seqs['bio_pages'] = pages([p for p in P if p['illus'] == 'B'])
    seqs['stars_pages'] = pages([p for p in P if p['illus'] == 'S'])
    return seqs

# ---------------- encoding simulator (positive controls) ----------------
def encode_lengths(units, rng, verbose=(1, 2), tok_len=5.0, drop=0.05, merge=0.10, split=0.10, layout_sd=0.15):
    """Simulate an opaque verbose cipher with re-segmented tokens from source letter counts:
    each letter -> 1 or 2 glyphs, glyph stream cut into tokens (mean tok_len glyphs),
    then entries dropped / merged with the next / split in two, and a lognormal layout jitter."""
    out = []
    i = 0
    while i < len(units):
        u = units[i]
        if rng.random() < drop: i += 1; continue
        c = u['c']
        if rng.random() < merge and i + 1 < len(units):
            c += units[i + 1]['c']; i += 1
        g = c + np.random.default_rng(rng.randrange(1 << 30)).binomial(c, 0.5)
        t = max(1, np.random.default_rng(rng.randrange(1 << 30)).binomial(g, 1.0 / tok_len))
        t = max(1, int(round(t * math.exp(rng.gauss(0, layout_sd)))))
        if rng.random() < split and t >= 6:
            a = rng.randint(2, t - 2); out += [a, t - a]
        else:
            out.append(t)
        i += 1
    return out

def zp(obs, null):
    null = np.asarray(null, float)
    sd = null.std(ddof=1) if len(null) > 1 else 0
    return ((obs - null.mean()) / sd if sd > 0 else 0.0), (1 + (null >= obs).sum()) / (1 + len(null))
