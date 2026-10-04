"""v28 TRY TO KILL v25 (glyph shape predicts glyph behaviour).

v25's negative controls were PRINTED fonts of standard alphabets.  v28 reruns the exact v25 pipeline
(v25_lib behaviour models + Mantel with random glyph-to-shape null; v25_image descriptors zone/frame/topo/combo)
on rivals that could align shape and behaviour for non-featural reasons:
  * medieval manuscripts transcribed at the level of graphic units (allographs, abbreviation marks,
    combining marks, minims as separate units);
  * Cyrillic, Glagolitic (same Old Church Slavonic text in two shape systems), Armenian, Ethiopic (abugida);
  * the Voynich itself under different segmentations of EVA.
Only the corpus and the glyph-image source change; behaviour, descriptors and statistics are v25's.

Data (scratch, not committed): CATMuS-medieval parquet (HF), allographetic late-medieval Castilian HTR set
(Zenodo 8406222, PAGE XML only), ENHG 15th-c. near-allographetic set (Zenodo 21257167), Wikimedia Wikipedia
parquets (am, ti, hy, ru, cu) on HF, fonts Junicode (CTAN), Noto (notofonts), FreeSerif, DejaVu.
"""
import os, sys, re, glob, json, pickle, unicodedata, random
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import v25_lib as L, v25_shapes as S, v25_image as I
from v18_lib import glyphs as vglyphs

ROOT = L.ROOT
CK = os.path.join(L.DATA, 'v28_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = L.SCR
LOOPS = L.LOOPS
TOK = L.TOK
FD = os.path.join(SCR, 'v28fonts')
FONT = dict(
    junicode=os.path.join(FD, 'junicode/opentype/Junicode-Regular.otf'),
    freeserif='/usr/share/fonts/truetype/freefont/FreeSerif.ttf',
    dejavuserif='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    notoeth=os.path.join(FD, 'NotoSansEthiopic.ttf'), notoserifeth=os.path.join(FD, 'NotoSerifEthiopic.ttf'),
    notoglag=os.path.join(FD, 'NotoSansGlagolitic.ttf'),
    notoarm=os.path.join(FD, 'NotoSansArmenian.ttf'), notoserifarm=os.path.join(FD, 'NotoSerifArmenian.ttf'),
    eva=I.FONTS['voynich'], unifont='/usr/share/fonts/opentype/unifont/unifont.otf')

NBSP = ' '


def disp(g):
    """string actually rendered for a unit: combining marks get a no-break-space carrier (ink = mark only)."""
    if g and unicodedata.combining(g[0]):
        return NBSP + g
    return g


# ------------------------------------------------------------------ image similarity for any font
def _render(s, font, size=96):
    f = ImageFont.truetype(font, size)
    im = Image.new('L', (size * 7, size * 3), 0)
    d = ImageDraw.Draw(im)
    d.text((size, int(size * 1.8)), s, font=f, fill=255, anchor='ls')
    return np.asarray(im) > 127, int(size * 1.8), size


def has_ink(s, font):
    a, _, _ = _render(disp(s), font)
    return a.sum() > 20


def covered(g, font):
    from fontTools.ttLib import TTFont
    key = font
    if key not in _CM:
        _CM[key] = TTFont(font, fontNumber=0).getBestCmap()
    return all(ord(c) in _CM[key] for c in g if c != NBSP)


_CM = {}


def image_sims(alph, font, xref):
    """v25_image descriptors with an arbitrary font; xref = reference x-height character."""
    old_render = I.render
    I.render = _render
    try:
        a, base, size = _render(xref, font)
        ys, _ = np.where(a)
        xh = max(ys.max() - ys.min(), 1)
        D = [I.descriptors(disp(g), font, xh) for g in alph]
    finally:
        I.render = old_render
    Zo = np.array([d[0] for d in D]); Fr = np.array([d[1] for d in D]); To = np.array([d[2] for d in D])
    Tz = (To - To.mean(0)) / (To.std(0) + 1e-9)
    out = dict(zone=L.cos_sim(Zo), frame=L.cos_sim(Fr), topo=-np.sqrt(((Tz[:, None] - Tz[None]) ** 2).sum(2)))
    out['combo'] = (I.zs(out['zone']) + I.zs(out['frame']) + I.zs(out['topo'])) / 3
    out['_topo_raw'] = To
    return out


# ------------------------------------------------------------------ word segmentation helpers
def units_nfd(word, keep_case=True):
    """graphic units: each base char and each combining mark is its own unit (NFD)."""
    w = unicodedata.normalize('NFD', word)
    return tuple(w if keep_case else w.lower())


def words_from_lines(lines, unit_fn, bad=r'[0-9\[\]⟦⟧^\uf000-\uf8ff]'):
    out = []
    for ln in lines:
        for w in re.split(r'[\s/·.,:;¶|‸⸗⁖\-‐"\'*#‧჻()]+', ln):
            if not w or re.search(bad, w):
                continue
            u = unit_fn(w)
            if u:
                out.append(u)
    return out


def finish(words, tok=TOK, minc=20):
    """truncate to tok glyph tokens; alphabet = units with >= minc tokens; keep only words fully in it."""
    words = L._truncate(words, tok * 2)
    c = Counter(g for w in words for g in w)
    A = sorted(g for g in c if c[g] >= minc)
    As = set(A)
    ws = [w for w in words if all(g in As for g in w)]
    ws = L._truncate(ws, tok)
    c = Counter(g for w in ws for g in w)
    A = sorted(g for g in A if c[g] >= minc)
    As = set(A)
    ws = [w for w in ws if all(g in As for g in w)]
    return ws, A


# ------------------------------------------------------------------ analysis (exact v25 statistics)
def position_profile(words, alph):
    idx = {g: i for i, g in enumerate(alph)}
    P = np.zeros((len(alph), 4))
    for w in words:
        n = len(w)
        for k, g in enumerate(w):
            if g not in idx:
                continue
            pos = 3 if n == 1 else (0 if k == 0 else (2 if k == n - 1 else 1))
            P[idx[g], pos] += 1
    return P / np.maximum(P.sum(1, keepdims=True), 1)


def analyse(tag, words, alph, sims, models=('ppmi', 'svd', 'potts'), nperm=5000, seed=5, extra_shapes=None, beh=None):
    """returns (rows, summary dict).  sims: dict name -> similarity matrix (image combo per font, hand ...)."""
    rng = np.random.default_rng(seed)
    B = beh if beh is not None else L.behaviour(words, alph, models=models)
    strata = L.freq_strata(B['_freq'])
    iu = np.triu_indices(len(alph), 1)
    Pp = position_profile(words, alph)
    posd = -np.abs(Pp[:, None, :] - Pp[None, :, :]).sum(2)
    rows, summ = [], {}
    for sname, Ssh in sims.items():
        cells = []
        for m in models:
            r, p, z, _ = L.mantel(Ssh, B[m], nperm=nperm, rng=rng)
            r2, p2, _, _ = L.mantel(Ssh, B[m], nperm=2000, rng=rng, strata=strata)
            pr = L.partial_spearman(Ssh[iu], B[m][iu], [posd[iu]])
            summ[(sname, m)] = (r, p, p2, pr)
            cells.append(f'{m} r {r:+.3f} p {p:.4f} (freq-strat p {p2:.4f}; partial|position r {pr:+.3f})')
        rows.append((tag, sname, len(alph), sum(len(w) for w in words), '; '.join(cells)))
    return rows, summ, B


def fmt_row(rid, method, result, verdict=''):
    return (rid, method, result, verdict)


def save(name, obj):
    pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def load(name):
    fn = os.path.join(CK, name)
    return pickle.load(open(fn, 'rb')) if os.path.exists(fn) else None
