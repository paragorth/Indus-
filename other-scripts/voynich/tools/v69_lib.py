"""v69 WHERE THE SCRIBE SLOWED DOWN: per-word writing-care measures from the page images.

Word boxes come from v18 (data/derived/v18_words.json: 25 Voynich pages, Q20 + f58;
v18_latin_words.json: 28 CREMMA-Medieval-Lat pages). For each word box this module measures,
on the deskewed line band (images stay in the scratch cache, only numbers are saved):
  wpg    box width per glyph (horizontal size / spacing)
  hgt    ink height: 10-90% spread of ink-pixel rows in a +-0.55 pitch band
  slant  shear that makes strokes most vertical (column-projection sharpness), px/px
  ncomp  ink components per glyph (pen lifts; ligatures lower it)
  swid   stroke width: ink area / skeleton length
  gapcv  CV of ink-run widths along the column profile (regularity of glyph spacing)
  irr    within-word CV of local column ink height (size regularity)
  dark   mean ink contrast on ink pixels
"""
import os, json, re, unicodedata, collections, math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.join(HERE, '..', 'data', 'derived')
CK = os.path.join(HERE, '..', 'data', 'v69_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v18'
NS = '{http://www.loc.gov/standards/alto/ns-v4#}'
EVA_MULTI = ['cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']
FEATS = ['wpg', 'hgt', 'slant', 'ncomp', 'swid', 'gapcv', 'irr', 'dark']


def vglyphs(w):
    out, i = [], 0
    while i < len(w):
        for m in EVA_MULTI:
            if w.startswith(m, i):
                out.append(m); i += len(m); break
        else:
            out.append(w[i]); i += 1
    return out


def lglyphs(w):
    return [c for c in w if not c.isspace()]


def load(fn, latin):
    im = Image.open(fn).convert('RGB')
    if latin:
        sc = 2000 / im.width
        im = im.resize((2000, int(im.height * sc)))
    im = np.asarray(im).astype(np.float32)
    gray = im.mean(axis=2)
    small = gray[::8, ::8]
    bg = ndi.median_filter(small, size=15)
    bg = ndi.zoom(bg, (gray.shape[0] / bg.shape[0], gray.shape[1] / bg.shape[1]), order=1)
    bg = bg[:gray.shape[0], :gray.shape[1]]
    con = np.clip(bg - gray, 0, None) / np.maximum(bg, 1)
    con[ndi.binary_dilation(bg < 110, iterations=20)] = 0
    return con


def img_path(which, folio):
    if which == 'V':
        return os.path.join(SCR, 'img', folio + '.jpg')
    fo, f = folio.split(':', 1)
    return os.path.join(SCR, 'lat', 'data', fo, f + '.jpg')


SHEARS = np.arange(-0.8, 0.801, 0.05)


def word_feats(con, x0, x1, yc, slope, s, g):
    H, W = con.shape
    hb = int(0.55 * s); hc = int(0.3 * s)
    xs = np.arange(max(0, x0), min(W, x1))
    if len(xs) < 4:
        return None
    ys = (yc + slope * (xs - (x0 + x1) / 2)).astype(int)
    rr = np.clip(ys[None, :] + np.arange(-hb, hb + 1)[:, None], 0, H - 1)
    c = con[rr, xs[None, :]]
    m = c > 0.15
    m = ndi.binary_opening(m, np.ones((2, 2)))
    if m.sum() < 30:
        return None
    core = m[hb - hc:hb + hc + 1]
    # height
    ry = np.nonzero(m)[0]
    hgt = float(np.percentile(ry, 90) - np.percentile(ry, 10))
    # slant: shear core band, maximise sum of squared column projection
    cy = np.arange(core.shape[0]) - core.shape[0] / 2
    pts_y, pts_x = np.nonzero(core)
    best, bs = -1, 0.0
    for sh in SHEARS:
        xx = np.round(pts_x + sh * cy[pts_y]).astype(int)
        xx -= xx.min()
        p = np.bincount(xx)
        v = float((p.astype(float) ** 2).sum())
        if v > best:
            best, bs = v, sh
    # components (8-connected, >= 8 px) in full band
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    ncomp = float((sizes >= 8).sum()) / g
    # stroke width
    sk = skeletonize(m)
    swid = float(m.sum()) / max(1, sk.sum())
    # spacing regularity on core column profile
    prof = core.sum(0)
    on = prof > 0
    runs, cur = [], 0
    for v in on:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    gapcv = float(np.std(runs) / np.mean(runs)) if len(runs) >= 2 else 0.0
    # size regularity: CV of per-column ink extent (top-to-bottom) where inked
    colh = []
    for j in np.nonzero(m.any(0))[0]:
        r = np.nonzero(m[:, j])[0]
        colh.append(r[-1] - r[0] + 1)
    colh = np.array(colh, float)
    irr = float(colh.std() / colh.mean()) if len(colh) > 2 else 0.0
    dark = float(c[m].mean())
    return {'wpg': (x1 - x0) / g, 'hgt': hgt, 'slant': float(bs), 'ncomp': ncomp, 'swid': swid,
            'gapcv': gapcv, 'irr': irr, 'dark': dark}


def page(args):
    which, pg = args
    con = load(img_path(which, pg['folio']), which == 'L')
    s = pg['pitch']
    gl = vglyphs if which == 'V' else lglyphs
    lines = collections.defaultdict(list)
    for w in pg['words']:
        lines[w['li']].append(w)
    out = []
    for li, ws in lines.items():
        ws = sorted(ws, key=lambda w: w['k'])
        if len(ws) >= 3:
            cx = np.array([(w['x0'] + w['x1']) / 2 for w in ws]); cy = np.array([w['y'] for w in ws])
            slope = float(np.polyfit(cx, cy, 1)[0])
        else:
            slope = 0.0
        for w in ws:
            g = max(1, len(gl(w['word'])))
            f = word_feats(con, w['x0'], w['x1'], w['y'], slope, s, g)
            if f is None:
                continue
            f.update({'folio': pg['folio'], 'li': w['li'], 'k': w['k'], 'nw': w['nw'], 'word': w['word'],
                      'g': g, 'x0': w['x0'], 'x1': w['x1'], 'y': w['y'], 'pitch': s,
                      'para_start': w.get('para_start', False), 'lcost': w['lcost']})
            out.append(f)
    return out


def extract(which, nw=2):
    fn = os.path.join(CK, f'feats_{which}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from concurrent.futures import ProcessPoolExecutor
    pages = json.load(open(os.path.join(DER, 'v18_words.json' if which == 'V' else 'v18_latin_words.json')))
    with ProcessPoolExecutor(nw) as ex:
        res = [r for rows in ex.map(page, [(which, p) for p in pages]) for r in rows]
    json.dump(res, open(fn, 'w'))
    return res


# ---------------- word frequencies (rarity) from whole corpora
def voynich_freq():
    L = json.load(open(os.path.join(DER, 'ZL3b_lines.json')))
    c = collections.Counter(w for r in L for w in r['words'])
    return c, L


def latin_freq():
    fn = os.path.join(CK, 'latin_freq.json')
    if os.path.exists(fn):
        return collections.Counter(json.load(open(fn)))
    c = collections.Counter()
    base = os.path.join(SCR, 'lat', 'data')
    for fo in os.listdir(base):
        for f in os.listdir(os.path.join(base, fo)):
            if not f.endswith('.xml'):
                continue
            try:
                root = ET.parse(os.path.join(base, fo, f)).getroot()
            except Exception:
                continue
            for s in root.iter(NS + 'String'):
                raw = s.get('CONTENT', '')
                txt = ''.join(' ' if unicodedata.category(ch)[0] in 'PZ' else (ch if unicodedata.category(ch)[0] in 'LN' else '') for ch in raw)
                c.update(w.lower() for w in txt.split() if w)
    json.dump(c, open(fn, 'w'))
    return c
