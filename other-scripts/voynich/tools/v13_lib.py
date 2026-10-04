"""v13 'information budget follows the picture': blind image features + text features.

Image features are computed blindly from reduced-size page scans (no text used):
page crop -> background-relative ink mask -> paint (chroma) mask -> split ink into
small glyph-sized components (text) and large components (drawing) -> drawing region.
"""
import io, json, math, os, re, zlib, random
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DER = os.path.join(ROOT, 'data', 'derived')
W_NORM = 600


# ------------------------------------------------------------------ image
def _longest_run(mask1d):
    best, cur, bs, s = (0, 0), 0, 0, 0
    for i, v in enumerate(mask1d):
        if v:
            if cur == 0:
                s = i
            cur += 1
            if cur > best[1] - best[0]:
                best = (s, i + 1)
        else:
            cur = 0
    return best


def crop_page(rgb):
    """Largest run of parchment-dominated columns/rows, then 3% inner trim."""
    L = rgb.mean(axis=2)
    parch = L > 0.42
    c0, c1 = _longest_run(parch.mean(axis=0) > 0.6)
    r0, r1 = _longest_run(parch[:, c0:c1].mean(axis=1) > 0.6)
    w, h = c1 - c0, r1 - r0
    c0 += int(0.04 * w); c1 -= int(0.04 * w); r0 += int(0.03 * h); r1 -= int(0.03 * h)
    return rgb[r0:r1, c0:c1]


def _block_bg(x, blk=24, q=75):
    h, w = x.shape
    nh, nw = max(1, h // blk), max(1, w // blk)
    small = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            small[i, j] = np.percentile(x[i * blk:(i + 1) * blk, j * blk:(j + 1) * blk], q)
    return ndi.zoom(small, (h / nh, w / nw), order=1)[:h, :w]


def load_norm(fn):
    im = Image.open(fn).convert('RGB')
    rgb = np.asarray(im, dtype=np.float64) / 255.0
    rgb = crop_page(rgb)
    h, w = rgb.shape[:2]
    im2 = Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_NORM, int(round(h * W_NORM / w))), Image.BILINEAR)
    return np.asarray(im2, dtype=np.float64) / 255.0


def segment(rgb, ink_delta=0.10, chroma_thr=0.10):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    L = (R + G + B) / 3
    a = R - G
    b = (R + G) / 2 - B
    bgL = _block_bg(L)
    bga, bgb = np.median(a), np.median(b)
    sharp = L < ndi.gaussian_filter(L, 3) - 0.025
    ink = (L < bgL - ink_delta) & ndi.binary_dilation(sharp, iterations=1)
    chroma = np.hypot(a - bga, b - bgb)
    # paint = strong chroma departure, not just darker brown ink
    paint = (chroma > chroma_thr) & ((a - bga < -0.03) | (a - bga > 0.10) | (b - bgb < -0.06))
    paint = ndi.binary_opening(paint, iterations=1)
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    objs = ndi.find_objects(lab)
    hts = np.array([o[0].stop - o[0].start for o in objs]) if objs else np.zeros(0)
    wds = np.array([o[1].stop - o[1].start for o in objs]) if objs else np.zeros(0)
    areas = ndi.sum(ink, lab, index=np.arange(1, n + 1)) if n else np.zeros(0)
    ok = areas >= 4
    medh = float(np.median(hts[ok])) if ok.any() else 6.0
    big = (hts > 4.0 * medh) | (wds > 20 * medh) | (areas > 60 * medh * medh)
    Hh, Ww = L.shape
    edge_touch = np.array([o[0].start <= 2 or o[1].start <= 2 or o[0].stop >= Hh - 2 or o[1].stop >= Ww - 2
                           for o in objs]) if objs else np.zeros(0, bool)
    big_ids = np.where(big & ok & ~edge_touch)[0] + 1
    draw_ink = np.isin(lab, big_ids)
    region = ndi.binary_dilation(draw_ink | paint, iterations=4)
    region = ndi.binary_fill_holes(region)
    # drop tiny region specks
    rl, rn = ndi.label(region)
    if rn:
        rs = ndi.sum(region, rl, index=np.arange(1, rn + 1))
        region = np.isin(rl, np.where(rs >= 400)[0] + 1)
    text_ink = ink & ~region & np.isin(lab, np.where(ok & ~big & ~edge_touch)[0] + 1)
    return dict(L=L, a=a, b=b, ink=ink, paint=paint & region, draw_ink=draw_ink & region,
                region=region, text_ink=text_ink, medh=medh, bga=bga, bgb=bgb, bgL=bgL)


def _entropy(c):
    c = np.asarray(c, float); c = c[c > 0]
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def features(rgb):
    s = segment(rgb)
    H, W = rgb.shape[:2]
    A = float(H * W)
    reg, paint, dink = s['region'], s['paint'], s['draw_ink']
    f = {}
    f['draw_area'] = reg.sum() / A
    f['paint_area'] = paint.sum() / A
    f['draw_ink'] = dink.sum() / A
    f['text_ink'] = s['text_ink'].sum() / A          # blind text-amount proxy (validation)
    pl, pn = ndi.label(paint)
    ps = ndi.sum(paint, pl, index=np.arange(1, pn + 1)) if pn else np.zeros(0)
    f['n_paint_blobs'] = int((ps >= 15).sum())
    dl, dn = ndi.label(dink, structure=np.ones((3, 3)))
    f['n_draw_comps'] = int(dn)
    rl, rn = ndi.label(reg)
    f['n_regions'] = int(rn)
    gy, gx = np.gradient(s['L'])
    edges = np.hypot(gx, gy) > 0.06
    f['draw_edges'] = (edges & reg).sum() / A
    f['edge_density'] = (edges & reg).sum() / max(1.0, reg.sum())
    # colour: hue histogram of paint pixels in (a,b) plane
    if paint.sum() > 0:
        da = s['a'][paint] - s['bga']; db = s['b'][paint] - s['bgb']
        ang = np.arctan2(db, da)
        hist, _ = np.histogram(ang, bins=12, range=(-math.pi, math.pi))
        f['hue_entropy'] = _entropy(hist)
        green = (da < -0.03)
        blue = (db < -0.06) & ~green
        red = (da > 0.10) & ~green & ~blue
        n = float(paint.sum())
        f['frac_green'] = green.sum() / n; f['frac_red'] = red.sum() / n; f['frac_blue'] = blue.sum() / n
        f['n_colours'] = int(sum(x.sum() / n > 0.05 for x in (green, red, blue)))
    else:
        f.update(hue_entropy=0.0, frac_green=0.0, frac_red=0.0, frac_blue=0.0, n_colours=0)
    # compressed size of the drawing only (text pixels whitened), 4-bit
    g = s['L'].copy()
    g[~reg] = 1.0
    q = np.clip(g * 15, 0, 15).astype(np.uint8)
    base = len(zlib.compress(np.full_like(q, 15).tobytes(), 9))
    f['draw_bytes'] = (len(zlib.compress(q.tobytes(), 9)) - base) / A
    # vertical zones of the drawing bbox: roots (bottom quarter) vs top quarter
    rows = np.where(reg.any(axis=1))[0]
    if len(rows) > 8:
        r0, r1 = rows[0], rows[-1] + 1
        hh = r1 - r0
        f['draw_height'] = hh / H
        top = slice(r0, r0 + hh // 4); bot = slice(r1 - hh // 4, r1)
        f['bottom_edges'] = (edges[bot] & reg[bot]).sum() / A
        f['bottom_unpainted'] = ((dink[bot]) & ~ndi.binary_dilation(paint, iterations=2)[bot]).sum() / A
        f['top_nongreen'] = (paint[top] & ~((s['a'][top] - s['bga']) < -0.03)).sum() / A
    else:
        f.update(draw_height=0.0, bottom_edges=0.0, bottom_unpainted=0.0, top_nongreen=0.0)
    return {k: float(v) for k, v in f.items()}


IMG_FEATS = ['draw_area', 'paint_area', 'draw_ink', 'n_paint_blobs', 'n_draw_comps', 'n_regions',
             'draw_edges', 'edge_density', 'hue_entropy', 'frac_green', 'frac_red', 'frac_blue',
             'n_colours', 'draw_bytes', 'draw_height', 'bottom_edges', 'bottom_unpainted', 'top_nongreen']


# ------------------------------------------------------------------ text
GLYPH_MULTI = [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]


def glyphs(w):
    for a, b in GLYPH_MULTI:
        w = w.replace(a, b)
    return list(w)


def load_pages(name='ZL3b'):
    recs = json.load(open(os.path.join(DER, name + '_lines.json')))
    pages = defaultdict(lambda: {'P': [], 'L': [], 'meta': None, 'nlines': 0})
    for r in recs:
        f = r['folio']
        p = pages[f]
        if p['meta'] is None:
            p['meta'] = {k: r[k] for k in ('quire', 'illus', 'lang', 'hand')}
        if r['ltype'] == 'L':
            p['L'] += r['words']
        else:
            p['P'] += r['words']
            p['nlines'] += 1
    return dict(pages)


def text_features(words, labels, nlines, global_counts, rng):
    t = {}
    n = len(words)
    t['n_words'] = n
    t['log_words'] = math.log1p(n)
    t['n_lines'] = nlines
    t['n_labels'] = len(labels)
    gl = [g for w in words for g in glyphs(w)]
    ng = max(1, len(gl))
    t['n_glyphs'] = len(gl)
    t['word_len'] = len(gl) / max(1, n)
    # rarefied vocabulary at 40 tokens
    if n >= 40:
        t['types40'] = float(np.mean([len(set(rng.sample(words, 40))) for _ in range(30)]))
    else:
        t['types40'] = float('nan')
    t['ms_hapax'] = sum(1 for w in words if global_counts[w] == 1) / max(1, n)
    c = Counter(gl)
    for name, fam in (('gallows', 'ktpf'), ('benched', 'TKPF'), ('chsh', 'CS'), ('e', 'e'), ('o', 'o'),
                      ('y', 'y'), ('d', 'd'), ('l', 'l'), ('r', 'r'), ('s', 's'), ('ain', 'ain'), ('m', 'm')):
        t['g_' + name] = sum(c[x] for x in fam) / ng
    t['w_qo'] = sum(w.startswith('qo') for w in words) / max(1, n)
    t['w_dy'] = sum(w.endswith('dy') for w in words) / max(1, n)
    t['w_aiin'] = sum(w.endswith('aiin') or w.endswith('ain') for w in words) / max(1, n)
    return t


TXT_FEATS = ['log_words', 'n_lines', 'n_labels', 'word_len', 'types40', 'ms_hapax', 'g_gallows', 'g_benched',
             'g_chsh', 'g_e', 'g_o', 'g_y', 'g_d', 'g_l', 'g_r', 'g_s', 'g_ain', 'g_m', 'w_qo', 'w_dy', 'w_aiin']


# ------------------------------------------------------------------ stats
def design(meta_rows, extra=None):
    """Covariate matrix: intercept + section + language + hand dummies (+extra cols)."""
    cols = [np.ones(len(meta_rows))]
    for key in ('illus', 'lang', 'hand'):
        levels = sorted(set(str(m[key]) for m in meta_rows))
        for lv in levels[1:]:
            cols.append(np.array([1.0 if str(m[key]) == lv else 0.0 for m in meta_rows]))
    if extra is not None:
        for e in extra:
            cols.append(np.asarray(e, float))
    return np.column_stack(cols)


def resid(Z, y):
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


def partial_r_matrix(Z, X, Y):
    """Partial correlation of every column of X with every column of Y given Z (NaN-free)."""
    Xr = np.column_stack([resid(Z, X[:, j]) for j in range(X.shape[1])])
    Yr = np.column_stack([resid(Z, Y[:, j]) for j in range(Y.shape[1])])
    Xr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)
    Yr = (Yr - Yr.mean(0)) / (Yr.std(0) + 1e-12)
    return Xr.T @ Yr / len(Xr)


def strata_perm(strata, rng):
    idx = np.arange(len(strata))
    out = idx.copy()
    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    for g in groups.values():
        p = list(g); rng.shuffle(p)
        out[g] = p
    return out


def cv_r2(Z, X, y, folds, lam=10.0):
    """Out-of-fold R^2 gain of ridge(X) on top of OLS(Z) for y."""
    n = len(y)
    pred0 = np.zeros(n); pred1 = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        b0, *_ = np.linalg.lstsq(Z[tr], y[tr], rcond=None)
        pred0[te] = Z[te] @ b0
        r = y[tr] - Z[tr] @ b0
        Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xt = (X[tr] - Xm) / Xs
        # residualize X on Z too
        Bx, *_ = np.linalg.lstsq(Z[tr], Xt, rcond=None)
        Xt = Xt - Z[tr] @ Bx
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ r)
        Xe = (X[te] - Xm) / Xs - Z[te] @ Bx
        pred1[te] = pred0[te] + Xe @ w
    ss = ((y - y.mean()) ** 2).sum()
    return (((y - pred0) ** 2).sum() - ((y - pred1) ** 2).sum()) / ss
