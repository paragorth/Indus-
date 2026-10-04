"""v38 'similar plants, similar text': segmentation, visual descriptors, text similarity,
confound-partialled Mantel tests. Images are read from a scratch cache only.
"""
import os, re, json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
from collections import Counter
from scipy import ndimage as ndi
from PIL import Image
from skimage import color, morphology, measure, feature, transform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DER = os.path.join(ROOT, 'data', 'derived')
CK = os.path.join(ROOT, 'data', 'v38_ckpt')
LOOPS = os.path.join(ROOT, 'loops')
os.makedirs(CK, exist_ok=True)


# ---------------------------------------------------------------- segmentation
def page_crop(rgb):
    """Crop to the vellum: largest bright, low-saturation-ish component."""
    hsv = color.rgb2hsv(rgb)
    v = hsv[..., 2]
    m = v > np.percentile(v, 35) * 0.85
    m = morphology.binary_opening(m, morphology.disk(3))
    lab, n = ndi.label(m)
    if n == 0:
        return rgb, (0, 0)
    sz = ndi.sum(m, lab, range(1, n + 1))
    k = 1 + int(np.argmax(sz))
    ys, xs = np.where(lab == k)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    y0 += int(.02 * h); y1 -= int(.02 * h); x0 += int(.05 * w); x1 -= int(.05 * w)
    return rgb[y0:y1, x0:x1], (y0, x0)


def voynich_plant(rgb, dE_thr=17, ink_drop=28):
    """Return (crop rgb, plant mask, paint mask, ink mask, vellum lab).
    Non-vellum pixels are grouped into connected components; glyph- and word-sized
    components (text) are dropped, large ones (drawing) kept."""
    rgb, _ = page_crop(rgb)
    lab = color.rgb2lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    vel = np.median(lab.reshape(-1, 3), axis=0)
    chroma_dev = np.hypot(a - vel[1], b - vel[2])
    dE = np.sqrt((L - vel[0]) ** 2 + (a - vel[1]) ** 2 + (b - vel[2]) ** 2)
    ink = (L < vel[0] - ink_drop)
    paint = (chroma_dev > 12) & ~ink
    nonvel = morphology.binary_dilation(dE > dE_thr, morphology.disk(1))
    H, W = nonvel.shape
    lbl = measure.label(nonvel)
    keep = np.zeros_like(nonvel)
    for r in measure.regionprops(lbl):
        y0, x0, y1, x1 = r.bbox
        big = r.area > 0.003 * H * W or (y1 - y0) > 0.09 * H
        edge = y0 <= 1 or x0 <= 1 or y1 >= H - 1 or x1 >= W - 1
        strip = edge and ((x1 - x0) < 0.06 * W or (y1 - y0) < 0.06 * H or r.extent < 0.08)
        if big and not strip and (y1 - y0) < 0.97 * H and (x1 - x0) < 0.97 * W:
            keep |= lbl == r.label
    keep = ndi.binary_fill_holes(morphology.binary_closing(keep, morphology.disk(2)))
    return rgb, keep, paint & keep, ink, vel


def masked_plant_image(rgb, mask, size=224):
    """Plant-only image: everything outside the (dilated) mask replaced by the vellum colour;
    cropped to the mask bounding box and padded square."""
    m = morphology.binary_dilation(mask, morphology.disk(3))
    if m.sum() < 50:
        m = np.ones_like(m)
    out = rgb.copy().astype(float)
    bg = np.median(rgb[~m].reshape(-1, 3), axis=0) if (~m).sum() > 100 else np.array([230, 220, 200.])
    out[~m] = bg
    ys, xs = np.where(m)
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = out.shape[:2]
    s = max(h, w)
    sq = np.ones((s, s, 3)) * bg
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = out
    return np.asarray(Image.fromarray(sq.clip(0, 255).astype(np.uint8)).resize((size, size), Image.BILINEAR))


# ---------------------------------------------------------------- hand-crafted descriptors
def shape_desc(mask):
    """Silhouette descriptors (scale-free)."""
    ys, xs = np.where(mask)
    if len(ys) < 50:
        return np.zeros(40)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    rows = transform.resize(m.astype(float), (16, 8), anti_aliasing=True)
    rowprof = rows.mean(1)
    colprof = rows.mean(0)
    hull = morphology.convex_hull_image(m)
    solidity = m.sum() / max(hull.sum(), 1)
    per = measure.perimeter(m)
    compact = per ** 2 / max(m.sum(), 1)
    nc = measure.label(m).max()
    mom = measure.moments_hu(measure.moments_normalized(measure.moments_central(m.astype(float)), 3))
    hu = -np.sign(mom) * np.log10(np.abs(mom) + 1e-12)
    return np.concatenate([[math.log(h / w), m.mean(), solidity, math.log(compact), math.log(1 + nc)],
                           rowprof, colprof, hu, [0, 0, 0, 0]])[:40]


def colour_desc(rgb, paint):
    """Paint colour: hue x saturation histogram (12 x 2) + mean Lab of paint + colour fractions by vertical third."""
    if paint.sum() < 30:
        return np.zeros(24 + 3 + 9)
    hsv = color.rgb2hsv(rgb)
    hh, ss = hsv[..., 0][paint], hsv[..., 1][paint]
    H, _, _ = np.histogram2d(hh, ss, bins=[12, 2], range=[[0, 1], [0, 1]])
    H = (H / H.sum()).ravel()
    lab = color.rgb2lab(rgb)[paint].mean(0) / 50.
    ys = np.where(paint)[0]
    y0, y1 = ys.min(), ys.max() + 1
    thirds = []
    for k in range(3):
        sl = np.zeros_like(paint)
        sl[y0 + (y1 - y0) * k // 3: y0 + (y1 - y0) * (k + 1) // 3] = True
        p = paint & sl
        hp = hsv[..., 0][p]
        n = max(len(hp), 1)
        green = ((hp > 0.17) & (hp < 0.45)).sum() / n
        red = ((hp < 0.08) | (hp > 0.9)).sum() / n
        blue = ((hp > 0.5) & (hp < 0.75)).sum() / n
        thirds += [green, red, blue]
    return np.concatenate([H, lab, thirds])


def texture_desc(img224):
    """HOG of the plant-only image (edge orientation layout: leaf shape, venation, serration)."""
    g = color.rgb2gray(img224)
    g = transform.resize(g, (96, 96), anti_aliasing=True)
    return feature.hog(g, orientations=8, pixels_per_cell=(24, 24), cells_per_block=(1, 1), feature_vector=True)


def edge_orient_desc(img224):
    """Global edge-orientation and curvature statistics (rotation-sensitive, location-free)."""
    g = color.rgb2gray(img224)
    gy, gx = np.gradient(ndi.gaussian_filter(g, 1.0))
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    w = mag > np.percentile(mag, 80)
    H, _ = np.histogram(ang[w], bins=12, range=(0, np.pi), weights=mag[w])
    H = H / max(H.sum(), 1e-9)
    lbp = feature.local_binary_pattern((g * 255).astype(np.uint8), 8, 1, 'uniform')
    L, _ = np.histogram(lbp, bins=10, range=(0, 10))
    L = L / L.sum()
    return np.concatenate([H, L])


# ---------------------------------------------------------------- text
def text_profiles(pages_words):
    """pages_words: list of token lists. Returns dict of page x feature matrices."""
    n = len(pages_words)
    tf = [Counter(w) for w in pages_words]
    df = Counter()
    for c in tf:
        df.update(set(c))
    vocab = sorted(df)
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((n, len(vocab)))
    for i, c in enumerate(tf):
        for w, k in c.items():
            X[i, vi[w]] = k
    B = (X > 0).astype(float)
    dfa = B.sum(0)
    idf = np.log(n / dfa)
    T = np.log1p(X) * idf
    tri = [Counter() for _ in range(n)]
    for i, ws in enumerate(pages_words):
        for w in ws:
            s = '<' + w + '>'
            for k in range(len(s) - 2):
                tri[i][s[k:k + 3]] += 1
    tv = sorted(set().union(*tri))
    ti = {t: j for j, t in enumerate(tv)}
    G = np.zeros((n, len(tv)))
    for i, c in enumerate(tri):
        for t, k in c.items():
            G[i, ti[t]] = k
    return dict(X=X, B=B, T=T, G=G, dfa=dfa, vocab=vocab)


def cos_sim(M):
    Z = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    return Z @ Z.T


def text_sims(prof):
    B, T, G, dfa = prof['B'], prof['T'], prof['G'], prof['dfa']
    inter = B @ B.T
    sz = B.sum(1)
    jac = inter / np.maximum(sz[:, None] + sz[None, :] - inter, 1)
    rare = ((dfa >= 2) & (dfa <= 4)).astype(float)
    Br = B * rare
    shared_rare = Br @ Br.T
    mid = ((dfa >= 2) & (dfa <= max(5, int(0.1 * len(B))))).astype(float)
    Bm = B * mid
    sm = Bm @ Bm.T
    smn = sm / np.sqrt(np.outer(np.maximum(Bm.sum(1), 1), np.maximum(Bm.sum(1), 1)))
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1)
    Gc = Gn - Gn.mean(0)
    return dict(jaccard=jac, tfidf=cos_sim(T), rare=shared_rare, midcos=smn, tri=cos_sim(Gc))


# ---------------------------------------------------------------- Mantel machinery
def upper(M):
    i, j = np.triu_indices(M.shape[0], 1)
    return M[i, j]


def zs(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1)


class Partial:
    """Partial Mantel by residualisation on pair-level confounds; permutation of page labels
    (optionally within strata) applied to the first matrix, then re-residualised."""

    def __init__(self, conf_mats, n):
        self.n = n
        self.iu = np.triu_indices(n, 1)
        X = [np.ones(len(self.iu[0]))] + [zs(upper(C)) for C in conf_mats]
        self.X = np.column_stack(X)
        self.P = np.linalg.pinv(self.X)

    def res(self, v):
        return v - self.X @ (self.P @ v)

    def r(self, A, B):
        a = zs(self.res(upper(A)))
        b = zs(self.res(upper(B)))
        return float(np.mean(a * b))

    def perm_null(self, A, B, nperm, rng, strata=None):
        b = zs(self.res(upper(B)))
        out = np.empty(nperm)
        for k in range(nperm):
            p = perm(self.n, rng, strata)
            Ap = A[np.ix_(p, p)]
            out[k] = np.mean(zs(self.res(upper(Ap))) * b)
        return out


def perm(n, rng, strata=None):
    if strata is None:
        return rng.permutation(n)
    p = np.arange(n)
    for s in set(strata):
        idx = np.where(np.asarray(strata) == s)[0]
        p[idx] = rng.permutation(idx)
    return p


def feat_sim(F):
    """Feature matrix -> similarity via standardised columns and negative Euclidean distance."""
    F = np.asarray(F, float)
    sd = F.std(0)
    F = (F - F.mean(0)) / np.where(sd > 0, sd, 1)
    d = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    return -d


# ---------------------------------------------------------------- Gerard (control)
def gerard_ocr(xml_path, wanted):
    """{scan index: (page_w, page_h, [(word, x0, y0, x1, y1)])} for wanted scan indices."""
    import xml.etree.ElementTree as ET
    out, k = {}, 0
    for ev, el in ET.iterparse(xml_path, events=('end',)):
        if el.tag == 'OBJECT':
            if k in wanted:
                ws = []
                for w in el.iter('WORD'):
                    c = [int(x) for x in w.get('coords').split(',')[:4]]
                    x0, y1, x1, y0 = c  # djvu coords: left, bottom, right, top
                    ws.append((w.text or '', x0, y0, x1, y1))
                out[k] = (int(el.get('width')), int(el.get('height')), ws)
            k += 1
            el.clear()
    return out


def gerard_plant(rgb, ocr):
    W0, H0, ws = ocr
    g = color.rgb2gray(rgb)
    H, W = g.shape
    sx, sy = W / W0, H / H0
    txt = np.zeros_like(g, bool)
    for w, x0, y0, x1, y1 in ws:
        if len(re.sub(r'[^A-Za-z]', '', w)) < 1:
            continue
        txt[max(0, int(y0 * sy) - 2):int(y1 * sy) + 2, max(0, int(x0 * sx) - 2):int(x1 * sx) + 2] = True
    bg = np.median(g)
    ink = g < bg - 0.22
    ink &= ~txt
    blk = morphology.binary_closing(ink, morphology.disk(3))
    lbl = measure.label(blk)
    keep = np.zeros_like(blk)
    for r in measure.regionprops(lbl):
        y0, x0, y1, x1 = r.bbox
        if r.area > 0.006 * H * W and (y1 - y0) > 0.06 * H and (x1 - x0) > 0.06 * W \
                and (x1 - x0) < 0.9 * W and r.extent > 0.04:
            keep |= lbl == r.label
    keep = ndi.binary_fill_holes(morphology.binary_closing(keep, morphology.disk(4)))
    return rgb, keep, np.zeros_like(keep), ink, None


# ---------------------------------------------------------------- shared analysis
VIS_FAMS = ['shape', 'colour', 'hog', 'edge', 'r18', 'dino']
TXT_METRICS = ['jaccard', 'tfidf', 'rare', 'midcos', 'tri']


def vis_sims(vis, keys, fams=VIS_FAMS):
    out = {}
    for f in fams:
        F = np.array([vis[k][f] for k in keys], float)
        if F.std() == 0:
            continue
        if f in ('r18', 'dino'):
            out[f] = cos_sim(F - F.mean(0))
        else:
            out[f] = feat_sim(F)
    return out


def mantel_table(V, T, part, nperm, rng, strata=None):
    """Partial Mantel r for every (visual family, text metric) with a joint permutation null
    (same page permutation for all families). Returns obs dict, null array, omnibus stats."""
    fams, mets = list(V), list(T)
    Tres = {m: zs(part.res(upper(T[m]))) for m in mets}
    obs = {(f, m): float(np.mean(zs(part.res(upper(V[f]))) * Tres[m])) for f in fams for m in mets}
    null = np.empty((nperm, len(fams), len(mets)))
    for k in range(nperm):
        p = perm(part.n, rng, strata)
        for i, f in enumerate(fams):
            a = zs(part.res(upper(V[f][np.ix_(p, p)])))
            for j, m in enumerate(mets):
                null[k, i, j] = np.mean(a * Tres[m])
    O = np.array([[obs[(f, m)] for m in mets] for f in fams])
    omni = O.mean()
    omni_null = null.mean((1, 2))
    mx = O.max()
    mx_null = null.max((1, 2))
    cell_p = {(f, m): float((1 + (null[:, i, j] >= O[i, j]).sum()) / (1 + nperm))
              for i, f in enumerate(fams) for j, m in enumerate(mets)}
    cell_z = {(f, m): float((O[i, j] - null[:, i, j].mean()) / null[:, i, j].std())
              for i, f in enumerate(fams) for j, m in enumerate(mets)}
    return dict(fams=fams, mets=mets, obs=O.tolist(),
                omni=float(omni), omni_z=float((omni - omni_null.mean()) / omni_null.std()),
                omni_p=float((1 + (omni_null >= omni).sum()) / (1 + nperm)),
                max=float(mx), max_p=float((1 + (mx_null >= mx).sum()) / (1 + nperm)),
                cell_p={'%s|%s' % k: v for k, v in cell_p.items()},
                cell_z={'%s|%s' % k: v for k, v in cell_z.items()},
                fam_z={f: float(np.mean([cell_z[(f, m)] for m in mets])) for f in fams},
                met_z={m: float(np.mean([cell_z[(f, m)] for f in fams])) for m in mets})


def plant_text(pages_words, Z, frac, rng, pool, beta=2.0):
    """Planted link: replace a fraction of each page's tokens with words drawn from a softmax
    over a word pool whose logits are a random linear function of the page's visual vector Z."""
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    W = rng.normal(size=(Z.shape[1], len(pool))) / np.sqrt(Z.shape[1])
    out = []
    for i, ws in enumerate(pages_words):
        lg = beta * (Z[i] @ W)
        p = np.exp(lg - lg.max()); p /= p.sum()
        ws = list(ws)
        k = int(round(frac * len(ws)))
        idx = rng.choice(len(ws), k, replace=False)
        rep = rng.choice(len(pool), k, p=p)
        for a, b in zip(idx, rep):
            ws[a] = pool[b]
        out.append(ws)
    return out
