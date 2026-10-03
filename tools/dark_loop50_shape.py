"""Shared glyph shape measures for S-DARK-50 (same definitions as tools/dark_loop50_c1.py)."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

def render_char(fontpath, ch, size=192, out=96):
    f = ImageFont.truetype(fontpath, size)
    im = Image.new('L', (size * 3, size * 3), 0); d = ImageDraw.Draw(im)
    d.text((size // 2, size // 2), ch, font=f, fill=255)
    a = np.array(im) > 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0: return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape; s = max(h, w)
    pad = np.zeros((s, s), bool); pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return np.array(Image.fromarray(pad.astype(np.uint8) * 255).resize((out, out), Image.BILINEAR)) > 60

def thin(img):
    im = img.copy().astype(np.uint8); changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(im, 1)
            p2 = P[:-2, 1:-1]; p3 = P[:-2, 2:]; p4 = P[1:-1, 2:]; p5 = P[2:, 2:]
            p6 = P[2:, 1:-1]; p7 = P[2:, :-2]; p8 = P[1:-1, :-2]; p9 = P[:-2, :-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0) if step == 0 else (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            m = (im == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any(): im[m] = 0; changed = True
    return im.astype(bool)

def skeleton_stats(sk):
    P = np.pad(sk.astype(np.uint8), 1)
    nb = sum(np.roll(np.roll(P, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))[1:-1, 1:-1]
    ends = int(((nb == 1) & sk).sum()); junc = (nb >= 3) & sk
    segs = sk & ~ndi.binary_dilation(junc, structure=np.ones((3, 3)))
    _, nseg = ndi.label(segs, structure=np.ones((3, 3))); _, njunc = ndi.label(junc, structure=np.ones((3, 3)))
    return ends, njunc, max(nseg, 1)

def measures_mask(g):
    if g is None or g.sum() < 10: return None
    g = ndi.binary_closing(g, iterations=1) | g
    filled = ndi.binary_fill_holes(g)
    _, holes = ndi.label(filled & ~g, structure=np.ones((3, 3)))
    _, comps = ndi.label(g, structure=np.ones((3, 3)))
    sm = ndi.gaussian_filter(g.astype(float), 1.5); gy, gx = np.gradient(sm)
    edge = g ^ ndi.binary_erosion(g)
    ang = (np.degrees(np.arctan2(gy[edge], gx[edge])) % 180.0); mag = np.hypot(gx[edge], gy[edge])
    h, _ = np.histogram(ang, bins=18, range=(0, 180), weights=mag); h = h / max(h.sum(), 1e-9)
    curv = 1.0 - np.sort(h)[-4:].sum()
    def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)
    sym = max(iou(g, g[:, ::-1]), iou(g, g[::-1, :]))
    Pm = float(edge.sum()); A = float(g.sum()); perim = Pm * Pm / (4 * math.pi * A)
    sk = thin(g); ends, njunc, nseg = skeleton_stats(sk)
    return dict(holes=int(holes), comps=int(comps), curv=float(curv), asym=float(1 - sym), perim=float(perim),
                strokes=int(nseg), logstrokes=math.log(nseg), ends=int(ends), junctions=int(njunc), ink=float(g.mean()))

KEYS5 = ('holes', 'curv', 'asym', 'perim', 'logstrokes')   # far_stroke needs a glyph-similarity matrix; omitted cross-script

def add_indices(rows, base=None, keys=KEYS5):
    """z-score within the script (over `base` rows or all) and add pict_shape5 (5-part) and pict_curv (curv+asym)."""
    base = base or rows
    mu = {k: np.mean([r[k] for r in base]) for k in keys}; sd = {k: np.std([r[k] for r in base]) + 1e-9 for k in keys}
    for r in rows:
        z = {k: (r[k] - mu[k]) / sd[k] for k in keys}
        r['pict_shape5'] = float(np.mean(list(z.values())))
        r['pict_curv'] = float((z['curv'] + z['asym']) / 2)
    return rows

def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
