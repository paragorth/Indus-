"""LA-80 shared helpers: document table, image fetch, outline measurement, text features."""
import json, re, os, sys, hashlib, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la80_ckpt')
IMG = os.path.join(CK, 'img')
os.makedirs(IMG, exist_ok=True)
sys.path.insert(0, HERE)
BASE = 'https://raw.githubusercontent.com/mwenge/lineara.xyz/master/'

def image_map():
    s = open(os.path.join(DATA, 'LinearAInscriptions.js')).read()
    out = {}
    for m in re.finditer(r'\["([^"]+)",\{(.*?)\n\}\]', s, re.S):
        im = re.search(r'"images": \[(.*?)\]', m.group(2), re.S)
        out[m.group(1)] = re.findall(r'"([^"]+)"', im.group(1)) if im else []
    return out

def fetch(doc_ids):
    im = image_map(); got = {}
    for d in doc_ids:
        paths = [p for p in im.get(d, []) if 'noimage' not in p]
        if not paths: continue
        p = paths[0]; fn = os.path.join(IMG, os.path.basename(p))
        if not os.path.exists(fn) or os.path.getsize(fn) < 500:
            subprocess.run(['curl', '-sS', '-f', '-o', fn, BASE + p], check=False)
        if os.path.exists(fn) and os.path.getsize(fn) > 500:
            got[d] = fn
    return got

def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

# ---------------- outline measurement ----------------
from PIL import Image
from scipy import ndimage as ndi

def load_gray(fn):
    im = Image.open(fn)
    dpi = im.info.get('dpi', (0, 0))[0]
    g = np.asarray(im.convert('L'), dtype=float)
    return g, float(dpi)

def segment(g, sig=3.0, frac=0.50):
    """Flood the background from the border on a smoothed image: background = pixels brighter
    than a threshold set between the border level and the central (object) level, plus low local
    grain; object = everything not reached. Falls back to a blurred marker watershed when the
    object is almost as light as the plate."""
    from skimage.filters import gaussian, sobel
    from skimage.segmentation import watershed
    b = gaussian(g, sig, preserve_range=True)
    H, W = b.shape
    bw = 4
    band = np.zeros(b.shape, bool); band[:bw] = band[-bw:] = True; band[:, :bw] = band[:, -bw:] = True
    # smooth background surface: quadratic fit to the brighter border pixels
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    bb = b[band]
    keep = band & (b >= np.percentile(bb, 40))
    A = np.c_[np.ones(keep.sum()), xx[keep], yy[keep], xx[keep] ** 2, yy[keep] ** 2, xx[keep] * yy[keep]]
    coef, *_ = np.linalg.lstsq(A, b[keep], rcond=None)
    surf = coef[0] + coef[1] * xx + coef[2] * yy + coef[3] * xx ** 2 + coef[4] * yy ** 2 + coef[5] * xx * yy
    b = b - surf + np.median(b[keep])
    bg = np.median(b[keep])
    cy, cx, hy, hx = H // 2, W // 2, H // 6, W // 6
    ob = np.median(b[cy - hy:cy + hy, cx - hx:cx + hx])
    m1 = ndi.uniform_filter(b, 15); m2 = ndi.uniform_filter(b * b, 15)
    lstd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    if bg - ob > 18:
        T = bg - frac * (bg - ob)
        cand = (b > T)
    else:
        cand = (np.abs(b - bg) < 8) & (lstd < np.percentile(lstd[band], 80) * 1.3)
    lab, n = ndi.label(cand)
    edge_labels = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    obj = ~np.isin(lab, list(edge_labels))
    obj = ndi.binary_opening(obj, iterations=3)
    l2, n = ndi.label(obj)
    if n == 0:
        return None
    sz = ndi.sum(obj, l2, range(1, n + 1))
    obj = l2 == (int(np.argmax(sz)) + 1)
    obj = ndi.binary_closing(obj, iterations=3)
    obj = ndi.binary_fill_holes(obj)
    # cut thin leaks into the plate margin: opening with a disk of 3 % of the object's size
    r = max(3, int(0.03 * np.sqrt(obj.sum())))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    obj2 = ndi.binary_opening(obj, structure=(xx * xx + yy * yy) <= r * r)
    l3, n3 = ndi.label(obj2)
    if n3:
        sz = ndi.sum(obj2, l3, range(1, n3 + 1))
        obj = l3 == (int(np.argmax(sz)) + 1)
    return obj

def _hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3: return np.array(pts, float)
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1], float)

def shape_features(obj, g, dpi):
    from skimage.measure import regionprops, find_contours
    H, W = obj.shape
    rp = regionprops(obj.astype(int))[0]
    area = rp.area
    ys, xs = np.nonzero(obj)
    pts = np.c_[xs, ys].astype(float)
    bd = obj ^ ndi.binary_erosion(obj)
    by, bx = np.nonzero(bd)
    from scipy.spatial import ConvexHull
    bp = np.c_[bx, by].astype(float)
    hull = bp[ConvexHull(bp).vertices]
    # min-area rectangle by rotating calipers over hull edges
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        if np.hypot(*e) == 0: continue
        a = np.arctan2(e[1], e[0])
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = hull @ R.T
        w_, h_ = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        if best is None or w_ * h_ < best[0]:
            best = (w_ * h_, a, R, q.min(0), w_, h_)
    rect_area, a, R, mn, rw, rh = best
    P = (pts @ R.T) - mn            # object pixels in rectangle frame
    long_is_x = rw >= rh
    L_, S_ = max(rw, rh), min(rw, rh)
    u = P[:, 0] if long_is_x else P[:, 1]   # coordinate along long axis
    v = P[:, 1] if long_is_x else P[:, 0]
    # corner fill: share of each corner square (side 0.2*short side) covered
    c = max(2.0, 0.2 * S_)
    fills = []
    for (u0, v0) in [(0, 0), (0, S_ - c), (L_ - c, 0), (L_ - c, S_ - c)]:
        inside = (u >= u0) & (u < u0 + c) & (v >= v0) & (v < v0 + c)
        fills.append(inside.sum() / (c * c))
    fills = np.clip(fills, 0, 1)
    # taper: width across short axis in end bands along long axis
    def band_w(lo, hi):
        s = (u >= lo * L_) & (u < hi * L_)
        if s.sum() < 5: return 0.0
        return np.ptp(v[s]) / S_
    w1, w2 = band_w(0, 0.15), band_w(0.85, 1.0)
    wm = band_w(0.45, 0.55)
    # perimeter ratios
    cont = find_contours(np.pad(obj, 2).astype(float), 0.5)
    cc = max(cont, key=len)
    per = np.sum(np.hypot(*np.diff(cc, axis=0).T))
    hp = np.sum(np.hypot(*np.diff(np.vstack([hull, hull[:1]]), axis=0).T))
    # radial profile Fourier energies (rotation-free magnitudes)
    cy, cx = cc[:, 0].mean(), cc[:, 1].mean()
    ang = np.arctan2(cc[:, 0] - cy, cc[:, 1] - cx)
    rad = np.hypot(cc[:, 0] - cy, cc[:, 1] - cx)
    o = np.argsort(ang)
    grid = np.linspace(-np.pi, np.pi, 256, endpoint=False)
    r = np.interp(grid, ang[o], rad[o], period=2 * np.pi)
    F = np.abs(np.fft.rfft(r / r.mean()))
    fe = F[1:13] / (np.sum(F[1:13]) + 1e-9)
    gv = g[obj]
    cm = 2.54 / dpi if dpi and dpi > 20 else np.nan
    f = dict(aspect=S_ / L_, rectness=area / rect_area, solidity=area / max(1, rp.area_convex),
             corner_mean=float(np.mean(fills)), corner_min=float(np.min(fills)), corner_sd=float(np.std(fills)),
             taper=min(w1, w2) / max(w1, w2, 1e-6), endw=(w1 + w2) / 2, waist=wm, rough=per / max(hp, 1),
             ecc=rp.eccentricity, extent=rp.extent, fill_img=area / (H * W),
             tone=float(np.mean(gv)), tone_sd=float(np.std(gv)),
             long_cm=L_ * cm, short_cm=S_ * cm, area_cm2=area * cm * cm)
    for i in range(1, 9):
        f['fd%d' % i] = float(fe[i - 1])
    return f

def measure(fn):
    g, dpi = load_gray(fn)
    obj = segment(g)
    if obj is None or obj.sum() < 400:
        return None, None
    f = shape_features(obj, g, dpi)
    # QC (geometry only, fixed before any text is looked at): share of object outline on the image frame
    fr = np.concatenate([obj[0], obj[-1], obj[:, 0], obj[:, -1]])
    f['qc_frame'] = float(fr.mean())
    f['qc_ok'] = int(f['qc_frame'] < 0.15 and f['fill_img'] < 0.92)
    return f, obj

def segment_fx(g):
    """Facsimile line drawing: dark lines, close gaps, flood the paper from the frame,
    object = what the flood cannot reach; a disk opening drops captions and loose strokes."""
    from skimage.filters import gaussian
    gg = gaussian(g, 1.0, preserve_range=True)
    lines = gg < 215
    lines = ndi.binary_dilation(lines, iterations=4)
    lab, n = ndi.label(~lines)
    edge_labels = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    obj = ~np.isin(lab, list(edge_labels))
    obj = ndi.binary_fill_holes(obj)
    r = max(4, int(0.04 * np.sqrt(max(obj.sum(), 1))))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    obj = ndi.binary_opening(obj, structure=(xx * xx + yy * yy) <= r * r)
    l2, n2 = ndi.label(obj)
    if n2 == 0:
        return None
    sz = ndi.sum(obj, l2, range(1, n2 + 1))
    return l2 == (int(np.argmax(sz)) + 1)

def measure_fx(fn):
    g, dpi = load_gray(fn)
    obj = segment_fx(g)
    if obj is None or obj.sum() < 400:
        return None, None
    f = shape_features(obj, g, dpi)
    fr = np.concatenate([obj[0], obj[-1], obj[:, 0], obj[:, -1]])
    f['qc_frame'] = float(fr.mean())
    return f, obj
