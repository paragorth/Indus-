"""pe84: physical features of the writing (not the outline) from full-resolution CDLI fat-cross photos.

Each face (obverse = biggest blob in the upper part, reverse = big blob below the bottom-edge view) is rescaled to
8 px/mm using the catalogue width, cut to its interior (2 mm eroded) and measured at fixed physical scales:
  E04/E08/E16   relief energy: std of difference-of-Gaussians at sigma 0.4 / 0.8 / 1.6 mm (impression wall shading)
  G1..G4        granulometry: black top-hat energy with square elements 1, 2, 3.5, 5 mm (stroke-width spectrum)
  ink           share of interior pixels inside detected impressions (black top-hat 3.5 mm > 3 robust sd)
  depth         mean top-hat contrast inside impressions / interior median brightness (depth / crispness proxy)
  wid           median impression width in mm (2 x max distance-to-edge per component)
  ncomp         impressions per cm2
  coh, ang      structure-tensor coherence and dominant stroke angle (deg, 0 = tablet axis) at sigma 0.8 mm
  diag          share of gradient energy at diagonal orientations (slant)
  gx, gy        mean brightness gradient (lighting direction; covariate)
  bright        interior median brightness (covariate)
  t0/t1/t2_*    E08 and depth in the top / middle / bottom third of the interior
usage: python3 pe84_feats.py <ids.txt> <img_dir> <out.json> [workers<=2]
"""
import sys, os, json, csv
import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
PXMM = 8.0
PLANT = set(x.strip() for x in open(os.environ['PE84_PLANT'])) if os.environ.get('PE84_PLANT') else set()


def blobs(g, thr, frac=0.01):
    m = g > thr
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        mk = lab[sl] == i + 1
        if mk.sum() > frac * g.size:
            out.append((sl, mk))
    return out


def otsu(g):
    h, e = np.histogram(g, 128, (0, 255))
    p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * e[:-1]); mt = mu[-1]
    s = (mt * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return e[int(np.argmax(s))]


def faces(L):
    """L: luminance at working size. Returns (obverse, reverse or None) as (slice, mask) on L."""
    border = np.concatenate([L[:5].ravel(), L[-5:].ravel(), L[:, :5].ravel(), L[:, -5:].ravel()])
    if np.median(border) > 100:  # light background: invert so clay is the bright object vs background
        g = np.abs(L - np.median(border))
        thr = max(25, otsu(g) * 0.8)
    else:
        g = L; thr = max(35, min(70, otsu(L) * 0.6))
    B = blobs(g, thr)
    H = L.shape[0]
    up = [b for b in B if (b[0][0].start + b[0][0].stop) / 2 < H * 0.55]
    if not up:
        return None, None
    ob = max(up, key=lambda b: b[1].sum())
    sl = ob[0]
    below = [b for b in B if b[0][0].start >= sl[0].stop - 3 and b[0][1].start < sl[1].stop and b[0][1].stop > sl[1].start]
    rv = None
    if below:
        big = max(below, key=lambda b: b[1].sum())
        if big[1].sum() >= 0.4 * ob[1].sum():
            rv = big
    return ob, rv


def face_feats(L, face, width_mm):
    sl, m = face
    sub = L[sl].copy()
    wpx = m.shape[1]
    s = PXMM * width_mm / wpx
    if not (0.15 < s < 3.0):
        return None
    ny, nx = max(8, int(round(m.shape[0] * s))), max(8, int(round(m.shape[1] * s)))
    sub = np.asarray(Image.fromarray(sub.astype(np.float32), mode='F').resize((nx, ny), Image.BILINEAR))
    mk = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((nx, ny), Image.NEAREST)) > 127
    inner = ndimage.binary_erosion(mk, iterations=int(2 * PXMM))
    if inner.sum() < 400:
        return None
    med = float(np.median(sub[inner]))
    if med < 5:
        return None
    X = sub / med  # exposure-normalised
    fill = X.copy(); fill[~mk] = 1.0
    f = {'s_orig': 1.0 / s, 'bright': med, 'area_cm2': float(inner.sum() / PXMM ** 2 / 100)}
    for sig, nm in ((0.4, 'E04'), (0.8, 'E08'), (1.6, 'E16')):
        d = ndimage.gaussian_filter(fill, sig * PXMM) - ndimage.gaussian_filter(fill, 2.5 * sig * PXMM)
        f[nm] = float(d[inner].std())
        if nm == 'E08':
            d08 = d
    th = {}
    for k, mm in ((1, 1.0), (2, 2.0), (3, 3.5), (4, 5.0)):
        sz = max(3, int(mm * PXMM) | 1)
        bt = ndimage.grey_closing(fill, size=(sz, sz)) - fill
        th[k] = bt
        f['G%d' % k] = float(bt[inner].mean())
    bt = th[3]
    v = bt[inner]; mad = np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-6
    imp = (bt > np.median(v) + 3 * mad) & inner
    imp = ndimage.binary_opening(imp, iterations=1)
    f['ink'] = float(imp.sum() / inner.sum())
    f['depth'] = float(bt[imp].mean()) if imp.sum() > 20 else None
    lab, n = ndimage.label(imp)
    if n:
        dt = ndimage.distance_transform_edt(imp)
        mx = ndimage.maximum(dt, lab, index=np.arange(1, n + 1))
        sizes = ndimage.sum(imp, lab, index=np.arange(1, n + 1))
        keep = sizes >= 6
        f['wid'] = float(np.median(2 * mx[keep]) / PXMM) if keep.any() else None
        f['ncomp'] = float(keep.sum() / (inner.sum() / PXMM ** 2 / 100))
    else:
        f['wid'] = None; f['ncomp'] = 0.0
    gy, gx = np.gradient(ndimage.gaussian_filter(fill, 0.8 * PXMM))
    f['gx'] = float(gx[inner].mean() * 1000); f['gy'] = float(gy[inner].mean() * 1000)
    gy2, gx2 = np.gradient(d08)
    Jxx = ndimage.gaussian_filter(gx2 * gx2, 2 * PXMM)[inner]; Jyy = ndimage.gaussian_filter(gy2 * gy2, 2 * PXMM)[inner]
    Jxy = ndimage.gaussian_filter(gx2 * gy2, 2 * PXMM)[inner]
    a, b, c = Jxx.sum(), Jyy.sum(), Jxy.sum()
    f['coh'] = float(np.sqrt((a - b) ** 2 + 4 * c ** 2) / (a + b + 1e-12))
    f['ang'] = float(np.degrees(0.5 * np.arctan2(2 * c, a - b)))
    th_ = np.arctan2(gy2[inner], gx2[inner]); mag = gx2[inner] ** 2 + gy2[inner] ** 2
    dg = np.abs(np.sin(2 * th_))  # 1 at diagonals, 0 at axes
    f['diag'] = float((mag * dg).sum() / (mag.sum() + 1e-12))
    rows = np.where(inner.any(1))[0]
    r0, r1 = rows[0], rows[-1] + 1
    cuts = np.linspace(r0, r1, 4).astype(int)
    for i in range(3):
        band = np.zeros_like(inner); band[cuts[i]:cuts[i + 1]] = True
        bi = band & inner
        f['t%d_E08' % i] = float(d08[bi].std()) if bi.sum() > 50 else None
        bj = band & imp
        f['t%d_depth' % i] = float(bt[bj].mean()) if bj.sum() > 15 else None
        f['t%d_ink' % i] = float(bj.sum() / max(1, bi.sum()))
    return f


def one(args):
    pid, path, width_mm = args
    try:
        im = Image.open(path).convert('L')
        if im.size[1] > 2600:
            im = im.resize((int(im.size[0] * 2600 / im.size[1]), 2600), Image.BILINEAR)
        L = np.asarray(im).astype(np.float32)
        ob, rv = faces(L)
        if ob is None:
            return pid, None
        if pid in PLANT and rv is not None:
            # planted 'written later' reverse: relief flattened by 25% and slightly blurred (image level)
            sl, m = rv
            sub = L[sl].copy(); med = float(np.median(sub[m]))
            fl = ndimage.gaussian_filter(med + 0.75 * (sub - med), 1.5)
            sub[m] = fl[m]; L = L.copy(); L[sl] = sub
        out = {'ob': face_feats(L, ob, width_mm), 'rv': face_feats(L, rv, width_mm) if rv is not None else None,
               'H': L.shape[0]}
        return pid, out
    except Exception as e:
        return pid, {'err': str(e)[:200]}


if __name__ == '__main__':
    ids = [x.strip() for x in open(sys.argv[1]) if x.strip()]
    d, outp = sys.argv[2], sys.argv[3]
    nw = min(2, int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    catp = sys.argv[5] if len(sys.argv) > 5 else '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/cdli_cat.csv'
    csv.field_size_limit(10 ** 9)
    W = {}
    for x in csv.DictReader(open(catp, encoding='utf-8', errors='replace')):
        try:
            W['P%06d' % int(x['id_text'])] = float(x['width'])
        except Exception:
            pass
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    if os.environ.get('PE84_FIXW'):
        W = {p: float(os.environ['PE84_FIXW']) for p in ids}
    todo = [(p, os.path.join(d, p + '.jpg'), W.get(p)) for p in ids
            if p not in res and os.path.exists(os.path.join(d, p + '.jpg')) and W.get(p)]
    print('todo', len(todo), flush=True)
    from multiprocessing import Pool
    with Pool(nw) as pool:
        for i, (pid, r) in enumerate(pool.imap_unordered(one, todo, chunksize=2)):
            res[pid] = r
            if i % 25 == 0:
                json.dump(res, open(outp, 'w')); print(i, flush=True)
    json.dump(res, open(outp, 'w'))
    print('done', len(res), sum(1 for v in res.values() if v and v.get('ob')), sum(1 for v in res.values() if v and v.get('rv')))
