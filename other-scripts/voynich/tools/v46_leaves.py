"""v46: blind leaf counter for herbal pages (low-resolution Yale IIIF scans, scratch only).
Green paint mask (hue 60-170 deg, chroma above page background) -> opening with a disk of radius R
(drops stems and thin ink) -> connected components above a minimum area = 'leaf blobs'.
Also counts all paint blobs (any hue) and the number of separate green regions before opening.
Usage: python3 v46_leaves.py IMGDIR OUT.json [overlay_dir]"""
import os, sys, json, glob, colorsys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

W = 600

def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r

def masks(fn):
    im = Image.open(fn).convert('RGB')
    w, h = im.size
    im = im.resize((W, int(round(h * W / w))), Image.BILINEAR)
    a = np.asarray(im, float) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(2), a.min(2)
    ch = mx - mn
    hue = np.zeros_like(mx)
    m = ch > 1e-6
    rc = np.where(m, (mx - r) / np.maximum(ch, 1e-6), 0); gc = np.where(m, (mx - g) / np.maximum(ch, 1e-6), 0); bc = np.where(m, (mx - b) / np.maximum(ch, 1e-6), 0)
    hh = np.where(r == mx, bc - gc, np.where(g == mx, 2 + rc - bc, 4 + gc - rc))
    hue = (hh / 6.0) % 1.0 * 360
    bgch = np.percentile(ch, 50)
    # trim a 4% border (page edge, binding shadow)
    H, Wd = ch.shape
    border = np.zeros_like(ch, bool); bh, bw = int(.04 * H), int(.04 * Wd)
    border[bh:H - bh, bw:Wd - bw] = True
    green = ((g - r) > 0.06) & (g >= b - 0.03) & border
    paint = (((g - r) > 0.06) | ((b - r) > 0.0) | ((r - g) > 0.25)) & border & (mx > 0.15)
    return im, green, paint

def count(fn, R=3, amin=60):
    im, green, paint = masks(fn)
    g0 = ndi.binary_closing(green, disk(1))
    g1 = ndi.binary_opening(g0, disk(R))
    lab, n = ndi.label(g1)
    sizes = ndi.sum(g1, lab, range(1, n + 1)) if n else np.array([])
    keep = [i + 1 for i, s in enumerate(sizes) if s >= amin]
    lab0, n0 = ndi.label(g0)
    s0 = ndi.sum(g0, lab0, range(1, n0 + 1)) if n0 else np.array([])
    labp, npb = ndi.label(ndi.binary_opening(paint, disk(2)))
    sp = ndi.sum(paint, labp, range(1, npb + 1)) if npb else np.array([])
    return dict(leaves=len(keep), green_regions=int((s0 >= amin).sum()), paint_blobs=int((sp >= amin).sum()),
                green_area=float(g0.mean())), (im, lab, keep)

if __name__ == '__main__':
    d, out = sys.argv[1], sys.argv[2]
    ov = sys.argv[3] if len(sys.argv) > 3 else None
    res = {}
    for fn in sorted(glob.glob(os.path.join(d, 'v_f*.jpg'))):
        f = os.path.basename(fn)[2:-4]
        r, (im, lab, keep) = count(fn)
        res[f] = r
        if ov:
            os.makedirs(ov, exist_ok=True)
            dr = ImageDraw.Draw(im)
            for k, i in enumerate(keep):
                ys, xs = np.nonzero(lab == i)
                dr.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=(255, 0, 0))
                dr.text((xs.min(), ys.min()), str(k + 1), fill=(255, 0, 0))
            im.save(os.path.join(ov, f + '.jpg'), quality=80)
    json.dump(res, open(out, 'w'), indent=0)
    print(len(res), 'pages')
