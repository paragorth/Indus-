#!/usr/bin/env python3
"""la34 per-occurrence shape features from SigLA drawings (low resolution, scratchpad images).
feat(mask) -> dict of scalar features + 'grid' (16x16 ink-occupancy of the ink bounding box, aspect kept by padding).
Scale-dependent features (modern pen width relative to sign size) are named with prefix 'S_' so the
scale-free arm can drop them.  An optional planted affine (shear, x-stretch, rotation) is applied before measuring."""
import numpy as np, os, json, sys
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.transform import AffineTransform, warp

MAXSIDE = 96


def load_mask(path):
    im = Image.open(path).convert('RGBA')
    w, h = im.size; f = min(1.0, MAXSIDE / max(w, h))
    a = np.asarray(im, float)
    ink = (a[..., 3] > 100) & (a[..., :3].mean(-1) < 128)
    if f < 1:
        ink = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)) > 100
    return ink


def plant(mask, shear=0.0, sx=1.0, rot=0.0):
    if shear == 0 and sx == 1 and rot == 0: return mask
    h, w = mask.shape; P = max(h, w) // 3
    m = np.pad(mask, P).astype(float)
    c = np.array(m.shape[::-1]) / 2
    T = AffineTransform(translation=-c) + AffineTransform(scale=(sx, 1), shear=shear, rotation=rot) + AffineTransform(translation=c)
    return warp(m, T.inverse, order=1) > 0.5


def clean(mask):
    lab, n = ndi.label(mask)
    if n == 0: return mask
    sz = ndi.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sz >= max(4, 0.03 * sz.max()))[0])
    return keep


def feat(mask):
    m = clean(mask)
    ys, xs = np.nonzero(m)
    if len(ys) < 10: return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    b = m[y0:y1, x0:x1]; H, W = b.shape
    f = {}
    f['aspect'] = np.log(W / H)
    sk = skeletonize(b)
    L = sk.sum()
    f['S_density'] = b.mean()
    f['S_width'] = b.sum() / max(L, 1) / max(H, W)
    f['skel_len'] = L / np.hypot(H, W)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode='constant') - 1
    f['ends'] = float(((nb == 1) & sk).sum()); f['junc'] = float(((nb >= 3) & sk).sum()) / 3
    yy, xx = np.nonzero(b)
    cy, cx = yy.mean() / H, xx.mean() / W
    f['cx'], f['cy'] = cx - .5, cy - .5
    Y = yy / max(H, W); X = xx / max(H, W)
    C = np.cov(np.vstack([X, Y]))
    ev, evec = np.linalg.eigh(C)
    ang = np.arctan2(evec[1, 1], evec[0, 1])
    f['axis_sin2'], f['axis_cos2'] = np.sin(2 * ang), np.cos(2 * ang)
    f['ecc'] = np.log((ev[1] + 1e-9) / (ev[0] + 1e-9))
    f['skew_x'] = float(((X - X.mean()) ** 3).mean() / (X.std() ** 3 + 1e-9))
    f['skew_y'] = float(((Y - Y.mean()) ** 3).mean() / (Y.std() ** 3 + 1e-9))
    f['shear_xy'] = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1] + 1e-12)
    # stroke-direction histogram from skeleton neighbour pairs (8 bins over 0-180 deg)
    g = ndi.gaussian_filter(b.astype(float), 1.0)
    gy, gx = np.gradient(g); mag = np.hypot(gx, gy)
    th = (np.arctan2(gy, gx) + np.pi / 2) % np.pi  # stroke direction = gradient + 90
    hist = np.bincount(np.minimum((th / np.pi * 8).astype(int), 7).ravel(), mag.ravel(), 8)
    hist = hist / (hist.sum() + 1e-9)
    for i in range(8): f[f'dir{i}'] = hist[i]
    # 16x16 grid, padded square
    S = max(H, W); sq = np.zeros((S, S), bool); oy, ox = (S - H) // 2, (S - W) // 2; sq[oy:oy + H, ox:ox + W] = b
    grid = np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((16, 16), Image.BOX), float) / 255
    out = {k: float(v) for k, v in f.items()}
    out['grid'] = [round(float(v), 3) for v in grid.ravel()]
    return out


def run(files, plant_args=None):
    res = []
    for p, pa in zip(files, plant_args or [None] * len(files)):
        try:
            m = load_mask(p)
            if pa: m = plant(m, *pa)
            res.append(feat(m))
        except Exception as e:
            res.append(None)
    return res


if __name__ == '__main__':
    # extract features for every downloaded occurrence (2 workers, batches), cache to la34_ckpt/feat.json
    from multiprocessing import Pool
    from la34_common import load, CK
    occ, meta, cid = load()
    out_p = os.path.join(CK, 'feat.json')
    cache = json.load(open(out_p)) if os.path.exists(out_p) else {}
    todo = [o['file'] for o in occ if os.path.exists(o['file']) and o['file'] not in cache]
    print('todo', len(todo), flush=True)
    B = 100
    with Pool(2) as pool:
        for i in range(0, len(todo), 2 * B):
            chunk = [todo[j:j + B] for j in range(i, min(i + 2 * B, len(todo)), B)]
            for fs, rs in zip(chunk, pool.map(run, chunk)):
                for f, r in zip(fs, rs): cache[f] = r
            json.dump(cache, open(out_p, 'w'))
            print(i + sum(map(len, chunk)), flush=True)
    print('cached', len(cache))
