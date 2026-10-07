"""pe81 cycle 2: scale-free outline features from CDLI fat-cross photos (obverse blob + top-edge blob).
usage: python3 pe81_shapes.py <dir_of_jpgs> [<dir2> ...] -> data/pe81_ckpt/shapes.json"""
import sys, os, json, glob
import numpy as np
from PIL import Image
from scipy import ndimage

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pe81_ckpt', 'shapes.json')


def blobs(a):
    m = a > 40
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        mask = lab[sl] == i + 1
        if mask.sum() > 0.01 * a.size:
            out.append((sl, mask))
    return out


def widths_at(mask, fracs):
    h = mask.shape[0]
    return [mask[min(h - 1, int(f * h))].sum() for f in fracs]


def feats(path):
    im = Image.open(path).convert('L')
    if im.size[1] > 600:
        im = im.resize((int(im.size[0] * 450 / im.size[1]), 450))
    a = np.asarray(im)
    B = blobs(a)
    if len(B) < 2:
        return None
    H = a.shape[0]
    up = [b for b in B if (b[0][0].start + b[0][0].stop) / 2 < H * 0.55]
    if not up:
        return None
    ob = max(up, key=lambda b: b[1].sum())
    sl, m = ob
    h, w = m.shape
    if h < 30 or w < 25:
        return None
    area = m.sum()
    f = {}
    f['rect'] = area / (h * w)
    hull_fill = []
    # convexity: area / convex hull area (via rows/cols spans)
    rows = [np.where(r)[0] for r in m]
    span = sum((r[-1] - r[0] + 1) if len(r) else 0 for r in rows)
    f['row_solid'] = area / max(1, span)
    c = max(2, int(0.15 * min(h, w)))
    corners = [m[:c, :c], m[:c, -c:], m[-c:, :c], m[-c:, -c:]]
    cf = [x.mean() for x in corners]
    f['corner_fill'] = float(np.mean(cf))
    f['corner_top_vs_bottom'] = float((cf[0] + cf[1]) - (cf[2] + cf[3]))
    wt, wm, wb = widths_at(m, [0.1, 0.5, 0.9])
    f['taper'] = (wt - wb) / max(1, wm)
    f['side_bulge'] = (wm - (wt + wb) / 2) / max(1, wm)
    cols = m.sum(0)
    ht, hm, hb = cols[int(0.1 * w)], cols[w // 2], cols[min(w - 1, int(0.9 * w))]
    f['vbulge'] = (hm - (ht + hb) / 2) / max(1, hm)
    f['lr_asym'] = float(abs(m[:, : w // 2].sum() - m[:, w - w // 2:].sum()) / area)
    f['aspect_px'] = h / w
    # top edge: blob above obverse overlapping horizontally
    top = [b for b in B if b[0][0].stop <= sl[0].start + 3 and b[0][1].start < sl[1].stop and b[0][1].stop > sl[1].start]
    if top:
        tsl, tm = max(top, key=lambda b: b[1].sum())
        tw = tm.shape[1]
        th = tm.sum(0)
        mid = th[tw // 2]
        ends = (th[int(0.15 * tw)] + th[min(tw - 1, int(0.85 * tw))]) / 2
        f['pillow'] = (mid - ends) / max(1, mid)
        f['thick_ratio_px'] = mid / max(1, w)
    else:
        f['pillow'] = None; f['thick_ratio_px'] = None
    return {k: (float(v) if v is not None else None) for k, v in f.items()}


if __name__ == '__main__':
    res = {}
    for d in sys.argv[1:]:
        for p in sorted(glob.glob(os.path.join(d, 'P*.jpg'))):
            pid = os.path.basename(p)[:-4]
            if pid in res:
                continue
            try:
                res[pid] = feats(p)
            except Exception as e:
                res[pid] = None
    json.dump(res, open(OUT, 'w'))
    ok = sum(1 for v in res.values() if v)
    print('images', len(res), 'ok', ok, 'with top edge', sum(1 for v in res.values() if v and v['pillow'] is not None))
