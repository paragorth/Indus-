"""pe83 cycle 3: outlines from a second, independent source: CDLI line art (hand drawings, mostly Scheil's MDP plates).
usage: python3 pe83_lineart.py <dir_of_P*_l.jpg> -> data/pe83_ckpt/lineart_feats.json"""
import sys, os, json, glob
import numpy as np
from PIL import Image
from scipy import ndimage
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe83_common as P


def measure(path, thr=160, dil=3):
    pid = os.path.basename(path)[:7]
    try:
        g = np.asarray(Image.open(path).convert('L')).astype(np.float32)
    except Exception:
        return pid, None
    if g.shape[0] > 700:
        im = Image.fromarray(g.astype(np.uint8)); im = im.resize((int(g.shape[1] * 600 / g.shape[0]), 600)); g = np.asarray(im).astype(np.float32)
    H, W = g.shape
    dark = g < thr
    m = ndimage.binary_dilation(dark, iterations=dil)
    lab, n = ndimage.label(m)
    cands = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if bh * bw > 0.10 * H * W and bh > 30 and bw > 30:
            cy = (sl[0].start + sl[0].stop) / 2; cx = (sl[1].start + sl[1].stop) / 2
            cands.append((sl, lab[sl] == i + 1, cy, cx))
    if not cands:
        return pid, None
    upper = [c for c in cands if c[2] < 0.6 * H] or cands
    sl, mk, cy, cx = min(upper, key=lambda c: (abs(c[3] - W / 2) / W > 0.25, -c[1].sum()))
    filled = ndimage.binary_fill_holes(mk)
    f = {'n_cand': len(cands), 'fill_ratio': float(filled.sum() / (mk.shape[0] * mk.shape[1]))}
    cfb = P.corner_metrics(filled)
    f['la_cf'] = float(np.mean(cfb)); f['la_cf_top'] = float((cfb[0] + cfb[1]) / 2); f['la_cf_bot'] = float((cfb[2] + cfb[3]) / 2)
    hf = P.hull_fill(mk)
    if hf:
        f['la_hull'] = float(np.mean(hf)); f['la_hull_top'] = float((hf[0] + hf[1]) / 2); f['la_hull_bot'] = float((hf[2] + hf[3]) / 2)
    f['la_aspect'] = mk.shape[0] / mk.shape[1]
    f['closed'] = float(f['fill_ratio'] > 0.55)
    return pid, f


if __name__ == '__main__':
    fs = sorted(glob.glob(os.path.join(sys.argv[1], 'P*_l.jpg')))
    with Pool(2) as pool:
        res = dict(pool.map(measure, fs, chunksize=20))
    json.dump(res, open(os.path.join(P.CK, 'lineart_feats.json'), 'w'))
    print('drawings', len(res), 'ok', sum(1 for v in res.values() if v), 'closed', sum(1 for v in res.values() if v and v['closed']))
