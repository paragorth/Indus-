"""pe81 cycle 2d: segmentation robustness. corner fill at thresholds 25/40/70 and obverse brightness; header link with brightness control."""
import sys, os, json, glob
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe81_engine as E, common
TN = sys.argv[1]
T = {t['id']: t for t in common.load() if t['lines']}
rows = []
for p in sorted(glob.glob(os.path.join(TN, 'P*.jpg'))):
    pid = os.path.basename(p)[:-4]
    if pid not in T: continue
    a = np.asarray(Image.open(p).convert('L')).astype(float)
    r = {'hd': common.header(T[pid]) is not None}
    for thr in (25, 40, 70):
        m = ndimage.binary_fill_holes(ndimage.binary_opening(a > thr, iterations=1))
        lab, n = ndimage.label(m); H = a.shape[0]
        best = None
        for i, sl in enumerate(ndimage.find_objects(lab)):
            mm = lab[sl] == i + 1
            if mm.sum() > 0.01 * a.size and (sl[0].start + sl[0].stop) / 2 < H * 0.55:
                if best is None or mm.sum() > best[1].sum(): best = (sl, mm)
        if best is None: break
        sl, mm = best; h, w = mm.shape; c = max(2, int(0.15 * min(h, w)))
        r[f'cf{thr}'] = float(np.mean([mm[:c, :c].mean(), mm[:c, -c:].mean(), mm[-c:, :c].mean(), mm[-c:, -c:].mean()]))
        if thr == 40:
            r['bright'] = float(a[sl][mm].mean()); r['la'] = float(np.log(mm.sum())); r['asp'] = float(np.log(h / w))
            r['nl'] = float(np.log(len(T[pid]['lines'])))
    if 'cf70' in r: rows.append(r)
hd = np.array([r['hd'] for r in rows], float)
Z = np.column_stack([np.ones(len(rows))] + [np.array([r[k] for r in rows]) for k in ('bright', 'la', 'asp', 'nl')])
res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
out = {'n': len(rows)}
for k in ('cf25', 'cf40', 'cf70'):
    y = rankdata([r[k] for r in rows])
    out[k] = {'raw_r': round(float(np.corrcoef(y, hd)[0, 1]), 3), 'partial_r_with_brightness': round(float(np.corrcoef(res(y), res(hd))[0, 1]), 3)}
out['bright_vs_header_r'] = round(float(np.corrcoef(Z[:, 1], hd)[0, 1]), 3)
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(E.CK, 'cycle2d.json'), 'w'), indent=1)
