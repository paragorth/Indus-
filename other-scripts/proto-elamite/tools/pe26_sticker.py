"""pe26 cycle 4: built-in colour checker. Many NMI photos show a white paper label on the tablet. Its colour measures the
lighting/white balance of that very shot. Detect bright, low-chroma blobs inside the object mask; record their Lab.
usage: python3 pe26_sticker.py ids.txt out.json"""
import sys, json, os, warnings, numpy as np
warnings.filterwarnings('ignore')
from PIL import Image
from skimage import color, measure, morphology
TN = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/pe26/tn'
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
out = {}
for p in ids:
    f = f'{TN}/{p}.jpg'
    if not os.path.exists(f): continue
    im = np.asarray(Image.open(f).convert('RGB')).astype(float) / 255
    lab = color.rgb2lab(im); L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    obj = L > 12
    clayL = np.percentile(L[obj], 75) if obj.any() else 50
    cand = (L > max(72, clayL + 18)) & (np.hypot(a, b) < 14)
    cand = morphology.binary_opening(cand, morphology.disk(1))
    lb = measure.label(cand); best = None
    for rp in measure.regionprops(lb):
        if rp.area < 25: continue
        if best is None or rp.area > best.area: best = rp
    if best is None: out[p] = None; continue
    m = lb == best.label; m = morphology.binary_erosion(m, morphology.disk(1)) if best.area > 60 else m
    out[p] = dict(area=int(best.area), L=float(np.median(L[m])), a=float(np.median(a[m])), b=float(np.median(b[m])))
json.dump(out, open(sys.argv[2], 'w'))
print('done', len(out), sum(v is not None for v in out.values()))
