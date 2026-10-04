"""pe26: per-tablet clay colour / texture features from CDLI thumbnails (scratchpad), plus photo-batch covariates.
usage: python3 pe26_features.py meta.json out.json"""
import sys, json, os, warnings, numpy as np
warnings.filterwarnings('ignore')
from PIL import Image
from skimage import color, morphology, feature, filters
TN = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/pe26/tn'
meta = json.load(open(sys.argv[1])); out = sys.argv[2]
res = json.load(open(out)) if os.path.exists(out) else {}

def feats(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(float) / 255.0
    H, W, _ = im.shape
    lab = color.rgb2lab(im)
    L = lab[..., 0]
    frame = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3), lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    bg = np.median(frame, 0)
    d = np.sqrt(((lab - bg) ** 2).sum(-1))
    try: t = filters.threshold_otsu(d)
    except Exception: t = 20
    t = max(t, 12)
    m = d > t
    m = morphology.binary_opening(m, morphology.disk(2))
    m = morphology.remove_small_objects(m, 200)
    core = morphology.binary_erosion(m, morphology.disk(4))
    if core.sum() < 500: return None
    P = lab[core]
    # drop red-ink/label pixels: very high chroma outliers
    ch = np.hypot(P[:, 1], P[:, 2])
    keep = ch < np.percentile(ch, 98)
    P = P[keep]
    f = {}
    rgbc = im[core]
    f['is_grey'] = float(np.abs(rgbc[:, 0] - rgbc[:, 2]).mean() < 0.01)
    for i, n in enumerate('Lab'):
        q = np.percentile(P[:, i], [10, 25, 50, 75, 90])
        f[f'{n}_p10'], f[f'{n}_p25'], f[f'{n}_med'], f[f'{n}_p75'], f[f'{n}_p90'] = q
        f[f'{n}_iqr'] = q[3] - q[1]
    f['chroma'] = float(np.median(np.hypot(P[:, 1], P[:, 2])))
    f['hue'] = float(np.degrees(np.arctan2(np.median(P[:, 2]), np.median(P[:, 1]))))
    f['a_over_L'] = f['a_med'] / max(f['L_med'], 1); f['b_over_L'] = f['b_med'] / max(f['L_med'], 1)
    # rg chromaticity (illumination-intensity invariant)
    rgb = im[core][keep] + 1e-3
    s = rgb.sum(1)
    f['r_chrom'] = float(np.median(rgb[:, 0] / s)); f['g_chrom'] = float(np.median(rgb[:, 1] / s))
    f['rg_corr'] = float(np.corrcoef(np.log(rgb[:, 0]), np.log(rgb[:, 2]))[0, 1])
    # texture on L inside core
    Ls = filters.gaussian(L, 1.0); Lb = filters.gaussian(L, 6.0)
    hp = (Ls - Lb)[core]
    f['hp_std'] = float(hp.std()); f['hp_rel'] = float(hp.std() / max(f['L_med'], 1))
    g = filters.sobel(Ls)[core]; f['grad_med'] = float(np.median(g)); f['grad_p90'] = float(np.percentile(g, 90))
    lbp = feature.local_binary_pattern((np.clip(L, 0, 100) * 2.55).astype(np.uint8), 8, 1, 'uniform')[core]
    h = np.bincount(lbp.astype(int), minlength=10)[:10] / len(lbp)
    for i in range(10): f[f'lbp{i}'] = float(h[i])
    # mottling: std of block means (8x8) inside core
    bm = []
    for y in range(0, H - 5, 5):
        for x in range(0, W - 5, 5):
            if core[y:y+5, x:x+5].all(): bm.append(L[y:y+5, x:x+5].mean())
    f['mottle'] = float(np.std(bm)) if len(bm) > 3 else float('nan')
    f['dark_frac'] = float((P[:, 0] < f['L_med'] - 25).mean())
    # batch covariates
    f['bg_L'], f['bg_a'], f['bg_b'] = map(float, bg)
    f['fg_frac'] = float(m.mean()); f['H'] = H; f['W'] = W; f['n_core'] = int(core.sum())
    return {k: float(v) for k, v in f.items()}

n = 0
for p, v in meta.items():
    if not v.get('photo') or p in res: continue
    try: r = feats(f'{TN}/{p}.jpg')
    except Exception as e: r = None; print(p, e)
    res[p] = r; n += 1
    if n % 100 == 0: json.dump(res, open(out, 'w')); print(n, flush=True)
json.dump(res, open(out, 'w')); print('done', len(res), sum(r is not None for r in res.values()))
