"""pe83: attack on pe81 (square corners <-> header). Shared loader, image segmentation with parameters, partial correlation."""
import sys, os, json, csv, re
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
TN = os.path.join(SCR, 'pe26', 'tn')
CK = os.path.join(common.DATA, 'pe83_ckpt')
os.makedirs(CK, exist_ok=True)
SH = json.load(open(os.path.join(common.DATA, 'pe81_ckpt', 'shapes.json')))
COVS = ('la', 'asp', 'tw', 'nl', 'dx', 'rev')


def num(s):
    try:
        return float(s)
    except Exception:
        return None


def catalogue(period_prefix='Proto-Elamite'):
    csv.field_size_limit(10 ** 9)
    return {'P%06d' % int(x['id_text']): x for x in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8', errors='replace'))
            if x.get('period', '').startswith(period_prefix)}


def rows():
    """Same tablet set and covariates as pe81 cycle 2c."""
    cat = catalogue(); out = []
    for t in common.load():
        sh = SH.get(t['id']); c = cat.get(t['id'])
        if not sh or not c or not t['lines']:
            continue
        h, w, th = num(c['height']), num(c['width']), num(c['thickness'])
        if not (h and w and th):
            continue
        nl = len(t['lines']); nx = sum(g == 'x' for l in t['lines'] for g in l['signs'])
        rev = any(l['surface'] != 'obverse' for l in t['lines'])
        m = re.match(r'Sb (\d+)', c['museum_no'] or '')
        col = c['collection']
        out.append(dict(id=t['id'], hd=float(common.header(t) is not None), cf=sh['corner_fill'], la=np.log(h * w), asp=np.log(h / w),
                        tw=th / w, nl=np.log(nl), dx=float(nx > 0), rev=float(rev), comp=c['object_preservation'] == 'complete',
                        site=t['provenience'], coll='Louvre' if 'Louvre' in col else ('Tehran' if 'Tehran' in col else 'other'),
                        batch=c['photo_up'] or 'none', lineart=bool(c['lineart_up']), sb=int(m.group(1)) if m else None,
                        n_entries=sum(1 for l in t['lines'] if l['numerals'] and l['signs']),
                        numonly=sum(1 for l in t['lines'] if l['numerals'] and not [g for g in l['signs'] if common.is_sign(g) or g == 'x']),
                        t=t))
    return out


def design(R, extra=()):
    Z = np.column_stack([np.ones(len(R))] + [np.array([r[k] for r in R], float) for k in COVS])
    Z = np.column_stack([Z, Z[:, 1] ** 2, Z[:, 4] ** 2] + [np.asarray(e, float) for e in extra])
    return Z


def strata(Z):
    return np.digitize(Z[:, 4], np.quantile(Z[:, 4], [.25, .5, .75])) * 4 + np.digitize(Z[:, 1], np.quantile(Z[:, 1], [.25, .5, .75]))


def partial(y, hd, Z, B=0, rng=None, s=None):
    """rank(y) vs hd, Z partialled out. Optional stratified permutation p (two-sided)."""
    ok = np.isfinite(y)
    y, hd, Z = y[ok], hd[ok], Z[ok]
    # drop constant/collinear columns gracefully via lstsq
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    a = res(rankdata(y)); b = res(hd)
    r = float(np.corrcoef(a, b)[0, 1])
    out = dict(n=int(ok.sum()), r=round(r, 3))
    if B:
        rng = rng or np.random.default_rng(83)
        s = strata(Z) if s is None else s[ok]
        cnt = 0
        for _ in range(B):
            hp = hd.copy()
            for k in np.unique(s):
                m = np.where(s == k)[0]; hp[m] = rng.permutation(hp[m])
            cnt += abs(np.corrcoef(a, res(hp))[0, 1]) >= abs(r)
        out['p'] = round((cnt + 1) / (B + 1), 5)
    return out


def load_img(pid, d=TN):
    im = Image.open(os.path.join(d, pid + '.jpg')).convert('RGB')
    if im.size[1] > 600:
        im = im.resize((int(im.size[0] * 450 / im.size[1]), 450))
    return np.asarray(im).astype(np.float32)


def blobs(g, thr=40, op=1, cl=0, frac=0.01):
    m = g > thr
    if op:
        m = ndimage.binary_opening(m, iterations=op)
    if cl:
        m = ndimage.binary_closing(m, iterations=cl)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        mk = lab[sl] == i + 1
        if mk.sum() > frac * g.size:
            out.append((sl, mk))
    return out


def corner_metrics(m, c=0.15):
    h, w = m.shape; k = max(2, int(c * min(h, w)))
    cf = [m[:k, :k].mean(), m[:k, -k:].mean(), m[-k:, :k].mean(), m[-k:, -k:].mean()]
    return cf


def hull_fill(m, c=0.15):
    """corner fill of the convex hull of the mask (ink inside the outline cannot change it)."""
    from scipy.spatial import ConvexHull
    from PIL import ImageDraw
    ys, xs = np.nonzero(m)
    pts = np.column_stack([xs, ys])
    try:
        hv = ConvexHull(pts)
    except Exception:
        return None
    h, w = m.shape
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts[hv.vertices]], fill=1, outline=1)
    hm = np.asarray(im).astype(bool)
    return corner_metrics(hm, c)


def pick_obverse(B, H):
    up = [b for b in B if (b[0][0].start + b[0][0].stop) / 2 < H * 0.55]
    if not up:
        return None
    return max(up, key=lambda b: b[1].sum())


def pick_reverse(B, H, ob):
    """the big blob directly below the obverse (fat cross: obverse, bottom edge, reverse)."""
    sl = ob[0]
    below = [b for b in B if b[0][0].start >= sl[0].stop - 3 and b[0][1].start < sl[1].stop and b[0][1].stop > sl[1].start]
    if len(below) < 1:
        return None
    big = max(below, key=lambda b: b[1].sum())
    if big[1].sum() < 0.4 * ob[1].sum():
        return None
    return big


def close_mask(m, r=6):
    p = np.pad(m, r + 2)
    st = ndimage.generate_binary_structure(2, 1)
    p = ndimage.binary_closing(p, structure=st, iterations=r)
    p = ndimage.binary_fill_holes(p)
    return p[r + 2:-(r + 2), r + 2:-(r + 2)] | m
