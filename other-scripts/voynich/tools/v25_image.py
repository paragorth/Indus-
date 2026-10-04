"""v25 image-based shape similarity: glyphs rendered from fonts, no hand-authored decomposition.

Voynich: G. Landini's EVA Hand A font (evaa.ttf, kept in the scratch directory, not committed);
Latin and Greek: DejaVu Serif; Hangul compatibility jamo: WenQuanYi Zen Hei.
Descriptors (all automatic):
  zone      ink bounding box padded to a square, 16x16 density grid (cosine)
  frame     horizontally cropped, vertical frame fixed relative to the baseline, 12x16 grid (keeps
            ascenders/descenders) (cosine)
  topo      automatic stroke counts: holes (closed loops), skeleton endpoints, skeleton junctions,
            connected components, ascender and descender extent, aspect, ink density (z-scored,
            similarity = -euclidean distance)
  combo     mean of the three z-scored similarity matrices
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import v25_lib as L, v25_shapes as S

FONTS = dict(voynich=os.path.join(L.SCR, 'evafont', 'evaa.ttf'),
             latin='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
             greek='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
             hangul='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
             latin_sans='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
             latin_free='/usr/share/fonts/truetype/freefont/FreeSerif.ttf')


def render(ch, font, size=96):
    f = ImageFont.truetype(font, size)
    im = Image.new('L', (size * 5, size * 3), 0)
    d = ImageDraw.Draw(im)
    d.text((size // 2, int(size * 1.8)), ch, font=f, fill=255, anchor='ls')
    a = np.asarray(im) > 127
    return a, int(size * 1.8), size


def descriptors(ch, font, xh_ref):
    a, base, size = render(ch, font)
    ys, xs = np.where(a)
    a = a[:, xs.min():xs.max() + 1]
    # zone
    b = a[ys.min():ys.max() + 1]
    h, w = b.shape; m = max(h, w)
    sq = np.zeros((m, m)); sq[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = b
    zone = np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((16, 16), Image.BILINEAR), float) / 255
    # frame: rows from base - 2.2 xh to base + 1.2 xh
    top = int(base - 2.4 * xh_ref); bot = int(base + 1.2 * xh_ref)
    fr = a[max(top, 0):bot]
    frame = np.asarray(Image.fromarray((fr * 255).astype(np.uint8)).resize((12, 16), Image.BILINEAR), float) / 255
    # topology
    from skimage.morphology import skeletonize
    lab_bg, nbg = ndi.label(~np.pad(b, 2))
    holes = nbg - 1
    _, ncomp = ndi.label(b, structure=np.ones((3, 3)))
    sk = skeletonize(np.pad(b, 2))
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode='constant') - 1
    ends = int(((nb == 1) & sk).sum()); junc_px = ((nb >= 3) & sk)
    junc = ndi.label(junc_px, structure=np.ones((3, 3)))[1]
    asc = max(0, (base - xh_ref) - ys.min()) / xh_ref
    desc = max(0, ys.max() - base) / xh_ref
    topo = np.array([holes, ends, junc, ncomp, asc, desc, w / xh_ref, b.mean()], float)
    return zone.ravel(), frame.ravel(), topo


def xheight(script):
    ref = dict(voynich='o', latin='x', greek='ο', hangul='ㅇ', latin_sans='x', latin_free='x')[script]
    a, base, size = render(ref, FONTS[script])
    ys, _ = np.where(a)
    return max(ys.max() - ys.min(), 1)


def zs(M):
    iu = np.triu_indices(len(M), 1); v = M[iu]
    Z = (M - v.mean()) / (v.std() + 1e-12)
    return Z


def image_sims(script, alph):
    xh = xheight(script)
    D = [descriptors(g, FONTS[script], xh) for g in alph]
    Zo = np.array([d[0] for d in D]); Fr = np.array([d[1] for d in D]); To = np.array([d[2] for d in D])
    To = (To - To.mean(0)) / (To.std(0) + 1e-9)
    out = dict(zone=L.cos_sim(Zo - 0), frame=L.cos_sim(Fr), topo=-np.sqrt(((To[:, None] - To[None]) ** 2).sum(2)))
    out['combo'] = (zs(out['zone']) + zs(out['frame']) + zs(out['topo'])) / 3
    out['_topo_raw'] = np.array([d[2] for d in D])
    return out


def montage(script, alph, fn):
    ims = []
    for g in alph:
        a, base, size = render(g, FONTS[script], 48)
        ys, xs = np.where(a)
        ims.append(a[max(base - 120, 0):base + 50, max(xs.min() - 2, 0):xs.max() + 3] if len(xs) else a[:10, :10])
    W = sum(i.shape[1] + 8 for i in ims)
    H = max(i.shape[0] for i in ims)
    can = np.zeros((H, W), bool); x = 0
    for i in ims:
        can[:i.shape[0], x:x + i.shape[1]] = i; x += i.shape[1] + 8
    Image.fromarray(((~can) * 255).astype(np.uint8)).save(fn)


if __name__ == '__main__':
    for sc, sh in (('voynich', S.VOYNICH), ('latin', S.LATIN), ('greek', S.GREEK), ('hangul', S.HANGUL)):
        A = list(sh)
        montage(sc, A, os.path.join(L.SCR, f'mont_{sc}.png'))
        o = image_sims(sc, A)
        print(sc, 'topo raw:', {g: list(np.round(t, 1)) for g, t in zip(A[:8], o['_topo_raw'][:8])})
