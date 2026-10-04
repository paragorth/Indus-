"""v44 automatic motor signatures from glyph images (no hand ductus): render each glyph in a font, skeletonize,
take the leftmost skeleton endpoint as pen entry and the rightmost as pen exit; stroke directions from the
skeleton 8 px in from each endpoint. A glyph without endpoints (closed loop) enters and exits at its top.
Voynich: Landini's EVA Hand A font (scratchpad, not committed). Output: data/v44_ckpt/sig_font_voynich.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import v25_image as I
import v44_lib as L


def walk(sk, p, steps=8):
    """follow the skeleton from endpoint p for `steps` pixels; return the reached pixel."""
    seen = {p}; cur = p
    for _ in range(steps):
        y, x = cur; nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if q in seen or not (0 <= q[0] < sk.shape[0] and 0 <= q[1] < sk.shape[1]): continue
                if sk[q]: nxt = q; break
            if nxt: break
        if nxt is None: break
        seen.add(nxt); cur = nxt
    return cur


def signature(ch, font, xh):
    a, base, size = I.render(ch, font)
    ys, xs = np.where(a)
    x0 = xs.min(); w = (xs.max() - xs.min()) / xh
    sk = skeletonize(a)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode='constant') - 1
    E = list(zip(*np.where((nb == 1) & sk)))
    Y = lambda y: (base - y) / xh
    X = lambda x: (x - x0) / xh
    if not E:
        top = min(zip(ys, xs)); p = (top[0], top[1])
        return (w, X(p[1]), Y(p[0]), 180.0, X(p[1]), Y(p[0]), 0.0)
    ent = min(E, key=lambda p: (p[1], p[0])); ext = max(E, key=lambda p: (p[1], -p[0]))
    q = walk(sk, ent); edir = np.degrees(np.arctan2(-(q[0] - ent[0]), q[1] - ent[1]))   # from endpoint inward
    r = walk(sk, ext); xdir = np.degrees(np.arctan2(-(ext[0] - r[0]), ext[1] - r[1]))  # from inside to endpoint
    return (w, X(ent[1]), Y(ent[0]), float(edir), X(ext[1]), Y(ext[0]), float(xdir))


if __name__ == '__main__':
    xh = I.xheight('voynich')
    out = {g: signature(g, I.FONTS['voynich'], xh) for g in L.VOYNICH_HAND}
    L.save('sig_font_voynich.json', out)
    for g, s in out.items(): print(g, ' '.join(f'{v:6.2f}' for v in s))
