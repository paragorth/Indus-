"""v70 synthetic hands: render any text in a Voynich-like stroke hand with pen jitter,
touching glyphs and known merged units, so the ink-unit pipeline can be checked against
ground truth. Output: contrast image (float, 0 = background), word boxes, per-glyph truth.

Shapes are stroke polylines in x-height units (baseline y=0, x-height y=1, ascenders to 2.2,
descenders to -0.9). Ligature units ('LIG:xy') are drawn as two shapes joined by a bench stroke.
"""
import math, random
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def arc(cx, cy, r, a0, a1, n=14, rx=None):
    rx = r if rx is None else rx
    return [(cx + rx * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]


# glyph: (advance width, [polylines])
G = {
    'o': (0.9, [arc(0.45, 0.5, 0.45, 0, 360, rx=0.38)]),
    'a': (0.95, [arc(0.42, 0.5, 0.45, 40, 330, rx=0.36), [(0.8, 0.95), (0.82, 0.0)]]),
    'e': (0.75, [arc(0.4, 0.5, 0.45, 40, 320, rx=0.32), [(0.62, 0.85), (0.7, 0.7)]]),
    'c': (0.7, [arc(0.4, 0.5, 0.45, 40, 320, rx=0.32)]),
    'i': (0.4, [[(0.2, 0.9), (0.2, 0.0)]]),
    'n': (0.75, [[(0.2, 0.9), (0.2, 0.0), (0.5, 0.3), (0.65, 1.0), (0.5, 1.2)]]),
    'r': (0.7, [[(0.25, 0.9), (0.2, 0.0)], [(0.22, 0.85), (0.6, 1.0), (0.6, 0.7)]]),
    'l': (0.75, [[(0.6, 1.9), (0.25, 0.4), (0.3, 0.0), (0.7, 0.1)]]),
    's': (0.8, [arc(0.4, 0.5, 0.45, 60, 320, rx=0.32), [(0.3, 0.95), (0.6, 1.6)]]),
    'y': (0.9, [arc(0.45, 0.55, 0.42, 0, 360, rx=0.33), [(0.78, 0.55), (0.7, -0.5), (0.3, -0.8)]]),
    'd': (0.95, [arc(0.42, 0.45, 0.42, 0, 360, rx=0.34), [(0.75, 0.5), (0.7, 1.4), (0.2, 1.9)]]),
    'k': (1.2, [[(0.25, 0.0), (0.25, 2.0)], arc(0.55, 1.3, 0.35, 180, 400, rx=0.3), [(0.85, 1.3), (0.9, 0.0)]]),
    't': (1.3, [[(0.3, 0.0), (0.3, 2.0)], arc(0.7, 1.3, 0.35, 160, 420, rx=0.4), [(1.05, 1.2), (1.1, 0.0)], arc(0.1, 1.5, 0.2, 0, 270)]),
    'p': (1.4, [[(0.3, 0.0), (0.3, 2.1)], arc(0.75, 1.4, 0.4, 170, 410, rx=0.45), [(1.15, 1.4), (1.15, 0.0)], [(-0.1, 0.9), (1.3, 0.9)]]),
    'f': (1.4, [[(0.3, 0.0), (0.3, 2.1)], arc(0.7, 1.5, 0.3, 170, 400, rx=0.4), [(1.05, 1.5), (1.15, 0.0)], [(0.0, 1.0), (1.3, 0.8)]]),
    'q': (0.9, [[(0.7, -0.6), (0.7, 1.1), (0.0, 0.3), (0.9, 0.3)]]),
    'm': (1.0, [[(0.2, 0.9), (0.2, 0.0)], [(0.5, 0.9), (0.5, 0.0), (0.9, 0.5), (0.8, 1.2)]]),
    'g': (0.9, [arc(0.45, 0.5, 0.4, 0, 360, rx=0.33), [(0.78, 0.5), (0.75, -0.7), (0.2, -0.6)]]),
    'x': (0.8, [[(0.1, 0.0), (0.7, 1.0)], [(0.1, 1.0), (0.7, 0.0)]]),
    'v': (0.8, [[(0.1, 1.0), (0.4, 0.0), (0.75, 1.0)]]),
    'z': (0.85, [[(0.1, 1.0), (0.75, 1.0), (0.1, 0.0), (0.75, 0.0)]]),
    'b': (0.9, [[(0.2, 2.0), (0.2, 0.0)], arc(0.5, 0.45, 0.42, 180, 540, rx=0.32)]),
    'u': (0.8, [[(0.15, 1.0), (0.15, 0.2), (0.4, 0.0), (0.65, 0.2), (0.65, 1.0)]]),
    'j': (0.5, [[(0.3, 1.0), (0.3, -0.6), (0.0, -0.8)]]),
    'h': (0.6, [[(0.1, 0.5), (0.5, 0.9), (0.5, 0.0)]]),
    'w': (1.1, [[(0.05, 1.0), (0.3, 0.0), (0.55, 0.8), (0.8, 0.0), (1.05, 1.0)]]),
}
# Voynich merged units: bench joins c c (ch), plume (sh), benched gallows
BENCH = {'ch': ('c', 'e'), 'sh': ('s', 'e')}


def unit_shape(u):
    """Return (advance, polylines) for a unit string; multi-glyph units are ligatures."""
    if u in G:
        return G[u]
    if u in BENCH:
        a, b = BENCH[u]
    elif len(u) == 3 and u[0] == 'c' and u[2] == 'h':   # cth etc: bench over gallows
        wa, pa = G['c']; wb, pb = G[u[1]]; wc, pc = G['e']
        pls = [p for p in pa] + [[(x + wa * 0.9, y) for x, y in pl] for pl in pb] + \
              [[(x + wa * 0.9 + wb * 0.9, y) for x, y in pl] for pl in pc]
        pls.append([(0.3, 0.95), (wa * 0.9 + wb * 0.9 + 0.4, 0.95)])
        return (wa * 0.9 + wb * 0.9 + wc, pls)
    else:
        a, b = u[0], u[1]
    wa, pa = G.get(a, G['x']); wb, pb = G.get(b, G['x'])
    pls = [p for p in pa] + [[(x + wa * 0.85, y) for x, y in pl] for pl in pb]
    pls.append([(0.35, 0.95), (wa * 0.85 + wb * 0.6, 0.95)])
    return (wa * 0.85 + wb, pls)


def render(lines_units, seed=0, xh=13.0, pitch=45, width=2000, margin=120, touch=0.18,
           jitter=0.05, pen=2.2, noise=0.03):
    """lines_units: list of lines; a line is a list of words; a word is a list of unit strings.
    Returns pages: list of dicts {con, words:[{x0,x1,y,li,k,units:[(u,xa,xb)]}], pitch}."""
    rng = random.Random(seed)
    S = 3  # supersampling
    pages, cur, li = [], None, 0
    H = 3000
    lines_per_page = (H - 2 * margin) // pitch

    def newpage():
        im = Image.new('L', (width * S, H * S), 0)
        return {'im': im, 'dr': ImageDraw.Draw(im), 'words': [], 'li': 0}
    for line in lines_units:
        if cur is None or cur['li'] >= lines_per_page:
            if cur is not None:
                pages.append(cur)
            cur = newpage()
        y0 = margin + cur['li'] * pitch + pitch * 0.6
        x = margin + rng.uniform(0, 10)
        slant = rng.uniform(-0.12, 0.05)
        for k, word in enumerate(line):
            if x > width - margin - 60:
                break
            wx0 = x; units = []
            sc = xh * rng.uniform(0.92, 1.08)
            for u in word:
                adv, pls = unit_shape(u)
                ua = x
                for pl in pls:
                    pts = []
                    for (px, py) in pl:
                        jx = px + rng.gauss(0, jitter); jy = py + rng.gauss(0, jitter)
                        X = x + (jx + slant * jy) * sc; Y = y0 - jy * sc
                        pts.append((X * S, Y * S))
                    cur['dr'].line(pts, fill=255, width=int(round(pen * S * rng.uniform(0.8, 1.2))), joint='curve')
                gapf = 0.0 if rng.random() < touch else rng.uniform(0.12, 0.3)
                x += (adv + gapf) * sc
                units.append((u, ua, x))
            cur['words'].append({'x0': int(wx0) - 2, 'x1': int(x) + 2, 'y': float(y0 - 0.5 * xh), 'li': cur['li'],
                                 'k': k, 'units': units, 'word': ''.join(u if len(u) == 1 else '[' + u + ']' for u in word)})
            x += rng.uniform(1.3, 2.0) * xh
        cur['li'] += 1
    if cur is not None and cur['words']:
        pages.append(cur)
    out = []
    nrng = np.random.default_rng(seed)
    for p in pages:
        a = np.asarray(p['im'].resize((width, H), Image.BOX)).astype(np.float32) / 255.0
        a = ndi.gaussian_filter(a, 0.6)
        a = a * nrng.uniform(0.4, 0.6) + nrng.normal(0, noise, a.shape).astype(np.float32)
        out.append({'con': np.clip(a, 0, None), 'words': p['words'], 'pitch': pitch})
    return out
