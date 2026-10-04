"""v34: re-measure the true right end of every Voynich text line on the Yale images.
v18 clipped line profiles at the text-block edge, so lines that run into the margin all
share one end. Here, starting from the v18 last-word start, ink in a band around the
line's fitted baseline is followed to the right until a gap of > GAP glyph units.
Writes data/v34_ckpt/lineends.json. Usage: python3 v34_lineends.py [overlay_page]
"""
import sys, os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v18_lib as L18
import v34_lib as V
IMG = os.path.join(V.SCR, 'v18', 'img')
GAP = 1.0


def page(args):
    folio, recs, pitch, ov = args
    gray, con, rb = L18.load(os.path.join(IMG, folio + '.jpg'))
    mask = ndi.binary_opening(con > 0.15, np.ones((2, 2)))
    out = measure(mask, recs, pitch, gray)
    if ov:
        overlay(os.path.join(IMG, folio + '.jpg'), out, os.path.join(V.SCR, 'v34_ov_' + folio + '.png'))
    return folio, out


def overlay(fn, out, dst, scale_to=2000):
    from PIL import Image, ImageDraw
    im = Image.open(fn).convert('RGB')
    im = im.resize((scale_to, int(im.height * scale_to / im.width))); d = ImageDraw.Draw(im)
    for n, r in out.items():
        y = r['a'] + r['b'] * r['xend']
        d.line([(r['xend'], y - 15), (r['xend'], y + 15)], fill=(255, 0, 0), width=3)
        d.line([(r['xend_v18'], y - 8), (r['xend_v18'], y + 8)], fill=(0, 0, 255), width=3)
    im.resize((1000, int(im.height * 1000 / im.width))).save(dst)


def measure(mask, recs, pitch, gray=None):
    """v18 ends are kept unless the line was clipped at the page's text-block edge
    (end within 3 px of the page maximum); clipped lines are followed to the right
    through ink runs >= 3 px wide separated by gaps <= GAP glyph units, stopping at a
    dark border (median gray < 90)."""
    H, W = mask.shape
    byl = {}
    for w in recs:
        byl.setdefault(w['li'], []).append(w)
    clip = max(max(w['x1'] for w in ws) for ws in byl.values())
    out = {}
    h = max(4, int(0.28 * pitch))
    for li, ws in byl.items():
        ws = sorted(ws, key=lambda w: w['k'])
        xc = np.array([(w['x0'] + w['x1']) / 2 for w in ws]); yc = np.array([w['y'] for w in ws])
        if len(ws) >= 3:
            b, a = np.polyfit(xc, yc, 1)
        else:
            b, a = 0.0, float(yc.mean())
        g = sum(len(V.glyphs(w['word'])) for w in ws) + 0.6 * (len(ws) - 1)
        unit = (ws[-1]['x1'] - ws[0]['x0']) / max(g, 1)
        xend, clipped, edge = int(ws[-1]['x1']), ws[-1]['x1'] >= clip - 3, False
        if clipped:
            xs = np.arange(int(ws[-1]['x1']) - 2, W - 15)
            ys = np.clip((a + b * xs).astype(int), h, H - h - 1)
            rows = ys[None, :] + np.arange(-h, h + 1)[:, None]
            col = mask[rows, xs[None, :]].sum(axis=0) >= 2
            lab, n = ndi.label(col)
            runs = [s_[0] for s_ in ndi.find_objects(lab) if s_[0].stop - s_[0].start >= 3]
            end = 0
            for r in runs:
                if r.start - end > GAP * unit:
                    break
                if gray is not None and np.median(gray[rows[:, r], xs[None, r]]) < 90:
                    edge = True; break
                end = r.stop
            xend = int(xs[0] + end) if end else xend
        out[str(ws[0]['n'])] = {'li': li, 'xend': xend, 'xend_v18': ws[-1]['x1'], 'unit': float(unit),
                                'a': float(a), 'b': float(b), 'clipped': bool(clipped), 'edge': edge,
                                'lcost': ws[0].get('lcost', 0)}
    return out


if __name__ == '__main__':
    ov = sys.argv[1] if len(sys.argv) > 1 else None
    pages = json.load(open(os.path.join(V.DATA, 'derived', 'v18_words.json')))
    jobs = [(p['folio'], p['words'], p['pitch'], p['folio'] == ov) for p in pages]
    with ProcessPoolExecutor(2) as ex:
        res = dict(ex.map(page, jobs))
    os.makedirs(V.CKPT, exist_ok=True)
    json.dump(res, open(os.path.join(V.CKPT, 'lineends.json'), 'w'))
    for f, o in res.items():
        d = [r['xend'] - r['xend_v18'] for r in o.values()]
        print(f, len(o), 'mean extension px', round(float(np.mean(d)), 1), 'max', max(d))
