"""Loop 47: crop, enlarge and mirror inscription bands of open-access seal photographs; render Wells glyphs for comparison.
Usage: python3 tools/dark_loop47_crop.py crop <image> <x0> <y0> <x1> <y1> <out.png> [scale] [--flip] [--rot deg]
       python3 tools/dark_loop47_crop.py glyphs <out.png> W1 W2 ...   (renders from tools/indus_font.ttf, U+E000 + Wells no.)
Images: data/derived/dark/loop47_images/ (SOURCES.txt). Fractions in [0,1] or pixel coordinates (> 1).
"""
import sys, os
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageEnhance, ImageFilter
Image.MAX_IMAGE_PIXELS = None
ROOT = '/home/user/Indus-/'
IMG = ROOT + 'data/derived/dark/loop47_images/'

def crop(args):
    f, x0, y0, x1, y1, out = args[:6]
    scale = float(args[6]) if len(args) > 6 and not args[6].startswith('--') else None
    flip = '--flip' in args
    rot = 0.0
    if '--rot' in args: rot = float(args[args.index('--rot') + 1])
    im = Image.open(f if os.path.exists(f) else IMG + f).convert('RGB')
    W, H = im.size
    x0, y0, x1, y1 = [float(v) for v in (x0, y0, x1, y1)]
    if max(x0, y0, x1, y1) <= 1.0: x0, x1 = x0 * W, x1 * W; y0, y1 = y0 * H, y1 * H
    if rot: im = im.rotate(rot, resample=Image.BICUBIC, expand=False)
    c = im.crop((int(x0), int(y0), int(x1), int(y1)))
    if scale is None: scale = max(1.0, 1400 / c.size[0])
    c = c.resize((int(c.size[0] * scale), int(c.size[1] * scale)), Image.LANCZOS)
    c = ImageOps.autocontrast(c, cutoff=1)
    c = ImageEnhance.Sharpness(c).enhance(1.5)
    if flip: c = ImageOps.mirror(c)
    c.save(out); print(out, c.size)

def glyphs(args):
    out = args[0]; signs = [int(s.lstrip('W')) for s in args[1:]]
    font = ImageFont.truetype(ROOT + 'tools/indus_font.ttf', 100)
    cell = 130; per = 8; rows = (len(signs) + per - 1) // per
    im = Image.new('RGB', (cell * min(per, len(signs)), rows * (cell + 30)), 'white'); d = ImageDraw.Draw(im)
    small = ImageFont.load_default(size=18) if hasattr(ImageFont, 'load_default') else ImageFont.load_default()
    for i, s in enumerate(signs):
        r, c = divmod(i, per)
        d.text((c * cell + 15, r * (cell + 30) + 5), chr(0xE000 + s), font=font, fill='black')
        d.text((c * cell + 15, r * (cell + 30) + cell + 2), 'W%d' % s, font=small, fill='red')
    im.save(out); print(out, im.size)

if __name__ == '__main__':
    if sys.argv[1] != 'sheet': {'crop': crop, 'glyphs': glyphs}[sys.argv[1]](sys.argv[2:])

def sheet(args):
    """sheet <image> <x0> <y0> <x1> <y1> <out.png> <split1,split2,...> : band split at fractional x positions into sign cells;
    row 1 = seal/photo orientation, row 2 = mirror image (impression orientation), cells reversed."""
    f, x0, y0, x1, y1, out, splits = args[:7]
    im = Image.open(f if os.path.exists(f) else IMG + f).convert('RGB'); W, H = im.size
    x0, y0, x1, y1 = [float(v) for v in (x0, y0, x1, y1)]
    if max(x0, y0, x1, y1) <= 1.0: x0, x1 = x0 * W, x1 * W; y0, y1 = y0 * H, y1 * H
    band = im.crop((int(x0), int(y0), int(x1), int(y1)))
    band = ImageOps.autocontrast(band, cutoff=1)
    cuts = [0.0] + [float(s) for s in splits.split(',')] + [1.0]
    cells = []
    for a, b in zip(cuts, cuts[1:]):
        c = band.crop((int(a * band.size[0]), 0, int(b * band.size[0]), band.size[1]))
        h = 420; c = c.resize((max(1, int(c.size[0] * h / c.size[1])), h), Image.LANCZOS)
        cells.append(ImageEnhance.Sharpness(c).enhance(1.4))
    wsum = sum(c.size[0] + 12 for c in cells)
    sh = Image.new('RGB', (wsum, 2 * 420 + 60), 'white'); d = ImageDraw.Draw(sh); x = 0
    for i, c in enumerate(cells):
        sh.paste(c, (x, 20)); d.text((x + 4, 2), f'{i+1}', fill='red'); x += c.size[0] + 12
    x = 0
    for i, c in enumerate(reversed(cells)):
        sh.paste(ImageOps.mirror(c), (x, 460)); d.text((x + 4, 442), f'mirror {len(cells)-i}', fill='blue'); x += c.size[0] + 12
    sh.save(out); print(out, sh.size)

if __name__ == '__main__' and sys.argv[1] == 'sheet':
    sheet(sys.argv[2:])
