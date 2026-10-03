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
    font = ImageFont.truetype(ROOT + 'tools/indus_font.ttf', 120)
    cell = 170
    im = Image.new('RGB', (cell * len(signs), cell + 30), 'white'); d = ImageDraw.Draw(im)
    small = ImageFont.load_default()
    for i, s in enumerate(signs):
        d.text((i * cell + 20, 10), chr(0xE000 + s), font=font, fill='black')
        d.text((i * cell + 20, cell), 'W%d' % s, font=small, fill='black')
    im.save(out); print(out, im.size)

if __name__ == '__main__':
    {'crop': crop, 'glyphs': glyphs}[sys.argv[1]](sys.argv[2:])
