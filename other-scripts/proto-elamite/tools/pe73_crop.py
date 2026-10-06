#!/usr/bin/env python3
"""pe73 helper: crop a CDLI photo or line art by fractions, contrast-stretch, upscale to a target long side.
Images stay in the scratchpad. usage: pe73_crop.py in.jpg out.jpg x0 y0 x1 y1 [target_long_side=1000] [--eq]"""
import sys
from PIL import Image, ImageOps
a = [x for x in sys.argv if x != '--eq']
im = Image.open(a[1]).convert('L')
W, H = im.size
x0, y0, x1, y1 = [float(v) for v in a[3:7]]
im = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
im = ImageOps.autocontrast(im, cutoff=1)
if '--eq' in sys.argv:
    im = ImageOps.equalize(im)
T = int(a[7]) if len(a) > 7 else 1000
s = T / max(im.size)
im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.LANCZOS)
im.save(a[2], quality=90)
print(im.size)
