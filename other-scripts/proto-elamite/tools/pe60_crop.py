#!/usr/bin/env python3
"""pe60 helper: crop a region of a CDLI photo (fractions of width/height), contrast-stretch, optional rotation,
and save a viewing copy (scratchpad only). Usage: pe60_crop.py in.jpg out.jpg x0 y0 x1 y1 [rot_deg] [maxside]"""
import sys
from PIL import Image, ImageOps
a = sys.argv
im = Image.open(a[1]).convert('L')
W, H = im.size
x0, y0, x1, y1 = [float(v) for v in a[3:7]]
im = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
rot = float(a[7]) if len(a) > 7 else 0
if rot:
    im = im.rotate(rot, expand=True)
im = ImageOps.autocontrast(im, cutoff=1)
im = ImageOps.equalize(im) if '--eq' in a else im
ms = int(a[8]) if len(a) > 8 and a[8].isdigit() else 1400
im.thumbnail((ms, ms))
im.save(a[2], quality=90)
print(im.size)
