#!/usr/bin/env python3
"""LA-7 step 1: render every syllabic sign (>=3 occurrences inside words) as a bare glyph under a
random code, so the depicted-object coding is done without the sign's value or contexts.
Writes contact sheets to <outdir>/sheet_k.png and the code key to data/la7_key.json (not opened
until coding is frozen)."""
import json, os, random, sys
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la7_common import sign_glyph_map, D

FONT = sys.argv[1]; OUT = sys.argv[2]
C = json.load(open(os.path.join(D, 'corpus.json')))
cnt = Counter(s for r in C for t in r['tokens'] if t['t'] == 'word' for s in t['s'])
gm = sign_glyph_map()
signs = sorted(s for s, n in cnt.items() if n >= 3 and s in gm)
rng = random.Random(7077)
codes = rng.sample(range(100, 1000), len(signs))
key = {('G%d' % c): s for c, s in zip(codes, signs)}
json.dump({'key': key, 'glyph': {s: gm[s][1] for s in signs}}, open(os.path.join(D, 'la7_key.json'), 'w'), indent=0)
items = sorted(key.items())
font = ImageFont.truetype(FONT, 150); small = ImageFont.load_default()
per = 20
for k in range(0, len(items), per):
    im = Image.new('RGB', (5 * 230, 4 * 250), 'white'); dr = ImageDraw.Draw(im)
    for j, (code, s) in enumerate(items[k:k + per]):
        x, y = (j % 5) * 230, (j // 5) * 250
        dr.text((x + 30, y + 10), gm[s][0], font=font, fill='black')
        dr.text((x + 90, y + 225), code, font=small, fill='red')
    im.save(os.path.join(OUT, 'sheet_%d.png' % (k // per)))
print(len(items), 'signs;', 'sheets', (len(items) + per - 1) // per)
print(' '.join(c for c, _ in items))
