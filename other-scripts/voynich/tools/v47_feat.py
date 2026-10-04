"""v47: per-unit embeddings with six networks (r18, dino, effb0, dinov2 = the v38 four; clip =
CLIP ViT-B/32 image tower, openai weights; mae = MAE ViT-B/16 self-supervised, mean patch token).
Sources: voynich (herbal pages, v38 segmentation), pharma (register units, v47_lib.pharma_units),
gerard (v38 scans), dodoens (Dodoens 1583, Latin; OCR word boxes blanked as for Gerard).
Writes data/v47_ckpt/vis_<src>.json (numbers only) and, for pharma, the unit texts.
Usage: python3 v47_feat.py CACHE voynich|pharma|gerard|dodoens
"""
import os, sys, json, re
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v47_lib import *

if __name__ == '__main__':
    cache, src = sys.argv[1], sys.argv[2]
    out = os.path.join(CK47, 'vis_%s.json' % src)
    done = json.load(open(out)) if os.path.exists(out) else {}
    if src == 'pharma':
        for u in pharma_units(cache):
            if u['key'] in done:
                continue
            rec = embed(u['img'])
            rec.update(area=u['area'], words=u['words'], labels=u['labels'], folio=u['folio'], para=u['para'], y=u['y'])
            done[u['key']] = rec
            print(u['key'], flush=True)
        json.dump(done, open(out, 'w'))
        sys.exit()
    pre = dict(voynich='v_', gerard='g_', dodoens='d_')[src]
    files = sorted(f for f in os.listdir(cache) if f.startswith(pre) and f.endswith('.jpg'))
    if src != 'voynich':
        ns = [int(re.findall(r'\d+', f)[0]) for f in files]
        ocr = gerard_ocr(os.path.join(cache, '%s_djvu.xml' % ('gerard' if src == 'gerard' else 'd')), set(n - 1 for n in ns))
    for i, f in enumerate(files):
        key = f[2:-4]
        if key in done:
            continue
        rgb = np.asarray(Image.open(os.path.join(cache, f)).convert('RGB'))
        if src == 'voynich':
            c, keep, paint, ink, vel = voynich_plant(rgb)
            rec = dict(area=float(keep.mean()))
        else:
            n = int(key)
            if n - 1 not in ocr:
                continue
            c, keep, paint, ink, vel = gerard_plant(rgb, ocr[n - 1])
            ws = [re.sub(r'[^a-z]', '', w[0].lower()) for w in ocr[n - 1][2]]
            rec = dict(area=float(keep.mean()), words=[w for w in ws if len(w) >= 3])
            if rec['area'] < 0.06 or len(rec['words']) < 30:
                done[key] = dict(area=rec['area'], skip=1)
                continue
        rec.update(embed(masked_plant_image(c, keep)))
        done[key] = rec
        if i % 10 == 0:
            json.dump(done, open(out, 'w')); print(i, key, flush=True)
    json.dump(done, open(out, 'w'))
    print('units', len(done))
