"""v38: leakage probe. Embeds the page with the plant BLANKED (text and margins only), with the
same DINO and ResNet-18, so cycle 7 can ask whether pictures of the handwriting alone predict
vocabulary (leak route) and partial that out of the plant link.
Writes data/derived/v38_vis3_voynich.json. Usage: python3 v38_feat3.py CACHE
"""
import os, sys, json
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *
import torch, torchvision, timm
torch.set_num_threads(2)
from v38_feat import nets, emb

if __name__ == '__main__':
    cache = sys.argv[1]
    out = os.path.join(DER, 'v38_vis3_voynich.json')
    done = json.load(open(out)) if os.path.exists(out) else {}
    models = nets()
    for i, f in enumerate(sorted(x for x in os.listdir(cache) if x.startswith('v_'))):
        key = f[2:-4]
        if key in done:
            continue
        rgb = np.asarray(Image.open(os.path.join(cache, f)).convert('RGB'))
        c, keep, paint, ink, vel = voynich_plant(rgb)
        m = morphology.binary_dilation(keep, morphology.disk(6))
        img = c.copy().astype(float)
        img[m] = np.median(c[~m].reshape(-1, 3), axis=0)
        h, w = img.shape[:2]; s = max(h, w)
        sq = np.ones((s, s, 3)) * np.median(c[~m].reshape(-1, 3), axis=0)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
        im = np.asarray(Image.fromarray(sq.clip(0, 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR))
        r18, dn = emb(models, im)
        done[key] = dict(txt_r18=np.round(r18, 5).tolist(), txt_dino=np.round(dn, 5).tolist())
        if i % 10 == 0:
            json.dump(done, open(out, 'w')); print(i, key, flush=True)
    json.dump(done, open(out, 'w'))
    print('pages', len(done))
