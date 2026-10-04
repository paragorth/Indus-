"""v38: per-page visual descriptors of the plant drawing (Voynich herbal pages, Gerard control).
Images are read from a scratch cache; only numeric vectors are written:
data/derived/v38_vis_<src>.json  {page: {family: [floats]}}
Families: shape (silhouette), colour (paint hue/sat), hog (edge layout), edge (orientation+LBP),
r18 (ImageNet ResNet-18 pooled), dino (self-supervised ViT-S/16 CLS).
Usage: python3 v38_feat.py CACHE voynich|gerard
"""
import os, sys, json, re
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *
import torch, torchvision, timm
torch.set_num_threads(2)


def nets():
    r18 = torchvision.models.resnet18(weights='IMAGENET1K_V1').eval()
    r18.fc = torch.nn.Identity()
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0).eval()
    return r18, dino


def emb(models, img):
    x = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1)[None]
    m = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    s = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (x - m) / s
    with torch.no_grad():
        return [mod(x)[0].numpy() for mod in models]


if __name__ == '__main__':
    cache, src = sys.argv[1], sys.argv[2]
    out = os.path.join(DER, 'v38_vis_%s.json' % src)
    done = json.load(open(out)) if os.path.exists(out) else {}
    files = sorted(f for f in os.listdir(cache) if f.startswith('v_' if src == 'voynich' else 'g_'))
    ocr = None
    if src == 'gerard':
        ns = [int(re.findall(r'\d+', f)[0]) for f in files]
        ocr = gerard_ocr(os.path.join(cache, 'gerard_djvu.xml'), set(n - 1 for n in ns))
    models = nets()
    for i, f in enumerate(files):
        key = f[2:-4]
        if key in done:
            continue
        rgb = np.asarray(Image.open(os.path.join(cache, f)).convert('RGB'))
        if src == 'voynich':
            c, keep, paint, ink, vel = voynich_plant(rgb)
        else:
            n = int(key)
            c, keep, paint, ink, vel = gerard_plant(rgb, ocr[n - 1])
        img = masked_plant_image(c, keep)
        r18, dn = emb(models, img)
        rec = dict(area=float(keep.mean()), shape=shape_desc(keep).tolist(), colour=colour_desc(c, paint).tolist(),
                   hog=texture_desc(img).tolist(), edge=edge_orient_desc(img).tolist(),
                   r18=r18.tolist(), dino=dn.tolist())
        done[key] = {k: (np.round(v, 5).tolist() if isinstance(v, list) else round(v, 5)) for k, v in rec.items()}
        if i % 10 == 0:
            json.dump(done, open(out, 'w'))
            print(i, key, flush=True)
    json.dump(done, open(out, 'w'))
    print('pages', len(done))
