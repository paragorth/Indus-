"""v38: second-round per-page vectors (scratch images in, numbers out).
Fresh networks never inspected in cycles 1-2: EfficientNet-B0 (ImageNet) 'effb0', DINOv2 ViT-S/14
'dinov2'. Content-only variants: DINO on the binary silhouette 'dino_sil' and on the grey plant
'dino_gray'. Adaptive segmentation (lower threshold for faint drawings) 'dino_ad'.
Production features 'prod' (vellum colour, text-ink colour, paint mean colour, page brightness
and contrast): what a session or batch shares, not what a plant looks like.
Writes data/derived/v38_vis2_<src>.json
Usage: python3 v38_feat2.py CACHE voynich|gerard
"""
import os, sys, json, re
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *
import torch, timm
torch.set_num_threads(2)


def emb(model, img):
    x = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1)[None]
    m = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    s = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    with torch.no_grad():
        return model((x - m) / s)[0].numpy()


if __name__ == '__main__':
    cache, src = sys.argv[1], sys.argv[2]
    out = os.path.join(DER, 'v38_vis2_%s.json' % src)
    done = json.load(open(out)) if os.path.exists(out) else {}
    files = sorted(f for f in os.listdir(cache) if f.startswith('v_' if src == 'voynich' else 'g_'))
    if src == 'gerard':
        keep_keys = set(json.load(open(os.path.join(DER, 'v38_vis_gerard.json'))))
        ns = [int(re.findall(r'\d+', f)[0]) for f in files]
        ocr = gerard_ocr(os.path.join(cache, 'gerard_djvu.xml'), set(n - 1 for n in ns))
    effb0 = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=0).eval()
    dinov2 = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).eval()
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0).eval()
    for i, f in enumerate(files):
        key = f[2:-4]
        if key in done:
            continue
        rgb = np.asarray(Image.open(os.path.join(cache, f)).convert('RGB'))
        if src == 'voynich':
            c, keep, paint, ink, vel = voynich_plant(rgb)
            if keep.mean() < 0.08:
                c2, keep2, paint2, _, _ = voynich_plant(rgb, dE_thr=11)
            else:
                keep2 = keep
            img_ad = masked_plant_image(c, keep2)
        else:
            c, keep, paint, ink, vel = gerard_plant(rgb, ocr[int(key) - 1])
            img_ad = None
        img = masked_plant_image(c, keep)
        sil = np.zeros_like(img) + 230
        mk = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).resize((1, 1)))  # placeholder
        ys, xs = np.where(keep) if keep.sum() > 50 else np.where(np.ones_like(keep))
        m = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = m.shape; s = max(h, w)
        sq = np.zeros((s, s), bool); sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
        sil = np.asarray(Image.fromarray(np.where(sq, 40, 230).astype(np.uint8)).resize((224, 224))).astype(float)
        sil = np.stack([sil] * 3, -1)
        gray = np.stack([color.rgb2gray(img) * 255] * 3, -1)
        lab = color.rgb2lab(c)
        txt_ink = ink & ~morphology.binary_dilation(keep, morphology.disk(4))
        paper = ~ink & ~keep
        prod = np.concatenate([np.median(lab[paper], 0), np.median(lab[txt_ink], 0) if txt_ink.sum() > 50 else np.zeros(3),
                               lab[paint].mean(0) if paint.sum() > 50 else np.zeros(3),
                               [lab[..., 0].mean(), lab[..., 0].std()]])
        rec = dict(effb0=emb(effb0, img), dinov2=emb(dinov2, img), dino_sil=emb(dino, sil), dino_gray=emb(dino, gray), prod=prod)
        if img_ad is not None:
            rec['dino_ad'] = emb(dino, img_ad)
        done[key] = {k: np.round(np.asarray(v, float), 5).tolist() for k, v in rec.items()}
        if i % 10 == 0:
            json.dump(done, open(out, 'w')); print(i, key, flush=True)
    json.dump(done, open(out, 'w'))
    print('pages', len(done))
