"""v47 'push the picture-text link until it breaks or gives words'.
Shared code: pharmaceutical-section units (each register of small root/leaf drawings paired with
the paragraph it sits on), network embeddings (the four v38 networks + a CLIP image tower + an MAE
self-supervised ViT), masked partial Mantel (pair subsets: held-out quires, hand-1 halves).
Images are read from a scratch cache only; only numbers are written.
"""
import os, sys, json, re, math
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *

CK47 = os.path.join(ROOT, 'data', 'v47_ckpt')
os.makedirs(CK47, exist_ok=True)

# ------------------------------------------------------------------ pharmaceutical panels
# (cache image, x0, x1 as fractions of the canvas width, candidate folios). Panel x-ranges were set
# by eye from the canvases BEFORE any text was looked at; ambiguous folio assignments (89v, 102v)
# are resolved by matching the detected text-line count to the transcribed paragraph-line count.
# f101v is split over two canvases with unknown line order and is left out.
PANELS = [('p_00', .05, .97, ['f88r']),
          ('p_01', .04, .30, ['f88v']), ('p_01', .31, .58, ['f89r1']), ('p_01', .585, .99, ['f89r2']),
          ('p_02', .03, .97, ['f89v1', 'f89v2']), ('p_03', .05, .385, ['f89v1', 'f89v2']),
          ('p_04', .04, .97, ['f99r']), ('p_05', .04, .97, ['f99v']), ('p_06', .04, .97, ['f100r']),
          ('p_07', .08, .385, ['f100v']), ('p_07', .39, .99, ['f101r']),
          ('p_09', .34, .645, ['f102r1']), ('p_09', .645, .99, ['f102r2']),
          ('p_10', .04, .97, ['f102v1', 'f102v2']), ('p_11', .04, .97, ['f102v1', 'f102v2'])]


def zl_paragraphs(folio):
    """Paragraphs (lists of lines of words) of a folio from the ZL3b line file (P lines only)."""
    recs = json.load(open(os.path.join(ROOT, 'data', 'derived', 'ZL3b_lines.json')))
    paras, cur = [], None
    for r in recs:
        if r['folio'] != folio or r['ltype'] != 'P':
            continue
        if r['para_start'] or cur is None:
            cur = []
            paras.append(cur)
        cur.append([w for w in r['words'] if '?' not in w])
    labels = [w for r in recs if r['folio'] == folio and r['ltype'] == 'L' for w in r['words']]
    return paras, labels


def panel_components(rgb, dE_thr=17, ink_drop=28):
    """Vellum-relative segmentation of a panel. Returns lab image, big (drawing) mask list,
    text (glyph-sized) mask, panel height."""
    lab = color.rgb2lab(rgb)
    vel = np.median(lab.reshape(-1, 3), axis=0)
    dE = np.sqrt(((lab - vel) ** 2).sum(-1))
    nonvel = morphology.binary_dilation(dE > dE_thr, morphology.disk(1))
    H, W = nonvel.shape
    lbl = measure.label(nonvel)
    big, txt = [], np.zeros_like(nonvel)
    for r in measure.regionprops(lbl):
        y0, x0, y1, x1 = r.bbox
        h, w = y1 - y0, x1 - x0
        edge = y0 <= 2 or x0 <= 2 or y1 >= H - 2 or x1 >= W - 2
        if r.area > 0.0015 * H * W or h > 0.05 * H:
            if edge and (w < 0.05 * W or h < 0.05 * H or r.extent < 0.08):
                continue           # binding shadow / page edge
            if h > 0.97 * H or w > 0.97 * W:
                continue
            jar = h > 1.4 * w and h > 0.08 * H and (x0 + x1) / 2 < 0.35 * W
            big.append(dict(mask=(lbl == r.label), bbox=r.bbox, jar=jar))
        elif h < 0.03 * H and w < 0.12 * W and r.area > 6:
            txt |= lbl == r.label
    return big, txt, H, W


def text_line_rows(txt, H, nb=40, cov=0.3):
    """Rows of paragraph text: glyph pixels spread over a wide stretch of the panel (labels and
    drawing fragments are narrow). Coverage = fraction of nb column bins with glyph pixels within
    a band of +-H/120 rows; peaks of the glyph profile inside high-coverage rows."""
    W = txt.shape[1]
    bins = np.array_split(np.arange(W), nb)
    occ = np.stack([txt[:, b].any(1) for b in bins], 1).astype(float)
    band = ndi.uniform_filter1d(occ, max(3, int(H / 60)), axis=0) > 0
    coverage = band.mean(1)
    prof = ndi.gaussian_filter1d(txt.sum(1).astype(float), H / 400.) * (coverage > cov)
    if not (prof > 0).any():
        return np.array([], int)
    pk, _ = find_peaks(prof, distance=max(3, int(H / 70)), height=0.2 * np.percentile(prof[prof > 0], 90))
    return pk


def tile_parts(rgb, parts, size=224):
    """Pack the register's drawings (largest first, up to 9) into a square grid, each part cropped to
    its own box on a vellum background, so small roots and leaves fill the network's view."""
    parts = sorted(parts, key=lambda c: -c['mask'].sum())[:9]
    k = int(np.ceil(np.sqrt(len(parts))))
    cell = size // k
    bg = np.median(rgb.reshape(-1, 3), axis=0)
    canvas = np.ones((size, size, 3)) * bg
    for i, c in enumerate(parts):
        y0, x0, y1, x1 = c['bbox']
        sub = rgb[y0:y1, x0:x1].astype(float).copy()
        mm = morphology.binary_dilation(c['mask'][y0:y1, x0:x1], morphology.disk(2))
        sub[~mm] = bg
        h, w = sub.shape[:2]; s = max(h, w)
        sq = np.ones((s, s, 3)) * bg
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = sub
        im = np.asarray(Image.fromarray(sq.clip(0, 255).astype(np.uint8)).resize((cell, cell), Image.BILINEAR))
        r, q = divmod(i, k)
        canvas[r * cell:(r + 1) * cell, q * cell:(q + 1) * cell] = im
    return canvas.clip(0, 255).astype(np.uint8)


def pharma_units(cache):
    """List of units: dict(folio, para index, words, labels, img (224 RGB), area, quire, y)."""
    from PIL import Image
    out, used = [], set()
    imgs = {}
    for ci, x0f, x1f, cands in PANELS:
        if ci not in imgs:
            imgs[ci] = np.asarray(Image.open(os.path.join(cache, ci + '.jpg')).convert('RGB'))
        full = imgs[ci]
        Hc, Wc = full.shape[:2]
        rgb = full[int(.02 * Hc):int(.98 * Hc), int(x0f * Wc):int(x1f * Wc)]
        big, txt, H, W = panel_components(rgb)
        rows = text_line_rows(txt, H)
        best = None
        for f in cands:
            if f in used:
                continue
            paras, labels = zl_paragraphs(f)
            L = sum(len(p) for p in paras)
            d = abs(L - len(rows))
            if best is None or d < best[0]:
                best = (d, f, paras, labels)
        _, folio, paras, labels = best
        used.add(folio)
        L = sum(len(p) for p in paras)
        P = len(rows)
        if P < 2:
            continue
        # paragraph j -> detected rows (proportional map of transcribed lines onto detected rows)
        spans, a = [], 0
        for p in paras:
            b = a + len(p)
            ia, ib = int(round(a * P / L)), max(int(round(b * P / L)) - 1, int(round(a * P / L)))
            spans.append((rows[min(ia, P - 1)], rows[min(ib, P - 1)]))
            a = b
        groups = [[] for _ in paras]
        for c in big:
            if c['jar']:
                continue
            y0, _, y1, _ = c['bbox']
            yc = (y0 + y1) / 2
            ds = []
            for (s0, s1) in spans:
                if yc < s0:
                    ds.append(s0 - yc)            # drawing above its paragraph: preferred
                elif yc > s1:
                    ds.append(1.5 * (yc - s1))
                else:
                    ds.append(0.)
            groups[int(np.argmin(ds))].append(c)
        for j, (p, g) in enumerate(zip(paras, groups)):
            ws = [w for l in p for w in l]
            if not g or len(ws) < 12:
                continue
            m = np.logical_or.reduce([x['mask'] for x in g])
            out.append(dict(key='%s.%d' % (folio, j), folio=folio, para=j, words=ws, labels=labels,
                            img=tile_parts(rgb, g), area=float(m.mean()), nparts=len(g),
                            y=float(np.mean(spans[j])) / H, panel=ci))
    return out


# ------------------------------------------------------------------ networks
_NETS = {}


def load_nets(which):
    import torch, torchvision, timm
    torch.set_num_threads(int(os.environ.get('V47_THREADS', 2)))
    for w in which:
        if w in _NETS:
            continue
        if w == 'r18':
            m = torchvision.models.resnet18(weights='IMAGENET1K_V1').eval(); m.fc = torch.nn.Identity()
        elif w == 'dino':
            m = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0).eval()
        elif w == 'effb0':
            m = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=0).eval()
        elif w == 'dinov2':
            m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).eval()
        elif w == 'mae':
            m = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=0).eval()
        elif w == 'clip':
            import open_clip
            m, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
            m = m.eval()
        _NETS[w] = m
    return _NETS


IMNET = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
CLIPN = (np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711]))


def embed(img, which=('r18', 'dino', 'effb0', 'dinov2', 'clip', 'mae')):
    import torch
    nets = load_nets(which)
    out = {}
    for w in which:
        mu, sd = CLIPN if w == 'clip' else IMNET
        x = torch.tensor(((img / 255. - mu) / sd).transpose(2, 0, 1)[None], dtype=torch.float32)
        with torch.no_grad():
            if w == 'clip':
                v = nets[w].encode_image(x)[0]
            elif w == 'mae':
                t = nets[w].forward_features(x)[0]
                v = t[1:].mean(0)            # mean of patch tokens (MAE CLS is untrained)
            else:
                v = nets[w](x)[0]
        out[w] = np.round(v.numpy().astype(float), 5).tolist()
    return out


CLIP_PROMPTS = {
    'root_bulb': 'a drawing of a plant with a round bulbous root',
    'root_tuber': 'a drawing of a plant with a thick swollen tuber root',
    'root_fibrous': 'a drawing of a plant with many thin fibrous roots',
    'root_tap': 'a drawing of a plant with a single long tap root',
    'leaf_round': 'a drawing of a plant with round leaves',
    'leaf_narrow': 'a drawing of a plant with long narrow grass-like leaves',
    'leaf_lobed': 'a drawing of a plant with deeply lobed or divided leaves',
    'leaf_serrate': 'a drawing of a plant with toothed serrated leaves',
    'flower': 'a drawing of a plant with flowers',
    'no_flower': 'a drawing of a plant without flowers',
    'berries': 'a drawing of a plant with berries or round fruits',
    'tree': 'a drawing of a tree or shrub',
    'climber': 'a drawing of a climbing vine',
    'blue': 'a plant drawing painted blue',
    'red': 'a plant drawing painted red',
}


def clip_text_matrix():
    import torch, open_clip
    nets = load_nets(['clip'])
    tok = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    with torch.no_grad():
        T = nets['clip'].encode_text(tok(list(CLIP_PROMPTS.values()))).numpy()
    return list(CLIP_PROMPTS), T / np.linalg.norm(T, axis=1, keepdims=True)


# ------------------------------------------------------------------ masked partial Mantel
class MPartial:
    """Partial Mantel on a subset of pairs (mask over upper-triangle pairs): residualise on
    confounds within the subset; permutation of page labels (within strata) on the visual side."""

    def __init__(self, conf_mats, n, pairmask=None):
        self.n = n
        self.iu = np.triu_indices(n, 1)
        self.m = np.ones(len(self.iu[0]), bool) if pairmask is None else pairmask[self.iu]
        X = [np.ones(self.m.sum())] + [upper(C)[self.m] for C in conf_mats]
        X = [x if x.std() == 0 else zs(x) for x in X]
        self.X = np.column_stack(X)
        self.P = np.linalg.pinv(self.X)

    def res(self, v):
        v = v[self.m]
        return v - self.X @ (self.P @ v)

    def stat(self, A, Tres):
        return float(np.mean([np.mean(zs(self.res(upper(Am))) * t) for Am, t in zip(A, Tres)]))

    def test(self, V, T, nperm, rng, strata=None):
        """Composite: mean partial r over all (network, text metric) cells; z vs permutation null."""
        Tres = [zs(self.res(upper(Tm))) for Tm in T.values()]
        def s(p):
            vs = [zs(self.res(upper(Vm[np.ix_(p, p)]))) for Vm in V.values()]
            return float(np.mean([np.mean(a * t) for a in vs for t in Tres]))
        idp = np.arange(self.n)
        obs = s(idp)
        null = np.array([s(perm(self.n, rng, strata)) for _ in range(nperm)])
        return dict(r=obs, z=float((obs - null.mean()) / null.std()), p=float((1 + (null >= obs).sum()) / (1 + nperm)), npairs=int(self.m.sum()))


def emb_sim_m(F):
    F = np.asarray(F, float)
    return cos_sim(F - F.mean(0))
