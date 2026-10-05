"""v70 THE TRANSCRIPTION ALPHABET IS WRONG; LET THE INK DECIDE: shared library.

Units are cut from the page images (Beinecke scans for the Voynich, CREMMA-Medieval-Lat for
the Latin control, synthetic hands rendered here) with no reference to any transcription
alphabet. Word boxes come from v18 (ink-gap blobs aligned to transliterated words; the
alignment uses word counts only, never glyph identities inside the word).

Per word box: deskewed band of +-0.6 line pitch; ink mask; units = column runs of the ink
mask separated at columns whose ink count <= theta (theta in pitch-scaled pixels) and at
deep local minima of the column profile; each unit becomes a 24x32 patch (height-normalised,
aspect kept up to 32 px, centred) plus its width/pitch.
Images stay in the scratch cache; only arrays of patches/ids go to data/v70_ckpt (git-ignored).
"""
import os, json, collections, math
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.join(HERE, '..', 'data', 'derived')
CK = os.path.join(HERE, '..', 'data', 'v70_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
PH, PW = 24, 32
EVA_MULTI = ['cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']


def vglyphs(w):
    out, i = [], 0
    while i < len(w):
        for m in EVA_MULTI:
            if w.startswith(m, i):
                out.append(m); i += len(m); break
        else:
            out.append(w[i]); i += 1
    return out


def load_con(fn, rescale_to=None):
    im = Image.open(fn).convert('RGB')
    if rescale_to:
        sc = rescale_to / im.width
        im = im.resize((rescale_to, int(im.height * sc)))
    im = np.asarray(im).astype(np.float32)
    gray = im.mean(axis=2)
    small = gray[::8, ::8]
    bg = ndi.median_filter(small, size=15)
    bg = ndi.zoom(bg, (gray.shape[0] / bg.shape[0], gray.shape[1] / bg.shape[1]), order=1)
    bg = bg[:gray.shape[0], :gray.shape[1]]
    con = np.clip(bg - gray, 0, None) / np.maximum(bg, 1)
    con[ndi.binary_dilation(bg < 110, iterations=20)] = 0
    return con


def word_band(con, x0, x1, yc, slope, s, T=0.15):
    H, W = con.shape
    hb = int(0.6 * s)
    xs = np.arange(max(0, x0 - 2), min(W, x1 + 2))
    if len(xs) < 4:
        return None
    ys = (yc + slope * (xs - (x0 + x1) / 2)).astype(int)
    rr = np.clip(ys[None, :] + np.arange(-hb, hb + 1)[:, None], 0, H - 1)
    c = con[rr, xs[None, :]]
    m = c > T
    m = ndi.binary_opening(m, np.ones((2, 2)))
    # drop specks
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n:
        sz = np.bincount(lab.ravel()); keep = sz >= 6; keep[0] = False
        m = keep[lab]
    return c, m


def cut_units(m, s, theta):
    """Cut a word mask into units. theta: cut where column ink <= theta*s/45 pixels and the
    column is a local minimum; theta=0 cuts only at blank columns. Units narrower than
    0.12*s are merged into the neighbour with the smaller gap."""
    prof = m.sum(0).astype(float)
    W = len(prof)
    thr = theta * s / 45.0
    sm = ndi.uniform_filter1d(prof, 3)
    cut = np.zeros(W, bool)
    cut[prof == 0] = True
    if theta > 0:
        for j in range(1, W - 1):
            if prof[j] <= thr and sm[j] <= sm[j - 1] and sm[j] <= sm[j + 1]:
                cut[j] = True
    segs, a = [], None
    for j in range(W):
        if not cut[j] and a is None:
            a = j
        elif cut[j] and a is not None:
            segs.append([a, j]); a = None
    if a is not None:
        segs.append([a, W])
    minw = max(2, 0.12 * s)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (a, b) in enumerate(segs):
            if b - a < minw:
                if i == 0:
                    j = 1
                elif i == len(segs) - 1:
                    j = i - 1
                else:
                    j = i - 1 if (a - segs[i - 1][1]) <= (segs[i + 1][0] - b) else i + 1
                lo, hi = min(i, j), max(i, j)
                segs[lo] = [segs[lo][0], segs[hi][1]]; del segs[hi]
                changed = True
                break
    return [(a, b) for a, b in segs if m[:, a:b].sum() >= 8]


def patch(c, m, a, b):
    sub = (c * m)[:, a:b]
    H, W = sub.shape
    sc = PH / H
    w2 = max(1, int(round(W * sc)))
    im = Image.fromarray((np.clip(sub / 0.6, 0, 1) * 255).astype(np.uint8))
    if w2 > PW:
        im = im.resize((PW, PH), Image.BILINEAR)
        w2 = PW
    else:
        im = im.resize((w2, PH), Image.BILINEAR)
    out = np.zeros((PH, PW), np.uint8)
    o = (PW - w2) // 2
    out[:, o:o + w2] = np.asarray(im)
    return out


def extract_words(con, words, s, thetas, T=0.15):
    """words: list of dicts with x0,x1,y and (li,k). Returns per theta: patches, widths,
    word index per unit, position in word."""
    res = {th: {'P': [], 'w': [], 'wid': [], 'pos': []} for th in thetas}
    bylines = collections.defaultdict(list)
    for i, w in enumerate(words):
        bylines[w['li']].append(i)
    for li, idx in bylines.items():
        ws = [words[i] for i in idx]
        if len(ws) >= 3:
            cx = np.array([(w['x0'] + w['x1']) / 2 for w in ws]); cy = np.array([w['y'] for w in ws])
            slope = float(np.polyfit(cx, cy, 1)[0])
        else:
            slope = 0.0
        for i in idx:
            w = words[i]
            r = word_band(con, w['x0'], w['x1'], w['y'], slope, s, T)
            if r is None:
                continue
            c, m = r
            for th in thetas:
                segs = cut_units(m, s, th)
                for p, (a, b) in enumerate(segs):
                    res[th]['P'].append(patch(c, m, a, b))
                    res[th]['w'].append((b - a) / s)
                    res[th]['wid'].append(i)
                    res[th]['pos'].append(p)
    return res


# ---------------------------------------------------------------- text battery
def ngram_H(seqs, order):
    """Conditional entropy H(u_t | previous order-1 units) within words, with '^' padding and '$' end."""
    c = collections.Counter(); cc = collections.Counter()
    for s in seqs:
        t = ['^'] * (order - 1) + list(s) + ['$']
        for i in range(order - 1, len(t)):
            ctx = tuple(t[i - order + 1:i]); c[(ctx, t[i])] += 1; cc[ctx] += 1
    n = sum(c.values())
    return -sum(v / n * math.log2(v / cc[k[0]]) for k, v in c.items())


def battery(words, ntok=4000, seed=0):
    """words: list of tuples (unit ids). Alphabet-size-aware feature vector."""
    rng = np.random.default_rng(seed)
    if len(words) > ntok:
        st = rng.integers(0, len(words) - ntok)
        words = words[st:st + ntok]
    words = [w for w in words if len(w)]
    units = [u for w in words for u in w]
    uc = collections.Counter(units)
    p = np.array(list(uc.values()), float); p /= p.sum()
    h1 = float(-(p * np.log2(p)).sum())
    keff = 2 ** h1
    H1w = ngram_H(words, 1); H2 = ngram_H(words, 2); H3 = ngram_H(words, 3)
    L = np.array([len(w) for w in words], float)
    wc = collections.Counter(words)
    fr = np.sort(np.array(list(wc.values()), float))[::-1]
    r = np.arange(1, len(fr) + 1)
    k = min(len(fr), 300)
    zipf = float(np.polyfit(np.log(r[:k]), np.log(fr[:k]), 1)[0]) if k > 5 else 0.0
    ttr = len(wc) / len(words)
    hap = sum(1 for v in wc.values() if v == 1) / max(1, len(wc))
    first = collections.Counter(w[0] for w in words); last = collections.Counter(w[-1] for w in words)

    def ent(cn):
        q = np.array(list(cn.values()), float); q /= q.sum(); return float(-(q * np.log2(q)).sum())
    return {'h1n': h1 / math.log2(max(2, len(uc))), 'h2r': H2 / H1w, 'h3r': H3 / H1w,
            'Lm': float(L.mean()), 'Lcv': float(L.std() / L.mean()), 'zipf': zipf, 'ttr': ttr,
            'hap': hap, 'fl': ent(first) / max(1e-9, ent(last)), 'keff': keff}


BFEATS = ['h1n', 'h2r', 'h3r', 'Lcv', 'zipf', 'ttr', 'hap', 'fl']
