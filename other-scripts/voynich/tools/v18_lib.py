"""v18 THE PEN REMEMBERS: ink-darkness per transliterated word from page images.

Pipeline per page: background-normalised ink contrast -> deskew -> text-line bands
(horizontal projection peaks, matched to the transliteration's line count) -> per line,
ink blobs separated by gaps -> dynamic-programming alignment of blob groups to the
transliterated words (expected width ~ glyph count) -> per-word ink measures.
Images stay in the scratch cache; only per-word numbers are saved.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks

EVA_MULTI = ['cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']


def glyphs(w):
    out, i = [], 0
    while i < len(w):
        for m in EVA_MULTI:
            if w.startswith(m, i):
                out.append(m); i += len(m); break
        else:
            out.append(w[i]); i += 1
    return out


def load(fn):
    im = np.asarray(Image.open(fn).convert('RGB')).astype(np.float32)
    gray = im.mean(axis=2)
    small = gray[::8, ::8]
    bg = ndi.median_filter(small, size=15)
    bg = ndi.zoom(bg, (gray.shape[0] / bg.shape[0], gray.shape[1] / bg.shape[1]), order=1)
    bg = bg[:gray.shape[0], :gray.shape[1]]
    contrast = np.clip(bg - gray, 0, None) / np.maximum(bg, 1)  # 0 = background, ~0.6 = black
    contrast[bg < 110] = 0  # dark scanner background / gutter shadow is not ink
    contrast[ndi.binary_dilation(bg < 110, iterations=20)] = 0
    # colour of ink: red-blue difference normalised (fresh iron-gall vs faded brown)
    rb = (im[..., 0] - im[..., 2]) / np.maximum(gray, 1)
    return gray, contrast, rb


def skew_angle(mask, x0, x1):
    best = (0, -1)
    sub = mask[:, x0:x1].astype(np.float32)[::2, ::2]
    for a in np.arange(-3, 3.01, 0.25):
        r = ndi.rotate(sub, a, reshape=False, order=0)
        v = r.sum(axis=1).var()
        if v > best[1]:
            best = (a, v)
    return best[0]


def text_columns(mask):
    """x-range of the main text block: columns with sustained ink density."""
    col = ndi.uniform_filter1d(mask.mean(axis=0), 25)
    thr = 0.35 * np.percentile(col, 95)
    idx = np.where(col > thr)[0]
    # largest contiguous run (allowing small holes)
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > 40:
            runs.append((start, prev)); start = i
        prev = i
    runs.append((start, prev))
    return max(runs, key=lambda r: r[1] - r[0])


def find_lines(mask, nlines, x0, x1):
    """Pick exactly nlines peaks of the row-ink profile (core columns only, to avoid
    margin stars) by DP: high profile, spacing close to the typical line pitch."""
    xa = x0 + int(0.25 * (x1 - x0))
    prof = ndi.gaussian_filter1d(mask[:, xa:x1].mean(axis=1), 4)
    cand, _ = find_peaks(prof, distance=15)
    cand = cand[prof[cand] > 0.12 * prof.max()]
    h = prof[cand] / prof.max()
    strong = cand[h > 0.4]
    s = np.median(np.diff(strong)) if len(strong) > 3 else 45.0
    m = len(cand)
    if m < nlines:
        return None, (s, m)
    INF = 1e9
    dp = np.full((nlines, m), INF); bk = np.zeros((nlines, m), int)
    dp[0] = -h
    for i in range(1, nlines):
        for j in range(i, m):
            d = (cand[j] - cand[:j]) / s
            pen = np.where(d < 0.75, 25 * (0.75 - d) ** 2 * 16, 0) + np.where(d > 1.35, 0.8 * (d - 1.35), 0) + 0.5 * (d - 1) ** 2
            tot = dp[i - 1, :j] + pen
            k = int(np.argmin(tot)); dp[i, j] = tot[k] - h[j]; bk[i, j] = k
    j = int(np.argmin(dp[-1])); out = [j]
    for i in range(nlines - 1, 0, -1):
        j = bk[i, j]; out.append(j)
    return cand[out[::-1]], (s, m, float(dp[-1].min()))


def blobs_in_line(cmask, y, half, x0, x1, gapmin):
    band = cmask[max(0, y - half):y + half, x0:x1]
    colink = band.sum(axis=0) > 0
    colink = ndi.binary_closing(colink, structure=np.ones(3))
    lab, n = ndi.label(colink)
    sl = ndi.find_objects(lab)
    spans = [(s[0].start + x0, s[0].stop + x0) for s in sl if s[0].stop - s[0].start >= 4]
    # merge spans separated by less than gapmin
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < gapmin:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def align(spans, words):
    """DP: assign consecutive span groups to words; leading/trailing spans may be skipped.
    A group may cover 1 word, or 2 words (split by glyph proportion) when no gap is visible.
    Returns list of (x_start, x_end) per word, cost per word, or None."""
    nw, ns = len(words), len(spans)
    if ns == 0:
        return None, None
    g = np.array([max(1, len(glyphs(w))) for w in words], float)
    totw = sum(b - a for a, b in spans)
    unit = (spans[-1][1] - spans[0][0]) / (g.sum() + 0.6 * (nw - 1))
    INF = 1e18
    # dp[i][j]: best cost having consumed i words and j spans
    dp = np.full((nw + 1, ns + 1), INF)
    bk = {}
    for j in range(0, ns + 1):  # skip leading spans
        dp[0][j] = 1.5 * j
    for i in range(nw):
        for j in range(ns):
            if dp[i][j] >= INF:
                continue
            for k in range(j + 1, min(ns, j + 6) + 1):  # group spans j..k-1
                width = spans[k - 1][1] - spans[j][0]
                for nwd in (1, 2):
                    if i + nwd > nw:
                        continue
                    exp = unit * (g[i:i + nwd].sum() + 0.6 * (nwd - 1))
                    c = np.log(width / exp) ** 2 * 4 + (k - j - 1) * 0.6 + (nwd - 1) * 1.5
                    if dp[i][j] + c < dp[i + nwd][k]:
                        dp[i + nwd][k] = dp[i][j] + c
                        bk[(i + nwd, k)] = (i, j, nwd)
    # trailing skips
    jbest = min(range(ns + 1), key=lambda j: dp[nw][j] + 1.5 * (ns - j))
    if dp[nw][jbest] >= INF:
        return None, None
    out = [None] * nw
    i, j = nw, jbest
    while i > 0:
        pi, pj, nwd = bk[(i, j)]
        a, b = spans[pj][0], spans[j - 1][1]
        if nwd == 1:
            out[pi] = (a, b)
        else:
            f = g[pi] / (g[pi] + g[pi + 1])
            m = int(a + f * (b - a))
            out[pi] = (a, m); out[pi + 1] = (m, b)
        i, j = pi, pj
    return out, (dp[nw][jbest] + 1.5 * (ns - jbest)) / nw


def word_measures(contrast, rb, cmask, y, half, a, b):
    band = slice(max(0, y - half), y + half)
    c = contrast[band, a:b]
    m = cmask[band, a:b]
    v = c[m]
    if v.size < 20:
        return None
    q = np.sort(v)
    return {
        'med': float(np.median(v)),                     # median ink contrast
        'top': float(q[int(0.7 * len(q)):].mean()),     # darkest 30%
        'p90': float(q[int(0.9 * len(q))]),
        'area': float(m.sum()),                         # ink pixels (stroke width x length)
        'rb': float(rb[band, a:b][m].mean()),           # ink hue
    }
