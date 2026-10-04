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


def components(mask, x0, x1):
    """Word-ish ink components: horizontal closing joins glyphs; returns list of
    (xa, xb, ya, yb, ycore, npix) where ycore is the densest row (x-height band)."""
    m = mask.copy(); m[:, :max(0, x0 - 80)] = 0; m[:, x1 + 60:] = 0
    j = ndi.binary_dilation(m, structure=np.ones((3, 9)))
    lab, n = ndi.label(j)
    out = []
    for k, sl in enumerate(ndi.find_objects(lab)):
        sub = (lab[sl] == k + 1) & m[sl]
        npx = int(sub.sum())
        if npx < 25:
            continue
        rows = ndi.uniform_filter1d(sub.sum(axis=1).astype(float), 7)
        out.append((sl[1].start, sl[1].stop, sl[0].start, sl[0].stop, sl[0].start + int(np.argmax(rows)), npx, k + 1))
    return out, lab


def assign_lines(comps, seeds, s, iters=4):
    """Assign components to seed lines; each line gets a robust linear y(x) fit."""
    nl = len(seeds)
    coef = [(0.0, float(y)) for y in seeds]
    xs = np.array([(c[0] + c[1]) / 2 for c in comps]); ys = np.array([c[4] for c in comps], float)
    w = np.array([c[5] for c in comps], float)
    for it in range(iters):
        pred = np.array([[a * x + b for (a, b) in coef] for x in xs])  # ncomp x nl
        d = np.abs(pred - ys[:, None])
        lab = d.argmin(axis=1); dm = d.min(axis=1)
        ok = dm < 0.45 * s
        newc = []
        for li in range(nl):
            sel = (lab == li) & ok
            if sel.sum() >= 3 and np.ptp(xs[sel]) > 200:
                A = np.vstack([xs[sel], np.ones(sel.sum())]).T
                sol, *_ = np.linalg.lstsq(A * np.sqrt(w[sel])[:, None], ys[sel] * np.sqrt(w[sel]), rcond=None)
                a = float(np.clip(sol[0], -0.04, 0.04)); b = float(np.average(ys[sel] - a * xs[sel], weights=w[sel]))
                newc.append((a, b))
            else:
                newc.append(coef[li])
        coef = newc
    return lab, ok, coef


def line_spans(comps, idx, gapmin):
    """Components of one line -> x-spans (merged when overlapping or closer than gapmin);
    each span keeps its component label ids."""
    cs = sorted((comps[i] for i in idx), key=lambda c: c[0])
    spans = []
    for c in cs:
        if spans and c[0] - spans[-1][1] < gapmin:
            a, b, ids = spans[-1]; spans[-1] = (a, max(b, c[1]), ids + [c[6]])
        else:
            spans.append((c[0], c[1], [c[6]]))
    return spans


def measure(contrast, rb, mask, lab, ids, a, b):
    ys, xs = None, None
    sub = lab[:, a:b]
    sel = np.isin(sub, ids) & mask[:, a:b]
    v = contrast[:, a:b][sel]
    if v.size < 30:
        return None
    q = np.sort(v)
    return {'med': float(np.median(v)), 'top': float(q[int(0.7 * len(q)):].mean()),
            'p90': float(q[int(0.9 * len(q))]), 'lo': float(q[:int(0.3 * len(q)) + 1].mean()),
            'area': float(v.size), 'rb': float(rb[:, a:b][sel].mean())}


def chain_lines(comps, s, maxgap=130, dyfrac=0.3):
    """Union components that are horizontal neighbours at the same core height; returns
    chains (list of comp-index lists) with a linear y(x) fit, sorted top to bottom."""
    n = len(comps)
    par = list(range(n))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    order = sorted(range(n), key=lambda i: comps[i][0])
    xa = np.array([comps[i][0] for i in order]); xb = np.array([comps[i][1] for i in order])
    yc = np.array([comps[i][4] for i in order])
    for p in range(n):
        q = p + 1
        while q < n and xa[q] < xb[p] + maxgap:
            if abs(yc[q] - yc[p]) < dyfrac * s and xa[q] > xa[p]:
                par[f(order[q])] = f(order[p])
            q += 1
    groups = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    chains = []
    for g in groups.values():
        xs = np.array([(comps[i][0] + comps[i][1]) / 2 for i in g]); ys = np.array([comps[i][4] for i in g], float)
        w = np.array([comps[i][5] for i in g], float)
        x_lo = min(comps[i][0] for i in g); x_hi = max(comps[i][1] for i in g)
        if len(g) >= 3 and np.ptp(xs) > 150:
            a = np.polyfit(xs, ys, 1, w=np.sqrt(w))[0]; a = float(np.clip(a, -0.05, 0.05))
        else:
            a = 0.0
        b = float(np.average(ys - a * xs, weights=w))
        chains.append({'idx': g, 'a': a, 'b': b, 'x0': x_lo, 'x1': x_hi, 'ink': float(w.sum())})
    return chains


def merge_chains(chains, s, tol=0.3):
    """Merge chains lying on the same text line (collinear, not overlapping in x)."""
    chains = sorted(chains, key=lambda c: c['x0'])
    changed = True
    while changed:
        changed = False
        for i in range(len(chains)):
            for j in range(len(chains)):
                if i == j:
                    continue
                A, B = chains[i], chains[j]
                if B['x0'] < A['x1'] - 15:
                    continue
                big = A if A['ink'] >= B['ink'] else B
                xm = (A['x1'] + B['x0']) / 2
                ya = big['a'] * xm + big['b']
                yb_ = (B['a'] * xm + B['b']) if big is A else (A['a'] * xm + A['b'])
                if abs(ya - yb_) < tol * s and B['x0'] - A['x1'] < 600:
                    A['idx'] = A['idx'] + B['idx']; A['x1'] = max(A['x1'], B['x1'])
                    A['a'], A['b'] = big['a'], big['b']; A['ink'] += B['ink']
                    del chains[j]; changed = True
                    break
            if changed:
                break
    return chains


def match_lines(chains, lines_glyphs, unit, skip_chain_pen, skip_line_pen=6.0):
    """Monotone DP matching chains (sorted top->bottom) to transliterated lines."""
    C = sorted(chains, key=lambda c: c['a'] * (c['x0'] + c['x1']) / 2 + c['b'])
    m, n = len(C), len(lines_glyphs)
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF); dp[0, 0] = 0
    bk = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i, j] >= INF:
                continue
            if j < m:  # skip chain
                c = dp[i, j] + skip_chain_pen(C[j])
                if c < dp[i, j + 1]:
                    dp[i, j + 1] = c; bk[(i, j + 1)] = (i, j, 'sc')
            if i < n:  # skip line
                c = dp[i, j] + skip_line_pen
                if c < dp[i + 1, j]:
                    dp[i + 1, j] = c; bk[(i + 1, j)] = (i, j, 'sl')
            if i < n and j < m:
                wd = C[j]['x1'] - C[j]['x0']
                c = dp[i, j] + 3 * np.log(wd / (unit * lines_glyphs[i])) ** 2
                if c < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = c; bk[(i + 1, j + 1)] = (i, j, 'm')
    i, j = n, m
    out = [None] * n
    while i > 0 or j > 0:
        pi, pj, t = bk[(i, j)]
        if t == 'm':
            out[pi] = C[pj]
        i, j = pi, pj
    return out, dp[n, m]


def chain_profile(lab, mask, comps, idx, a, b, s, band=0.3):
    """Column ink profile of a chain's components (real ink pixels, not the dilation),
    counted only within +-band*s of the core line y = a x + b."""
    xa = min(comps[i][0] for i in idx); xb = max(comps[i][1] for i in idx)
    ids = [comps[i][6] for i in idx]
    prof = np.zeros(xb - xa)
    for i in idx:
        c = comps[i]
        sub = (lab[c[2]:c[3], c[0]:c[1]] == c[6]) & mask[c[2]:c[3], c[0]:c[1]]
        yy = np.arange(c[2], c[3])[:, None]; xx = np.arange(c[0], c[1])[None, :]
        sub &= np.abs(yy - (a * xx + b)) < band * s
        prof[c[0] - xa:c[1] - xa] += sub.sum(axis=0)
    nz = np.nonzero(prof)[0]
    if len(nz) == 0:
        return xa, xb, prof, ids
    return xa + nz[0], xa + nz[-1] + 1, prof[nz[0]:nz[-1] + 1], ids


def align_gaps(xa, prof, words, mingap=3, gapw=1.0):
    """Choose len(words)-1 word boundaries among the blank-column runs of the profile."""
    nw = len(words)
    ink = prof > 0
    ink = ndi.binary_closing(ink, np.ones(2))
    # gaps = runs of empty columns
    lab, n = ndi.label(~ink)
    gaps = []
    for sl in ndi.find_objects(lab):
        a, b = sl[0].start, sl[0].stop
        if a == 0 or b == len(prof) or b - a < mingap:
            continue
        gaps.append(((a + b) / 2, b - a))
    if nw == 1:
        return [(xa, xa + len(prof))], 0.0
    if len(gaps) < nw - 1:
        return None, None
    g = np.array([max(1, len(glyphs(w))) for w in words], float)
    W = len(prof)
    unit = W / (g.sum() + 0.6 * (nw - 1))
    pos = np.array([p for p, _ in gaps]); wid = np.array([w for _, w in gaps], float)
    medw = np.median(np.sort(wid)[-(nw - 1):])
    K = len(gaps)
    INF = 1e18
    # dp[i][k]: boundary i (0-based) placed at gap k
    dp = np.full((nw - 1, K), INF); bk = np.zeros((nw - 1, K), int)
    exp = unit * g
    for k in range(K):
        dp[0, k] = 4 * np.log(max(pos[k], 1) / exp[0]) ** 2 - gapw * min(wid[k] / medw, 1.5)
    for i in range(1, nw - 1):
        for k in range(i, K):
            wds = pos[k] - pos[:k]
            c = dp[i - 1, :k] + 4 * np.log(np.maximum(wds, 1) / exp[i]) ** 2
            j = int(np.argmin(c)); dp[i, k] = c[j] - gapw * min(wid[k] / medw, 1.5); bk[i, k] = j
    last = dp[-1] + 4 * np.log(np.maximum(W - pos, 1) / exp[-1]) ** 2
    k = int(np.argmin(last)); cost = last[k]
    bnd = [k]
    for i in range(nw - 2, 0, -1):
        k = bk[i, k]; bnd.append(k)
    bnd = [pos[k] for k in bnd[::-1]]
    edges = [0] + bnd + [W]
    out = [(int(xa + edges[i]), int(xa + edges[i + 1])) for i in range(nw)]
    return out, float(cost / nw)


def match_lines2(chains, lines_glyphs, unit, s, skip_chain_pen, skip_line_pen=6.0, wsp=6.0):
    """Like match_lines, with a spacing term: consecutive matched lines should be
    (number of lines apart) x s apart vertically."""
    C = sorted(chains, key=lambda c: c['a'] * (c['x0'] + c['x1']) / 2 + c['b'])
    yc = np.array([c['a'] * (c['x0'] + c['x1']) / 2 + c['b'] for c in C])
    sk = np.array([skip_chain_pen(c) for c in C]); csk = np.concatenate([[0], np.cumsum(sk)])
    m, n = len(C), len(lines_glyphs)
    wd = np.array([c['x1'] - c['x0'] for c in C], float)
    INF = 1e18
    dp = np.full((n, m), INF); bk = {}
    mc = lambda i, j: 3 * np.log(wd[j] / (unit * lines_glyphs[i])) ** 2
    for i in range(min(3, n)):
        for j in range(m):
            dp[i, j] = i * skip_line_pen + csk[j] + mc(i, j)
    for i in range(1, n):
        for j in range(1, m):
            best = dp[i, j]; arg = None
            for di in (1, 2, 3):
                ip = i - di
                if ip < 0:
                    break
                jp = np.arange(j)
                dy = (yc[j] - yc[jp]) / s
                c = dp[ip, jp] + (csk[j] - csk[jp + 1]) + wsp * (dy - di) ** 2 + (di - 1) * skip_line_pen
                k = int(np.argmin(c))
                if c[k] + mc(i, j) < best:
                    best = c[k] + mc(i, j); arg = (ip, k)
            dp[i, j] = best
            if arg:
                bk[(i, j)] = arg
    tot = np.full((n, m), INF)
    for i in range(n):
        for j in range(m):
            tot[i, j] = dp[i, j] + (n - 1 - i) * skip_line_pen + (csk[m] - csk[j + 1])
    i, j = np.unravel_index(np.argmin(tot), tot.shape)
    cost = tot[i, j]
    out = [None] * n
    while True:
        out[i] = C[j]
        if (i, j) not in bk:
            break
        i, j = bk[(i, j)]
    return out, float(cost)


def line_pitch(mask, x0, x1):
    p = mask[:, x0:x1].mean(axis=1); p = p - p.mean()
    ac = np.correlate(p, p, 'full')[len(p) - 1:]
    lag = 28 + int(np.argmax(ac[28:80]))
    return float(lag)


def strip_tracks(mask, x0, x1, s, sw=200, step=100):
    """Row-profile peaks in overlapping vertical strips, linked left->right into tracks."""
    centers, peaks = [], []
    for xa in range(x0, x1 - sw // 2, step):
        xb = min(x1, xa + sw)
        p = ndi.gaussian_filter1d(mask[:, xa:xb].mean(axis=1), 3)
        pk, pr = find_peaks(p, distance=int(0.55 * s), prominence=0.02)
        pk = pk[p[pk] > 0.06]
        centers.append((xa + xb) / 2); peaks.append([(int(y), float(p[y])) for y in pk])
    tracks = []  # each: list of (strip index, y, height)
    active = []
    for si, pl in enumerate(peaks):
        used = set()
        cand = []
        for ti in active:
            t = tracks[ti]
            if si - t[-1][0] > 3:
                continue
            # predicted y: last y plus local slope
            if len(t) >= 2:
                sl = (t[-1][1] - t[-2][1]) / max(1, t[-1][0] - t[-2][0])
                sl = float(np.clip(sl, -0.04 * step, 0.04 * step))
            else:
                sl = 0
            yp = t[-1][1] + sl * (si - t[-1][0])
            for k, (y, h) in enumerate(pl):
                d = abs(y - yp)
                if d < 0.3 * s:
                    cand.append((d, ti, k))
        cand.sort()
        taken_t = set()
        for d, ti, k in cand:
            if ti in taken_t or k in used:
                continue
            tracks[ti].append((si, pl[k][0], pl[k][1])); taken_t.add(ti); used.add(k)
        for k, (y, h) in enumerate(pl):
            if k not in used:
                tracks.append([(si, y, h)]); active.append(len(tracks) - 1)
        active = [ti for ti in active if si - tracks[ti][-1][0] <= 3]
    out = []
    for t in tracks:
        xs = np.array([centers[a] for a, _, _ in t]); ys = np.array([y for _, y, _ in t], float)
        out.append({'xs': xs, 'ys': ys, 'x0': xs[0] - sw / 2, 'x1': xs[-1] + sw / 2,
                    'ink': float(sum(h for _, _, h in t)), 'n': len(t),
                    'ym': float(np.median(ys))})
    return out


def track_y(t, x):
    return np.interp(x, t['xs'], t['ys'])


def track_profile(mask, t, x0, x1, s, band=0.25):
    xa = int(max(x0, t['x0'] - 60)); xb = int(min(x1, t['x1'] + 60))
    xx = np.arange(xa, xb)
    yy = track_y(t, xx)
    h = int(band * s)
    prof = np.zeros(len(xx))
    for k in range(-h, h + 1):
        r = np.clip((yy + k).astype(int), 0, mask.shape[0] - 1)
        prof += mask[r, xx]
    nz = np.nonzero(prof >= 2)[0]
    if len(nz) == 0:
        return xa, xa, prof[:0]
    return xa + nz[0], xa + nz[-1] + 1, prof[nz[0]:nz[-1] + 1]


def suppress_ascenders(tr, s):
    """Drop weak tracks lying within 0.7 pitch of a much stronger overlapping track
    (gallows ascenders and descender rows form weak profile peaks)."""
    for t in tr:
        t['h'] = t['ink'] / max(1, t['n'])
    keep = []
    for t in tr:
        bad = False
        for u in tr:
            if u is t:
                continue
            ov = min(t['x1'], u['x1']) - max(t['x0'], u['x0'])
            if ov <= 0.5 * (t['x1'] - t['x0']):
                continue
            xm = (max(t['x0'], u['x0']) + min(t['x1'], u['x1'])) / 2
            if abs(track_y(t, xm) - track_y(u, xm)) < 0.85 * s and u['h'] > 1.3 * t['h'] and u['h'] * u['n'] > 1.5 * t['h'] * t['n']:
                bad = True; break
        if not bad:
            keep.append(t)
    return keep


def align_gaps_lead(xa, prof, words, unit_page=None, maxlead=260, pen=0.08):
    """align_gaps, also trying to drop leading ink (margin stars, stalks) up to maxlead px:
    the start may move to just after any blank run of >=8 px in the first maxlead px.
    Alternatives are compared with a total-width term against the page glyph unit."""
    G = sum(max(1, len(glyphs(w))) for w in words) + 0.6 * (len(words) - 1)
    tw = (lambda W: 1.0 * np.log(W / (unit_page * G)) ** 2) if unit_page else (lambda W: 0.0)
    best = align_gaps(xa, prof, words)
    best = (best[0], best[1] + tw(len(prof)) if best[1] is not None else 1e9)
    ink = ndi.binary_closing(prof > 0, np.ones(2))
    lab, n = ndi.label(~ink)
    for sl in ndi.find_objects(lab):
        a, b = sl[0].start, sl[0].stop
        if a == 0 or b >= min(len(prof), maxlead) or b - a < 8:
            continue
        rest = prof[b:]
        nz = np.nonzero(rest)[0]
        if len(nz) < 10:
            continue
        al, c = align_gaps(xa + b + nz[0], rest[nz[0]:nz[-1] + 1], words)
        if al is not None and c + pen + tw(nz[-1] - nz[0] + 1) < best[1]:
            best = (al, c + pen + tw(nz[-1] - nz[0] + 1))
    return best
