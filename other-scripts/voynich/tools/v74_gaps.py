"""v74 cycle 1b: physical width of every written space between words, from the page images.
For each aligned line (v18 word boxes: Voynich Q20+f58 from data/derived/v18_words.json, extra pages from
data/v74_ckpt/words_extra.json; Latin CREMMA pages from v18_latin_words.json) the ink column profile of the
core band (+-0.25 line pitch around the line track through the word centres) is rebuilt with the v18 mask
(contrast > 0.15, 2x2 opening). Every word boundary of the v18 alignment sits at the centre of a blank-column
run; that run's width is the written space (px). Recorded per junction:
  gap      blank-run width at the aligned boundary (px)
  gap0     the same after re-aligning the line with the gap-width bonus switched off (gapw=0: boundaries chosen
           by expected word widths only), to check that the aligner's preference for wide gaps does not
           decide the result
  hgt      local glyph height (10-90% ink-row spread in +-0.55 pitch, mean of the two neighbouring words)
  lmed     median gap of the line; r = gap / lmed (frozen P1 statistic: r < 1)
Images stay in the scratch cache. Out: data/v74_ckpt/gaps_V.json, gaps_L.json"""
import os, sys, json, collections
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v74_lib as X
import v18_lib as V
DER = os.path.join(X.ROOT, 'data', 'derived')
SC18 = os.path.join(X.SCR, 'v18')


def lglyphs(w):
    return [c for c in w if not c.isspace()]


def img(which, folio):
    if which == 'V':
        for d in (os.path.join(SC18, 'img'), os.path.join(X.SCR, 'v74', 'img')):
            p = os.path.join(d, folio + '.jpg')
            if os.path.exists(p): return p, None
        return None, None
    fo, f = folio.split(':', 1)
    return os.path.join(SC18, 'lat', 'data', fo, f + '.jpg'), 2000


def runs(prof):
    ink = ndi.binary_closing(prof > 0, np.ones(2))
    lab, n = ndi.label(~ink)
    out = []
    for sl in ndi.find_objects(lab):
        a, b = sl[0].start, sl[0].stop
        if a == 0 or b == len(prof): continue
        out.append((a, b))
    return out


def word_hgt(mask, x0, x1, yfun, s):
    hb = int(0.55 * s)
    xs = np.arange(max(0, x0), min(mask.shape[1], x1))
    if len(xs) < 3: return np.nan
    ys = yfun(xs).astype(int)
    rr = np.clip(ys[None, :] + np.arange(-hb, hb + 1)[:, None], 0, mask.shape[0] - 1)
    m = mask[rr, xs[None, :]]
    ry = np.nonzero(m)[0]
    if len(ry) < 20: return np.nan
    return float(np.percentile(ry, 90) - np.percentile(ry, 10))


def page(args):
    which, pg = args
    fn, wide = img(which, pg['folio'])
    if fn is None or not os.path.exists(fn): return []
    if wide:
        im = Image.open(fn); sc = 2000 / im.width
        tmp = os.path.join(X.SCR, 'v74', 'tmp_' + pg['folio'].replace(':', '_').replace('/', '_') + '.png')
        im.convert('RGB').resize((2000, int(im.height * sc))).save(tmp)
        gray, con, rb = V.load(tmp); os.remove(tmp)
    else:
        gray, con, rb = V.load(fn)
    mask = ndi.binary_opening(con > 0.15, np.ones((2, 2)))
    s = pg['pitch']
    gl = V.glyphs if which == 'V' else lglyphs
    lines = collections.defaultdict(list)
    for w in pg['words']: lines[w['li']].append(w)
    out = []
    for li, ws in lines.items():
        ws = sorted(ws, key=lambda w: w['k'])
        if len(ws) < 3 or [w['k'] for w in ws] != list(range(len(ws))): continue
        cx = np.array([(w['x0'] + w['x1']) / 2 for w in ws], float); cy = np.array([w['y'] for w in ws], float)
        t = {'xs': cx, 'ys': cy, 'x0': ws[0]['x0'], 'x1': ws[-1]['x1']}
        yfun = lambda x: np.interp(x, cx, cy)
        xa = int(ws[0]['x0']); xb = int(ws[-1]['x1'])
        xx = np.arange(xa, xb); h = int(0.25 * s)
        prof = np.zeros(len(xx))
        for k in range(-h, h + 1):
            r = np.clip((yfun(xx) + k).astype(int), 0, mask.shape[0] - 1)
            prof += mask[r, xx]
        R = runs(prof)
        if not R: continue
        cen = np.array([(a + b) / 2 for a, b in R]); wid = np.array([b - a for a, b in R], float)

        def at(B):
            j = int(np.argmin(np.abs(cen + xa - B)))
            return (float(wid[j]) if abs(cen[j] + xa - B) <= 3 else np.nan), j

        # gapw=0 re-alignment of the same words on the same profile
        old = V.glyphs; V.glyphs = gl
        try:
            al0, _ = V.align_gaps(xa, prof, [w['word'] for w in ws], gapw=0.0)
        finally:
            V.glyphs = old
        hg = [word_hgt(mask, w['x0'], w['x1'], yfun, s) for w in ws]
        rec = []
        for k in range(len(ws) - 1):
            g, j = at(ws[k]['x1'])
            g0 = at(al0[k][1])[0] if al0 is not None else np.nan
            rec.append(dict(folio=pg['folio'], li=li, k=k, nw=len(ws), a=ws[k]['word'], b=ws[k + 1]['word'],
                            x=int(ws[k]['x1']), gap=g, gap0=g0, same0=bool(al0 is not None and abs(al0[k][1] - ws[k]['x1']) <= 3),
                            hgt=float(np.nanmean([hg[k], hg[k + 1]])), pitch=s, lcost=ws[k].get('lcost', 0.0)))
        gs = np.array([r['gap'] for r in rec]); g0s = np.array([r['gap0'] for r in rec])
        lm = np.nanmedian(gs) if np.isfinite(gs).sum() >= 2 else np.nan
        lm0 = np.nanmedian(g0s) if np.isfinite(g0s).sum() >= 2 else np.nan
        for r in rec:
            r['lmed'] = float(lm); r['r'] = float(r['gap'] / lm) if lm and np.isfinite(lm) else np.nan
            r['lmed0'] = float(lm0); r['r0'] = float(r['gap0'] / lm0) if lm0 and np.isfinite(lm0) else np.nan
        out += rec
    return out


def run(which, pages, name, nw=2):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(nw) as ex:
        res = [r for rows in ex.map(page, [(which, p) for p in pages]) for r in rows]
    X.jsave(name, res)
    return res


if __name__ == '__main__':
    os.makedirs(os.path.join(X.SCR, 'v74'), exist_ok=True)
    what = sys.argv[1:] or ['V', 'L', 'VX']
    if 'V' in what:
        r = run('V', json.load(open(os.path.join(DER, 'v18_words.json'))), 'gaps_V.json'); print('V', len(r), flush=True)
    if 'VX' in what:
        ex = [p for p in X.jload('words_extra.json') if p.get('words')]
        r = run('V', ex, 'gaps_VX.json'); print('VX', len(r), flush=True)
    if 'L' in what:
        r = run('L', json.load(open(os.path.join(DER, 'v18_latin_words.json'))), 'gaps_L.json'); print('L', len(r), flush=True)
