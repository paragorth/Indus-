"""v18 control: per-word ink measures for known-language Latin manuscripts (HTR-United
CREMMA-Medieval-Lat, ALTO line baselines + line transcriptions; images stay in scratch).
Same word-alignment and ink measures as v18_extract.py; line tracks come from the ALTO
baselines (scaled to 2000 px width). Glyph count = letters + abbreviation marks.
Usage: python3 v18_latin_extract.py DATA_DIR FOLDER [FOLDER ...]
"""
import sys, os, re, json, unicodedata, numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from scipy import ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v18_lib as V
OUT = os.path.join(HERE, '..', 'data', 'derived', 'v18_latin_words.json')
NS = '{http://www.loc.gov/standards/alto/ns-v4#}'
T = 0.15


def lglyphs(w):
    return [c for c in w if not c.isspace()]


def parse(xmlf):
    root = ET.parse(xmlf).getroot()
    tags = {t.get('ID'): t.get('LABEL') for t in root.iter(NS + 'OtherTag')}
    lines = []
    for tb in root.iter(NS + 'TextBlock'):
        lab = tags.get(tb.get('TAGREFS'), '')
        if 'Main' not in lab:
            continue
        for tl in tb.iter(NS + 'TextLine'):
            bl = [int(v) for v in tl.get('BASELINE', '').split()]
            s = tl.find(NS + 'String')
            if s is None or len(bl) < 4:
                continue
            raw = s.get('CONTENT', '')
            txt = ''.join(' ' if unicodedata.category(c)[0] in 'PZ' else (c if unicodedata.category(c)[0] in 'LN' else '') for c in raw)
            words = [w.lower() for w in txt.split() if w]
            if words:
                lines.append({'bl': list(zip(bl[::2], bl[1::2])), 'words': words})
    return lines


def page(args):
    jpg, xmlf, folio = args
    lines = parse(xmlf)
    if len(lines) < 5:
        return None
    im = Image.open(jpg); sc = 2000 / im.width
    tmp = jpg + '.2000.png'
    im.convert('RGB').resize((2000, int(im.height * sc))).save(tmp)
    gray, con, rb = V.load(tmp); os.remove(tmp)
    mask = ndi.binary_opening(con > T, np.ones((2, 2)))
    mlo = con > 0.08
    # line pitch: distance to the next line below that overlaps in x (multi-column safe)
    ext = [(min(x for x, _ in l['bl']) * sc, max(x for x, _ in l['bl']) * sc, np.mean([y for _, y in l['bl']]) * sc) for l in lines]
    ds = []
    for a0, a1, ay in ext:
        below = [by - ay for b0, b1, by in ext if by > ay + 3 and min(a1, b1) - max(a0, b0) > 0.3 * (a1 - a0)]
        if below:
            ds.append(min(below))
    s = float(np.median(ds))
    tracks = []
    for l in lines:
        xs = np.array([x for x, _ in l['bl']], float) * sc; yb = np.array([y for _, y in l['bl']], float) * sc
        o = np.argsort(xs)
        tracks.append({'xs': xs[o], 'ys': yb[o] - 0.17 * s, 'x0': xs.min(), 'x1': xs.max()})
    x0, x1 = int(min(t['x0'] for t in tracks)) - 20, int(max(t['x1'] for t in tracks)) + 20
    profs = {}
    for li, t in enumerate(tracks):
        xa, xb, prof = V.track_profile(mask, t, max(0, x0), min(mask.shape[1], x1), s)
        if len(prof) >= 20:
            profs[li] = (xa, prof)
    ups = [len(p) / (sum(len(lglyphs(w)) for w in lines[li]['words']) + 0.6 * (len(lines[li]['words']) - 1))
           for li, (xa, p) in profs.items() if len(lines[li]['words']) >= 4]
    if not ups:
        return None
    unit_page = float(np.percentile(ups, 35))
    out = []
    old = V.glyphs
    V.glyphs = lglyphs  # word widths counted in letters
    try:
        for li, (xa, prof) in profs.items():
            ws = lines[li]['words']; t = tracks[li]
            al, cc = V.align_gaps_lead(xa, prof, ws, unit_page, maxlead=60)
            if al is None:
                continue
            h = int(0.3 * s)
            for k, (a, b) in enumerate(al):
                g = len(lglyphs(ws[k]))
                xx = np.arange(a, b); yy = V.track_y(t, xx).astype(int)
                rows = np.clip(yy[None, :] + np.arange(-h, h + 1)[:, None], 0, mask.shape[0] - 1)
                cm = mlo[rows, xx[None, :]]; cv = con[rows, xx[None, :]][cm]; rv = rb[rows, xx[None, :]][cm]
                gv = gray[rows, xx[None, :]][cm]
                if cv.size < 20:
                    m = None
                else:
                    q = np.sort(cv)
                    m = {'med': float(np.median(cv)), 'top': float(q[int(0.7 * len(q)):].mean()),
                         'p90': float(q[int(0.9 * len(q))]), 'areag': float(cv.size / max(1, g)),
                         'rb': float(rv.mean()), 'gray': float(np.median(gv))}
                out.append({'folio': folio, 'n': li + 1, 'li': li, 'k': k, 'nw': len(ws), 'word': ws[k],
                            'x0': int(a), 'x1': int(b), 'y': float(V.track_y(t, (a + b) / 2)),
                            'wexp': float(unit_page * g), 'lcost': float(cc), 'm': m,
                            'para_start': False, 'para_end': False})
    finally:
        V.glyphs = old
    return {'folio': folio, 'pitch': s, 'nlines': len(lines), 'matched': len(profs), 'cost': 0.0, 'words': out}


if __name__ == '__main__':
    base = sys.argv[1]
    jobs = []
    for fo in sys.argv[2:]:
        d = os.path.join(base, fo)
        for f in sorted(os.listdir(d)):
            if f.endswith('.jpg'):
                x = os.path.join(d, f[:-4] + '.xml')
                if os.path.exists(x):
                    jobs.append((os.path.join(d, f), x, fo + ':' + f[:-4]))
    with ProcessPoolExecutor(2) as ex:
        res = [r for r in ex.map(page, jobs) if r]
    for r in res:
        print(r['folio'], round(r['pitch'], 1), r['nlines'], r['matched'], len(r['words']))
    json.dump(res, open(OUT, 'w'))
