"""v18 cycle 1a: per-word ink measures for the Voynich text pages (Q20 + f58).
Writes data/derived/v18_words.json (numbers only; images stay in the scratch cache).
Usage: python3 v18_extract.py IMGDIR [OVERLAYDIR]
"""
import sys, os, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v18_lib import *
DATA = os.path.join(HERE, '..', 'data', 'derived')
T = 0.15


def page(args):
    fn, folio, lines, ovdir = args
    gray, con, rb = load(fn)
    mask = ndi.binary_opening(con > T, np.ones((2, 2)))
    mlo = con > 0.08  # measuring mask keeps faint strokes
    x0, x1 = text_columns(mask)
    s = line_pitch(mask, x0, x1)
    tr = [t for t in strip_tracks(mask, x0, x1, s) if t['n'] >= 2]
    for t in tr:
        a, b, p = track_profile(mask, t, x0, x1, s); t['x0'], t['x1'] = a, b
    tr = suppress_ascenders([t for t in tr if t['x1'] - t['x0'] > 60], s)
    for t in tr:
        t['a'] = 0; t['b'] = t['ym']
    gl = [sum(len(glyphs(w)) for w in r['words']) + 0.6 * (len(r['words']) - 1) for r in lines]
    big = [t for t in tr if t['x1'] - t['x0'] > 0.6 * (x1 - x0)]
    unit = np.median([t['x1'] - t['x0'] for t in big]) / np.percentile(gl, 75)
    ml, cost = match_lines2(tr, gl, unit, s, lambda c: min(8, c['n'] * 0.8))
    out = []
    profs = {}
    for li, (r, t) in enumerate(zip(lines, ml)):
        if t is not None:
            xa, xb, prof = track_profile(mask, t, x0, x1, s)
            if len(prof) >= 20:
                profs[li] = (xa, prof)
    # left text margin: most lines start there; stars and their stalks sit further left
    margin = np.percentile([v[0] for v in profs.values()], 60)
    for li, (r, t) in enumerate(zip(lines, ml)):
        if li not in profs:
            continue
        xa, prof = profs[li]
        cut = int(margin - 25 - xa)
        if cut > 0:
            prof = prof[cut:]; xa += cut
            nz = np.nonzero(prof)[0]
            if len(nz) < 10:
                continue
            prof = prof[nz[0]:nz[-1] + 1]; xa += nz[0]
        al, cc = align_gaps(xa, prof, r['words'])
        if al is None:
            continue
        h = int(0.3 * s)
        for k, (a, b) in enumerate(al):
            g = len(glyphs(r['words'][k]))
            xx = np.arange(a, b); yy = track_y(t, xx).astype(int)
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
            out.append({'folio': folio, 'n': r['n'], 'li': li, 'k': k, 'nw': len(r['words']),
                        'word': r['words'][k], 'x0': int(a), 'x1': int(b), 'y': float(track_y(t, (a + b) / 2)),
                        'wexp': float(unit * g), 'lcost': float(cc), 'm': m,
                        'para_start': r['para_start'], 'para_end': r['para_end']})
    return {'folio': folio, 'pitch': s, 'nlines': len(lines), 'matched': sum(m is not None for m in ml),
            'cost': cost, 'margin': float(margin), 'words': out}


if __name__ == '__main__':
    imgdir = sys.argv[1]
    L = json.load(open(os.path.join(DATA, 'ZL3b_lines.json')))
    jobs = []
    for f in sorted(os.listdir(imgdir)):
        folio = f[:-4]
        lines = [r for r in L if r['folio'] == folio and r['ltype'] == 'P']
        if lines:
            jobs.append((os.path.join(imgdir, f), folio, lines, None))
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(page, jobs))
    for r in res:
        print(r['folio'], r['pitch'], r['nlines'], r['matched'], round(r['cost'], 1), len(r['words']))
    json.dump(res, open(os.path.join(DATA, 'v18_words.json'), 'w'))
