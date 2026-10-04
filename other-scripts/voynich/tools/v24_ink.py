"""v24 cycle 3a: local ink anomalies (overwrite candidates) inside words, Yale IIIF images (2000 px, scratch only).
For each word box from v18 (data/derived/v18_words.json) the word's ink columns are split into as many
equal segments as the word has glyph units; each segment gets a darkness value (mean of its darkest 30%
stroke pixels).  Darkness is residualised on glyph type x page (glyphs differ in stroke overlap), and the
anomaly of a glyph = residual minus the word's median residual, scaled by the page MAD.
Planted control: a random glyph segment in random words is darkened (double-ink simulation: c' = 1-(1-c)^2
on its stroke pixels); recovery rate and localisation are measured on the same pipeline.
Writes data/v24_ckpt/ink_glyphs.json.  Usage: python3 v24_ink.py IMGDIR
"""
import os, sys, json, random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v18_lib as V
import v24_lib as L

T = 0.15


def seg_values(con, w, pitch, plant=None):
    h = int(0.3 * pitch)
    y = int(w['y']); x0, x1 = w['x0'], w['x1']
    reg = con[max(0, y - h):y + h + 1, x0:x1].copy()
    if reg.size == 0:
        return None
    g = V.glyphs(w['word'])
    n = len(g)
    m = reg > T
    cols = np.nonzero(m.any(0))[0]
    if n == 0 or len(cols) < n * 2:
        return None
    a, b = cols[0], cols[-1] + 1
    edges = np.linspace(a, b, n + 1).astype(int)
    if plant is not None:
        k = plant
        sub = reg[:, edges[k]:edges[k + 1]]
        mm = sub > T
        sub[mm] = 1 - (1 - sub[mm]) ** 2
        reg[:, edges[k]:edges[k + 1]] = sub
    out = []
    for k in range(n):
        s = reg[:, edges[k]:edges[k + 1]]
        v = s[s > T]
        if v.size < 8:
            out.append(None)
        else:
            q = np.sort(v)
            out.append(float(q[int(0.7 * len(q)):].mean()))
    return g, out


def page(args):
    fn, pg, seed = args
    gray, con, rb = V.load(fn)
    rng = random.Random(seed)
    recs = []
    for w in pg['words']:
        r = seg_values(con, w, pg['pitch'])
        if r is None:
            continue
        g, vals = r
        rec = dict(folio=w['folio'], n=w['n'], k=w['k'], word=w['word'], glyphs=g, vals=vals, plant=None)
        recs.append(rec)
        if rng.random() < 0.1 and len(g) >= 2:     # planted twin of this word
            kk = rng.randrange(len(g))
            r2 = seg_values(con, w, pg['pitch'], plant=kk)
            if r2 is not None:
                recs.append(dict(rec, vals=r2[1], plant=kk))
    return recs


if __name__ == '__main__':
    imgdir = sys.argv[1]
    pages = json.load(open(os.path.join(L.DATA, 'derived', 'v18_words.json')))
    jobs = [(os.path.join(imgdir, p['folio'] + '.jpg'), p, i) for i, p in enumerate(pages)]
    with ProcessPoolExecutor(2) as ex:
        res = [r for rr in ex.map(page, jobs) for r in rr]
    json.dump(res, open(os.path.join(L.CK, 'ink_glyphs.json'), 'w'))
    print('words', sum(r['plant'] is None for r in res), 'planted twins', sum(r['plant'] is not None for r in res))
