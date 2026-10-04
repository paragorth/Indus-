"""v34: same line-end re-measurement for the CREMMA Latin pages (positive control), on
images rescaled to 2000 px as in v18. Writes data/v34_ckpt/lineends_lat.json.
Usage: python3 v34_lineends_lat.py [overlay_folio]
"""
import sys, os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v18_lib as L18
import v34_lib as V
from v34_lineends import measure, overlay


def jpg_of(folio):
    ms, name = folio.split(':', 1)
    return os.path.join(V.LATDIR, ms, name + '.jpg')


def page(args):
    folio, recs, pitch, ov = args
    fn = jpg_of(folio)
    im = Image.open(fn); sc = 2000 / im.width
    tmp = os.path.join(V.SCR, 'v34_tmp_%d.png' % os.getpid())
    im.convert('RGB').resize((2000, int(im.height * sc))).save(tmp)
    gray, con, rb = L18.load(tmp); os.remove(tmp)
    mask = ndi.binary_opening(con > 0.15, np.ones((2, 2)))
    out = measure(mask, recs, pitch, gray)
    if ov:
        overlay(fn, out, os.path.join(V.SCR, 'v34_ovL_' + folio.replace(':', '_') + '.png'))
    return folio, out


if __name__ == '__main__':
    ov = sys.argv[1] if len(sys.argv) > 1 else None
    pages = json.load(open(os.path.join(V.DATA, 'derived', 'v18_latin_words.json')))
    jobs = [(p['folio'], p['words'], p['pitch'], p['folio'] == ov) for p in pages]
    with ProcessPoolExecutor(2) as ex:
        res = dict(ex.map(page, jobs))
    json.dump(res, open(os.path.join(V.CKPT, 'lineends_lat.json'), 'w'))
    for f, o in res.items():
        d = [r['xend'] - r['xend_v18'] for r in o.values()]
        print(f, len(o), 'mean ext px', round(float(np.mean(d)), 1), 'max', max(d))
