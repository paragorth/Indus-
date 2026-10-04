"""v38: fetch low-resolution page images (scratch only, never the repo).
Voynich herbal pages from the Yale IIIF manifest 2002046 (canvas list cached in
data/derived/v13_canvases.json); Gerard, The Herball (1636), archive.org item
mobot31753000817756 page images + djvu.xml OCR.
Usage: python3 v38_fetch.py CACHE voynich|gerard [first last step]
"""
import json, os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v8_lib import voynich_pages, DATA

SIZE = '!600,600'
ITEM = 'mobot31753000817756'


def herbal_canvases():
    can = {c['label']: c for c in json.load(open(os.path.join(DATA, 'derived', 'v13_canvases.json')))['canvases']
           if c['single']}
    out = {}
    for p in voynich_pages(min_tokens=1):
        if p['illus'] != 'H':
            continue
        lab = p['id'][1:]
        if lab in can:
            out[p['id']] = can[lab]['iiif']
    return out


def curl(url, fn):
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        return fn
    subprocess.run(['curl', '-sSL', '--retry', '4', '-m', '120', '-o', fn, url], check=True)
    return fn


if __name__ == '__main__':
    cache, which = sys.argv[1], sys.argv[2]
    os.makedirs(cache, exist_ok=True)
    if which == 'voynich':
        hc = herbal_canvases()
        jobs = [(u + '/full/%s/0/default.jpg' % SIZE, os.path.join(cache, 'v_%s.jpg' % f)) for f, u in hc.items()]
    else:
        a, b, s = map(int, sys.argv[3:6])
        jobs = [('https://iiif.archive.org/iiif/%s$%d/full/%s/0/default.jpg' % (ITEM, n, SIZE),
                 os.path.join(cache, 'g_%04d.jpg' % n)) for n in range(a, b, s)]
        x = os.path.join(cache, 'gerard_djvu.xml')
        if not os.path.exists(x):
            curl('https://archive.org/download/%s/%s_djvu.xml' % (ITEM, ITEM), x)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: curl(*j), jobs))
    print('fetched', len(jobs))
