"""v47: fetch page images into the scratch cache (never the repo).
voynich: herbal pages (!600,600) + pharmaceutical canvases (!1600,1600, small drawings).
gerard:  the v38 Gerard scans (keys of data/derived/v38_vis_gerard.json) + djvu.xml.
dodoens: Dodoens, Stirpium historiae pemptades sex (Antwerp 1583, Latin; archive.org
         mobot31753000817947), every 3rd scan 40-890 + djvu.xml.
Usage: python3 v47_fetch.py CACHE voynich|gerard|dodoens
"""
import json, os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_fetch import herbal_canvases, curl
from v8_lib import DATA

PH = list(range(160, 164)) + list(range(174, 182))   # canvas indices of the pharmaceutical section


def ia(item, a, b, s, cache, pre, size='!600,600'):
    jobs = [('https://iiif.archive.org/iiif/%s$%d/full/%s/0/default.jpg' % (item, n, size),
             os.path.join(cache, '%s_%04d.jpg' % (pre, n))) for n in range(a, b, s)]
    x = os.path.join(cache, '%s_djvu.xml' % pre)
    curl('https://archive.org/download/%s/%s_djvu.xml' % (item, item), x)
    return jobs


if __name__ == '__main__':
    cache, which = sys.argv[1], sys.argv[2]
    os.makedirs(cache, exist_ok=True)
    if which == 'voynich':
        jobs = [(u + '/full/!600,600/0/default.jpg', os.path.join(cache, 'v_%s.jpg' % f)) for f, u in herbal_canvases().items()]
        can = json.load(open(os.path.join(DATA, 'derived', 'v13_canvases.json')))['canvases']
        for i, ci in enumerate(PH):
            c = can[ci]
            sz = '2400,' if c['w'] > 1.5 * c['h'] else '!1600,1600'
            jobs.append((c['iiif'] + '/full/%s/0/default.jpg' % sz, os.path.join(cache, 'p_%02d.jpg' % i)))
    elif which == 'gerard':
        keys = json.load(open(os.path.join(DATA, 'derived', 'v38_vis_gerard.json')))
        jobs = [('https://iiif.archive.org/iiif/mobot31753000817756$%d/full/!600,600/0/default.jpg' % int(k),
                 os.path.join(cache, 'g_%04d.jpg' % int(k))) for k in keys]
        curl('https://archive.org/download/mobot31753000817756/mobot31753000817756_djvu.xml', os.path.join(cache, 'gerard_djvu.xml'))
    else:
        jobs = ia('mobot31753000817947', 40, 890, 3, cache, 'd')
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: curl(*j), jobs))
    print('fetched', len(jobs))
