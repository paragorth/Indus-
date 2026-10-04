"""v13: fetch reduced-size (<=1000 px) Beinecke MS 408 page images via Yale IIIF.

Manifest: https://collections.library.yale.edu/manifests/2002046
Images go to a scratch cache (NOT the repo); only derived numbers are stored.
Usage: python3 v13_fetch.py CACHE_DIR   (2 parallel workers, resumable)
"""
import json, os, re, sys, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor

MANIFEST = 'https://collections.library.yale.edu/manifests/2002046'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'data', 'derived', 'v13_canvases.json')


def canvases():
    m = json.load(urllib.request.urlopen(MANIFEST))
    rows = []
    for c in m['items']:
        lab = c['label']['none'][0]
        body = c['items'][0]['items'][0]['body']
        iid = body['service'][0]['@id']
        rows.append({'label': lab, 'iiif': iid, 'w': body['width'], 'h': body['height'],
                     'single': bool(re.fullmatch(r'\d+[rv]', lab))})
    return rows


def get(row, cache):
    fn = os.path.join(cache, row['iiif'].rsplit('/', 1)[1] + '.jpg')
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        return fn
    url = row['iiif'] + '/full/!1000,1000/0/default.jpg'
    subprocess.run(['curl', '-sS', '--retry', '3', '-o', fn, url], check=True)
    return fn


if __name__ == '__main__':
    cache = sys.argv[1]
    os.makedirs(cache, exist_ok=True)
    rows = canvases()
    json.dump({'manifest': MANIFEST, 'size': '!1000,1000', 'canvases': rows}, open(OUT, 'w'), indent=0)
    todo = [r for r in rows if r['single']]
    with ThreadPoolExecutor(2) as ex:
        for fn in ex.map(lambda r: get(r, cache), todo):
            pass
    print('fetched', len(todo))
