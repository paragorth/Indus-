"""v18: fetch 2000-px-wide Beinecke MS 408 text pages (Yale IIIF) to a scratch cache.
Manifest https://collections.library.yale.edu/manifests/2002046 (canvas list in data/derived/v13_canvases.json).
Images stay in the scratch cache, never in the repo. Usage: python3 v18_fetch.py CACHE
"""
import json, os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
CAN = json.load(open(os.path.join(HERE, '..', 'data', 'derived', 'v13_canvases.json')))['canvases']
PAGES = ['58r', '58v'] + [f'{n}{s}' for n in list(range(103, 109)) + list(range(111, 117)) for s in 'rv']
PAGES = [p for p in PAGES if p != '116v']

def get(row, cache):
    fn = os.path.join(cache, 'f' + row['label'] + '.jpg')
    if os.path.exists(fn) and os.path.getsize(fn) > 10000:
        return fn
    subprocess.run(['curl', '-sS', '--retry', '3', '-o', fn, row['iiif'] + '/full/2000,/0/default.jpg'], check=True)
    return fn

if __name__ == '__main__':
    cache = sys.argv[1]; os.makedirs(cache, exist_ok=True)
    rows = [r for r in CAN if r['label'] in PAGES]
    with ThreadPoolExecutor(2) as ex:
        print(list(ex.map(lambda r: get(r, cache), rows)))
