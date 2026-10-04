"""v46: fetch non-herbal Voynich page images (Yale IIIF manifest 2002046) at low
resolution into the scratchpad (never the repo). Canvas list cached in
data/derived/v13_canvases.json.  Usage: python3 v46_fetch.py CACHE [SIZE]"""
import json, os, re, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')

def leafnum(lab):
    m = re.match(r'(\d+)', lab)
    return int(m.group(1)) if m else -1

def jobs(cache, size):
    out = []
    for i, c in enumerate(json.load(open(os.path.join(DATA, 'derived', 'v13_canvases.json')))['canvases']):
        n = leafnum(c['label'])
        if n < 57 or n > 116:   # herbal 1-56 cached by v38; 57-116 = cosmo/astro/zodiac/bio/pharma/stars (+late herbal)
            continue
        name = re.sub(r'[^0-9a-z]+', '_', c['label'].lower()).strip('_') + '_%03d' % i
        out.append((c['iiif'] + '/full/%s/0/default.jpg' % size, os.path.join(cache, 'n_%s.jpg' % name)))
    return out

def curl(url, fn):
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        return
    subprocess.run(['curl', '-sSL', '--retry', '4', '-m', '120', '-o', fn, url], check=True)

if __name__ == '__main__':
    cache = sys.argv[1]; size = sys.argv[2] if len(sys.argv) > 2 else '!1000,1000'
    os.makedirs(cache, exist_ok=True)
    J = jobs(cache, size)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: curl(*j), J))
    print('fetched', len(J))
