"""v74 cycle 1a: extend the v18 line/word segmentation to more text pages that hold the frozen P1 junctions.
Fetches 2000-px Beinecke MS 408 images (Yale IIIF manifest 2002046, canvas list v13_canvases.json) into the
scratch cache, runs the unchanged v18 page pipeline (v18_extract.page: line tracks, gap-based DP word
alignment to the ZL3b words) and saves the word boxes to data/v74_ckpt/words_extra.json (numbers only).
Two workers."""
import os, sys, json, subprocess
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v74_lib as X
import v18_extract as E
IMG = os.path.join(X.SCR, 'v74', 'img'); os.makedirs(IMG, exist_ok=True)
CAN = json.load(open(os.path.join(X.ROOT, 'data', 'derived', 'v13_canvases.json')))['canvases']
PAGES = [f'{n}{s}' for n in range(75, 85) for s in 'rv'] + ['26r', '31r', '33r', '39r', '39v', '40r', '46r', '55r', '55v',
         '99r', '99v', '57r', '26v', '31v', '33v', '34r', '34v', '40v', '15v', '1v', '3r', '7v', '10r', '13v', '16r']


def get(r):
    fn = os.path.join(IMG, 'f' + r['label'] + '.jpg')
    if not (os.path.exists(fn) and os.path.getsize(fn) > 10000):
        subprocess.run(['curl', '-sS', '--retry', '3', '-o', fn, r['iiif'] + '/full/2000,/0/default.jpg'], check=True)
    return fn


def safe(job):
    try:
        return E.page(job)
    except Exception as e:
        return {'folio': job[1], 'error': repr(e), 'words': []}


if __name__ == '__main__':
    rows = [r for r in CAN if r['label'] in PAGES]
    L = json.load(open(os.path.join(X.ROOT, 'data', 'derived', 'ZL3b_lines.json')))
    jobs = []
    for r in rows:
        fn = get(r); folio = 'f' + r['label']
        lines = [x for x in L if x['folio'] == folio and x['ltype'] == 'P']
        if lines: jobs.append((fn, folio, lines, None))
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(safe, jobs))
    for r in res:
        print(r['folio'], r.get('error', ''), r.get('pitch'), r.get('nlines'), r.get('matched'), len(r['words']), flush=True)
    X.jsave('words_extra.json', res)
