"""v13 real-herbal control: Gerard, The Herball (1636 ed., Missouri Botanical Garden copy),
archive.org item mobot31753000817756 (public domain). Page images via
https://iiif.archive.org/iiif/mobot31753000817756$N/full/!1000,1000/0/default.jpg (scratch cache only),
OCR per page from mobot31753000817756_djvu.xml. Same blind image features as the Voynich.
Usage: python3 v13_gerard.py CACHE_DIR [first last]
Writes data/derived/v13_gerard.json {n: {'img': {...}, 'ocr': [words]}} (numbers + OCR word list).
"""
import json, os, sys, subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v13_lib import DER, load_norm, features

ITEM = 'mobot31753000817756'
OUT = os.path.join(DER, 'v13_gerard.json')


def one(args):
    n, cache = args
    fn = os.path.join(cache, 'g%04d.jpg' % n)
    if not (os.path.exists(fn) and os.path.getsize(fn) > 1000):
        url = 'https://iiif.archive.org/iiif/%s$%d/full/!1000,1000/0/default.jpg' % (ITEM, n)
        subprocess.run(['curl', '-sSL', '--retry', '3', '-o', fn, url], check=True)
    try:
        return n, features(load_norm(fn))
    except Exception as e:
        return n, {'error': str(e)}


if __name__ == '__main__':
    cache = sys.argv[1]
    a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (100, 500)
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if 'ocr_done' not in done:
        ocr = {}
        k = 0
        for ev, el in ET.iterparse(os.path.join(cache, 'djvu.xml'), events=('end',)):
            if el.tag == 'OBJECT':
                if a <= k < b:
                    ocr[str(k)] = [w.text or '' for w in el.iter('WORD')]
                k += 1; el.clear()
        done = {'ocr_done': True, 'pages': {n: {'ocr': ws} for n, ws in ocr.items()}}
    todo = [(n, cache) for n in range(a, b) if 'img' not in done['pages'].get(str(n), {})]
    with ProcessPoolExecutor(2) as ex:
        for i, (n, f) in enumerate(ex.map(one, todo)):
            done['pages'].setdefault(str(n), {})['img'] = f
            if i % 25 == 0:
                json.dump(done, open(OUT, 'w'))
    json.dump(done, open(OUT, 'w'))
    print('pages', len(done['pages']))
