"""v13: blind image features for every single-page canvas (2 workers, checkpointed).
Usage: python3 v13_imgfeat.py CACHE_DIR
Writes data/derived/v13_imgfeat.json  {label: {feature: value}}  (numbers only, no images).
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v13_lib import DER, load_norm, features

OUT = os.path.join(DER, 'v13_imgfeat.json')


def one(args):
    lab, fn = args
    try:
        return lab, features(load_norm(fn))
    except Exception as e:
        return lab, {'error': str(e)}


if __name__ == '__main__':
    cache = sys.argv[1]
    can = json.load(open(os.path.join(DER, 'v13_canvases.json')))['canvases']
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [(c['label'], os.path.join(cache, c['iiif'].rsplit('/', 1)[1] + '.jpg'))
            for c in can if c['single'] and c['label'] not in done]
    with ProcessPoolExecutor(2) as ex:
        for i, (lab, f) in enumerate(ex.map(one, todo)):
            done[lab] = f
            if i % 20 == 0:
                json.dump(done, open(OUT, 'w'))
    json.dump(done, open(OUT, 'w'), indent=0)
    print('pages', len(done))
