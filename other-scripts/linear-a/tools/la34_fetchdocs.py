#!/usr/bin/env python3
"""la34: faster, uniform route to SigLA occurrence drawings: fetch each SigLA document drawing once
(throttled, one request at a time) into the scratchpad and cut every occurrence out of it with SigLA's own
box (the published occurrence PNG is exactly this crop: 99.4 % identical pixels on HT 13 #1).
All occurrence images used by la34 come from this one route, so sources are not mixed."""
import os, json, time, urllib.parse, urllib.request, collections
from PIL import Image
from la34_common import CK, fname
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
DOCS = os.path.join(SCR, 'la34docs'); OUT = os.path.join(SCR, 'la34crop')


def main():
    os.makedirs(DOCS, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    occ = json.load(open(os.path.join(CK, 'occ_index.json')))
    by = collections.defaultdict(list)
    for o in occ: by[o['src'].rsplit('_', 1)[0] + '.png'].append(o)
    print('docs', len(by), flush=True)
    for i, (src, os_) in enumerate(sorted(by.items())):
        p = os.path.join(DOCS, fname(src))
        if not os.path.exists(p):
            url = 'https://sigla.phis.me/' + urllib.parse.quote(src)
            try:
                open(p, 'wb').write(urllib.request.urlopen(url, timeout=60).read())
            except Exception as e:
                print('err', url, e, flush=True); time.sleep(3); continue
            time.sleep(0.35)
        try:
            im = Image.open(p)
            for o in os_:
                q = os.path.join(OUT, fname(o['src']))
                if os.path.exists(q): continue
                x, y, w, h = o['rect']
                im.crop((int(x), int(y), int(x + w), int(y + h))).save(q)
        except Exception as e:
            print('crop err', src, e, flush=True)
        if i % 50 == 0: print(i, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
