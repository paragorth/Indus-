"""pe57: harvest Zenodo records (3dbigdataspace / Objaverse 3D models) for a query; keep records whose
title matches a pattern. Output JSON list under data/pe57_ckpt/."""
import json, sys, time, re, urllib.request, urllib.parse, os
q, pat, out = sys.argv[1], sys.argv[2], sys.argv[3]
rx = re.compile(pat, re.I)
res = {}
for page in range(1, 60):
    url = "https://zenodo.org/api/records?" + urllib.parse.urlencode({'q': q, 'size': 25, 'page': page})
    for tr in range(4):
        try:
            d = json.load(urllib.request.urlopen(url, timeout=60)); break
        except Exception as e:
            print('retry', page, e, flush=True); time.sleep(10)
    else:
        break
    hs = d['hits']['hits']
    for h in hs:
        m = h['metadata']
        if rx.search(m['title'] + ' ' + m.get('description', '')):
            res[h['id']] = {'title': m['title'], 'desc': m.get('description', ''),
                            'files': [(f['key'], f['size'], f['links']['self']) for f in h.get('files', [])]}
    print(page, len(hs), len(res), flush=True)
    if len(hs) < 25:
        break
    time.sleep(1.2)
json.dump(res, open(out, 'w'), indent=1)
