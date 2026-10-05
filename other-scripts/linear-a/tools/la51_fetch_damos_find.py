#!/usr/bin/env python3
"""la51: fetch find-place metadata (find_area, find_area_name, object, series) for every DAMOS
item already in data/damos_items.jsonl. Output data/la51_ckpt/damos_find.jsonl. 3 threads, resumable."""
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
OUT = os.path.join(D, 'la51_ckpt', 'damos_find.jsonl')
KEYS = ['heading', 'find_area', 'find_area_name', 'area_code', 'find_place', 'find_place2', 'object',
        'series', 'subseries', 'provenience', 'context', 'chronology1', 'hand_easy']
def get(i):
    for a in range(3):
        try:
            req = urllib.request.Request(f'https://damos.hf.uio.no/ajaxitem/{i}/', headers={'User-Agent': 'script-research/0.1'})
            it = json.loads(urllib.request.urlopen(req, timeout=30).read()).get('item') or {}
            return {'id': i, **{k: it.get(k) for k in KEYS}}
        except Exception as e:
            err = str(e); time.sleep(2)
    return {'id': i, 'error': err}
ids = [json.loads(l)['id'] for l in open(os.path.join(D, 'damos_items.jsonl')) if 'error' not in l]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)['id'] for l in open(OUT) if '"error"' not in l}
todo = [i for i in ids if i not in done]
with open(OUT, 'a') as f, ThreadPoolExecutor(3) as ex:
    for r in ex.map(get, todo):
        f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
print('done', len(todo))
