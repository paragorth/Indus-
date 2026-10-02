#!/usr/bin/env python3
"""Fetch Linear B document texts from DAMOS (Database of Mycenaean at Oslo,
https://damos.hf.uio.no, CC BY-NC-SA 4.0) through its public JSON endpoint
/ajaxitem/{id}/. Keeps heading + transliterated content only.
Output: data/damos_items.jsonl  (one {"id","heading","content"} per line)
Polite: 4 parallel requests, resumable.
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'damos_items.jsonl')

def get(i):
    url = f'https://damos.hf.uio.no/ajaxitem/{i}/'
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'script-research/0.1'})
            raw = urllib.request.urlopen(req, timeout=30).read()
            if raw.lstrip().startswith(b'<'): return i, None      # HTML page = no such item
            d = json.loads(raw)
            it = d.get('item') or {}
            if not it: return i, None
            return i, {'id': i, 'heading': it.get('heading'), 'content': it.get('content'),
                       'collectionid': it.get('collectionid')}
        except Exception as e:
            err = e; time.sleep(2)
    return i, {'id': i, 'error': str(err)}

def main():
    hi = int(sys.argv[1]) if len(sys.argv) > 1 else 7000
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            if 'error' not in r: done.add(r['id'])
    todo = [i for i in range(1, hi + 1) if i not in done]
    n = 0
    with open(OUT, 'a') as f, ThreadPoolExecutor(4) as ex:
        for i, r in ex.map(get, todo):
            if r: f.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
    print('fetched', n)

if __name__ == '__main__':
    main()
