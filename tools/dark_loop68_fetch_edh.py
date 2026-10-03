"""S-DARK-68: fetch EDH (CC BY-SA 4.0) inscriptions per Roman province through the open API, keep only the fields needed
for name-pool divergence (id, province, findspots, country, type, dates, transcription). One jsonl per province.
Usage: python3 tools/dark_loop68_fetch_edh.py"""
import json, os, sys, time, urllib.request, urllib.parse

OUT = '/home/user/Indus-/data/derived/dark/loop68_corpora/edh/'
os.makedirs(OUT, exist_ok=True)
PROV = ['Rom', 'LaC', 'Etr', 'VeH', 'ApC', 'Sam', 'Umb', 'Pic', 'Aem', 'Lig', 'Tra', 'BrL', 'Sic', 'Sar',
        'Dal', 'PaS', 'PaI', 'GeS', 'GeI', 'Bae', 'HiC', 'Lus', 'Nar', 'Bri', 'Nor', 'Rae', 'Bel', 'Lug', 'Aqu',
        'Afr', 'Num', 'Dac', 'MoS', 'MoI', 'Mak', 'Ach', 'Asi', 'Thr']
KEEP = ['id', 'province_label', 'findspot_ancient', 'findspot_modern', 'modern_region', 'country', 'type_of_inscription',
        'not_before', 'not_after', 'language', 'transcription']

def get(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            print('retry', t, url[-60:], e, file=sys.stderr); time.sleep(5 * (t + 1))
    return None

for p in PROV:
    fn = OUT + f'{p}.jsonl'
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        continue
    rows = []; off = 0
    while True:
        d = get(f'https://edh.ub.uni-heidelberg.de/data/api/inschrift/suche?provinz={p}&limit=1000&offset={off}')
        if d is None: break
        items = d.get('items', [])
        for it in items:
            rows.append({k: it.get(k) for k in KEEP} | {'prov': p})
        off += 1000
        if off >= int(d.get('total', 0)) or not items: break
    with open(fn, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(p, len(rows), os.path.getsize(fn), flush=True)
print('done')
