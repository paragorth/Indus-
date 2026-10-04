"""pe26: collect control-set catalogue rows (non-Proto-Elamite tablets in the same museums) from the CDLI search API."""
import json, subprocess, sys, urllib.parse
OUT = '/home/user/Indus-/other-scripts/proto-elamite/data/pe26_ckpt/controls_cat' + (sys.argv[1] if len(sys.argv) > 1 else '') + '.json'
Q = [('louvre_urukIII', {'collection': 'Louvre', 'period': 'Uruk III'}),
     ('louvre_urukIV', {'collection': 'Louvre', 'period': 'Uruk IV'}),
     ('louvre_urukV', {'collection': 'Louvre', 'period': 'Uruk V'}),
     ('tehran_all', {'collection': 'National Museum, Tehran'}),
     ('susa_urukIV', {'provenience': 'Susa', 'period': 'Uruk IV'}),
     ('susa_urukV', {'provenience': 'Susa', 'period': 'Uruk V'})]
MAXP = 30
if len(sys.argv) > 1 and sys.argv[1] == '2':
    MAXP = 4
    Q = [(f'louvre_{p}', {'collection': 'Louvre', 'period': p}) for p in ['Ur III', 'Old Akkadian', 'ED IIIb', 'Old Babylonian', 'Lagash II', 'ED IIIa']] + \
        [(f'louvre_susa_{p}', {'collection': 'Louvre', 'provenience': 'Susa', 'period': p}) for p in ['Old Babylonian', 'Old Akkadian', 'Ur III', 'Middle Elamite', 'Neo-Elamite', 'Lagash II']]
res = {}
for name, q in Q:
    for page in range(1, MAXP):
        qq = dict(q, limit=100, page=page)
        url = 'https://cdli.earth/search?' + urllib.parse.urlencode(qq)
        r = subprocess.run(['curl', '-sS', '-m', '120', '-H', 'Accept: application/json', url], capture_output=True, text=True)
        try: d = json.loads(r.stdout)
        except Exception: print(name, page, 'bad', r.stdout[:200]); break
        for a in d:
            pid = 'P%06d' % a['id']
            res.setdefault(pid, {'q': name, 'museum_no': a.get('museum_no'),
                'provenience': (a.get('provenience') or {}).get('provenience', ''),
                'collection': ';'.join(c['collection']['collection'] for c in a.get('collections', [])),
                'period': (a.get('period') or {}).get('period', ''),
                'type': (a.get('artifact_type') or {}).get('artifact_type', ''),
                'h': a.get('height'), 'w': a.get('width')})
        print(name, page, len(d), len(res), flush=True)
        if len(d) < 100: break
json.dump(res, open(OUT, 'w'), indent=0)
