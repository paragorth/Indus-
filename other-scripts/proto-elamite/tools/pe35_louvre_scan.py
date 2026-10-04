"""pe35: scan Louvre collection JSON records (one ark at a time, polite rate) and keep Suse III / proto-Elamite
tablets and sealings: object number, title, description, Amiet/MDP bibliography, image URLs.
Output (git-ignored checkpoint): data/pe35_ckpt/louvre_scan.jsonl. Usage: pe35_louvre_scan.py START END [STEP]"""
import json, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe35_ckpt')
os.makedirs(CK, exist_ok=True)
OUT = os.path.join(CK, 'louvre_scan.jsonl')
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        done.add(json.loads(l)['ark'])
a, b = int(sys.argv[1]), int(sys.argv[2])
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
f = open(OUT, 'a')
for n in range(a, b, step):
    ark = 'cl%09d' % n
    if ark in done:
        continue
    rec = {'ark': ark}
    try:
        with urllib.request.urlopen('https://collections.louvre.fr/ark:/53355/%s.json' % ark, timeout=30) as r:
            d = json.loads(r.read())
        on = d.get('objectNumber') or [{}]
        rec['num'] = on[0].get('value')
        txt = json.dumps(d, ensure_ascii=False).lower()
        rec['keep'] = ('suse iii' in txt or 'proto-' in txt or 'proto\\u00e9' in txt)
        if rec['keep']:
            rec.update(title=d.get('title'), desc=d.get('description'), date=d.get('displayDateCreated'),
                       bib=[(x.get('bibliographyRef', '')[:80], x.get('bibliographyDetail')) for x in d.get('bibliography', [])],
                       img=[x.get('urlImage') for x in d.get('image', [])])
    except Exception as e:
        rec['err'] = str(e)[:80]
        if '429' in str(e) or '403' in str(e):
            f.write(json.dumps(rec) + '\n'); f.flush(); time.sleep(60)
            continue
    f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
    time.sleep(0.4)
