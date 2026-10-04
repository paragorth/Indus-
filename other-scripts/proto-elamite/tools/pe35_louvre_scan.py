"""pe35: scan Louvre collection JSON records (one ark at a time, polite rate) and keep Suse III / proto-Elamite
objects: object number, title, description, bibliography (Amiet MDP 43, Scheil MDP 6/17, Legrain MDP 16...), image URLs.
Output (git-ignored checkpoint): data/pe35_ckpt/louvre_scan.jsonl. Usage: pe35_louvre_scan.py START END [STEP]"""
import json, os, sys, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe35_ckpt')
os.makedirs(CK, exist_ok=True)
OUT = os.path.join(CK, 'louvre_scan.jsonl')


def fetch(ark):
    r = subprocess.run(['curl', '-sS', '-m', '30', '-w', '\n%{http_code}',
                        'https://collections.louvre.fr/ark:/53355/%s.json' % ark], capture_output=True, text=True)
    body, _, code = r.stdout.rpartition('\n')
    return code, body


if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            d = json.loads(l)
            if 'urllib' not in d.get('err', ''):
                done.add(d['ark'])
    a, b = int(sys.argv[1]), int(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    f = open(OUT, 'a')
    for n in range(a, b, step):
        ark = 'cl%09d' % n
        if ark in done:
            continue
        rec = {'ark': ark}
        for attempt in range(3):
            code, body = fetch(ark)
            if code in ('403', '429', '000', ''):
                time.sleep(10 * (attempt + 1))
                continue
            break
        if code != '200':
            rec['err'] = 'HTTP ' + code
            if code in ('403', '429', '000', ''):
                continue          # not recorded: retried on a later run
        else:
            d = json.loads(body)
            on = d.get('objectNumber') or [{}]
            rec['num'] = on[0].get('value')
            txt = body.lower()
            rec['keep'] = ('suse iii' in txt or 'proto-' in txt or 'proto-\\u00e9' in txt or 'amiet' in txt)
            if rec['keep']:
                rec.update(title=d.get('title'), desc=d.get('description'), date=d.get('displayDateCreated'),
                           bib=[(x.get('bibliographyRef', '')[:80], x.get('bibliographyDetail')) for x in d.get('bibliography', [])],
                           img=[x.get('urlImage') for x in d.get('image', [])])
        f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
        time.sleep(0.5)
