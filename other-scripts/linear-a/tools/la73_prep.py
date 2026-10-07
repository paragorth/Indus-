"""la73 data prep: Linear A site incidence (rd / read) and Ur III dated name incidence (calibration)."""
import json, csv, re, os, collections as C
HERE = os.path.dirname(os.path.abspath(__file__)); LA = os.path.join(HERE, '..')
CK = os.path.join(LA, 'data', 'la73_ckpt')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
NONTAB = {'Nodule', 'Roundel', 'Sealing', 'Clay vessel', 'Stone vessel', 'Metal object', 'Stone object',
          'Architecture', 'Graffito', 'ivory object', 'Inked inscription'}

def la_docs(site, ver='rd'):
    d = json.load(open(os.path.join(LA, 'data', 'corpus_ra.json')))
    ok = ('read', 'damaged') if ver == 'rd' else ('read',)
    out = []
    for x in d:
        if x['site'] != site or x['support'] in NONTAB:
            continue
        w = ['-'.join(t['s']) for t in x['tokens'] if t['t'] == 'word' and t.get('st') in ok
             and len(t['s']) >= 2 and 'bridge' not in t.get('fl', []) and 'erased' not in t.get('fl', [])]
        if w:
            out.append(dict(id=x['id'], findspot=x.get('findspot', ''), scribe=x.get('scribe', ''), w=w))
    return out

def ur3():
    fn = os.path.join(CK, 'ur3_dated.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    names = C.defaultdict(list); site = {}
    for line in open('/home/user/Indus-/data/derived/dark/loop76_corpora/ur3_admin_names.jsonl'):
        r = json.loads(line); names[r['pid']].append(r['name']); site[r['pid']] = r['site']
    csv.field_size_limit(10 ** 9)
    off = {'Ur-Namma': 0, 'Šulgi': 18, 'Amar-Suen': 66, 'Šū-Suen': 75, 'Shu-Suen': 75, 'Ibbi-Suen': 84}
    out = []
    for row in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8')):
        pid = row['id_text'].lstrip('0') if row['id_text'] else ''
        key = pid if pid in names else ('%06d' % int(pid) if pid.isdigit() and ('%06d' % int(pid)) in names else None)
        if not key:
            continue
        m = re.match(r'^\s*([^\W\d][\w\-]*)\.(\d\d)\.(\d\d)', row['dates_referenced'] or '')
        if not m or m.group(1) not in off or m.group(2) == '00' or not m.group(3).isdigit():
            continue
        mo = int(m.group(3))
        if not 1 <= mo <= 12:
            continue
        t = (off[m.group(1)] + int(m.group(2)) - 1) * 12 + (mo - 1)   # absolute month index
        out.append(dict(id=key, site=site[key], t=t, w=names[key]))
    json.dump(out, open(fn, 'w'))
    return out

if __name__ == '__main__':
    for s in ('Haghia Triada', 'Khania', 'Zakros', 'Phaistos'):
        D = la_docs(s); print(s, len(D), sum(len(x['w']) for x in D))
    U = ur3(); print('ur3 dated', len(U), C.Counter(x['site'] for x in U).most_common(5))
