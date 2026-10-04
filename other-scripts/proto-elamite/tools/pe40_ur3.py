"""pe40 Ur III rota control: Drehem tablets that name a governor's province (ensi2 / bala lines).
Players = province names (GN{ki}) on ensi2/bala lines; truth = month (cyclic, 12). Date lines are never read as players."""
import csv, re, json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe40_ckpt')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

def build():
    out = os.path.join(CK, 'ur3_bala.json')
    if os.path.exists(out):
        return json.load(open(out))
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III') and 'Puzri' in row['provenience']:
            m = re.match(r'^([^.]+)\.(\d\d)\.(\d\d)', row['dates_referenced'])
            if m and m.group(3) not in ('00',) and m.group(3).isdigit() and 1 <= int(m.group(3)) <= 12:
                keep['P%06d' % int(row['id_text'])] = dict(king=m.group(1), year=m.group(2), month=int(m.group(3)))
    T = collections.defaultdict(list); cur = None
    for raw in open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            cur = raw[1:8] if raw[1:8] in keep else None; continue
        if not cur or not raw[:1].isdigit():
            continue
        tk = [re.sub(r'[#!?\[\]<>]', '', x) for x in raw.split()[1:]]
        if not tk or tk[0] in ('iti', 'mu', 'u4') or tk[0].startswith('u4-') or tk[0].startswith('mu-'):
            continue
        if 'ensi2' in tk or 'bala' in tk or any(x.startswith('bala') for x in tk):
            for x in tk:
                if x.endswith('{ki}') and x != 'nibru{ki}':
                    T[cur].append(x)
    res = [dict(id=p, month=keep[p]['month'], king=keep[p]['king'], year=keep[p]['year'], players=sorted(set(v)))
           for p, v in T.items() if v]
    json.dump(res, open(out, 'w'))
    return res

if __name__ == '__main__':
    R = build()
    print(len(R))
    c = collections.Counter(x for r in R for x in r['players'])
    print(c.most_common(30))
    print(collections.Counter(len(r['players']) for r in R))
    # month profile of top provinces
    for gn, n in c.most_common(12):
        mc = collections.Counter(r['month'] for r in R if gn in r['players'])
        print(gn, n, [mc.get(m, 0) for m in range(1, 13)])
