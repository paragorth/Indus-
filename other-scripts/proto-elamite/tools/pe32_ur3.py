"""pe32 Ur III control corpus: ration / wage lines of the form '<count> <word> ... <amount>-ta' (CDLI dump in the
scratchpad, not committed).  Each line becomes a PE-like event (word = the counted sign, x = the count,
y = x * per-unit amount in sila).  The recipient words and their usual sizes are never given to the blind
method; they serve only as the truth it must give back."""
import os, re, json, csv
from pe27_common import SCRATCH, ur_int, ur_cap, NUMTOK, _toks
from pe32_common import CK

LINE = re.compile(r"^\d+'?\.\s+(.*)$")


def build():
    fn = os.path.join(CK, 'ur3_ration_events.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    csv.field_size_limit(10 ** 9)
    keep = set()
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III'):
            keep.add('P%06d' % int(row['id_text']))
    ev, cur = [], None
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            cur = raw[1:8] if raw[1:8] in keep else None
            continue
        if not cur or not raw[:1].isdigit():
            continue
        m = LINE.match(raw.rstrip())
        if not m:
            continue
        body = m.group(1)
        if '[' in body or '...' in body or ' x ' in f' {body} ' or '-ta' not in body:
            continue
        tk = _toks(body)
        x, j = ur_int(tk, 0)
        if not x or j >= len(tk) or NUMTOK.match(tk[j]):
            continue
        word = tk[j]
        mm = re.search(r'((?:\d+(?:/\d+)?\((?:barig|ban2|disz|asz)\)\s*)+(?:sila3)?)-ta\b', body)
        if not mm:
            continue
        rt = mm.group(1).strip()
        if '/' in rt:
            continue
        rtk = rt.split()
        if rtk[-1] != 'sila3':
            rtk = rtk + ['sila3'] if not re.search(r'(barig|ban2)\)$', rt) else rtk + ['sila3']
        r = ur_cap(rtk)
        if not r or r > 300:
            continue
        ev.append({'tid': cur, 'pfin': word, 'x': x, 'r': r, 'y': x * r})
    json.dump(ev, open(fn, 'w'))
    return ev


if __name__ == '__main__':
    from collections import Counter
    E = build()
    print(len(E), len({e['tid'] for e in E}))
    c = Counter(e['pfin'] for e in E)
    for w, n in c.most_common(15):
        print(w, n, Counter(e['r'] for e in E if e['pfin'] == w).most_common(5))
