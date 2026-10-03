"""pe4: build calibration corpora from the CDLI ATF dump (Ur III).

Positive control (real factorial item codes): Drehem (Puzrish-Dagan) livestock
lines "N(disz) udu niga sig5 ..." -> head animal word + attribute tokens, and
Ur III textile lines "N tug2 ..." (all sites) -> head + attribute tokens.
Negative control (personal names from the same archive and conventions):
Drehem "ki PN-ta" and "giri3 PN" lines -> PN tokens.
Each record keeps tablet id, head word, attribute tokens and the numeral value
(sexagesimal count, disz/u/gesz2 only), so quantity checks are possible.

usage: python3 pe4_build_controls.py <cdli.atf> <cdli_cat.csv>
writes data/pe4_controls.json
"""
import csv, json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'pe4_controls.json')

ANIMALS = {'udu', 'u8', 'masz2', 'ud5', 'sila4', 'kir11', 'gukkal', 'gu4', 'ab2',
           'amar', 'anse', 'dara4', 'masz2-gal', 'udu-nita2', 'munus', 'masz-da3',
           'sze6-gi6', 'kun-gid2', 'udu-a-lum', 'a-lum'}
NUMV = {'disz': 1, 'asz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600}
NUMRE = re.compile(r"^(\d+(?:/\d+)?)\(([a-z0-9']+)\)$")


def clean(w):
    w = re.sub(r'[#!?*\[\]<>]', '', w)
    w = re.sub(r'\{[^}]*\}', lambda m: m.group(0).strip('{}') + '.', w)  # determinatives as tokens
    return w


def toks(words):
    out = []
    for w in words:
        w = clean(w)
        for s in re.split(r'[-.]', w):
            if s and s not in ('x', '...'):
                out.append(s)
    return out


def num_prefix(words):
    val, i = 0, 0
    while i < len(words):
        m = NUMRE.match(clean(words[i]))
        if not m:
            break
        n, u = m.groups()
        if '/' in n or u not in NUMV:
            return None, i
        val += int(n) * NUMV[u]
        i += 1
    return (val if i else None), i


def main(atf, cat):
    csv.field_size_limit(10 ** 9)
    period, prov = {}, {}
    with open(cat, newline='') as f:
        for row in csv.DictReader(f):
            if row['period'].startswith('Ur III'):
                pid = 'P%06d' % int(row['id_text']) if row['id_text'].isdigit() else row['id_text']
                period[pid] = 1
                prov[pid] = row.get('provenience', '')
    herd, tex, names = [], [], []
    cur = None
    with open(atf, errors='replace') as f:
        for line in f:
            if line.startswith('&P'):
                cur = line[1:8]
                continue
            if cur not in period or not re.match(r"^\d+'?\.", line):
                continue
            body = line.split('.', 1)[1].strip()
            if '[' in body or 'x' in body.split():
                continue  # broken line
            words = body.split()
            drehem = prov.get(cur, '').startswith('Puzri')
            val, i = num_prefix(words)
            rest = words[i:]
            if val and rest:
                head = clean(rest[0])
                if drehem and head in ANIMALS:
                    herd.append({'t': cur, 'head': head, 'attr': toks(rest[1:]), 'words': [clean(w) for w in rest[1:]], 'n': val})
                elif head == 'tug2' or head.startswith('tug2'):
                    ws = rest[1:] if head == 'tug2' else [head[5:]] + rest[1:]
                    tex.append({'t': cur, 'head': 'tug2', 'attr': toks(ws),
                                'words': [clean(w) for w in ws], 'n': val})
            elif drehem and not val and len(words) == 2:
                pn = None
                if words[0] == 'ki' and len(words) == 2 and words[1].endswith('-ta'):
                    pn = words[1][:-3]
                elif words[0] in ('giri3', 'gir3') and len(words) == 2:
                    pn = words[1]
                if pn:
                    names.append({'t': cur, 'head': 'PN', 'attr': toks([pn]), 'words': [clean(pn)], 'n': None})
    json.dump({'herd': herd, 'textile': tex, 'names': names}, open(OUT, 'w'))
    for k, v in (('herd', herd), ('textile', tex), ('names', names)):
        ty = Counter(tuple(r['attr']) for r in v)
        print(k, len(v), 'records', len(ty), 'types', Counter(r['head'] for r in v).most_common(6))
        print('   top attr strings', ty.most_common(8))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
