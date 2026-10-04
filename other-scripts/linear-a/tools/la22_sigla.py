#!/usr/bin/env python3
"""la22: parse SigLA document pages (Salgarella & Castellan, CC BY-NC-SA; fetched page by page
into the git-ignored la22_ckpt/sigla/) into sign sequences with SigLA's own word grouping.

Only sign identities (AB/A codes), roles, word grouping (the alternating even/odd colour class)
and SigLA's sure/unsure flag are used.  Output: la22_ckpt/sigla.json
"""
import os, re, json, glob, html

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'la22_ckpt')

OCC = re.compile(r'<span style="[^"]*" class="popup popup-(?:left|right) ([a-z ]+?)" id="occ-(\d+)">'
                 r'<span class="info-title">#\d+: (?:<a [^>]*reading-pattern:\(([A-Z0-9]+), (?:false|true)\)[^>]*>)?'
                 r'<span class="(sure|unsure)-reading">([^<]*)</span>')


def parse(path):
    s = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'<div class="title">([^<]*)</div>', s)
    name = html.unescape(m.group(1)) if m else os.path.basename(path)
    occ = []
    for cls, n, code, sure, rd in OCC.findall(s):
        parts = cls.split()
        occ.append({'n': int(n), 'role': parts[0], 'par': parts[1] if len(parts) > 1 else '',
                    'code': code or '', 'read': html.unescape(rd), 'sure': sure == 'sure'})
    occ.sort(key=lambda o: o['n'])
    words = []; cur = []; last = None
    for o in occ:
        if o['role'] == 'syllabogram':
            if cur and o['par'] != last:
                words.append(cur); cur = []
            cur.append(o); last = o['par']
        else:
            if cur: words.append(cur); cur = []
            last = None
    if cur: words.append(cur)
    return {'name': name, 'occ': occ, 'words': [[(o['code'], o['read'], o['sure']) for o in w] for w in words]}


def main():
    out = {}
    for p in sorted(glob.glob(os.path.join(CK, 'sigla', '*.html'))):
        r = parse(p)
        if r['occ']: out[r['name']] = r
    json.dump(out, open(os.path.join(CK, 'sigla.json'), 'w'), ensure_ascii=False)
    nocc = sum(len(r['occ']) for r in out.values())
    print('docs', len(out), 'occurrences', nocc,
          'syllabograms', sum(1 for r in out.values() for o in r['occ'] if o['role'] == 'syllabogram'),
          'unsure', sum(1 for r in out.values() for o in r['occ'] if not o['sure']),
          'words', sum(len(r['words']) for r in out.values()))


if __name__ == '__main__':
    main()
