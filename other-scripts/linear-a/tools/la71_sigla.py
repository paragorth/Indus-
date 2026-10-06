#!/usr/bin/env python3
"""la71: re-parse SigLA document pages (la22_ckpt/sigla/*.html, CC BY-NC-SA) without the la22 bug.

la22_sigla.py's pattern read the sign reading as [^<]*, so every occurrence whose reading carries a
subscript (ra<sub>2</sub>, pa<sub>3</sub>, ta<sub>2</sub> ...) was silently dropped (la71 cycle 1).
This parser keeps every 'occ-N' popup: code, role, reading (subscripts flattened), sure/unsure.
Output: data/la71_ckpt/sigla71.json  {doc: {'occ': [...]}}
"""
import os, re, json, glob, html

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
SIG = os.path.join(DATA, 'la22_ckpt', 'sigla')
CK = os.path.join(DATA, 'la71_ckpt')
POP = re.compile(r'<span style="[^"]*" class="popup popup-(?:left|right) ([a-z ]+?)" id="occ-(\d+)">(.*?)<span class="role">', re.S)


def parse(path):
    s = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'<div class="title">([^<]*)</div>', s)
    name = html.unescape(m.group(1)) if m else os.path.basename(path)
    occ = []
    for cls, n, body in POP.findall(s):
        code = re.search(r'reading-pattern:\(([A-Za-z0-9]+), (?:false|true)\)', body)
        rd = re.search(r'<span class="(sure|unsure)-reading">(.*?)</span>(?:</a>)?</span>', body, re.S)
        parts = cls.split()
        read = html.unescape(re.sub(r'<sub>(.*?)</sub>', r'\1', rd.group(2))) if rd else ''
        read = re.sub(r'<[^>]+>', '', read)
        occ.append({'n': int(n), 'role': parts[0], 'par': parts[1] if len(parts) > 1 else '',
                    'code': code.group(1) if code else '', 'read': read,
                    'sure': (rd.group(1) == 'sure') if rd else False, 'parsed': bool(rd)})
    occ.sort(key=lambda o: o['n'])
    return name, occ


def main():
    os.makedirs(CK, exist_ok=True)
    out = {}
    for p in sorted(glob.glob(os.path.join(SIG, '*.html'))):
        name, occ = parse(p)
        if occ: out[name] = {'occ': occ}
    json.dump(out, open(os.path.join(CK, 'sigla71.json'), 'w'), ensure_ascii=False)
    old = json.load(open(os.path.join(DATA, 'la22_ckpt', 'sigla.json')))
    from collections import Counter
    n = sum(len(v['occ']) for v in out.values()); no = sum(len(v['occ']) for v in old.values())
    print('docs', len(out), 'occurrences', n, '(la22 parse kept %d; lost %d)' % (no, n - no))
    print('roles', Counter(o['role'] for v in out.values() for o in v['occ']))
    print('unsure', Counter(o['role'] for v in out.values() for o in v['occ'] if not o['sure']))
    print('unparsed reading', sum(1 for v in out.values() for o in v['occ'] if not o['parsed']))
    print('no code', Counter(o['read'] for v in out.values() for o in v['occ'] if not o['code']).most_common(10))


if __name__ == '__main__':
    main()
