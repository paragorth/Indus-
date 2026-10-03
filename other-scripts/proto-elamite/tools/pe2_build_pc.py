"""Build a proto-cuneiform (Uruk IV/III administrative) corpus in the same line
schema as data/pe_corpus.json, so both systems can be profiled with one code.

Source: CDLI bulk ATF (cdli-gh/data, cdliatf_unblocked.atf) + cdli_cat.csv.
Filter: '#atf: lang qpc' AND catalogue period starts with 'Uruk III' or
'Uruk IV' AND genre 'Administrative'. (lang qpc also tags the Proto-Elamite
tablets; the period filter keeps them out.)

usage: python3 pe2_build_pc.py <cdli.atf> <cdli_cat.csv>
output: ../data/pe2_pc_corpus.json
"""
import csv, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'pe2_pc_corpus.json')
NUM = re.compile(r'^([\d/]+|n)\((N\d+[^)]*)\)$')


def norm_code(c):
    # N39~a -> N39A, N30~c@b -> N30C@b ; keep @-modifiers
    m = re.match(r'^(N\d+)(?:~([a-z]))?(.*)$', c)
    if not m:
        return c
    n, v, rest = m.groups()
    n = 'N%02d' % int(n[1:])
    return n + (v.upper() if v else '') + rest


def main(atf, cat):
    csv.field_size_limit(10 ** 9)
    meta = {}
    for r in csv.DictReader(open(cat, errors='ignore')):
        per = r['period']
        if (per.startswith('Uruk III') or per.startswith('Uruk IV')) and r['genre'].startswith('Administrative'):
            meta['P' + r['id_text'].zfill(6)] = {'period': per.split(' (')[0], 'provenience': r['provenience'],
                                                 'designation': r['designation']}
    T, cur, lang, surf = [], None, None, 'obverse'

    def flush():
        if cur and lang == 'qpc' and cur['id'] in meta and cur['lines']:
            cur.update(meta[cur['id']])
            T.append(cur)

    for l in open(atf, errors='ignore'):
        l = l.rstrip('\n')
        if l.startswith('&'):
            flush()
            cur = {'id': l[1:8], 'lines': []}; lang = None; surf = 'obverse'
            continue
        if cur is None:
            continue
        if l.startswith('#atf: lang'):
            lang = l.split()[-1]; continue
        if l.startswith('@'):
            w = l[1:].split()
            if w and w[0] in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'seal'):
                surf = w[0]
            continue
        m = re.match(r"^([0-9][0-9a-z.']*?)\.\s+(.*)$", l)
        if not m:
            continue
        label, body = m.groups()
        lac = '...' in body or '[' in body
        dam = '#' in body or '?' in body
        clean = re.sub(r'[#?!\[\]<>*]', '', body)
        if ' , ' in ' ' + clean + ' ':
            left, right = (' ' + clean + ' ').split(' , ', 1)
        else:
            left, right = '', clean
        nums, signs = [], []
        for side in (left, right):
            for w in side.split():
                if w in (',', '...', 'N') or w.startswith('...'):
                    continue
                mm = NUM.match(w)
                if mm:
                    k, c = mm.groups()
                    c = norm_code(c)
                    if side is left:
                        try:
                            k = int(k)
                        except ValueError:
                            k = 1
                        nums.append([k, c])
                    else:
                        signs.append(c)   # counted signs written in the sign field (e.g. N57)
                    continue
                if w == 'x' or w == 'X':
                    signs.append('x'); continue
                signs.append(w)
        cur['lines'].append({'surface': surf, 'label': label, 'signs': signs, 'numerals': nums,
                             'lacuna': lac, 'damaged': dam, 'raw': body})
    flush()
    json.dump(T, open(OUT, 'w'))
    nl = sum(len(t['lines']) for t in T)
    print('tablets', len(T), 'lines', nl)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
