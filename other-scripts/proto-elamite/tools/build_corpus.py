#!/usr/bin/env python3
"""Build a clean Proto-Elamite corpus JSON from the CDLI bulk dump.

Source: github.com/cdli-gh/data (cdliatf_unblocked.atf, cdli_cat.csv; Git LFS files).
Usage:
  python3 build_corpus.py CDLI_ATF CDLI_CAT_CSV
Writes ../data/pe_raw.atf (the PE subset of the ATF, verbatim) and ../data/pe_corpus.json.

Per tablet: id, designation, provenience, lines. Each line: surface, column, label,
signs (non-numerical tokens, M-signs / compounds / x), numerals [[count, code], ...],
flags (header comment follows, damaged, lacuna).
"""
import csv, json, re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data')

NUM_RE = re.compile(r'^\[?(\d+|n)\]?\((N[0-9A-Z]+)\)')


def clean(tok):
    t = re.sub(r'[\[\]#?!*<>]', '', tok)
    return t


def parse_side(seg):
    """Split a text segment into sign tokens and numeral tokens."""
    signs, nums, lac = [], [], False
    for tok in seg.split():
        if '...' in tok:
            lac = True
            continue
        if tok.startswith('|'):
            signs.append(clean(tok))
            continue
        t = clean(tok)
        m = re.match(r'^(\d+|n)\(((?:N[0-9A-Z]+|n)(?:@[a-z])?)\)$', t)
        if m:
            c = m.group(1)
            nums.append([int(c) if c != 'n' else None, m.group(2)])
            continue
        if t in ('', '+'):
            continue
        signs.append(t)
    return signs, nums, lac


def main(atf_path, cat_path):
    cat = {}
    with open(cat_path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['period'].startswith('Proto-Elamite'):
                cat['P%06d' % int(r['id_text'])] = r
    txt = open(atf_path, encoding='utf-8').read()
    blocks = re.split(r'\n(?=&P\d{6})', txt)
    pe = [b for b in blocks if b[1:8] in cat]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'pe_raw.atf'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(b.rstrip() for b in pe) + '\n')
    tablets = []
    for b in pe:
        L = b.split('\n')
        pid = L[0][1:8]
        r = cat[pid]
        surface, column = 'obverse', 1
        lines = []
        for raw in L[1:]:
            s = raw.strip()
            if s.startswith('@'):
                kw = s[1:].split()
                if kw and kw[0] in ('obverse', 'reverse', 'top', 'bottom', 'left', 'right', 'edge', 'seal', 'surface'):
                    surface = kw[0]
                    column = 1
                elif kw and kw[0] == 'column':
                    try:
                        column = int(kw[1])
                    except Exception:
                        pass
                continue
            if s.startswith('# header') and lines:
                lines[-1]['header_comment'] = True
                continue
            if s.startswith('#') or s.startswith('$') or not s:
                continue
            m = re.match(r'^(\S+?)\.\s+(.*)$', s)
            if not m:
                continue
            label, body = m.group(1), m.group(2)
            if ',' in body:
                left, right = body.split(',', 1)
            else:
                left, right = body, ''
            s1, n1, lac1 = parse_side(left)
            s2, n2, lac2 = parse_side(right)
            # numerals written on the sign side (e.g. a bare total line) go to numerals
            lines.append({
                'surface': surface, 'column': column, 'label': label,
                'signs': s1 + s2, 'numerals': n1 + n2,
                'has_comma': ',' in body,
                'lacuna': lac1 or lac2, 'damaged': '#' in body or '[' in body,
                'raw': body,
            })
        tablets.append({
            'id': pid, 'designation': r['designation'],
            'provenience': r['provenience'], 'genre': r['genre'],
            'object_type': r['object_type'], 'lines': lines,
        })
    json.dump(tablets, open(os.path.join(OUT, 'pe_corpus.json'), 'w'), indent=0)
    # counts
    ntab = len(tablets)
    tok = [s for t in tablets for l in t['lines'] for s in l['signs']]
    msign = [s for s in tok if s.startswith('M') or s.startswith('|')]
    base = set(re.match(r'M\d+', s).group(0) for s in msign if re.match(r'M\d+', s))
    nnum = sum(1 for t in tablets for l in t['lines'] for n in l['numerals'])
    print('catalogue PE entries', len(cat), '; with ATF', ntab)
    print('non-numerical sign tokens', len(tok), '(M/compound', len(msign), ')')
    print('distinct sign forms (with ~variants, compounds)', len(set(msign)))
    print('distinct base M numbers', len(base))
    print('numeral tokens (count(code) groups)', nnum)
    import collections
    print(collections.Counter(t['provenience'] for t in tablets).most_common(8))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
