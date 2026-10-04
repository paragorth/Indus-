"""pe8: build per-tablet find metadata for the 'filled-in forms' tests.

From cdli_cat.csv (CDLI bulk catalogue, commit d66b12b) take, for every Proto-Elamite entry:
museum_no, excavation_no, findspot_square, stratigraphic_level, provenience_remarks, collection, seal_id,
primary_publication.  From pe_raw.atf take surface-level '$' notes (blank / broken / seal / scribal design)
and '@column' use.  Output: data/pe8_meta.json   {P-number: {...}}

usage: python3 pe8_build.py <cdli_cat.csv>
"""
import csv, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')


def parse_atf():
    out, cur, surf = {}, None, None
    for line in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        s = line.rstrip('\n').strip()
        if s.startswith('&'):
            cur = s[1:].split('=')[0].strip()
            out[cur] = {'notes': [], 'surfaces': [], 'cols': set(), 'seal_note': False, 'sd': False}
            surf = None
            continue
        if cur is None:
            continue
        d = out[cur]
        if s.startswith('@'):
            tag = s[1:].split()[0] if len(s) > 1 else ''
            if tag in ('obverse', 'reverse', 'top', 'left', 'bottom', 'right', 'edge', 'seal', 'surface'):
                surf = tag
                d['surfaces'].append(tag)
            elif tag == 'column':
                d['cols'].add((surf, s.split()[1] if len(s.split()) > 1 else '1'))
            continue
        if s.startswith('$'):
            d['notes'].append((surf, s[1:].strip()))
            if 'seal' in s:
                d['seal_note'] = True
            if 'scribal design' in s:
                d['sd'] = True
            continue
        if s.startswith('#') and 'seal' in s:
            d['seal_note'] = True
    for d in out.values():
        d['cols'] = sorted([list(c) for c in d['cols']])
    return out


def main(catf):
    csv.field_size_limit(10 ** 9)
    atf = parse_atf()
    meta = {}
    with open(catf, newline='') as f:
        for row in csv.DictReader(f):
            if not row['period'].startswith('Proto-Elamite'):
                continue
            pid = 'P%06d' % int(row['id_text']) if row['id_text'].isdigit() else row['id_text']
            if pid not in atf:
                continue
            m = {k: row[k] for k in ('museum_no', 'excavation_no', 'findspot_square', 'stratigraphic_level',
                                     'provenience_remarks', 'collection', 'seal_id', 'primary_publication',
                                     'designation', 'provenience')}
            mm = re.match(r'\s*(Sb|AO|NMI BK|NMI|MDP|UM|IM|EMSP)\D*?(\d+)', m['museum_no'] or '')
            m['mus_prefix'] = mm.group(1) if mm else None
            m['mus_num'] = int(mm.group(2)) if mm else None
            dm = re.match(r'(.*?),?\s*(\d+)\s*$', m['designation'] or '')
            m['pub_vol'] = dm.group(1).strip() if dm else m['designation']
            m['pub_num'] = int(dm.group(2)) if dm else None
            m.update(atf[pid])
            meta[pid] = m
    json.dump(meta, open(os.path.join(DATA, 'pe8_meta.json'), 'w'))
    from collections import Counter
    print(len(meta), 'tablets with metadata')
    print(Counter(m['mus_prefix'] for m in meta.values()).most_common())
    print(Counter(m['pub_vol'] for m in meta.values()).most_common(8))


if __name__ == '__main__':
    main(sys.argv[1])
