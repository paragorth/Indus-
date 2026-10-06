#!/usr/bin/env python3
"""pe60 cycle 1: inventory of Proto-Elamite texts NOT in the training corpus (data/pe_raw.atf).

Sources (fetched live, 6 Oct 2026):
  CDLI catalogue  https://cdli.earth/search?period=Proto-Elamite&limit=2000&format=csv
  CDLI ATF        https://cdli.earth/search?period=Proto-Elamite&limit=2000&format=atf&aspect=inscriptions
  CDLI images     https://cdli.earth/dl/photo/<P>.jpg , https://cdli.earth/dl/lineart/<P>_l.jpg (HTTP status only)
Compared with: data/pe_raw.atf (training corpus) and data/pe22_ckpt/catalog.json (2023 CDLI dump).
Images are NOT stored in the repo (licensed CDLI photos stay in the scratchpad).
Usage: python3 pe60_inventory.py <download_dir>   (download_dir holds pe_cat.csv, pe_live.atf, imgchk.txt)
Writes data/pe60_inventory.json and data/pe60_ckpt/new_atf.atf.
"""
import csv, json, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'pe60_ckpt')
os.makedirs(CK, exist_ok=True)


def atf_blocks(path):
    t = open(path, encoding='utf8', errors='replace').read()
    out = {}
    for b in re.split(r'\n(?=&P)', t):
        m = re.match(r'&(P\d+)', b.strip())
        if m:
            out[m.group(1)] = b.strip() + '\n'
    return out


def content_lines(b):
    return [l for l in b.splitlines() if re.match(r"^\d+'?\.", l.strip())]


def main(dl):
    live = atf_blocks(os.path.join(dl, 'pe_live.atf'))
    old = atf_blocks(os.path.join(D, 'pe_raw.atf'))
    cat = {'P%06d' % int(r['artifact_id']): r for r in csv.DictReader(open(os.path.join(dl, 'pe_cat.csv')))}
    oldcat = json.load(open(os.path.join(D, 'pe22_ckpt', 'catalog.json')))
    img = {}
    for l in open(os.path.join(dl, 'imgchk.txt')):
        p, a, b = l.split()
        img[p] = (a.split('=')[1].split(':')[0] in ('200', '206'), b.split('=')[1].split(':')[0] in ('200', '206'))
    inv = []
    for p, r in sorted(cat.items()):
        if p in old:
            continue
        has_atf = p in live
        n_lines = len(content_lines(live[p])) if has_atf else 0
        ph, la = img.get(p, (None, None))
        if has_atf and n_lines > 0:
            status = 'new transliteration on CDLI (not in training corpus)'
        elif has_atf:
            status = 'CDLI ATF stub only (to be completed / no linguistic content)'
        elif ph or la:
            status = 'image only, no transliteration'
        else:
            status = 'catalogue entry only (no image, no ATF)'
        inv.append({'P': p, 'designation': r['designation'], 'museum_no': r['museum_no'],
                    'provenience': r['provenience'], 'collection': r['collections'],
                    'type': r['artifact_type'], 'pub_key': r['publications_key'],
                    'in_2023_dump': p in oldcat, 'atf_lines': n_lines,
                    'photo': ph, 'lineart': la, 'status': status})
    with open(os.path.join(CK, 'new_atf.atf'), 'w') as f:
        for x in inv:
            if x['atf_lines']:
                f.write(live[x['P']] + '\n')
    summ = collections.Counter((x['status'], x['type']) for x in inv)
    out = {'source': 'CDLI live 6 Oct 2026 vs data/pe_raw.atf', 'n_live_catalogue': len(cat),
           'n_live_atf': len(live), 'n_training_atf': len(old),
           'summary': {f'{a} | {b}': n for (a, b), n in summ.most_common()}, 'items': inv}
    json.dump(out, open(os.path.join(D, 'pe60_inventory.json'), 'w'), indent=1)
    for k, v in out['summary'].items():
        print(v, k)


if __name__ == '__main__':
    main(sys.argv[1])
