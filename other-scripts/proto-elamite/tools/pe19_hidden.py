"""pe19: build the HIDDEN chronology labels (never read by any fitting code).

Sources (data only, no readings):
  - CDLI catalogue field stratigraphic_level (Le Brun's Acropole I soundings,
    CahDAFI 1; Malyan ABC and TUV building levels).
  - Site-level early group: Tepe Sofalin and Ozbaki (published as early PE
    sites; whole-site attribution, weakest label).
Level codes (larger = later):
  Susa Acropole I: 17A-16 contact 0, 16 / 16C 1, 16-15B contact 1.5, 15B 2,
                   15A 2.5, 14B 3.
  Malyan: building levels numbered from the top (ABC BL 5 is the earliest
          excavated), so level 3 -> 0 (earlier), level 2 -> 1 (later).
usage: python3 pe19_hidden.py <cdli_cat.csv>  -> ../data/pe19_hidden.json
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'pe19_hidden.json')
SUSA = {'17A-16 (contact)': 0, '16': 1, '16C': 1, '16-15B (contact)': 1.5,
        '15B': 2, '15A': 2.5, '14B': 3}


def main(cat):
    csv.field_size_limit(10 ** 9)
    out = {'susa_level': {}, 'malyan_level': {}, 'early_site': {}, 'raw': {}}
    for r in csv.DictReader(open(cat, errors='ignore')):
        if not r['period'].startswith('Proto-Elamite'):
            continue
        pid = 'P' + r['id_text'].zfill(6)
        lv, prov = r['stratigraphic_level'].strip(), r['provenience']
        if lv:
            out['raw'][pid] = [prov, lv, r['excavation_no'], r['findspot_square']]
        if prov.startswith('Susa') and lv in SUSA:
            out['susa_level'][pid] = SUSA[lv]
        if 'Malyan' in prov and lv in ('2', '3'):
            out['malyan_level'][pid] = 1 if lv == '2' else 0
        if 'Sofalin' in prov or 'Ozbaki' in prov:
            out['early_site'][pid] = 1
    json.dump(out, open(OUT, 'w'), indent=1)
    print({k: len(v) for k, v in out.items()})


if __name__ == '__main__':
    main(sys.argv[1])
