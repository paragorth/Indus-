#!/usr/bin/env python3
"""LA-10 typology table: data only, no proposed readings.
Sources (CLDF, downloaded 4 Oct 2026 into ../data/la10/wals and ../data/la10/grambank):
  WALS Online v2020 (Dryer & Haspelmath eds.), https://github.com/cldf-datasets/wals  (CC BY 4.0)
  Grambank v1 (Skirgard et al. 2023), https://github.com/grambank/grambank             (CC BY 4.0)
WALS has no Bronze Age language; modern relatives stand in (marked 'proxy'). Grambank has Akkadian, Ancient Greek,
Etruscan, Hieroglyphic Luwian, Hurrian, Phoenician, Sumerian, Ugaritic. Neither database codes vowel harmony or OCP.
Output ../data/la10/typology.txt and typology.json
"""
import csv, json, os, collections
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'la10')
W = {'grk': 'Greek (Modern) [proxy for Greek]', 'gcy': 'Greek (Cypriot) [proxy]', 'cop': 'Coptic [proxy for Egyptian]',
     'heb': 'Hebrew (Modern) [proxy for NW Semitic]', 'ams': 'Arabic (MSA) [proxy for Semitic]', 'geo': 'Georgian [Kartvelian]',
     'laz': 'Laz [Kartvelian]', 'sva': 'Svan [Kartvelian]', 'mgl': 'Mingrelian [Kartvelian]', 'abk': 'Abkhaz [NW Caucasian]',
     'kab': 'Kabardian [NW Caucasian]', 'chc': 'Chechen [Nakh]', 'lez': 'Lezgian [Daghestanian]', 'arm': 'Armenian (Eastern)',
     'bsq': 'Basque [isolate]', 'tur': 'Turkish [vowel-harmony reference, not Bronze Age]', 'prs': 'Persian'}
WF = ['1A', '2A', '3A', '12A', '26A', '27A', '51A']
G = {'akka1240': 'Akkadian', 'anci1242': 'Ancient Greek', 'etru1241': 'Etruscan', 'hier1240': 'Hieroglyphic Luwian',
     'hurr1240': 'Hurrian', 'phoe1239': 'Phoenician', 'sume1241': 'Sumerian', 'ugar1238': 'Ugaritic',
     'geor1241': 'Georgian', 'basq1248': 'Basque', 'nucl1301': 'Turkish', 'mode1248': 'Modern Greek', 'egyp1246': 'Egyptian (Ancient)',
     'copt1239': 'Coptic', 'hatt1246': 'Hattic', 'hitt1242': 'Hittite', 'elam1244': 'Elamite', 'urar1245': 'Urartian'}
PRE = ['GB079', 'GB090', 'GB092', 'GB094', 'GB430', 'GB431']
SUF = ['GB080', 'GB089', 'GB091', 'GB093', 'GB432', 'GB433']
RED = ['GB158', 'GB159']

def main():
    codes = {r['ID']: r['Name'] for r in csv.DictReader(open(os.path.join(D, 'wals', 'codes.csv')))}
    pn = {r['ID']: r['Name'] for r in csv.DictReader(open(os.path.join(D, 'wals', 'parameters.csv')))}
    wv = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(D, 'wals', 'values.csv'))):
        if r['Language_ID'] in W and r['Parameter_ID'] in WF: wv[r['Language_ID']][r['Parameter_ID']] = codes.get(r['Code_ID'], r['Value'])
    gv = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(D, 'grambank', 'values.csv'))):
        if r['Language_ID'] in G and r['Parameter_ID'] in PRE + SUF + RED: gv[r['Language_ID']][r['Parameter_ID']] = r['Value']
    lines = ['WALS (v2020 CLDF) features: ' + '; '.join(f'{f} {pn[f]}' for f in WF)]
    for k, nm in W.items():
        lines.append(f'  {nm}: ' + ' | '.join(f'{f}={wv[k].get(f, "-")}' for f in WF))
    lines.append('\nGrambank v1: prefix index = share of coded prefix features present (' + ','.join(PRE) + '); suffix index likewise (' +
                 ','.join(SUF) + '); reduplication ' + ','.join(RED))
    out = {'wals': wv, 'grambank': {}}
    for k, nm in G.items():
        v = gv.get(k)
        if not v: lines.append(f'  {nm} ({k}): not in Grambank'); continue
        def idx(fs):
            c = [v[f] for f in fs if v.get(f) in ('0', '1')]; return (sum(x == '1' for x in c), len(c))
        p, s, rd = idx(PRE), idx(SUF), idx(RED)
        out['grambank'][nm] = dict(prefix=p, suffix=s, redup=rd)
        lines.append(f'  {nm}: prefix {p[0]}/{p[1]}  suffix {s[0]}/{s[1]}  reduplication {rd[0]}/{rd[1]}')
    langs = {r['ID']: r['Genus'] for r in csv.DictReader(open(os.path.join(D, 'wals', 'languages.csv')))}
    want = ('Greek', 'Semitic', 'Egyptian-Coptic', 'Kartvelian', 'Northwest Caucasian', 'Nakh', 'Lezgic', 'Avar-Andic-Tsezic',
            'Armenian', 'Basque', 'Turkic')
    agg = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for r in csv.DictReader(open(os.path.join(D, 'wals', 'values.csv'))):
        gn = langs.get(r['Language_ID'])
        if gn in want and r['Parameter_ID'] in ('2A', '12A', '26A'): agg[gn][r['Parameter_ID']][codes[r['Code_ID']]] += 1
    lines.append('\nWALS by genus (all languages of the genus; counts): 2A vowel qualities | 12A syllable structure | 26A prefixing vs suffixing')
    for gn in want:
        lines.append(f'  {gn}: ' + ' | '.join(f'{p} ' + ', '.join(f'{k} {v}' for k, v in agg[gn][p].items()) for p in ('2A', '12A', '26A')))
    out['wals_genus'] = {g: {p: dict(c) for p, c in d.items()} for g, d in agg.items()}
    open(os.path.join(D, 'typology.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(out, open(os.path.join(D, 'typology.json'), 'w'), indent=1)
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
