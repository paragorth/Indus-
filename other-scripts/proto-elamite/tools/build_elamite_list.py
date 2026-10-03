#!/usr/bin/env python3
"""Build and FREEZE two name lists before any PE matching is run.

1. data/elamite_name_list_frozen.json  -- Elamite personal / divine names and
   name elements from cuneiform Elamite (and Elamite names in Sumero-Akkadian
   texts), with sources and periods.
2. data/control_name_list_frozen.json  -- non-Elamite (Akkadian / Sumerian)
   personal names from CDLI, the 'random onomasticon' for control (b).

Inputs (cached copies; the URLs are recorded in the output):
  zadok.txt     archive.org/download/TheElamiteOnomasticon1984/Zadok1984TheElamiteOnomasticon_djvu.txt
  kings.wiki    en.wikipedia.org/w/index.php?title=List_of_Elamite_kings&action=raw
  cdli.atf      CDLI bulk ATF (github.com/cdli-gh/data, commit d66b12b)
  cdli_cat.csv  CDLI catalogue (same commit)
Usage: build_elamite_list.py ZADOK_TXT KINGS_WIKI CDLI_ATF CDLI_CAT
Only names (data) are taken from these sources; no etymologies or readings.
"""
import csv, hashlib, json, os, re, sys, datetime, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
zadok_p, kings_p, atf_p, cat_p = sys.argv[1:5]
csv.field_size_limit(10 ** 9)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def clean_syl(s):
    """ATF / OCR syllable string -> lower-case hyphenated syllables (no indices)."""
    s = s.replace('§', 'š').replace('sz', 'š').replace('SZ', 'š').replace('s,', 's').replace('t,', 't')
    s = re.sub(r'\{[^}]*\}', '-', s)        # determinatives become boundaries
    s = re.sub(r'[\[\]#?!*<>()\'^:]', '', s)
    s = re.sub(r'(?<=[a-zšḫ])[0-9x]+', '', s)  # sign indices
    s = s.lower().replace('ḫ', 'h')
    s = re.sub(r'-+', '-', s).strip('-')
    return s


SYL_OK = re.compile(r'^[aeiouhkgptdbszšmnlrwy]{1,5}(-[aeiouhkgptdbszšmnlrwy]{1,5})*$')

# ---------------------------------------------------------------- Elamite
names = []      # dicts: form (hyphenated syllables), period, source, kind
elements = []   # dicts: form (phonemic, Zadok-normalised or broad), source, period

# A. Zadok 1984: element headwords (Zadok's own normalised stems)
zt = open(zadok_p, encoding='utf-8', errors='replace').read()
zt2 = re.sub(r'\s+', ' ', re.sub(r'-\s*\n\s*', '-', zt))
body = zt2[zt2.find('B.'):]
for num, hw in re.findall(r'(?<![\w(])(\d{1,3}[a-z]?)[.,] ?\(?((?:[A-Z§Š]|\([A-Za-z§]{1,2}\))'
                          r'(?:[A-Z§Š\'/]|\([A-Za-z§]{1,2}\))+)(?=[\s.,(:;-])', zt2):
    if hw in ('OB', 'RAE', 'ME', 'NE', 'MB', 'NA', 'OAKK', 'C/', 'N/LB', 'SB'):
        continue
    f = re.sub(r'\(([A-Za-z§])\)', r'\1', hw).split('/')[-1]
    f = f.replace('§', 'š').replace('Š', 'š').lower()
    f = {'atfa': 'atta', 'atlata': 'attata'}.get(f, f)   # OCR fixes (Zadok 18, 18a)
    if re.fullmatch(r'[a-zš]{2,10}', f):
        elements.append({'form': f, 'source': 'Zadok 1984 headword §' + num, 'period': '3rd mill.-Achaemenid'})

# B. Zadok 1984: name spellings cited under each element
pat = re.compile(r"(?<![\w-])((?:[df] ?)?[A-Z§][a-z§'0-9]{0,4}(?:-(?:d ?)?[a-z§'0-9]{1,5})+)(?![\w-])")
PER = re.compile(r'\b(OAkk|Ur III|OB|MB|ME|NE|NA|RAE|N/LB|SB)\b')
for m in pat.finditer(zt2):
    raw = m.group(1)
    raw2 = re.sub(r'^[df] ?', '', raw)
    raw2 = re.sub(r'-d ?', '-', raw2)
    form = clean_syl(raw2)
    if not SYL_OK.match(form) or form.count('-') < 1:
        continue
    after = zt2[m.end():m.end() + 60]
    par = re.match(r'\s*\(([^)]*)\)', after)
    p = PER.search(par.group(1)) if par else None
    names.append({'form': form, 'period': p.group(1) if p else '?', 'source': 'Zadok 1984 (OCR)',
                  'kind': 'DN' if raw.startswith('d') else 'PN'})

# C. CDLI Elamite-language texts: {disz}/{m} personal names and {d}/{dingir} divine names
per = {}
for row in csv.DictReader(open(cat_p, encoding='utf-8', errors='replace')):
    if 'Elamite' in (row.get('language') or ''):
        per['P%06d' % int(row['id_text'])] = row['period']
cur = None
for line in open(atf_p, encoding='utf-8', errors='replace'):
    if line.startswith('&P'):
        cur = line[1:8]
        continue
    if cur not in per or not re.match(r"^\d+'?\.", line):
        continue
    for w in line.split('.', 1)[1].split():
        w2 = re.sub(r'[#?!\[\]<>()]', '', w)
        mm = re.match(r'\{(disz|m|d|dingir|an)\}(.+)', w2)
        if not mm or re.search(r'[A-Z_]|\.\.\.|\bx\b', mm.group(2)):
            continue
        rest = re.sub(r'\{(d|dingir|an)\}', '-', mm.group(2))
        form = clean_syl(rest)
        if not SYL_OK.match(form):
            continue
        pr = per[cur]
        names.append({'form': form, 'period': pr.split(' (')[0], 'source': 'CDLI ' + cur,
                      'kind': 'PN' if mm.group(1) in ('disz', 'm') else 'DN'})

# D. Rulers of Elam (Awan .. Neo-Elamite), broad transcriptions from the Wikipedia list
wk = open(kings_p, encoding='utf-8').read()
kings = [re.sub(r'\[\[[^|\]]*\||\[\[|\]\]', '', k) for k in re.findall(r"'''(\[\[[^\]\n]+\]\]|[^'\n\[{}|]+)'''", wk)]
EXCLUDE = {'kings of Elam', 'Epirmupi', 'Ili-ishmani', 'Inshushinak-shar-ili', 'Açina', 'Martiya',
           'Ummanigash', 'Ummanunu', 'Teumman', 'Urtak', 'Indabibi', 'Bahuri', 'Shalla', 'Kidinu',
           'Darius', 'Tigraios', 'Okkonapses', 'Pittit', 'Anzaze', 'Phraates', 'Osroes', 'Ulfan',
           'Abar-Basi', 'Khwasak', 'Kamnaskires-Orodes'}
seen = set()
for k in kings:
    k = re.sub(r'\s+[IVX]+$', '', k.strip())
    if k in EXCLUDE or k.startswith(('Kamnaskires', 'Orodes', 'Imazu')) or k in seen:
        continue
    seen.add(k)
    if 'Humban-haltash' in k or 'Tepti-Humban' in k or 'Shutur' in k or 'Hallutash' in k \
            or 'Atta-hamiti' in k or 'Humban-' in k and 'numena' not in k and 'menanu' not in k:
        prd = 'Neo-Elamite'
    elif k in ('Hishep-ratep', 'Luh-ishan', 'Puzur-Inshushinak', 'Emahshini', 'Autalummash', 'Khita',
               "Hi'elu", "Hita'a", 'Tata', 'Ukkutahesh', 'Hishur', 'Shushuntarana', 'Napilhush',
               'Kikku-siwe-tempt', 'Peyli'):
        prd = 'Old Elamite (Awan)'
    else:
        prd = 'Old/Middle Elamite'
    names.append({'form': k.lower().replace('sh', 'š').replace("'", ''), 'period': prd,
                  'source': 'Wikipedia "List of Elamite kings" (broad transcription)', 'kind': 'royal PN',
                  'broad': True})

# E. Task-specified common elements and gods (broad transcription; well-known attestations)
for f in ['kutir', 'kudur', 'šilhak', 'untaš', 'hutelutuš', 'kuk', 'temti', 'tepti', 'attahušu',
          'inšušinak', 'šušinak', 'napiriša', 'napir', 'humban', 'huban', 'kiririša', 'šimut',
          'pinikir', 'nahhunte', 'hutran', 'ruhuratir', 'lagamar', 'šutruk', 'hallutuš', 'šilhaha',
          'idattu', 'ebarti', 'halki', 'nahiti', 'manzat', 'išnikarap', 'tan', 'siwe', 'atta', 'puzur']:
    elements.append({'form': f, 'source': 'task brief / standard Elamite royal and divine names',
                     'period': 'Old-Neo Elamite'})
# components of royal names
for n in names:
    if n.get('broad'):
        for part in n['form'].split('-'):
            elements.append({'form': part, 'source': 'component of ' + n['form'], 'period': n['period']})
# components of CDLI pre-Achaemenid DNs / PNs (whole DN forms)
for n in names:
    if n['source'].startswith('CDLI') and n['period'] != 'Achaemenid' and n['kind'] == 'DN':
        elements.append({'form': n['form'].replace('-', ''), 'source': 'CDLI DN ' + n['form'],
                         'period': n['period']})

# dedupe
def dd(L, key):
    out, s = [], set()
    for x in L:
        k = key(x)
        if k in s:
            continue
        s.add(k)
        out.append(x)
    return out

names = dd(names, lambda x: (x['form'], x['period']))
elements = dd(elements, lambda x: x['form'])
cnt = collections.Counter(n['period'] for n in names)
OLD = {'OAkk', 'Ur III', 'OB', 'Old Akkadian', 'Old Elamite (Awan)', 'Old/Middle Elamite', 'ME', 'MB',
       'Middle Elamite', 'NE', 'Neo-Elamite'}
out = {
    '_frozen': datetime.datetime.utcnow().isoformat() + 'Z',
    '_note': 'Frozen BEFORE the PE matching test (tools/attack_names.py). Do not edit after the test.',
    '_sources': {
        'zadok1984': {'url': 'https://archive.org/download/TheElamiteOnomasticon1984/Zadok1984TheElamiteOnomasticon_djvu.txt',
                      'sha256': sha(zadok_p),
                      'what': 'R. Zadok, The Elamite Onomasticon (AION Suppl. 40, 1984), archive.org OCR text. Element headwords + cited name spellings; OCR noisy (š often lost).'},
        'cdli': {'what': 'CDLI bulk ATF, texts whose catalogue language includes Elamite; {disz}/{m} and {d}/{dingir} words', 'atf_sha256': sha(atf_p)},
        'kings': {'url': 'https://en.wikipedia.org/w/index.php?title=List_of_Elamite_kings&action=raw', 'sha256': sha(kings_p),
                  'what': 'royal names only (Awan to Neo-Elamite), Akkadian governors and post-Achaemenid rulers dropped'},
        'manual': 'task brief list of famous elements/gods',
    },
    'counts': {'names': len(names), 'elements': len(elements), 'names_by_period': dict(cnt),
               'names_pre_achaemenid': sum(1 for n in names if n['period'] in OLD),
               'names_period_unknown': cnt.get('?', 0)},
    'primary_set_rule': 'test uses ALL names except period Achaemenid/RAE/N/LB/NA/SB (mostly non-Elamite or foreign-context); elements: all with >=2 vowels',
    'names': names,
    'elements': elements,
}
json.dump(out, open(os.path.join(DATA, 'elamite_name_list_frozen.json'), 'w'), ensure_ascii=False, indent=0)
print('Elamite:', out['counts'])

# ---------------------------------------------------------------- control onomasticon
ctrl = collections.Counter()
lang = None
cur = None
for line in open(atf_p, encoding='utf-8', errors='replace'):
    if line.startswith('&P'):
        cur = line[1:8]; lang = None
        continue
    if line.startswith('#atf: lang'):
        parts = line.split()
        lang = parts[2] if len(parts) > 2 else None
        continue
    if lang not in ('akk', 'sux') or cur in per or not re.match(r"^\d+'?\.", line):
        continue
    for w in re.findall(r'\{(?:disz|m)\}([^\s]+)', line):
        w2 = re.sub(r'[#?!\[\]<>()]', '', w)
        if re.search(r'[A-Z_]|\.\.\.|\bx\b|\{(?!d\})', w2):
            continue
        form = clean_syl(re.sub(r'\{d\}', '-', w2))
        if SYL_OK.match(form) and '-' in form:
            ctrl[(form, lang)] += 1
eln = {n['form'] for n in names}
cl = [{'form': f, 'lang': l, 'n': c} for (f, l), c in ctrl.items() if f not in eln]
json.dump({'_frozen': out['_frozen'],
           '_note': 'Non-Elamite control onomasticon: syllabically spelled {disz}/{m} personal names in CDLI Akkadian and Sumerian texts (Elamite-language texts excluded; exact Elamite-list forms removed). Some foreign (incl. Elamite) names in Assyrian letters may remain.',
           'count': len(cl), 'by_lang': dict(collections.Counter(x['lang'] for x in cl)), 'names': cl},
          open(os.path.join(DATA, 'control_name_list_frozen.json'), 'w'), ensure_ascii=False)
print('control:', len(cl))
