"""Loop 80 cycle 1: tabulate every CDLI text that mentions Meluhha.
Input: scratchpad loop80_hits.json (from tools/dark_loop80_extract.py).
Third-millennium and Early OB administrative/legal entries are annotated by hand from the ATF
(persons, profession, goods, quantity, unit); later texts are classed by genre automatically.
Output: data/derived/dark/loop80_meluhha.csv
Usage: python3 tools/dark_loop80_c1.py HITS.json"""
import json, csv, sys, re
from collections import Counter

# P: (category, persons, relation_to_Meluhha, profession, goods, quantity, unit, note)
# category: PERSON-ETHNIC (lu2 me-luh-ha = 'man of Meluhha'), PERSON-PATRONYM (dumu me-luh-ha),
# PERSON-NAMED (personal name 'Meluhha'), SHIP (ma2 me-luh-ha crew), VILLAGE (e2-duru5 / i3-dub me-luh-ha, Girsu),
# GOOD (X me-luh-ha), TRAVEL (going to Meluhha), ROYAL, UNCLEAR
A = {
 'P453801': ('PERSON-ETHNIC', 'na-na-sa3; sa6-ma-ar; a-li-a-hi (wife of sa6-ma-ar)', 'lu2 me-luh-ha{ki}-me', 'sipa a-dara4 (shepherds of bezoar goats), a-ru-a lugal', 'i3-gesz (sesame oil) ration', '1; 1; 1/2', 'sila3', 'foreign names na-na-sa3, sa6-ma-ar; wife has an Akkadian name'),
 'P454137': ('PERSON-ETHNIC', '(group, unnamed)', 'lu2 me-luh-ha-me', '', 'barley ration', '1(asz) 1(ban2) = 1 gur 10 sila (Irisagrig)', 'gur', 'in a list of musicians (nar), lamentation priests (gala), doorkeepers'),
 'P212982': ('PERSON-ETHNIC', 'lu2-sun2-zi-da', 'lu2 me-luh-ha', '', 'silver compensation for a broken tooth', '10', 'gin2 (shekel)', 'Sumerian name, 14 attestations in CDLI; pays to ur-tesz2 (Ur-Teš, Nisqum)'),
 'P253274': ('PERSON-ETHNIC', 'i3-li2-a-hi', 'lu2 me-luh', '', 'ration (list of royal soldiers aga3-us2 lugal)', '9', '(units, OAkk @c/@t numeral)', 'Akkadian name'),
 'P217616': ('PERSON-ETHNIC', '(group, unnamed)', 'me-luh-ha-me', '', 'dabin flour of Akkade', '3', 'ban2', 'Adab, OAkk'),
 'P108448': ('PERSON-PATRONYM', 'ur-{d}lamma', 'dumu me-luh-ha', '', 'wool', '41 2/3 ma-na; remainder 2 gu2 50 ma-na', 'ma-na', 'Girsu; Ur-Lamma son of Meluhha (Sumerian name)'),
 'P102519': ('PERSON-PATRONYM', 'ur-{d}lamma', 'dumu me-luh-ha', '', 'barley', '30', 'gur', ''),
 'P123177': ('PERSON-PATRONYM', 'ur-{d}lamma', 'dumu me-luh-ha', 'sealing official (kiszib3)', 'barley', '4(gesz2) 25 gur 1 barig 5 ban2 5 sila (= 265+ gur)', 'gur', ''),
 'P202783': ('PERSON-PATRONYM', 'ur-{d}lamma', 'dumu me-luh-ha', 'sealing official (kiszib3)', 'barley', '4(gesz2) 25 gur ...', 'gur', 'duplicate account of P123177'),
 'P136196': ('PERSON-PATRONYM', 'ur-{d}lamma', 'dumu me-luh-ha', 'sealing official (kiszib3)', 'barley', '1(gesz2) = 60 gur 2 barig 4 ban2', 'gur', ''),
 'P207485': ('PERSON-PATRONYM', 'ur-{d}lamma', 'dumu me-luh-ha', '', 'oil?', '3 ban2 4 sila', 'ban2', ''),
 'P318542': ('PERSON-PATRONYM', 'ur-{d}lamma', 'dumu me-luh-ha', 'a-ru-a (dedicated) worker', 'ration', '3 ban2 3 sila', 'ban2', ''),
 'P124739': ('PERSON-PATRONYM', 'ur-{d}ig-alim', 'dumu me-luh-ha', '', 'workers of the house of Nin-mar', '?', '', 'Girsu, Ninmar district (Guabba)'),
 'P135727': ('PERSON-PATRONYM', 'ma2-gur8-re', 'dumu me-luh-ha', 'erin2 e2 {d}nansze (worker of the Nanshe temple), clay-carrier', 'barley', '1(asz) 1(barig)', 'gur', ''),
 'P467759': ('PERSON-PATRONYM', 'lu2-mar-za', 'dumu me-luh-ha', '', '?', '2 1/2', '?', 'Ur, OAkk; Lu-marza is a common Sumerian name (12 CDLI)'),
 'P112283': ('PERSON-NAMED', 'me-luh-ha (son of ur-{d}na-ru2-a)', 'personal name', 'worker under foreman nam-mah-ni', '', '1', 'person', 'a Sumerian father names his son Meluhha'),
 'P110774': ('PERSON-NAMED', 'me-luh-ha (female worker)', 'personal name', 'geme2 (female worker), daughter of ...', 'barley ration', '3 ban2 3 sila', 'ban2', 'Girsu'),
 'P115261': ('PERSON-NAMED', 'me-luh-ha', 'personal name', 'ugula (foreman of 6 ARAD2 of Nanshe)', '', '6', 'workers', 'OAkk Girsu'),
 'P329097': ('PERSON-NAMED', 'me-luh-ha', 'personal name (in a list of persons each receiving 1 sheep)', '', 'sheep', '1', 'udu', 'Adab, OAkk'),
 'P326554': ('PERSON-NAMED', 'me-lu-ha', 'personal name (seized by szesz-szesz; legal)', '', 'silver', '10?', 'gin2', 'Adab, OAkk'),
 'P135796': ('UNCLEAR', 'e-lum-me-luh', 'possible personal name ending in -me-luh', 'lu2-[...]', 'messenger rations: beer, bread, oil, naga', '5; 5; 2; 2', 'sila3 / gin2', 'Umma; standard messenger rations'),
 'P212842': ('SHIP', 'da-di3', 'lu2-tukul ma2 me-luh-ha-ka (weapon-bearer of the Meluhha ship)', 'lu2 tukul (armed man)', 'oil', '1', 'sila3', 'Akkade period'),
 'P326408': ('SHIP', '(captain, unnamed)', 'nu-banda3 ma2 me-luh-ha', 'nu-banda3 (overseer)', 'sheep', '1', 'udu', 'Adab'),
 'P217502': ('SHIP', '(4 men)', 'gurusz ma2 me-luh-ha', '', 'barley ration', '5 (for 4 men)', 'gur', 'Adab'),
 'P323644': ('SHIP', '', 'ninda ma2 me-luh-ha', '', 'bread', '5', 'gur(?) (5(asz@c))', 'Adab'),
 'P382354': ('SHIP', 'giri3 gen-na', 'ma2 me-luh-ha-sze3 (for the Meluhha ship)', '', 'lard/oil', '10', 'sila3', 'Adab'),
 'P215605': ('SHIP', '', 'n x me-luh-ha / n sila3 ma2-bi', '', '?', 'n', 'sila3', 'Susa, OAkk'),
 'P320486': ('TRAVEL', '{d}utu-illat', 'sukkal going to Meluhha', 'sukkal (envoy)', 'garments', '2', 'tug2', 'Umma, Šulgi 41'),
 'P512784': ('TRAVEL', 'tu-ra-am-i3-li2? / nin9-kal-la', '[me]-luh-ha-ta (from Meluhha?)', '', 'aromatics', '5', 'gin2', 'Drehem; reading uncertain'),
 'P213104': ('PERSON-ETHNIC', 'ur-ab-ba', 'ra-gaba me-luh-ha (courier/rider of Meluhha)', 'ra-gaba (courier)', 'chair', '1', 'piece', 'Lagash II'),
 'P217715': ('GOOD', '', 'masz2 ga me-luh-ha', '', 'suckling kid of Meluhha', '1', 'animal', 'OAkk Girsu'),
 'P511999': ('UNCLEAR', '', 'me-luh-ha [x?]', '', 'oxen', '?', '', 'OAkk'),
 'P136689': ('GOOD', '', 'uruda me-luh-ha', '', 'copper of Meluhha', '6', 'ma-na', 'Ur'),
 'P136752': ('GOOD', 'ur-gu2-edin-na; a-hu-wa-qar', '{gesz}ab-ba me-luh-ha', 'craftsmen archive', 'ab-ba wood (handle)', '2', 'piece', 'Ur, IS 15'),
 'P136982': ('GOOD', 'amar-{d}iszkur; a-hu-wa-qar', '{gesz}dur2 {gesz}ab-ba me-luh-ha', '', 'ab-ba wood seat', '1', 'piece', 'Ur'),
 'P137023': ('GOOD', '', '{gesz}mu10-us2 ... me-luh-ha', '', 'ab-ba(?) wood fitting', '3', 'piece', 'Ur'),
 'P137076': ('GOOD', '', 'ab-ba me-luh-ha', '', 'ab-ba wood', '?', '', 'Ur'),
 'P137081': ('GOOD', 'dingir-su-ra-bi2; a-hu-wa-qar', 'dar{muszen} me-luh-ha', '', 'Meluhha bird figurine (ivory)', '1', 'piece', 'Ur'),
 'P137085': ('GOOD', 'dingir-su-ra-bi2; a-hu-wa-qar', 'dar me-luh-ha', '', 'Meluhha bird figurine (ivory)', '3', 'piece', 'Ur'),
 'P137088': ('GOOD', 'dingir-su-ra-bi2; a-hu-wa-qar', 'dar me-luh-ha', '', 'Meluhha bird figurine (ivory)', '1', 'piece', 'Ur'),
 'P137092': ('GOOD', 'dingir-su-ra-bi2; a-hu-wa-qar', 'dar me-luh-ha', '', 'Meluhha bird figurine (ivory)', '1', 'piece', 'Ur'),
 'P137094': ('GOOD', 'a-hu-wa-qar', 'dar me-luh-ha', '', 'Meluhha bird figurine (ivory)', '1', 'piece', 'Ur'),
 'P137142': ('GOOD', '{d}nanna-kam; a-hu-wa-qar', 'mes me-luh-ha', '', 'mes wood chair', '1', 'piece', 'Ur'),
 'P137152': ('GOOD', '', '{gesz}ab-ba me-luh-ha', '', 'ab-ba wood', '?', '', 'Ur'),
 'P137566': ('GOOD', '', 'mes me-luh-ha', '', 'mes wood bed', '1', 'piece', 'Ur'),
 'P137823': ('GOOD', '{d}nanna-kam; dingir-su-ra-bi2; ur-{d}ba-ba6', 'mes me-luh-ha / me-luh-ha{ki}', '', 'mes wood furniture', '1; 1', 'piece', 'Ur, IS 15'),
 'P140868': ('GOOD', '{d}ma-an-isz-di2-su (Maništušu cult)', '{gesz}gu-za {gesz}ab-ba saga me-luh-ha', '', 'ab-ba wood chair', '1', 'piece', 'Umma'),
 'P107404': ('GOOD', '', '{gesz}guzza {gesz}ab-ba me-luh-ha', '', 'ab-ba wood chair', '1', 'piece', 'Umma'),
 'P249043': ('GOOD', '', '{gesz}gu-za-nita {gesz}ab-ba me-luh-ha', '', 'ab-ba wood chair', '1', 'piece', 'Umma, ŠS 9'),
 'P283844': ('GOOD', '', 'kab-bi {gesz}ab-ba me-luh-ha', '', 'ab-ba wood (parts of chairs)', '2', 'piece', 'Girsu'),
 'P216994': ('GOOD', '', '{gesz}banszur me-luh-ha', '', 'Meluhha table', '1', 'piece', 'Lagash II'),
 'P236658': ('GOOD', 'an-ti-iri-nag', '{gesz}giri3-gub {gesz}ab-ba me-luh-ha', '', 'ab-ba wood footstool', '1', 'piece', 'Isin, Ishbi-Erra 14'),
 'P390701': ('GOOD', 'szu-{d}nin-kar-ak; lu2-{d}nin-szubur', '{gesz}giri3-gub {gesz}ab-ba me-luh-ha', '', 'ab-ba wood footstool', '1', 'piece', 'Isin, Ishbi-Erra 12'),
 'P130369': ('VILLAGE', '', '{gesz}kiri6 me-luh-ha {d}nin-mar{ki}-ka', '', 'orchard of Meluhha of Ninmar', '', '', 'Girsu, Guabba district'),
}
VILLAGE_P = ['P100892', 'P102031', 'P108484', 'P110238', 'P110574', 'P114609', 'P115266', 'P115571', 'P116633', 'P116735', 'P116995', 'P123010', 'P132876', 'P135969', 'P206053', 'P218242', 'P235705', 'P340534', 'P374962', 'P378169', 'P492732', 'P500263']
ROYAL_EARLY = ['P227514', 'P232275', 'P232277', 'P232300', 'P232301', 'P431881', 'P431882', 'P431884', 'P431886', 'P432309', 'P461937', 'P461938', 'P461954']

d = json.load(open(sys.argv[1]))
rows = []
for r in d:
    p = r['p']; per = r.get('period', ''); gen = r.get('genre', '')
    line = r['hits'][0]['line']
    if p in A:
        cat, pers, rel, prof, goods, q, unit, note = A[p]
    elif p in VILLAGE_P:
        m = re.search(r'([\d()a-z\'/ ]+?)\s*(?:sze\s*)?gur', ' '.join(r['hits'][0]['ctx']))
        cat, pers, rel, prof, goods, q, unit, note = 'VILLAGE', '', 'i3-dub / e2-duru5 me-luh-ha (granary / village of Meluhha, Girsu)', '', 'barley', '', 'gur', 'Girsu granary account'
    elif p in ROYAL_EARLY or 'Royal' in gen:
        cat, pers, rel, prof, goods, q, unit, note = 'ROYAL', '', 'royal inscription', '', '', '', '', ''
    elif 'Lexical' in gen:
        cat, pers, rel, prof, goods, q, unit, note = 'LEXICAL', '', 'word list', '', re.sub(r'\s+', ' ', line)[:60], '', '', ''
    elif 'Literary' in gen:
        cat, pers, rel, prof, goods, q, unit, note = 'LITERARY', '', '', '', '', '', '', ''
    elif 'Scientific' in gen or 'Mathem' in gen:
        cat, pers, rel, prof, goods, q, unit, note = 'SCIENTIFIC', '', 'stone lists / magic', '', 'gug me-luh-ha (carnelian)', '', '', ''
    else:
        cat, pers, rel, prof, goods, q, unit, note = 'OTHER-LATE', '', '', '', '', '', '', ''
    rows.append({'cdli_p': p, 'designation': r.get('designation', ''), 'period': per, 'provenience': r.get('provenience', ''),
                 'date': r.get('dates_referenced', ''), 'genre': gen, 'category': cat, 'persons': pers, 'relation': rel,
                 'profession': prof, 'goods': goods, 'quantity': q, 'unit': unit, 'note': note, 'meluhha_line': line})
out = 'data/derived/dark/loop80_meluhha.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), 'texts ->', out)
early = [x for x in rows if any(k in x['period'] for k in ('ED', 'Akkadian', 'Lagash', 'Ur III', 'Early Old'))]
print('3rd mill + Early OB:', len(early)); print(Counter(x['category'] for x in early).most_common())
print('all:', Counter(x['category'] for x in rows).most_common())
missing = [x['cdli_p'] for x in early if x['category'] in ('OTHER-LATE',)]
print('unannotated early:', missing)
