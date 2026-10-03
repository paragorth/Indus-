"""Loop 71 (S-DARK-71): rebuilding the Lothal warehouse sealing archive.

Shared data layer for the loop-71 cycles.
Sources joined per sealing (CISI number L-124 ... L-216, the 93 sealings of Frenez & Tosi 2005):
  FT   Frenez & Tosi 2005, Table 1 (number of impressions, fastening/back type, context W/W1/W2/O/?), already
       transcribed in data/derived/lothal-sealings-frenez-tosi2005.json; plus statements in their text
       (pp. 73-84) about individual sealings, coded below in FT_NOTES.
  RAO  Rao 1985 (Lothal vol. II, MASI 78, archive.org scan), 'Plates - Sealings' pp. 325-327: per plate the
       antiquity number, number of impressions, emblem, reverse description, phase, find-spot and depth;
       transcribed by eye below in RAO_PLATES.
  W    Wells (data/raw/inscriptions.csv): faces n.k, excavation id (antiquity no. + Rao plate), find-spot.
  IM77 Mahadevan 1977 (data/im77): Lothal sealings 7201-7283, sides; linked to L numbers only where the
       loop-24 aligner accepted the pair (otherwise unlinked; the IM77 network is analysed on its own).
Signs: Wells numbers in reading order (raw tokens reversed), 0 = lost sign kept as a wildcard for die matching.
"""
import csv, json, re, collections, sys
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import ROOT, RAW, merge_maps, load_faces, load_im77, OPEN, pval

FTJ = ROOT + '/data/derived/lothal-sealings-frenez-tosi2005.json'
OUT = ROOT + '/data/derived/dark/'
CAT = OUT + 'loop71_lothal_sealings.csv'
L24 = OUT + 'loop24_pairs.json'

# ---- Rao 1985, pp. 325-327 (Plates - Sealings). antiquity no. -> (plate, n_imp text, emblem, reverse, phase, findspot, depth)
RAO_PLATES = [
    ('CLXI-E', 13191, '1', 'unicorn horn + signs', 'interlacing reed-marks and textile impression', 'IV', 'SRG 3, L 5, layer 8', "0'10\""),
    ('CLXI-F', 1830, '1', 'elephant walking + signs', 'groove', 'II-III', 'SRG 3, GX 10, warehouse, layer 2', "2'0\""),
    ('CLXII-A1', 1292, '2 (same seal)', 'swastika (multiple lines)', 'plain', 'III', 'SRG 2, B 6, layer 11', "5'6\""),
    ('CLXII-A2', 1833, '1', 'compartmental geometric design', 'deep reed-mark', 'III', 'SRG 3, GX 10, layer 2', "2'5\""),
    ('CLXII-B', 1831, '1', 'unicorn + fire altar + signs', 'deep reed and cord marks', 'III', 'SRG 3, GX 10, layer 2', "2'5\""),
    ('CLXII-C1', 1984, '2 (same seal)', 'unicorn + fire altar + sign', 'hollow depression', 'III', 'unstratified, GX 10, layer 2', "2'5\""),
    ('CLXII-C2', 722, '2 (same seal)', 'unicorn + fire altar + signs', 'reed-mark', 'III', 'SRG 2, B 2, layer 3', "1'8\""),
    ('CLXII-D', 2077, '2 (two seals)', 'signs (faint)', 'reed-mark', 'III', 'SRG 3, GX 10, layer 2', "3'4\""),
    ('CLXII-E', 1888, '3 (Rao: of a seal)', 'unicorn + fire altar + signs', 'deep reed-mark', 'III', 'SRG 3, GX 10, layer 2', "3'4\""),
    ('CLXII-F', 14586, '1', 'Maltese cross in geometric frame', 'reed mark', 'IV', 'SRG 2, GX 10, unstratified', "3'4\""),
    ('CLXIII-A1', 1877, '1', 'unicorn + fire altar + signs (faint)', 'reed mark', 'III', 'SRG 3, GX 10, layer 2', "2'6\""),
    ('CLXIII-A2', 1879, '1', 'bull with manger + signs', 'three deep reed marks', 'III', 'SRG 3, GX 10, layer 2', "2'6\""),
    ('CLXIII-A3', 1898, '1', 'unicorn + fire altar + signs', 'broad reed marks', 'III', 'SRG 3, GX 10, layer 2', "3'4\""),
    ('CLXIII-A4', 1854, '1', 'unicorn + fire altar + signs', 'broad reed marks', 'III', 'SRG 3, GX 10, layer 2', "2'6\""),
    ('CLXIII-B1', 1880, '1', 'unicorn horn + signs', 'broad reed mark', 'II-III', 'SRG 3, GX 10, layer 2', "3'1\""),
    ('CLXIII-B2', 1926, '1', 'unicorn (partial) + signs', 'reed marks', 'II-III', 'SRG 3, GX 10, layer 2', "2'6\""),
    ('CLXIII-B3', 3282, '1', 'unicorn head (partial) + signs', 'stick mark', 'IV', 'SRG 2, E 2, layer 4', "2'0\""),
    ('CLXIII-B4', 1876, '1', 'unicorn head + signs', 'plain', 'II-III', 'SRG 3, GX 10, layer 2', "2'6\""),
    ('CLXIII-C1', 1855, '1', 'fire altar', 'plain', 'III', 'SRG 3, GX 10, layer 3', "3'2\""),
    ('CLXIII-C2', 1835, '1', 'hind part of bull', 'reed mark', 'III', 'SRG 3, GX 10, layer 3', "2'0\""),
    ('CLXIII-C3', 1837, '1', 'hind part of bull', 'reed marks', 'III', 'SRG 3, GX 10, layer 3', "2'0\""),
    ('CLXIII-D1', 3694, '3 (Rao: of a seal)', 'signs', 'deep reed marks', 'III', 'SRG 2, B 1y, layer 3', "1'8\""),
    ('CLXIII-D2', 5242, '2 (Rao: of a seal)', 'signs', 'deep reed marks', 'III', 'SRG 2, E 2, layer 6', "3'0\""),
    ('CLXIII-D3', 1883, '2 (Rao: of a seal)', 'signs', 'reed mark', 'III', 'SRG 3, GX 10, layer 2', "2'6\""),
    ('CLXIII-E1', 1838, '1', 'signs (faint)', 'irregular', 'II-III', 'SRG 3, GX 10, layer 3', "2'9\""),
    ('CLXIII-E2', 1873, '1', 'signs (faint)', 'hollow reed mark', 'II-III', 'SRG 3, GX 10, layer 3', "3'4\""),
    ('CLXIII-E3', 1853, '1', 'signs', 'reed marks', 'II-III', 'SRG 3, GX 10, layer 2', "2'6\""),
    ('CLXIII-F', 1895, '4 (seals)', 'signs', 'deep reed mark', 'II-III', 'SRG 3, GX 10, layer 3', "3'4\""),
    ('CLXIII-G', 16845, '5 (seals)', 'signs (faint), recessed square all round', 'plain', 'IV', 'surface collection', ''),
    ('CLXIV-A', 13881, '1', 'unicorn + fire altar + signs', 'plain (sling ball?)', 'III', 'SRG 3, J 5, layer 4', "2'0\""),
    ('CLXIV-B', 13051, '1', 'composite animal + signs', 'flat pitted surface', 'III', 'SRG 2, BX 4, layer 2', "1'9\""),
    ('CLXIV-C', 800, '1', 'unicorn + signs; textile impression on obverse', 'flat, use-marks', 'III', 'SRG 2, A 1, pit sealed by layer 9', "4'6\""),
    ('CLXIV-E1', None, '5 on two facets', 'signs in four impressions, unicorn horn in one', 'cord and reed marks', 'III', 'SRG 3, GX 10', ''),
    ('CLXIV-E2', 1870, '3', 'signs + head of bull', 'deep reed marks', 'II-III', 'SRG 3, GX 10, layer 3', "3'4\""),
    ('CLXIV-F1', 15348, '1', 'unicorn horn + signs (faint)', 'flat', 'IV', 'SRG 2, surface find', ''),
    ('CLXIV-F2', 16912, '1', 'signs', 'irregular; unbaked', 'III', 'SRG 3, surface find', ''),
]

# ---- statements in the Frenez & Tosi 2005 text about single sealings (pp. 73-84) ----
FT_NOTES = {
    'L-206': 'peg-on-wall, 2 impressions; with the most recurrent seal (p.84)',
    'L-179': 'peg-on-wall (stone peg in wooden surface), 2 badly preserved impressions',
    'L-175': 'peg-on-wall; square geometric seal, grid pattern (cf. RJ-2 DA 8546)',
    'L-173': 'peg-on-wall; geometric seal, labyrinth cross-like motif',
    'L-154': 'peg-on-wall, stepped peg; flattened rope',
    'L-190': 'structures (between two parallel sticks); >= 3 different seals, the same 3 as on L-189; gaur impression',
    'L-189': 'structures; 4 seals on 2 faces: 3 unicorns + 1 gaur; the 3 seals of L-190 recur here',
    'L-191': 'gaur impression',
    'L-130': 'structures (corner recess); possible faint geometric impression on side',
    'L-145': 'structures; surface find; string (leather?) impression; removed while still plastic',
    'L-193': 'structures/lock: 3 superimposed impressions; with the most recurrent seal',
    'L-198': 'structures or locker; 2 superimposed impressions; with the most recurrent seal',
    'L-141': 'massive polygonal lump',
    'L-207': 'massive polygonal lump',
    'L-158': 'locker; single well-centred impression',
    'L-144': 'quadrangular locker with movable wooden stakes',
    'L-148': 'small locker, pointed peg impression',
    'L-149': 'locker on vertical plank; unicorn',
    'L-209': 'crescent lump; 2 parallel impressions of one inscribed miniature framed tablet (not a seal)',
    'L-146': 'two small canes + string + fine fabric; unicorn (cf. Umma sealing)',
    'L-143': 'flat; horizontal cane/reed bundle + cord',
    'L-142': 'plano-convex pill; small unicorn seal; 5 fingernail tallies; braided strings',
    'L-211': '3 overlapped impressions (only inscriptions kept visible); knotted strings at corner of polished wooden box',
    'L-125': 'single impression of the most recurrent seal; fingerprints over the unicorn',
    'L-124': 'small vessel closed with interwoven straw; the most recurrent seal alone',
    'L-126': 'leather sack; the most recurrent seal alone',
    'L-174': 'outside warehouse; rim of small vessel (12 cm); 2 impressions of one geometric swastika-like seal',
    'L-204': 'wicker lid; 2 impressions (of the same large unicorn seal?)',
    'L-208': 'directly on a rope; >= 2 overlapped impressions; outside warehouse',
    'L-151': 'thin square sheet on a soft material; worn unicorn seal 2.5 cm',
    'L-195': 'pottery; with the most recurrent seal',
    'L-197': 'pottery; with the most recurrent seal',
}
for k in range(161, 173):
    FT_NOTES['L-%d' % k] = 'elephant seal (single impression); knotted strings on polished wooden box' + (
        '; strings round the edge' if k in (163, 167) else '') + ('; no fingernail tallies' if k in (165, 170) else '; fingernail tallies')
# FT p.84: the most recurrent seal (16 impressions) is named on these sealings
FT_HUB = ['L-124', 'L-125', 'L-126', 'L-195', 'L-197', 'L-189', 'L-190', 'L-206', 'L-193', 'L-198']
# FT p.83 summary counts
FT_SUMMARY = dict(sealings=93, impressions=130, imps_single=63, imps_multi=67, recurrent_seals=13, recurrent_imps=76,
                  seals_max=67, seals_min=57, warehouse_seals='25-30', hub_imps=16, elephant_imps=12)

HUB = (705, 500, 741, 1, 55, 220, 740, 90)
ELE = (817, 2, 48, 740)


def antiq(exid):
    """antiquity (register) number from the Wells excavation id: '01853(T027) CLXIII-E3' -> 1853"""
    m = re.match(r'\s*0*(\d{2,5})', exid or '')
    if not m: return None
    n = int(m.group(1))
    if exid.startswith('01891179'): n = 1891          # '01891179' = 1891 + stray digits
    if exid.startswith('1980/5C'): return None        # a museum/accession style number, not Rao's register
    return n


def plate(exid):
    m = re.search(r'(CL[XVI]+-?[A-G]\d?)', exid or '')
    return m.group(1).replace('CLXIV', 'CLXIV').replace('CLXIII', 'CLXIII') if m else ''


def wells_lothal(level='seq_raw'):
    """CISI -> list of Wells rows (faces) for Lothal TAG objects; tokens in reading order (0 kept)."""
    mp = merge_maps()[level]
    W = collections.OrderedDict()
    for r in csv.DictReader(open(RAW)):
        if r['site'] != 'Lothal' or not r['type'].startswith('TAG'): continue
        toks = [int(t) for t in re.findall(r'\d{3}', r['text'])][::-1]
        toks = [0 if t in (0, 999) else mp.get(t, t) for t in toks]
        key = r['cisi'] if r['cisi'] not in ('-', '') else 'W' + r['id'].split('.')[0]
        W.setdefault(key, []).append(dict(id=r['id'], toks=toks, text=r['text'], symbol=r['symbol'], type=r['type'],
                                          area=r['area-section'], exid=r['excavation-idno'], layer=r['period'],
                                          phase=r['phase'], depth=r['depth'], pres=r['preservation'],
                                          complete=r['complete'] == 'Y'))
    return W


# ---- die (seal identity) matching with 0 as wildcard ----
def fits(a, b):
    """impression a (tokens, 0 wildcard) fits inside text b at some offset; >= 2 non-zero tokens must agree."""
    na = [x for x in a if x]
    if len(na) < 2: return False
    # trim leading/trailing zeros of a (lost edges) so a fragment can sit anywhere inside b
    i, j = 0, len(a)
    while i < j and a[i] == 0: i += 1
    while j > i and a[j - 1] == 0: j -= 1
    core = a[i:j]
    for off in range(len(b) - len(core) + 1):
        ok = True; agree = 0
        for x, y in zip(core, b[off:off + len(core)]):
            if x and y and x != y: ok = False; break
            if x and y: agree += 1
        if ok and agree >= 2: return True
    return False


def overhang(a, b, kmin=3):
    """a (with lost edges) overlaps b with a's core running past one end of b; >= kmin agreeing signs, no conflict."""
    i, j = 0, len(a)
    while i < j and a[i] == 0: i += 1
    while j > i and a[j - 1] == 0: j -= 1
    core = a[i:j]
    for off in range(-len(core) + 1, len(b)):
        agree = 0; ok = True; inside = True
        for p, x in enumerate(core):
            q = off + p
            if q < 0 or q >= len(b): inside = False; continue
            if x and b[q] and x != b[q]: ok = False; break
            if x and b[q]: agree += 1
        if ok and not inside and agree >= kmin: return True
    return False


def assign_dies(imps, allow_overhang=True):
    """imps: list of (key, toks). Returns key -> die label. A die is founded by a 'full' text (no 0, >= 3 signs);
    each impression is read as the founder it fits uniquely (contained with >= 2 agreeing signs, or - for broken
    fragments - overlapping one end with >= 3 agreeing signs). Founders contained in a longer founder are merged.
    Impressions fitting no founder or several stay their own die ('u:<text>') -- conservative."""
    full = sorted({tuple(t) for k, t in imps if t and 0 not in t and len(t) >= 3}, key=len, reverse=True)
    rep = {}
    for t in full:
        hosts = [u for u in full if len(u) > len(t) and fits(list(t), list(u))]
        if len(hosts) == 1: rep[t] = hosts[0]
    for t in list(rep):
        while rep[t] in rep: rep[t] = rep[rep[t]]
    founders = [t for t in full if t not in rep]
    lab = {}
    for k, t in imps:
        tt = tuple(t)
        if tt in rep: lab[k] = rep[tt]; continue
        if tt in founders: lab[k] = tt; continue
        c = [u for u in founders if fits(list(t), list(u))]
        if not c and allow_overhang: c = [u for u in founders if overhang(list(t), list(u))]
        lab[k] = c[0] if len(c) == 1 else ('u',) + tt
    return lab


def fmt(t):
    if not t: return '(none)'
    if t[0] == 'u': return 'u:' + '-'.join(map(str, t[1:]))
    return '-'.join(map(str, t))
