"""pe33: per-tablet seal-motif codes (data only, from what the impression depicts).

Source of motifs: Legrain 1921, Empreintes de cachets elamites (MDP 16), catalogue entries for
figures 1-339 (archive.org item mmoires16franuoft, OCR checked against page images).
Links tablet -> figure:
  (a) CDLI ATF seal notes 'seal 1 = PESnnnn': for nnnn <= 339 PES = Legrain figure number
      (verified: 29/29 'Scheil, N' entries of Legrain's Table des figures that are in our corpus
      are seal-flagged MDP 17, N tablets, and every one with a CDLI PES id <= 339 has PES = fig).
  (b) Legrain's Table des figures 'Scheil, N' = MDP 17, N (tablets then in press).
  (c) SENSITIVITY ONLY: 'AO 512, n N' read as MDP 06, N (2 of 3 checkable cases disagree with CDLI).
Feature codes describe depicted content only (no interpretation of the scene, no sign readings).
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'pe33_seal_motifs.json')

# figure -> (short description of depicted content, feature set)
FIG = {
 36: ('circular palmettes separated by long rectangles', {'GEOM'}),
 52: ('fish, net and proto-Elamite signs', {'WATER'}),
 53: ('fish swimming', {'WATER'}),
 88: ('frieze of large bulls; small calf or goat', {'BOVID'}),
 89: ('same pastoral scene: bull raising head', {'BOVID'}),
 93: ('bulls butting, calves leaping', {'BOVID'}),
 94: ('bull charging above two facing bulls; branch', {'BOVID'}),
 96: ('bulls walking in frieze', {'BOVID'}),
 97: ('kneeling long-haired bovids before mountain with conifer; bovid and antelope in field', {'BOVID', 'CAPRID'}),
 98: ('frieze of bovids (bison)', {'BOVID'}),
 104: ('bulls lying in pasture; branches', {'BOVID'}),
 105: ('bulls lying', {'BOVID'}),
 110: ('bulls (fragment)', {'BOVID'}),
 125: ('large male goat; shrub', {'CAPRID'}),
 143: ('cypress between two ibexes(?)', {'CAPRID'}),
 147: ('ibexes standing and lying', {'CAPRID'}),
 152: ('head of deer(?)', {'CAPRID'}),
 154: ('mouflon', {'CAPRID'}),
 155: ('wild sheep(?)', {'CAPRID'}),
 159: ('lion hunting ibexes and antelopes', {'FELINE', 'CAPRID', 'PREDATION'}),
 162: ('lion threatening bull (same seal as 161/168)', {'FELINE', 'BOVID', 'PREDATION'}),
 163: ('bull pursued by lions', {'FELINE', 'BOVID', 'PREDATION'}),
 164: ('wild bull pursued by eagle and lion', {'FELINE', 'BOVID', 'BIRD', 'PREDATION'}),
 168: ('replica of 161: lion and bull', {'FELINE', 'BOVID', 'PREDATION'}),
 173: ('lion and ibex', {'FELINE', 'CAPRID', 'PREDATION'}),
 182: ('symmetric lions and conifers', {'FELINE'}),
 196: ('ibex seized by two canids; second ibex escaping', {'CAPRID', 'CANID', 'PREDATION'}),
 198: ('two bulls, a goat, three small human figures with vessels, milk vessel and pail, a lion', {'BOVID', 'CAPRID', 'HUMAN', 'VESSEL', 'FELINE'}),
 222: ('two domed storage buildings on wooden frame, man on ladder carrying load, crouching figure', {'BUILDING', 'HUMAN'}),
 223: ('replica of 222', {'BUILDING', 'HUMAN'}),
 258: ('bulls in human postures: seated bull, bull holding bow and arrows', {'BOVID', 'ANTHRO'}),
 259: ('bull in human posture', {'BOVID', 'ANTHRO'}),
 265: ('two seated felines holding long cords', {'FELINE', 'ANTHRO'}),
 266: ('frieze of lions standing like men, paws raised; mountains and cypress', {'FELINE', 'ANTHRO'}),
 267: ('same: lions standing like men, with two seated bulls', {'FELINE', 'ANTHRO', 'BOVID'}),
 268: ('lions (or large dogs) holding hairy triangles in their paws', {'FELINE', 'ANTHRO'}),
 270: ('two facing composite beasts: lion body, eagle wings', {'MONSTER'}),
 281: ('two heads, bull and antelope, on a triangular base over a double volute', {'BOVID', 'CAPRID'}),
 314: ('bulls and mountains; a cross', {'BOVID'}),
 322: ('ibexes or antelopes; leaves; rosette of dots', {'CAPRID'}),
 325: ('lion and bull, head turned, mountain landscape', {'FELINE', 'BOVID', 'PREDATION'}),
 329: ('frieze of composite beasts: griffin and long-horned antelope with bull tail; snake, branch', {'MONSTER', 'CAPRID'}),
 330: ('bull between two lions and lion between two bulls; hairy triangle', {'FELINE', 'BOVID'}),
 333: ('three felines kneeling in boats holding oars; reeds', {'FELINE', 'ANTHRO', 'WATER'}),
 334: ('bovid in a boat with oars, nets, a fish; reeds', {'BOVID', 'ANTHRO', 'WATER'}),
 336: ('bovids (bison) standing between mountains and crosses, wearing belts', {'BOVID', 'ANTHRO'}),
 265.1: None,
}
FIG.pop(265.1)
# Legrain Table des figures: 'Scheil, N' (MDP 17, N) -> figure
SCHEIL = {36: [38], 52: [57], 89: [33], 93: [7], 96: [246], 97: [417], 98: [427], 104: [13],
          105: [209], 110: [465], 125: [16], 143: [439], 147: [444], 152: [256], 154: [468],
          155: [437], 162: [474], 163: [10], 164: [239], 173: [4], 182: [431], 196: [321],
          223: [191], 258: [322], 259: [257], 267: [44], 268: [461], 270: [2], 281: [421]}
# 'AO 512, n N' -> figure (sensitivity only)
AO512 = {53: [212], 88: [395], 94: [218], 168: [288], 182: [213, 318], 265: [206], 266: [229],
         314: [204, 225], 159: [378]}


def build():
    import re, sys
    sys.path.insert(0, HERE)
    meta = json.load(open(os.path.join(HERE, '..', 'data', 'pe8_meta.json')))
    by = {(v['pub_vol'], v['pub_num']): p for p, v in meta.items()}
    raw = open(os.path.join(HERE, '..', 'data', 'pe_raw.atf'), encoding='utf-8').read()
    pes = {}
    for b in re.split(r'\n(?=&P)', raw):
        m = re.match(r'&(P\d+)', b)
        if m:
            ids = sorted({int(x) for x in re.findall(r'PES0*(\d+)', b, re.I)})
            if ids:
                pes[m.group(1)] = ids
    rows = {}

    def add(p, fig, how, primary):
        r = rows.setdefault(p, {'id': p, 'designation': meta[p]['designation'], 'figs': [], 'links': [],
                                'features': set(), 'primary': False, 'pes_all': pes.get(p, [])})
        if fig not in r['figs']:
            r['figs'].append(fig)
            r['features'] |= FIG[fig][1]
        r['links'].append(how)
        r['primary'] |= primary
    for p, ids in pes.items():
        for i in ids:
            if i in FIG:
                add(p, i, 'CDLI PES%04d = MDP 16 fig. %d' % (i, i), True)
    for f, L in SCHEIL.items():
        for n in L:
            p = by.get(('MDP 17', n))
            if p:
                add(p, f, "MDP 16 Table des figures: fig. %d 'Scheil, %d'" % (f, n), True)
    for f, L in AO512.items():
        for n in L:
            p = by.get(('MDP 06', n))
            if p and not (p in rows and rows[p]['primary']):
                add(p, f, "MDP 16 Table des figures: fig. %d 'AO 512, n %d' read as MDP 06 %d (unverified)" % (f, n, n), False)
    out = []
    for p, r in sorted(rows.items()):
        r['features'] = sorted(r['features'])
        r['describes'] = [FIG[f][0] for f in r['figs']]
        r['unidentified_other_seals'] = [i for i in r['pes_all'] if i not in FIG]
        out.append(r)
    doc = {'source': 'L. Legrain 1921, Empreintes de cachets elamites, MDP 16 (Paris: Leroux); archive.org mmoires16franuoft. '
                     'Seal ids PESnnnn from CDLI ATF seal notes. Codes describe depicted content only.',
           'features': ['BOVID', 'CAPRID', 'FELINE', 'CANID', 'BIRD', 'MONSTER', 'HUMAN', 'ANTHRO', 'BUILDING', 'VESSEL', 'WATER', 'GEOM', 'PREDATION'],
           'tablets': out}
    json.dump(doc, open(OUT, 'w'), indent=1)
    return doc


if __name__ == '__main__':
    d = build()
    T = d['tablets']
    print(len(T), 'coded;', sum(t['primary'] for t in T), 'primary')
    import collections
    c = collections.Counter(f for t in T if t['primary'] for f in t['features'])
    print(c.most_common())
    for t in T:
        print(t['id'], t['designation'], t['figs'], t['features'], 'P' if t['primary'] else 's', t['unidentified_other_seals'])
