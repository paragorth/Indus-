#!/usr/bin/env python3
"""LA-27 data: published masses (grams) of Aegean Bronze Age balance weights and ingots.

Masses only; no interpretation is taken over. Sources (open access, read 4 Oct 2026):
 MI90  Michailidou, 'The lead weights from Akrotiri: the archaeological record', IGRA 1990,
       Table I: West House lead discs, mass computed from cleaned diameter and height
       (m = v * 11.35 g/cm3). helios.eie.gr/helios/bitstream/10442/8467
 MIWC  Michailidou, 'Stone balance weights? The evidence from Akrotiri' (Weights in Context
       2006), weighed masses cited in the text. helios.eie.gr/.../10442/8480
 MIME  Michailidou, 'On the Minoan economy' (Knossos and Akrotiri items cited with masses).
       helios.eie.gr/.../10442/8485
 DZ98  de Zwarte, Talanta 30-31 (1998-99), Table 1 (Akrotiri stone set, after Hiller von
       Gaertringen/Lehmann 1901), Table 5 (Akrotiri lead set, inv. 648, 1298-1303, masses
       from Petruso 1978), Table 10 (Katsambas, LM IIIA2).
 CTX   CONTEXT.md: typical oxhide ingot (HT, Zakros, Tylissos) 20-30 kg, mean c. 29 kg.
Flags: core = Neopalatial (MM III-LM I) Aegean context, worked as a weight; approx = heavy
pieces only given to the nearest kg or half kg; restored = original mass restored by author.
"""
W = []
def add(src, site, mats, mass, core=True, note=''):
    W.append(dict(src=src, site=site, mat=mats, g=float(mass), core=core, note=note))

for m in [16.087, 33.002, 48.127, 42.824, 65.507, 85.56, 120.78, 192.79, 207.91, 260.61,
          342.59, 360.07, 754.35, 1027, 1593.2, 3144.3, 4444.1, 6352.8]:
    add('MI90', 'Akrotiri West House', 'lead', m, note='computed from dimensions')
for m in [105, 139, 175, 212, 320, 425, 535, 840, 956, 1167, 1288]:
    add('DZ98-T1', 'Akrotiri (old collection)', 'stone', m)
for m in [65.0, 88.1, 216.0, 704.6, 1021.2, 1162.2, 1408.6]:
    add('DZ98-T5', 'Akrotiri Lilies room', 'lead', m)
for m in [5.7, 10.2, 10.5, 28.0, 48.9]:
    add('DZ98-T10', 'Katsambas', 'stone', m, core=False, note='LM IIIA2')
# Akrotiri stones and leads cited with masses (MIWC); natural pebbles and game pieces flagged
for m, n, c in [(478, 'sphendonoid', True), (21.9, 'cylinder', True), (62.1, 'marble disc', True),
                (52.5, 'lead', True), (20.2, 'disc No 370', True), (20.2, 'disc No 1971', True),
                (66.5, 'elongated stone', True), (1028, 'marble sphere', True),
                (272.11, 'restored', True), (114.7, 'No 1754', True), (11.6, 'No 1767', True),
                (8.1, 'disc No 1804', True), (167.8, 'stone', True), (168.3, 'No 3832', True),
                (26, 'marble No 1740', True), (46.8, 'marble disc', True), (59.2, 'No 4324', True),
                (268.5, 'stone', True), (151.1, 'No 1446 Potamos', True), (18.5, 'No 3830', True),
                (6.2, 'marble No 3774', True), (101.7, 'lead', True), (21.9, 'lead No 9196', True),
                (3.3, 'gypsum cylinder', True), (817, 'House of Ladies', True),
                (11.5, 'sphere', True), (11.5, 'sphere', True), (11.5, 'disc', True),
                (39, 'natural pebble', False), (35, 'natural pebble', False), (1, 'pebble', False),
                (3.2, 'pebble No 3004', False), (91, 'game token?', False)]:
    add('MIWC', 'Akrotiri', 'stone/lead', m, core=c, note=n)
for m, s, n in [(59, 'Knossos', 'frustum'), (62.26, 'Knossos', 'frustum with dot'),
                (19, 'Knossos', 'alabaster disc with dot'), (12.6, 'Knossos', 'Evans no.'),
                (204, 'Knossos', 'three holes'), (1421, 'Zakros', 'stone'),
                (378, 'Miletus', 'marble disc, six circles'), (480, 'Akrotiri', 'stone (MI90 text)'),
                (1050, 'Akrotiri', 'stone (MI90 text)'), (720, 'Akrotiri', 'stone (MI90 text)')]:
    add('MIME', s, 'stone', m, note=n)
for m in [15000]:  # 3, 4.5, 6, 5.5 kg duplicate MI90 discs
    add('MI90/MIWC', 'Akrotiri', 'lead', m, note='approx heavy piece')
add('CTX', 'HT/Zakros/Tylissos', 'copper ingot', 29000, note='typical oxhide ingot')

if __name__ == '__main__':
    import json, os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'la27_weights.json')
    json.dump(W, open(out, 'w'), indent=0)
    print(len(W), 'masses;', sum(w['core'] for w in W), 'core')
