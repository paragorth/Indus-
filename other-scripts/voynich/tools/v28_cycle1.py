"""v28 cycle 1: medieval handwritten Latin-script manuscripts transcribed at the level of graphic units.
Writes loops/v28_cycle1.txt from cached results (run first: python3 v28_run.py <corpus keys>)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v28_report as RP, v28_lib as X, v25_lib as L

HANDS = [('Sev_Z', 'Castilian hand Sev_Z'), ('Mad_A-Hand_A', 'Castilian Mad_A hand A'), ('Val_S-Hand_C', 'Castilian Val_S hand C'), ('Val_S-Hand_D', 'Castilian Val_S hand D')]
LEV = [('allo', 'ALLOGRAPHETIC (long s/round s, r rotunda, uncial d, insular t, z/ezh, u/v, i/j, capitals, tironian et, per/pro marks, every combining abbreviation mark a separate unit)'),
       ('graph', 'GRAPHEMIC + MARKS (allographs collapsed; abbreviation and combining marks kept as units)'),
       ('plain', 'PLAIN (allographs collapsed, combining marks dropped)'),
       ('minim', 'MINIM (allographetic, and m n u i split into minim strokes, like EVA i-strings)')]
rows = [RP.row('V-28.1.0a', 'voy:ZL3b:v25', 'REPRODUCTION of v25 on Voynich ZL with the v28 code'),
        RP.row('V-28.1.0b', 'latinprint', 'Printed Latin (v25 corpus) under FreeSerif, Junicode, Unifont')]
k = 1
rows.append(RP.row(f'V-28.1.{k}', 'cast:pool:allo', 'Late-medieval Castilian, all hands pooled, allographetic (Zenodo 8406222)')); k += 1
for h, hl in HANDS:
    for lv, ll in LEV:
        rows.append(RP.row(f'V-28.1.{k}', f'cast:{h}:{lv}', f'{hl}: {ll}')); k += 1
for lv, ll in LEV[:3]:
    rows.append(RP.row(f'V-28.1.{k}', f'enhg:{lv}', f'Early New High German 15th-c. formulary, near-allographetic (Zenodo 21257167, 92k tokens = whole set): {ll}')); k += 1
rows.append(RP.row(f'V-28.1.{k}', 'catmus:allo', 'CATMuS-medieval Latin, 7 MSS 12th-15th c. (graphemic: allographs merged by the editors, but capitals, tironian et, per/pro/con/us marks and every combining abbreviation mark kept as units)')); k += 1
rows.append(RP.row(f'V-28.1.{k}', 'catmus:plain', 'CATMuS Latin with combining marks dropped and capitals lowered (precomposed abbreviation letters kept)')); k += 1
hdr = ('# v28 cycle 1 - medieval handwritten manuscripts transcribed as graphic units (try to kill v25).\n'
       '# Same code path as v25 (v25_lib behaviour + Mantel; v25_image descriptors); glyph images rendered from Junicode, FreeSerif, Unifont (MUFI coverage). r reported per font x behaviour model.\n'
       '# Voynich reference (v25, reproduced V-28.1.0a): image combo ppmi +0.28, svd +0.30, potts +0.43.\n'
       '| row | method and control | result | verdict |\n|---|---|---|---|')
L.write_rows(os.path.join(X.LOOPS, 'v28_cycle1.txt'), rows, hdr)
print(open(os.path.join(X.LOOPS, 'v28_cycle1.txt')).read()[:3000])
