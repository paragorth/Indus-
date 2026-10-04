"""v28 cycle 2: other scripts (Ethiopic abugida as near-featural positive control; Cyrillic, OCS Cyrillic vs the same
text in Glagolitic, Armenian), Voynich under different EVA segmentations, and BPE-merged multi-letter units in printed
Latin (does segmentation into composite units create the effect?).  Writes loops/v28_cycle2.txt from cached results."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v28_report as RP, v28_lib as X, v25_lib as L

SETS = [('am', 'ETHIOPIC (Amharic Wikipedia), every fidel syllable a unit; Noto Sans/Serif Ethiopic, FreeSerif; hand = featural decomposition from the Unicode layout (consonant row + vowel order); near-featural POSITIVE control'),
        ('ti', 'ETHIOPIC (Tigrinya Wikipedia), replication of the abugida control'),
        ('ru', 'Russian Cyrillic (Wikipedia), FreeSerif / DejaVu Serif / Unifont'),
        ('cu', 'Old Church Slavonic in CYRILLIC (cu Wikipedia; iotated and ligature letters kept: ꙗ ѥ ѩ ѭ ю ꙑ)'),
        ('glag', 'the SAME OCS text mapped letter-for-letter into GLAGOLITIC (behaviour fixed, shape system changed); Noto Sans Glagolitic, FreeSerif, Unifont'),
        ('hy', 'Armenian (Wikipedia), Noto Sans/Serif Armenian, FreeSerif'),
        ('voy:ZL3b:split', 'Voynich ZL, every EVA character a unit (benches split: ch = c+h, sh = s+h, cth = c+t+h)'),
        ('voy:IT2a:split', 'Voynich IT, every EVA character a unit'),
        ('voy:ZL3b:istr', 'Voynich ZL, i-runs (+n/r/l/m) and e-runs as single units (iin, ee ...)'),
        ('voy:IT2a:istr', 'Voynich IT, i-runs and e-runs as units'),
        ('voy:ZL3b:benchsplit', 'Voynich ZL, benched gallows split into ch + gallows'),
        ('voy:IT2a:v25', 'Voynich IT, v25 units (replication)'),
        ('voy:ZL3b:split+bpe6', 'Voynich ZL split, then 6 greedy BPE merges (frequent chunks as units)'),
        ('latinprint+bpe10', 'Printed Latin with 10 BPE merges (multi-letter units such as qu, us, er rendered as strings)'),
        ('latinprint+bpe25', 'Printed Latin with 25 BPE merges')]
rows = [RP.row(f'V-28.2.{k}', n, lab) for k, (n, lab) in enumerate(SETS, 1)]
hdr = ('# v28 cycle 2 - other scripts, Voynich segmentations, composite-unit (BPE) control. Same v25 pipeline.\n'
       '| row | method and control | result | verdict |\n|---|---|---|---|')
L.write_rows(os.path.join(X.LOOPS, 'v28_cycle2.txt'), rows, hdr)
