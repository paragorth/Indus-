"""v35 cycle 2: a 17th-c. cipher of Voynich size (Borg), planted homophonic ciphers (calibration), and size-matched /
mark-free re-tests of every corpus (v35_post.py).  Writes loops/v35_cycle2.txt from cached results."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v35_report as R, v28_lib as X, v25_lib as L

SETS = [
    ('borg', 'BORG CIPHER (17th c., Vatican Borg.lat.898; 27 symbols: Greek letters, zodiac and planet signs, digits; the closest in age and inventory size to the Voynich); SU/Uppsala transcription 0001r-0204v (scratch only), bracketed cleartext removed; glyphs rendered from Unicode as drawn in the project key (DejaVu Sans, FreeSerif); hand = v25 primitives', 'WEAK / MIXED: image ppmi +0.15-0.18 (p 0.03), svd +0.10-0.14, potts ~0; hand ~0; top-23 n.s. Below the Voynich on every model. Grade B (glyph images are modern renderings of the key, not the hand)'),
    ('plant_rand', 'PLANTED cipher, random design: printed Latin enciphered homophonically onto the 84 Copiale glyphs, homophones per letter in proportion to frequency, glyphs dealt at random', 'NULL as it must be (calibration)'),
    ('plant_family', 'PLANTED cipher, shape-family design: same, but each letter\'s homophones are a cluster of shape-similar glyphs (greedy on the v35 hand Jaccard). The "cipher designer drew homophones as variants of one shape" mechanism', 'Hand +0.14 / +0.11 / +0.06 (82 units), top-23 units +0.56 / +0.50 / +0.46; image ~0 (families defined on strokes are not seen by image descriptors). A homophone-family cipher CAN make the effect (see cycle 3 for the discriminating test)'),
    ('copiale', 'Copiale size-matched and mark-free re-tests', 'Null in every subset'),
    ('tengwar', 'TENGWAR re-tests: top-23 units and KILL CONTROL without tehtar (consonant grid only)', 'Tehtar carry most of it: without them +0.08 / +0.14 / +0.11 image, +0.09 / +0.18 / +0.20 hand (p 0.01-0.1). The purely featural consonant grid of Tengwar gives LESS than the Voynich (+0.28 / +0.30 / +0.43). Grade A'),
    ('cree', 'SYLLABICS re-tests: top-23 units and without finals / dots', 'Survives without finals: hand +0.13 / +0.09 / +0.16 (p <= 0.002); top-23 hand +0.15 / +0.19 / +0.35. Grade A'),
    ('shavian', 'SHAVIAN re-tests (top-23 units)', 'Image vanishes at 23 units (-0.06 to -0.02); hand +0.16 / +0.06 / +0.10. Weak'),
    ('deseret', 'DESERET re-tests', 'Null'),
    ('cherokee', 'CHEROKEE re-tests', 'Null'),
]
rows = []
for k, (n, lab, v) in enumerate(SETS, 1):
    res = R.summary(n) if k <= 3 else ''
    ps = R.post_summary(n)
    rows.append((f'V-35.2.{k}', f'{lab}. Corpus key {n}. {R.PIPE}; post-hoc: Mantel on the 23 most frequent units (2,000 perms), mean over 300 random 23-unit subsets, named subsets',
                 (res + ' || ' if res else '') + ps, v))
rows.append(('V-35.2.V', 'Cycle 2 verdict', 'Voynich +0.28 / +0.30 / +0.43 (23 units, no diacritics). Like-for-like (no marks, ~23 units): Tengwar consonants +0.08 to +0.20, syllabics +0.09 to +0.35 (hand), Borg <= +0.18, Shavian <= +0.16, Copiale / Deseret / Cherokee ~0. Only the planted shape-family cipher, scored with the strokes used to build it, reaches Voynich level',
             'No real invented or cipher script reaches the Voynich on all three models; the Voynich is at the top of the scale even against Tolkien\'s featural Tengwar once Tengwar\'s vowel marks are removed. Grade A for the scale, B for the ranking against Tengwar (one text, one mode)'))
hdr = ('# v35 cycle 2 - Borg cipher, planted homophonic ciphers, size-matched and mark-free re-tests (4 Oct 2026). Tools v35_lib.py, v35_post.py, v35_report.py, v35_cycle2.py.\n'
       '| row | method and control | result | verdict |\n|---|---|---|---|')
L.write_rows(os.path.join(L.LOOPS, 'v35_cycle2.txt'), rows, hdr)
