"""v35 cycle 1: the v25 pipeline on invented scripts and cipher alphabets.  Writes loops/v35_cycle1.txt from cached results."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v35_report as R, v28_lib as X, v25_lib as L

SETS = [
    ('voy', 'VOYNICH ZL, v25 units (replication of v25 inside v35 code)', 'REPRODUCES v25 exactly (+0.28 / +0.30 / +0.43). Reference'),
    ('voy@73000', 'Voynich truncated to the Copiale token count (73k)', 'Size does not matter. Reference'),
    ('voy_lines@73000', 'Voynich in Copiale FORMAT: word spaces removed, a line is one unit string, 73k tokens', 'Format does not create or remove it (+0.32 / +0.40 / +0.46)'),
    ('copiale', 'COPIALE CIPHER (1760s homophonic cipher, 84 symbols: Roman letters with dot / circumflex / underline / umlaut variants, Greek letters, invented symbols); HTR line transcription (learnable-typewriter set, 73k symbols); glyphs = the Copiale font embedded in Knight-Megyesi-Schaefer 2011; hand = v25 book-hand primitives + marks, written from the font montage first', 'NULL: a designed cipher alphabet whose symbols come in visible families (a/a-circumflex, n/n-dot/n-underlined) shows ~0 (+0.04 at most). Grade A'),
    ('copiale_merge', 'KILL CONTROL: Copiale with every marked variant merged into its base letter', 'Null either way'),
    ('tengwar', 'TENGWAR (Tolkien; deliberately featural: stem height = manner, bow side/closure = place) writing English (Pride and Prejudice + Frankenstein) in an orthographic tehta mode, tehtar as separate units; FreeMonoTengwar; hand = stem/bow/side/closure primitives', 'POSITIVE CONTROL WORKS: image +0.45 / +0.39 / +0.54, hand +0.35 / +0.35 / +0.44, at or above the Voynich. Grade A (see cycle 2 for the tehta confound)'),
    ('shavian', 'SHAVIAN (designed 1960, partly featural: tall/deep/short heights, rotated pairs) on the same English text (ReadLex spellings); Noto Sans Shavian; hand = height class + stroke primitives', 'Weak positive: image +0.10 / +0.11 / +0.13, hand +0.30 / +0.28 / +0.30 (height class ~ vowel/consonant). Grade A'),
    ('deseret', 'DESERET (designed 1850s, shapes arbitrary to sound) on the SAME phoneme stream as Shavian (one-to-one map), behaviour essentially fixed, shape system changed; Noto Sans Deseret', 'NULL (0.00 / -0.01 / +0.01). Same text in Shavian is positive: the link follows the shape design, not the text. Grade A'),
    ('cherokee', 'CHEROKEE SYLLABARY (invented 1821 by Sequoyah, shapes adapted from Latin letters, arbitrary to sound); ChrEn monolingual + parallel Cherokee side; Noto Sans Cherokee, FreeSerif', 'NULL (-0.01 to +0.04, n.s.). An invented script with arbitrary shapes does not show it. Grade A'),
    ('cree', 'CANADIAN SYLLABICS (Evans 1840, featural by rotation: series shape + orientation = vowel, dots = length / w); Northern East Cree New Testament (eBible crl, scratch only); Noto Sans Canadian Aboriginal, FreeSans; hand = series + orientation + dots + final', 'POSITIVE but weaker than the Voynich: hand +0.15 / +0.13 / +0.21, image +0.06-0.08 / +0.06 / +0.13-0.15. Rotation features are invisible to rotation-sensitive image descriptors. Grade A'),
]
rows = [(f'V-35.1.{k}', f'{lab}. Corpus key {n}. {R.PIPE}', R.summary(n), v) for k, (n, lab, v) in enumerate(SETS, 1)]
rows.append(('V-35.1.V', 'Cycle 1 verdict', 'Designed-featural positive controls are positive (Tengwar >= Voynich; Shavian, Cree syllabics +0.1 to +0.3); designed-arbitrary scripts (Deseret, Cherokee) and the Copiale cipher alphabet are null', 'Being INVENTED does not produce the effect; being FEATURALLY designed does. The Voynich sits with Tengwar (top of the scale), above syllabics and Shavian. Grade A'))
hdr = ('# v35 cycle 1 - shape predicts behaviour in invented scripts and cipher alphabets (4 Oct 2026). Tools v35_lib.py, v35_shapes.py, v35_run.py, v35_report.py.\n'
       '| row | method and control | result | verdict |\n|---|---|---|---|')
L.write_rows(os.path.join(L.LOOPS, 'v35_cycle1.txt'), rows, hdr)
