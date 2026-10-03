"""S-DARK-57 legwork: build data/derived/dark/loop57_earlymarks.csv from open publications of pre-Mature
(Early Harappan / Kot Diji / Ravi / Mehrgarh IV-VII / Sarai Khola IA-II) pottery marks, each transcribed by eye
against the Wells inventory rendered from tools/indus_font.ttf (scratchpad glyph sheets), with a Wells number where a
shape match exists and 'non-Wells' + a shape description otherwise. Confidence per sign: A (shape identical to one Wells
sign, no plausible rival), B (same family, rival Wells numbers exist: the number given is the family head), C (loose
match: stroke-count or orientation ambiguous, or the Wells sign is only a shape neighbour).
Sources (all open access, see loop57_images/SOURCES.txt):
  Q80  Quivron 1980, Paleorient 6: 269-280, Tableau 4 (p. 277): 50 categories of pre-firing marks, Mehrgarh periods
       IV, V, V-VI, VI, VI-VII, VII (+ surface), with counts per category per period (857 marks read; paper says 850).
  H72  Halim 1972, Pakistan Archaeology 8: 95-99, Table 11 (40 mark types) + catalogue nos 1-40 with sherd counts;
       Sarai Khola Period IA (transitional) and II (Kot Diji related); 68 sherds (2 IA, 59 II, 7 surface).
  K06  Kenoyer 2006, 'The origin, context and function of the Indus script: recent insights from Harappa',
       Figs 3-4 (pp. 22-23): Harappa Period 1 (Ravi) and Period 2 (Kot Diji) post-firing graffiti and pre-firing
       potter's marks (drawings, no per-mark counts; each drawn mark counted once).
  K65  Khan 1965, Pakistan Archaeology 2, pl. XXIV nos 2-3: Kot Diji levels, incised marks visible in the photo.
Usage: python3 tools/dark_loop57_build.py
"""
import csv, os
ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/loop57_earlymarks.csv'

rows = []
def add(src, site, phase, date_bc, mark_id, shape, wells, conf, n, firing, pub, fig, url, note=''):
    rows.append(dict(source=src, site=site, phase=phase, approx_date_BC=date_bc, mark_id=mark_id, shape=shape,
                     signs_W=wells, confidence_per_sign=conf, n_marks=n, firing=firing, publication=pub,
                     page_or_figure=fig, url=url, notes=note))

# ---------------------------------------------------------------- Mehrgarh (Quivron 1980, Tableau 4)
Q = 'Quivron, G. 1980. Les marques incisees sur les poteries de Mehrgarh au Baluchistan. Paleorient 6: 269-280'
QURL = 'https://www.persee.fr/doc/paleo_0153-9345_1980_num_6_1_4281 (page image renderPage .../T1_0277_0000_1200.jpg)'
QP = ['IV', 'V', 'V-VI', 'VI', 'VI-VII', 'VII']
QD = {'IV': '3500', 'V': '3300', 'V-VI': '3200', 'VI': '3100', 'VI-VII': '2900', 'VII': '2800-2500', 'surface': '?'}
# (category, shape description, Wells, confidence, counts IV..VII [+surface])
QT = [
 ('q01', 'one vertical stroke', '31', 'B', [0, 0, 3, 1, 3, 18], 'tall/short height not recorded; W31 (tall 1) or W1'),
 ('q02', 'one horizontal stroke', 'non-Wells', '-', [0, 0, 0, 0, 0, 22], 'no horizontal-bar sign in Wells'),
 ('q03', 'two vertical strokes', '32', 'B', [1, 0, 1, 2, 6, 58], 'W32 (tall 2) or W2'),
 ('q04', 'two horizontal strokes', 'non-Wells', '-', [0, 0, 0, 0, 0, 6], ''),
 ('q05', 'stroke with short tick at top', 'non-Wells', '-', [0, 0, 0, 0, 1, 0], ''),
 ('q06', 'stroke with two bars at top', 'non-Wells', '-', [0, 0, 0, 0, 0, 1], 'W400 comb family only if teeth are read as bars'),
 ('q07', 'three vertical strokes', '33', 'B', [2, 3, 8, 7, 32, 67], 'W33 (tall 3) or W3'),
 ('q08', 'three strokes converging at base', '33', 'C', [0, 0, 1, 10, 10, 6], 'fan of three; counted as 3-stroke numeral'),
 ('q09', 'four vertical strokes', '34', 'B', [0, 0, 0, 0, 1, 15], ''),
 ('q10', 'six vertical strokes', '36', 'B', [0, 0, 0, 0, 0, 1], ''),
 ('q11', 'stroke with top bar and side bars', 'non-Wells', '-', [0, 0, 0, 0, 0, 1], ''),
 ('q12', 'stroke with many short hatches', '413', 'C', [0, 0, 0, 0, 0, 1], 'W413 hatched stem'),
 ('q13', 'one arc', '899', 'B', [0, 0, 0, 1, 5, 43], 'W899/900 single arc'),
 ('q14', 'two nested arcs', '904', 'B', [1, 0, 7, 4, 7, 42], 'W904/905 double arc'),
 ('q15', 'three nested arcs', '908', 'B', [2, 0, 0, 1, 6, 14], ''),
 ('q16', 'four nested arcs', '909', 'C', [0, 0, 1, 0, 4, 7], ''),
 ('q17', 'single wavy line', '450', 'C', [0, 0, 3, 1, 1, 16], 'wave / zigzag family W450-453'),
 ('q18', 'three stacked wavy lines', '451', 'C', [0, 0, 0, 0, 0, 4], ''),
 ('q19', 'X with curved arms', '645', 'C', [1, 1, 0, 4, 4, 6], 'X family'),
 ('q20', 'curved X with central bar', '646', 'C', [0, 0, 0, 0, 1, 1], ''),
 ('q21', 'lens / eye shape (bow over bow)', '790', 'C', [0, 0, 0, 0, 1, 2], 'oval W790 is the nearest'),
 ('q22', 'curve with upward tail (bird-foot curve)', 'non-Wells', '-', [0, 0, 0, 0, 0, 41], 'the dominant period VII workshop mark (16 pots in the store room?)'),
 ('q23', 'double flourish curve', 'non-Wells', '-', [0, 0, 0, 0, 0, 1], ''),
 ('q24', 'wave with arc', 'non-Wells', '-', [0, 0, 0, 0, 0, 1], ''),
 ('q25', 'wave with two arcs', 'non-Wells', '-', [0, 0, 0, 0, 0, 1], ''),
 ('q26', 'stem with three branches each side (tree)', '390', 'B', [0, 0, 0, 0, 0, 1], 'W390/405 tree'),
 ('q27', 'T (vertical with top bar)', 'non-Wells', '-', [0, 0, 1, 2, 2, 17], 'W70/W610 are boxed or barbed T forms, not a plain T'),
 ('q28', 'arch with one inner stroke (T under arch)', 'non-Wells', '-', [0, 0, 1, 2, 2, 59], ''),
 ('q29', 'arch with three inner strokes', 'non-Wells', '-', [0, 0, 0, 0, 0, 17], ''),
 ('q30', 'V', '700', 'B', [0, 0, 3, 4, 7, 23], 'W700 plain V/U cup; W697-699 are doubled'),
 ('q31', 'V with tail (V~)', '700', 'C', [0, 1, 1, 0, 0, 66], 'V plus a short tail; the commonest period VII mark'),
 ('q32', 'V with inner stroke', '706', 'B', [0, 0, 0, 2, 0, 3], 'W703/706 U with one central stroke'),
 ('q33', 'y (V on a stem)', '64', 'B', [0, 0, 0, 0, 0, 3], ''),
 ('q34', 'IV (stroke + V)', '31-700', 'C', [0, 0, 0, 0, 0, 22], 'a stroke beside a V; also readable as one sign'),
 ('q35', 'IN (stroke + N)', 'non-Wells', '-', [0, 0, 0, 0, 0, 1], ''),
 ('q36', 'W (double V)', '697', 'B', [0, 0, 0, 0, 2, 5], ''),
 ('q37', 'VIV', 'non-Wells', '-', [0, 0, 0, 0, 0, 1], ''),
 ('q38', 'zigzag line', '452', 'B', [0, 0, 0, 0, 0, 1], ''),
 ('q39', 'X', '645', 'A', [1, 0, 1, 7, 19, 33], ''),
 ('q40', 'X with vertical through it', '647', 'C', [0, 0, 0, 0, 0, 1], ''),
 ('q41', 'double X side by side', '645-645', 'C', [1, 0, 2, 0, 3, 1, 0], 'two X; W552/554 are X-pairs in a frame'),
 ('q42', 'X with central vertical (star of 3)', '685', 'C', [1, 0, 0, 0, 0, 0, 0], 'asterisk family W681/685'),
 ('q43', 'X with horizontal bar (hourglass on its side)', '646', 'C', [0, 0, 0, 0, 4, 0, 0], ''),
 ('q44', 'lens crossed by diagonal', 'non-Wells', '-', [0, 0, 0, 2, 1, 0, 0], ''),
 ('q45', 'bowtie (two triangles tip to tip)', '540', 'B', [0, 0, 0, 0, 1, 1, 0], ''),
 ('q46', 'double bowtie', '545', 'C', [0, 0, 0, 0, 0, 1, 0], ''),
 ('q47', 'grid (hash in a box)', '615', 'B', [0, 0, 0, 0, 0, 0, 1], 'surface find'),
 ('q48', 'rectangle with diagonals (crossed box)', 'non-Wells', '-', [0, 0, 0, 0, 0, 1, 0], ''),
 ('q49', 'circle', '790', 'C', [0, 0, 0, 0, 0, 1, 0], 'oval W790 nearest; no plain circle in Wells'),
 ('q50', 'dotted circle', '792', 'C', [0, 0, 0, 0, 0, 1, 0], ''),
 ('q51', 'hook / spiral', '445', 'C', [0, 0, 0, 0, 0, 1, 0], ''),
]
for cat, shape, w, conf, counts, note in QT:
    per = QP + (['surface'] if len(counts) == 7 else [])
    for ph, n in zip(per, counts):
        if n:
            add('Q80', 'Mehrgarh', 'MR ' + ph, QD[ph], cat, shape, w, conf, n, 'pre-firing', Q, 'Tableau 4, p. 277', QURL, note)

# ---------------------------------------------------------------- Sarai Khola (Halim 1972, Table 11 + catalogue)
H = 'Halim, M. A. 1972. The pottery of Periods IA and II with incised potter\'s or graffiti marks. Pakistan Archaeology 8: 95-99'
HURL = 'https://archive.org/details/in.gov.ignca.69948 (69948_text.pdf pp. 113-117 of the scan)'
# (table no, shape, Wells, conf, n sherds, phase, firing, note); n from the catalogue ('two more sherds' etc.)
HT = [
 (1, 'one short stroke (oval incision)', '1', 'B', 3, 'II', 'pre-firing', 'cat. 1: + two more sherds (Sq. 18/V (6), surface)'),
 (2, 'two strokes', '2', 'B', 4, 'II', 'pre-firing', 'cat. 2: + three more sherds'),
 (3, 'three strokes', '3', 'B', 3, 'II', 'pre-firing', 'cat. 3: + two more sherds'),
 (4, 'two strokes, offset', '2', 'C', 1, 'surface', 'pre-firing', 'cat. 4'),
 (5, 'three strokes, scattered', '3', 'C', 2, 'surface', 'pre-firing', 'cat. 5: + one sherd Sq. 17/V (4)'),
 (6, 'four strokes, scattered', '4', 'C', 1, 'II', 'unknown', 'cat. 6'),
 (7, 'cross (+)', 'non-Wells', '-', 4, 'III?', 'pre-firing', 'cat. 7: Sq. 18/Y (8) Period III + three more sherds; plus sign, no Wells equivalent'),
 (8, 'stroke on a horizontal bar (inverted T)', 'non-Wells', '-', 2, 'II', 'pre-firing', 'cat. 8: + one sherd'),
 (9, 'T', 'non-Wells', '-', 1, '?', 'unknown', 'cat. 9'),
 (10, 'U with flat base (cup with two verticals)', '700', 'C', 1, '?', 'pre-firing', 'cat. 10'),
 (11, 'two tall strokes joined by a long horizontal (H with long bar)', 'non-Wells', '-', 1, 'II', 'pre-firing', 'cat. 11 (Type IXB)'),
 (12, 'one oblique stroke', '1', 'C', 1, '?', 'pre-firing', 'cat. 12'),
 (13, 'inverted V (chevron)', '480', 'B', 1, '?', 'pre-firing', 'cat. 13 = Kot Diji pl. XXIV no. 2 parallel'),
 (14, 'chevron with hatched arm', '480', 'C', 1, 'IA', 'pre-firing', 'cat. 14, Period IA'),
 (15, 'cluster of short strokes under a dotted arc', 'non-Wells', '-', 1, 'II', 'pre-firing', 'cat. 15'),
 (16, 'three-armed star (Y with lower arm)', 'non-Wells', '-', 1, 'II', 'unknown', 'cat. 16'),
 (17, 'arrow (stem with two barbs)', '515', 'B', 2, 'II', 'post-firing', 'cat. 17: post-firing on the shoulder; a similar pre-firing mark on a surface sherd'),
 (18, 'W (two V)', '697', 'B', 1, 'II', 'unknown', 'cat. 18'),
 (19, 'W, asymmetric', '697', 'C', 1, 'II', 'unknown', 'cat. 19 (+ one surface sherd)'),
 (20, 'V', '700', 'B', 5, '?', 'unknown', 'cat. 20: + four other sherds'),
 (21, 'Y', '64', 'B', 1, '?', 'unknown', 'cat. 21'),
 (22, 'V with inner stroke', '706', 'B', 2, 'II', 'unknown', 'cat. 22: + one sherd'),
 (23, 'N (zigzag of three strokes)', '452', 'C', 2, 'II', 'unknown', 'cat. 23: + one sherd'),
 (24, 'V with dotted arm (W?)', '700', 'C', 1, 'II', 'unknown', 'cat. 24'),
 (25, 'V, open', '700', 'B', 1, 'II', 'unknown', 'cat. 25'),
 (26, 'V with two inner strokes', '706', 'C', 1, 'II', 'unknown', 'cat. 26'),
 (27, 'W (zigzag)', '697', 'C', 1, 'IA', 'unknown', 'cat. 27, Period IA'),
 (28, 'N', '452', 'C', 1, 'II', 'unknown', 'cat. 28'),
 (29, 'two oblique strokes', '2', 'C', 2, 'II', 'unknown', 'cat. 29: + one sherd'),
 (30, 'W with long arms', '697', 'C', 1, 'II', 'pre-firing', 'cat. 30'),
 (31, 'W with short arms', '697', 'C', 1, 'II', 'unknown', 'cat. 31'),
 (32, 'two strokes, one tall one short', '2', 'C', 1, 'II', 'unknown', 'cat. 32'),
 (33, 'short oblique + tall stroke', '2', 'C', 1, 'II', 'pre-firing', 'cat. 33'),
 (34, 'stroke with three branches on one side (half tree)', '390', 'C', 1, 'II', 'unknown', 'cat. 34'),
 (35, 'two strokes crossing (N-like)', '2', 'C', 1, 'II', 'unknown', 'cat. 35'),
 (36, 'E-shaped curve (hook)', '445', 'C', 1, 'II', 'unknown', 'cat. 36'),
 (37, 'single wave', '450', 'C', 1, '?', 'unknown', 'cat. 37'),
 (38, 'double wave (zigzag)', '451', 'C', 1, 'II', 'unknown', 'cat. 38'),
 (39, 'double spiral with central strokes', 'non-Wells', '-', 1, 'II', 'unknown', 'cat. 39'),
 (40, 'one stroke; two strokes (two marks on one sherd)', '1-2', 'C', 1, '?', 'post-firing', 'cat. 40: graffiti on shoulder and rim'),
]
for no, shape, w, conf, n, ph, fire, note in HT:
    add('H72', 'Sarai Khola', 'SKh ' + ph, '3000-2600' if ph in ('II', 'IA', '?') else ('2600-2000' if ph.startswith('III') else '?'),
        f'h{no:02d}', shape, w, conf, n, fire, H, 'Table 11 p. 96; catalogue pp. 97-99', HURL, note)

# ---------------------------------------------------------------- Harappa Ravi / Kot Diji (Kenoyer 2006, Figs 3-4)
K = 'Kenoyer, J. M. 2006. The origin, context and function of the Indus script: recent insights from Harappa. Proc. Pre-symposium of RIHN / 7th ESCA Harvard-Kyoto Roundtable: 9-27'
KURL = 'https://www.harappa.com/content/origin-context-and-function-indus-script-recent-insights-harappa (PDF Kenoyer2006_..., read in the Chromium viewer, page 9 of 11)'
KT = [
 # Ravi phase, post-firing graffiti (Fig. 3)
 ('r-g1', 'Ravi', 'sherd 1: three tree/branch signs in a row (Y-fork, three-branch stem, Y-fork)', '64-390-64', 'C-B-C', 'post-firing', 'Figure 3 no. 1; multi-sign'),
 ('r-g2', 'Ravi', 'sherd 2: two curved strokes with hatches', 'non-Wells', '-', 'post-firing', 'Figure 3 no. 2'),
 ('r-g3', 'Ravi', 'circle quartered by a cross, hatched', '818', 'B', 'post-firing', 'Figure 3'),
 ('r-g4', 'Ravi', 'V with inner stroke (arrow-like)', '706', 'B', 'post-firing', 'Figure 3'),
 ('r-g5', 'Ravi', 'fan of five strokes meeting at base', 'non-Wells', '-', 'post-firing', 'Figure 3'),
 ('r-g6', 'Ravi', 'X with extra stroke', '646', 'C', 'post-firing', 'Figure 3'),
 ('r-g7', 'Ravi', 'asterisk (six arms)', '685', 'B', 'post-firing', 'Figure 3'),
 ('r-g8', 'Ravi', 'one oblique stroke', '1', 'C', 'post-firing', 'Figure 3'),
 ('r-g9', 'Ravi', 'one vertical stroke', '31', 'B', 'post-firing', 'Figure 3'),
 ('r-g10', 'Ravi', 'two vertical strokes', '32', 'B', 'post-firing', 'Figure 3'),
 # Ravi phase, pre-firing potter's marks
 ('r-p1', 'Ravi', 'arrow pointing right (stroke with barbed head)', '515', 'B', 'pre-firing', 'Figure 3'),
 ('r-p2', 'Ravi', 'V', '700', 'B', 'pre-firing', 'Figure 3'),
 ('r-p3', 'Ravi', 'V with inner stroke', '706', 'B', 'pre-firing', 'Figure 3'),
 ('r-p4', 'Ravi', 'X', '645', 'A', 'pre-firing', 'Figure 3'),
 ('r-p5', 'Ravi', 'cross (+)', 'non-Wells', '-', 'pre-firing', 'Figure 3'),
 ('r-p6', 'Ravi', 'two vertical strokes', '32', 'B', 'pre-firing', 'Figure 3'),
 # Kot Diji phase, post-firing graffiti (Fig. 4)
 ('k-g1', 'Kot Diji', 'sherd 1: V with forked tips (two bird-foot V)', '706', 'C', 'post-firing', 'Figure 4 no. 1'),
 ('k-g2', 'Kot Diji', 'sherd 2: V with forked tips', '706', 'C', 'post-firing', 'Figure 4 no. 2'),
 ('k-g3', 'Kot Diji', 'sherd 3: V with forked tips between two arcs', '899-706-900', 'C-C-C', 'post-firing', 'Figure 4 no. 3; multi-sign'),
 ('k-g4', 'Kot Diji', 'V with inner stroke', '706', 'B', 'post-firing', 'Figure 4'),
 ('k-g5', 'Kot Diji', 'fan of four strokes meeting at base', 'non-Wells', '-', 'post-firing', 'Figure 4'),
 ('k-g6', 'Kot Diji', 'stroke with two bars (double cross)', 'non-Wells', '-', 'post-firing', 'Figure 4'),
 ('k-g7', 'Kot Diji', 'asterisk', '685', 'B', 'post-firing', 'Figure 4'),
 ('k-g8', 'Kot Diji', 'one oblique stroke', '1', 'C', 'post-firing', 'Figure 4'),
 ('k-g9', 'Kot Diji', 'one vertical stroke', '31', 'B', 'post-firing', 'Figure 4'),
 ('k-g10', 'Kot Diji', 'two vertical strokes', '32', 'B', 'post-firing', 'Figure 4'),
 ('k-g11', 'Kot Diji', 'sherd 4: two U with forked tips', '706-706', 'C-C', 'post-firing', 'Figure 4 no. 4'),
 ('k-g12', 'Kot Diji', 'sherd 5: stroke + cup with inner strokes (dotted)', '1-706', 'C-C', 'post-firing', 'Figure 4 no. 5'),
 ('k-g13', 'Kot Diji', 'V with inner stroke', '706', 'B', 'post-firing', 'Figure 4'),
 ('k-g14', 'Kot Diji', 'V with inner stroke (second)', '706', 'B', 'post-firing', 'Figure 4'),
 ('k-g15', 'Kot Diji', 'inverted V', '480', 'B', 'post-firing', 'Figure 4'),
 ('k-g16', 'Kot Diji', 'hash (#)', '615', 'C', 'post-firing', 'Figure 4; W615 is a boxed grid'),
 ('k-g17', 'Kot Diji', 'comb (stem with four teeth)', '401', 'B', 'post-firing', 'Figure 4; W400/401 comb'),
 ('k-g18', 'Kot Diji', 'Z-like hook (angled line)', '440', 'C', 'post-firing', 'Figure 4'),
 ('k-g19', 'Kot Diji', 'N', '452', 'C', 'post-firing', 'Figure 4'),
 ('k-g20', 'Kot Diji', 'sherd 6: oblique stroke + two crossed ovals (bowtie-like)', '1-540-540', 'C-C-C', 'post-firing', 'Figure 4 no. 6'),
 ('k-g21', 'Kot Diji', 'sherd 7: stroke + V + V', '1-700-700', 'C-B-B', 'post-firing', 'Figure 4 no. 7'),
 ('k-g22', 'Kot Diji', 'sherd 8: arrow + short stroke', '515-1', 'B-C', 'post-firing', 'Figure 4 no. 8'),
 ('k-g23', 'Kot Diji', 'hooked square (angular spiral)', 'non-Wells', '-', 'post-firing', 'Figure 4'),
 ('k-g24', 'Kot Diji', 'arc + crossed oval + oblique', '899-645-1', 'C-C-C', 'post-firing', 'Figure 4; multi-sign'),
 ('k-g25', 'Kot Diji', 'stroke + X', '31-645', 'B-A', 'post-firing', 'Figure 4'),
 ('k-g26', 'Kot Diji', 'bowtie (hourglass)', '540', 'B', 'post-firing', 'Figure 4'),
 ('k-g27', 'Kot Diji', 'square with diagonal cross', 'non-Wells', '-', 'post-firing', 'Figure 4'),
 ('k-g28', 'Kot Diji', 'sherd 9: three strokes + W (zigzag)', '33-697', 'B-C', 'post-firing', 'Figure 4 no. 9'),
 ('k-g29', 'Kot Diji', 'sherd 10: stroke + three inverted V', '31-480-480-480', 'B-C-C-C', 'post-firing', 'Figure 4 no. 10'),
 ('k-g30', 'Kot Diji', 'fan of strokes in an arc (feather)', 'non-Wells', '-', 'post-firing', 'Figure 4'),
 ('k-g31', 'Kot Diji', 'grid 2x2 (hash with closed cells)', '615', 'C', 'post-firing', 'Figure 4'),
 ('k-g32', 'Kot Diji', 'hash (#) second', '615', 'C', 'post-firing', 'Figure 4'),
 ('k-g33', 'Kot Diji', 'sun (dotted circle with rays)', 'non-Wells', '-', 'post-firing', 'Figure 4'),
 ('k-g34', 'Kot Diji', 'crossed triangle on a stem', 'non-Wells', '-', 'post-firing', 'Figure 4'),
 # Kot Diji phase, pre-firing potter's marks
 ('k-p1', 'Kot Diji', 'sherd 11: three oblique strokes + comb/fan', '3-401', 'C-C', 'pre-firing', 'Figure 4 no. 11'),
 ('k-p2', 'Kot Diji', 'sherd 12: oblique stroke + figure-8 + branching mark', '1-793-non-Wells', 'C-C--', 'pre-firing', 'Figure 4 no. 12'),
 ('k-p3', 'Kot Diji', 'sherd 13: arc with inner cross (dotted outline)', '818', 'C', 'pre-firing', 'Figure 4 no. 13'),
 ('k-p4', 'Kot Diji', 'I-beam (two bars joined)', 'non-Wells', '-', 'pre-firing', 'Figure 4'),
 ('k-p5', 'Kot Diji', ')( (two arcs back to back)', '906', 'B', 'pre-firing', 'Figure 4'),
 ('k-p6', 'Kot Diji', 'X with extra strokes (multi-crossed)', '646', 'C', 'pre-firing', 'Figure 4'),
 ('k-p7', 'Kot Diji', 'asterisk (six arms)', '685', 'B', 'pre-firing', 'Figure 4'),
 ('k-p8', 'Kot Diji', 'X', '645', 'A', 'pre-firing', 'Figure 4'),
 ('k-p9', 'Kot Diji', 'three vertical strokes', '33', 'B', 'pre-firing', 'Figure 4'),
 ('k-p10', 'Kot Diji', 'two vertical strokes', '32', 'B', 'pre-firing', 'Figure 4'),
 ('k-p11', 'Kot Diji', 'one vertical stroke', '31', 'B', 'pre-firing', 'Figure 4'),
 ('k-p12', 'Kot Diji', 'two arcs crossing (curved X)', '645', 'C', 'pre-firing', 'Figure 4'),
 ('k-p13', 'Kot Diji', 'shallow V (curved)', '700', 'C', 'pre-firing', 'Figure 4'),
 ('k-p14', 'Kot Diji', 'V + small v', '700-700', 'B-C', 'pre-firing', 'Figure 4'),
 ('k-p15', 'Kot Diji', 'three nested arcs', '908', 'B', 'pre-firing', 'Figure 4'),
 ('k-p16', 'Kot Diji', 'three oblique strokes', '3', 'C', 'pre-firing', 'Figure 4'),
]
for mid, ph, shape, w, conf, fire, fig in KT:
    add('K06', 'Harappa', 'HP ' + ph, '3300-2800' if ph == 'Ravi' else '2800-2600', mid, shape, w, conf, 1, fire, K, fig, KURL,
        'drawing only; each drawn mark counted once (the figure is a selection, not a census)')

# ---------------------------------------------------------------- Kot Diji (Khan 1965 pl. XXIV)
KD = 'Khan, F. A. 1965. Excavations at Kot Diji. Pakistan Archaeology 2: 11-85, pl. XXIV'
KDURL = 'https://archive.org/details/dli.ernet.108437 (108437-Pakistan Archaeology No.2_text.pdf p. 102 of the scan)'
add('K65', 'Kot Diji', 'KD Kot Diji levels', '3000-2600', 'kd-2', 'inverted V (chevron) on a rimless jar', '480', 'C', 1, 'unknown', KD, 'pl. XXIV no. 2', KDURL, 'poor halftone; Halim 1972 Table 12 cites it as the parallel of Sarai Khola no. 13')
add('K65', 'Kot Diji', 'KD Kot Diji levels', '3000-2600', 'kd-3', 'two crossed strokes (X) beside a vertical', '645', 'C', 1, 'unknown', KD, 'pl. XXIV no. 3', KDURL, 'poor halftone')

with open(OUT, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
tot = sum(r['n_marks'] for r in rows)
print(f'{len(rows)} rows, {tot} marks ->', OUT)
import collections
print(collections.Counter(r['source'] for r in rows))
print('marks per source:', {s: sum(r['n_marks'] for r in rows if r['source'] == s) for s in ('Q80', 'H72', 'K06', 'K65')})
