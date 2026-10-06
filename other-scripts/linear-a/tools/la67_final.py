#!/usr/bin/env python3
"""LA-67 final: the clean table of every grade-C Linear A guess in FINDINGS.md, its kill test, the result
and the rewritten grade.  Results are copied from loops/la67_cycle1-3.txt (the la44 row is read from the
la44 re-run read-out if present).  Grades: C+ = survived a held-out / decoy-calibrated kill test;
C = survived only literally or weakly (decoys pass as often, or the test is near its line); C- = passes the
stated line but not beyond matched decoys; KILLED = failed its own kill line or the strongest existing-data
test; UNTESTABLE-NOW = needs data that does not exist yet (grade unchanged)."""
import os, re, json
HERE = os.path.dirname(os.path.abspath(__file__))
LOOPS = os.path.join(HERE, '..', 'loops')
OUT = os.path.join(LOOPS, 'la67_final.txt')

T = [
    # (source, guess, test, result, outcome, new grade)
    ('la61', '*318 is a commodity marker', 'COM role alone in the la60 grammar, 4 fresh splits, vs 20 frequency-matched single decoy signs (1i)', '-10.8 bits; 15/20 decoys higher', 'killed', 'KILLED'),
    ('la52', 'OLE+RI dies under copying faster than its frequency predicts', '2 new-seed sets of gentle iterated-learning chains (3 learner types x 240 chains each) (3g)', 'z -0.73 and -1.37 (line: > -1 kills); no word below -2 repeats across seeds', 'killed', 'KILLED'),
    ('la53', 'PA-DE is a specially marked entry', 'la53 ranking re-run, fresh seeds, 1,000 permutations, 31 same-n decoys (3h)', 'p 0.004, rank 1 of 175, 0 decoys as low; q 0.61', 'survived', 'C'),
    ('la40', 'KU-RE, U, A, DA, KA-NA are heading words', 'first line + no count vs 40 matched decoys, 20 fresh halves; power check (1a, 1j)', 'none passes (KU-RE, U, KA-NA score 0; A 0.22, DA 0.25 at decoy level); *516, *307, A-DU do pass', 'killed', 'KILLED'),
    ('la54', 'HT 95b is a grain document', 'commodity vote from its words on OTHER tablets, calibrated on 107 known documents (1g)', 'GRA, confidence 0.60; such votes right 0.44 vs GRA base 0.13 (rests on HT 86/95)', 'survived', 'C'),
    ('la45', 'OLE+U, OLE+MI, OLE+DI are entries, not commodities', 'literal kill line + after-word share vs 25 decoy logograms, 20 halves (1b)', '4 / 3 / 4 uses already after a word (HT 2 A-KA-RU OLE+U 20); share 0.15-0.33 vs decoys 0.30, P 0.35-0.58', 'killed', 'KILLED'),
    ('la65', '*28B-NU-MA-RE and SI-PI-KI are fixed words of the Zakros wine template', 'literal over all documents + identical-tablet-set pairs vs within-site re-dealing (1f)', '0 off-template uses; the only identical pair in the corpus, P 0.0005', 'survived', 'C'),
    ('la63', '*308 is a fraction-bound commodity sign', 'literal (bare integer) + 70 decoy signs + 1,000 within-document number shuffles (1c)', '9/9 with fraction, 0 bare; decoys P 0.056, within-document P 0.034', 'survived', 'C'),
    ('la63', 'SI (with CYP), NI, TA2 predict the commodity logogram', 'commodity fixed/chosen on half A, site-stratified test on half B, 20 halves, 29 decoys (1d)', 'full-data P 0.008 / 0.001 / 0.30, but held-out 7/20, 8/20, 0/17', 'killed', 'KILLED'),
    ('la64', 'VIN unit 7-24 l (C-)', 'needs vessel capacities (THE Zb 13, KN Zb 35)', '-', 'untestable', 'C- (untestable-now)'),
    ('la60/61', 'TA-I goes with AROM', 'word->logogram tablets vs 1,000 site-stratified logogram shuffles; 21 decoy words (1e)', 'P 0.002 on 2 tablets; 29 % of decoy words also pass', 'survived', 'C'),
    ('la44', 'given affixation, LA leans to Greek-type suffixing', 'new v2 bank (fresh seeds), ISOL removed, robust panel, 240 shuffled-LA targets (3k)', '@LA44@', '@LA44O@', '@LA44G@'),
    ('la27', 'D = 1/5 and B = 1/3', 'literal count of repetitions in every amount (1h)', 'max D 4 (HT 115a), max B 2; nothing kills', 'survived', 'C (decisive re-reading untestable-now)'),
    ('la10', 'repeated consonants avoided inside roots', 'same-consonant sign pairs in HT and non-HT word types vs re-dealt signs and within-word shuffles (2b)', 'HT P 0.002, non-HT P 0.008 vs re-dealt; within-word order adds nothing (P 0.56, 0.07)', 'survived', 'C+ (avoidance inside the word, not adjacency)'),
    ('la10', 'consonant-final stems', 'no written word-final consonant', '-', 'untestable', 'C (untestable-now)'),
    ('la12', 'HT 116a GRA 109 = HT 1 KU-PA3-NU 109; HT 27a VIR 140 = HT 27b', '170 HT internal sums vs the same sums shifted +-1..5 (2a)', 'matched 0.53 vs 0.41-0.52', 'killed', 'KILLED'),
    ('la13', 'HT KA vs other JA (dialect rule)', 'single pair vs tablet relabelings, then family-wise max over 501 pairs (2c, 3a)', 'single P 0.037; family-wise P 0.70', 'killed', 'KILLED'),
    ('la16', 'KU~WA and MA~ME sign equations', 'cross-site substitutions vs 300 tablet relabelings (2c)', '0 (P 1.0) and 2 (P 0.71)', 'killed', 'KILLED'),
    ('la18', 'SI- is an alternating prefix', 'within HT alone: context cosine of 10 SI-X/X pairs vs 2,000 matched decoy pairings (2d)', 'P 0.092 (line: q > 0.1 kills)', 'survived', 'C'),
    ('la20', '*307 is a top-rank (grain-class) commodity', 'literal kill line over all documents (2e)', 'OLE written before *307 on PE Wy 5 and PE Zg 5', 'killed', 'KILLED'),
    ('la24', 'HT 9b repeats HT 9a recipients at 4/5 (J = 1/2)', 'pairs with >= 3 same-ratio entries vs 300 within-document shuffles (2f)', '3 pairs (HT 9a/b x4/5, HT 86/95 x1/2) vs 1.74, P 0.043', 'survived', 'C (re-reading untestable-now)'),
    ('la28', 'KA, KU, SI, I stand unnumbered in HT ledgers', 'stricter parse vs 2,000 decoy sign sets (2g)', '0.34 vs 0.23, P 0.13', 'killed', 'KILLED'),
    ('la33', 'grain lists are rations', 'amount dispersion of recurring GRA entry words vs shuffles, HT 86/95 out; OLE/CYP/VIN as decoys (2h)', 'GRA P 0.47; CYP P 0.018 and VIN P 0.013 are the near-equal ones', 'killed', 'KILLED'),
    ('la33', 'LA groups sit at the demand end', 'needs LB ration lists in LA format', '-', 'untestable', 'C (untestable-now)'),
    ('la37', 'SI~TI, TI~TE, RE~ME alternate word-finally', 'final vs non-final counts vs 344 decoy sign pairs (2i)', '5/4, 3/1, 2/1 final/non-final; decoy pairs are 0.55 final; P 0.28-0.55', 'decoy-level', 'C-'),
    ('la48', 'entry words are not standing accounts with quotas', '(word, amount) repeats on 2+ tablets vs 1,000 within-site shuffles (2j)', '64 vs 57.0, P 0.097 without HT 86/95', 'survived', 'C'),
    ('la48', 'PE Wy 5 and PE Zg 5 record one transfer', 'needs seal / clay / hand data', '-', 'untestable', 'C (untestable-now)'),
    ('la50', 'LA shares the LB staple order grain > olive > figs > wine', 'concordance, rank among 24 orders, 20 fresh halves (2k)', '0.79, rank 1/24, 20/20 halves (4 VIN-before-olive pairs already known)', 'survived', 'C+'),
    ('la50', 'figs and olives share fewer tablets than chance', 'NI/OLIV tablets vs 5,000 within-site re-dealings among commodity tablets; 120 decoy pairs (3b)', '3 vs 3.46, P 0.53', 'killed', 'KILLED'),
    ('la57', 'DI-DE-RU, DA-SI-*118, *28B-NU-MA-RE, U-*325-ZA, TE-TU are commodity words (C-)', 'literal kill / support lines (2l)', 'all 16 uses word + number; none heads a logogram list, none in a logogram slot', 'survived', 'C- (unchanged)'),
    ('la60', 'entry words return with the same commodity', 'commodity chosen on train half, hit on held-out half, 20 halves, 43 decoy words (2m)', 'named words 0.47; decoy words 0.35 (P 0.14); site default 0.37', 'decoy-level', 'C-'),
    ('la14', 'outside HT, consecutive entries share their first sign', '2,000 order shuffles, 20 halves, last/second sign as decoys (2n)', '0.163 vs 0.078, P 0.0005, 12/20 halves; decoy positions null', 'survived', 'C+'),
    ('la51', 'QA2 and TE with the Villa magazines; NI and A-DU with houses', 'deposit-label permutations, 40 held-out half-sets, 73 decoy terms (3c)', 'P 0.005 / 0.009 / 0.043 / 0.009; direction 40/40; decoy false-survival 0.01', 'survived', 'C+'),
    ('la51', 'CYP, O and *86 with houses', 'same (3c)', 'P 0.26 / 0.22 / n 1', 'killed', 'KILLED'),
    ('la23', 'Casa Room 9 lists words met nowhere else', 're-dealing of HT documents among deposits, 1,000 fresh (3d)', '0.62, P 0.33', 'killed', 'KILLED'),
    ('la23', '~370-670 named people at HT', 'needs a new person list', '-', 'untestable', 'C (untestable-now)'),
    ('la47', 'layout profiles separate la45 classes where order does not', 'fresh seed, 1,000 within-page permutations (3e)', '0.436 vs 0.335, P 0.044 (line: P > 0.2 kills)', 'survived', 'C'),
    ('la49', 'commodity and libation groups in model geometry', 'count-embedding proxy with duplicates and stone vessels held out (3f)', 'proxy finds neither group even on the full corpus: no power', 'untestable', 'C (untestable-now)'),
    ('la21', 'QA behaves like a pure vowel', '1,000-restart grid (fresh tag), 3 within-word shuffles, all non-vowel signs as decoys (3j)', 'QA 0.63 vs shuffles 0.05-0.07; decoy signs P 0.017', 'survived', 'C+'),
    ('la54', '~0.16 bits/doc about the unwritten commodity beyond site', '2 fresh real runs + 11 fresh within-site nulls (3i)', 'real 0.332 / 0.205 vs nulls 0.165 +- 0.025; P 0.05 (line: P > 0.2 kills)', 'survived', 'C-'),
    ('la31', 'words follow the easier sailing leg among big archives', 'partial Mantel without HT, 5,000 fresh permutations, 200 decoy leg re-pairings (3l)', 'r +0.19 (line: r <= 0 kills), P 0.13', 'survived', 'C'),
    ('photo', 'J = 1/2 cannot be ruled out (visual read)', 'needs specialist re-reading of PH 9b, PH 22a, HT 104', '-', 'untestable', 'C (untestable-now)'),
    ('la17', 'date-distant sites share fewer words', 'needs more MM II-III material', '-', 'untestable', 'C (untestable-now)'),
    ('la22', 'break-edge restorations', 'needs new photographs or joins', '-', 'untestable', 'C (untestable-now)'),
    ('la25', 'Zakros olives fix an autumn-winter fire', 'needs archaeobotany', '-', 'untestable', 'C (untestable-now)'),
    ('la26', 'ring pairs have alike sign profiles; RO on S68 nodules', 'exact QAP / one seal: needs new sites or seals', '-', 'untestable', 'C (untestable-now)'),
    ('la29', 'vine-suitable land as best predictor', 'needs new commodity documents', '-', 'untestable', 'C (untestable-now)'),
    ('la32', 'variants on one tablet look alike', 'needs a larger variant set', '-', 'untestable', 'C (untestable-now)'),
    ('la34', 'hand items (variants not one writer, tablets vs sealings, look-alikes)', 'needs photographs / hand attributions', '-', 'untestable', 'C (untestable-now)'),
    ('la35', 'an LB-sized plural suffix cannot be excluded', 'needs 80+ W / W+X pairs', '-', 'untestable', 'C (untestable-now)'),
    ('la38', 'weak vowel fit at HT; TO / PI / WA flags', 'needs new HT documents', '-', 'untestable', 'C (untestable-now)'),
    ('la55', 'commodities saturate, headers threshold; towns sublinear', 'needs new small deposits / sites', '-', 'untestable', 'C (untestable-now)'),
    ('la56', 'HT 127b running total; KU-RO adds name logograms', 'needs collation', '-', 'untestable', 'C (untestable-now)'),
    ('la57', 'khipu top cords follow the cuneiform convention', 'needs more top-cord khipus (not Linear A)', '-', 'untestable', 'C (untestable-now)'),
    ('la58', 'separate site administrations', 'needs a new LM IB archive', '-', 'untestable', 'C (untestable-now)'),
    ('la65', 'dossier slot typing agrees with la60 (C-)', 'depends on the la60 frozen roles; LB control passes 1/3', '-', 'untestable', 'C- (untestable-now)'),
]


def la44_fill():
    s = open(os.path.join(LOOPS, 'la67_cycle3.txt')).read()
    m = re.search(r'\| LA-67\.3k \|.*?\| (LA FUS_SUF[^|]*)\| ([^|]*)\|', s)
    if not m:
        return 'not finished', 'untestable', 'C (re-run unfinished)'
    res, verd = m.group(1).strip(), m.group(2).strip()
    if verd.startswith('KILLED'):
        return res, 'killed', 'KILLED'
    if verd.startswith('SURVIVES'):
        return res, 'survived', 'C+'
    return res, 'survived', 'C-'


def main():
    r, o, g = la44_fill()
    rows = []
    for src, guess, test, res, out, grade in T:
        if res == '@LA44@':
            res, out, grade = r, o, g
        rows.append((src, guess, test, res, out, grade))
    tested = [x for x in rows if x[4] != 'untestable']
    surv = [x for x in tested if x[4] == 'survived']
    plus = [x for x in surv if x[5].startswith('C+')]
    killed = [x for x in tested if x[4] == 'killed']
    dec = [x for x in tested if x[4] == 'decoy-level']
    unt = [x for x in rows if x[4] == 'untestable']
    with open(OUT, 'w') as f:
        f.write('LA-67 final: THE KILL SWEEP (6 Oct 2026). Every grade-C (C+, C-) Linear A guess in FINDINGS.md, tested with its own would-kill line, or, where that needs new data,\n'
                'the strongest existing-data test (fresh seeds, matched decoys, held-out tablet halves, permutation nulls). Decoys went through the identical criterion.\n'
                'Scripts: tools/la67_lib.py, la67_c1.py, la67_c1b.py, la67_s318.py, la67_c2.py, la67_c3.py, la67_c3b.py, la67_chains.py, la67_ilz.py, la67_pade.py, la67_ws.py, la67_wrap.py, la67_final.py.\n'
                'Rows: loops/la67_cycle1-3.txt. Checkpoints: data/la67_ckpt/ (git-ignored).\n\n'
                'Grades: C+ = survived a held-out / decoy-calibrated test; C = survived only literally, weakly or at its line; C- = passes its line but not beyond matched decoys;\n'
                'KILLED = failed; untestable-now = needs data that do not exist yet (grade unchanged).\n\n')
        f.write('| # | source | guess | kill test | result | new grade |\n|---|---|---|---|---|---|\n')
        for i, x in enumerate(rows, 1):
            f.write('| %d | %s | %s | %s | %s | %s |\n' % (i, x[0], x[1], x[2], x[3], x[5]))
        f.write('\n| LA-67.final | %d grade-C guesses (groups) listed; %d tested now, %d untestable-now. | Survived %d (C+ %d: %s); passed the line but not beyond decoys %d (C-); killed %d (%s). | '
                'NOT CRACKED. The sweep removes %d of %d testable guesses; the survivors are structural (word-internal consonant avoidance, QA in the vowel row, first-sign runs outside HT, the staple order, room links), none is a reading. |\n' % (
                    len(rows), len(tested), len(unt), len(surv), len(plus), '; '.join(x[1] for x in plus), len(dec), len(killed), '; '.join(x[1] for x in killed),
                    len(killed), len(tested)))
    print(open(OUT).read())


if __name__ == '__main__':
    main()
