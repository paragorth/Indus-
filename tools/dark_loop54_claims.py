"""Loop 54: the quantitative claim behind each WORKING-DICTIONARY.md entry (A and B, plus the testable C closers) and each
GRAMMAR.md statement, as a statistic that can be computed on the corpus and on null corpora.
kind: 'adjacency' = a bigram-with-start/end fact (a first-order chain reproduces it in expectation by construction);
      'position' = where the sign stands (start/end; a chain with lengths kept does not know the end);
      'distance' = co-occurrence, exclusion, order or repetition at distance >= 2 (beyond a first-order chain);
      'trigram'  = three-sign run (beyond order 1, within order 2);
      'corpus'   = whole-corpus statistic.
'outside' names the outside fact (object type, site, Gulf find, emblem, second transcription) the entry also rests on."""
from dark_loop54_common import *

def claims(lib, S, tr=lambda ws: set(ws)):
    """lib = mk(S); tr translates a set of Wells signs to the working sign space (identity for W, bridge for M)."""
    L = lib; t = tr
    def T(*ws): return t(set(ws))
    C = []
    def add(entry, grade, claim, kind, f, outside='', srow=''):
        C.append({'entry': entry, 'grade': grade, 'claim': claim, 'kind': kind, 'f': f, 'outside': outside, 'srow': srow})
    short = T(1, 3, 4, 5, 16, 17, 18); tall = T(31, 32, 33, 34); num = S['NUM']
    tree = T(390, 405, 406, 407); jar = T(740); fishq = T(235, 240, 233, 231); fish = T(235, 240, 233, 231, 220)
    counted = T(390, 405, 406, 407, 900, 220, 740, 700)
    op = T(817, 861, 820); mark = T(2, 60); cl = S['CL']
    # ---- 1. numbers
    add('W1-W5,W16-W18 numerals 1-8', 'A', 'short numeral followed by a counted item (next in COUNTED)', 'adjacency', L['nxt'](short, counted), '', 'S23,S70')
    add('W1-W5,W16-W18 numerals 1-8', 'A', 'distinct numerals before the tree (counted = many)', 'adjacency', L['numvar'](tree), '', 'S70,S199')
    add('W1-W5,W16-W18 numerals 1-8', 'A', 'two numeral tokens in one text, non-adjacent (numerals mutually exclusive: low)', 'distance', L['two_tokens_nonadj'](num), '', 'S-DARK-26,S366')
    add('W31-W34 tall strokes', 'A', 'tall numeral followed by W700 (voucher count)', 'adjacency', L['nxt'](tall, T(700)), 'Harappa tablets only (S218): site/object fact', 'S197-S217')
    add('W31-W34 tall strokes', 'A', 'W700 preceded by a tall numeral', 'adjacency', L['prv'](tall, T(700)), 'Harappa tablets only (S218)', 'S197-S217')
    add('W55 twelve', 'A number / B fixed term', 'W55 followed by the jar or by 255 (title slot), not by a counted good', 'adjacency', L['nxt'](T(55), T(740, 255)), 'enriched abroad (S273, S284): Gulf fact; 50x frequency = unigram, untestable', 'S273')
    add('W55 twelve', 'A number / B fixed term', 'W55 followed by a counted good (should be low)', 'adjacency', L['nxt'](T(55), counted - jar), '', 'S273')
    add('W56 twenty-four', 'B', 'W56 followed by W2 (the Kalibangan 91-56-2 formula)', 'adjacency', L['nxt'](T(56), T(2)), 'Kalibangan moulded tablets K-69-75 (S149): object/site fact; 8 copies of one text', 'S149')
    add('W2 connective', 'A connective', 'W2 in second position', 'position', L['second'](T(2)), '', 'GRAMMAR')
    add('W2 connective', 'A connective', 'W2 preceded by an opener', 'adjacency', L['prv'](T(817, 861, 820, 920, 692), T(2)), '', 'GRAMMAR')
    add('W2 connective', 'A connective', 'W2 initial (never: low)', 'position', L['init'](T(2)), '', 'GRAMMAR')
    add('W2 connective', 'A connective', 'W2 twice in one text per 1000 texts with W2 (once rule: low)', 'distance', L['repeat'](T(2)), '', 'S-DARK-15')
    add('W2 connective', 'A connective', 'W2 text also carries marked jar 741/742 at any distance (exclusion: low)', 'distance', L['cooc'](T(2), T(741, 742), 1), '', 'S-DARK-15,S333')
    add('W60 connective', 'A', 'W60 in second position', 'position', L['second'](T(60)), '', 'S139,S331')
    add('W60 connective', 'A', 'W60 preceded by an opener (817/861/820/920/692)', 'adjacency', L['prv'](T(817, 861, 820, 920, 692), T(60)), '', 'S331')
    add('W60 connective', 'A', 'W60 twice per 1000 texts with W60 (low)', 'distance', L['repeat'](T(60)), '', 'S-DARK-15')
    add('W60 connective', 'A', 'W60 initial (low)', 'position', L['init'](T(60)), '', 'GRAMMAR')
    # ---- 2. frame words
    add('W817/W861 opener', 'A function', 'W817/861 initial', 'position', L['init'](T(817, 861)), 'owned seals; on sealings at half the seal rate (S318): object fact', 'S29,S66')
    add('W817/W861 opener', 'A function', 'W817/861 followed by W2', 'adjacency', L['nxt'](T(817, 861), T(2)), '', 'GRAMMAR')
    add('W817/W861 opener', 'A function', 'opener text also carries suffix W400 at any distance (exclusion: low)', 'distance', L['cooc'](T(817, 861, 820), T(400), 1), 'seal vs tablet sub-systems (S29, S66)', 'S29,S50')
    add('W817/W861 opener', 'A function', 'opener text also carries a closer (jar etc.) at distance >= 2', 'distance', L['cooc'](T(817, 861, 820), cl, 2), '', 'GRAMMAR')
    add('W817/W861 opener', 'A function', 'opener twice per 1000 texts with it (low)', 'distance', L['repeat'](T(817, 861, 820)), '', 'S88')
    add('W820 wheel opener', 'A function', 'W820 initial', 'position', L['init'](T(820)), '', 'S331')
    add('W820 wheel opener', 'A function', 'W820 final (suffix stripped; S286 opener-as-closer)', 'position', L['final'](T(820)), '', 'S286')
    add('W820 wheel opener', 'A function', 'W820 followed by W2 or W60', 'adjacency', L['nxt'](T(820), mark), '', 'S331')
    add('W740 jar closer', 'A function / C word', 'W740 final (suffix stripped)', 'position', L['final'](jar), 'also counted T2+jar x46: bigram', 'S289')
    add('W740 jar closer', 'A function / C word', 'jar text also carries another closer at any distance (mutual exclusion: low)', 'distance', L['cooc'](jar, cl - jar, 1), '', 'S289,S291')
    add('W740 jar closer', 'A function / C word', 'W740 twice per 1000 jar texts (low)', 'distance', L['repeat'](jar), '', 'S88')
    add('W740 jar closer', 'A function / C word', 'S289 closer-paradigm count (signs >=40% final with jar O/E <= 0.5)', 'corpus', L['closer_paradigm'], 'IM77 prior prediction S291 (transcription-robust)', 'S289')
    add('W400 suffix', 'A function', 'W400 raw-final', 'position', L['final'](T(400), strip=False), 'tablets/rods 21-25% vs sealings 1% (S318): object fact', 'S29')
    add('W400 suffix', 'A function', 'W400 preceded by a closer', 'adjacency', L['prv'](cl, T(400)), '', 'S24,S29')
    add('W400 suffix', 'A function', 'W400 text also carries an opener (exclusion: low)', 'distance', L['cooc'](T(400), T(817, 861, 820), 1), 'seal vs tablet (S29)', 'S29,S50')
    add('W90 man suffix', 'B', 'W90 raw-final', 'position', L['final'](T(90), strip=False), 'person first on Gulf seals (S47, S74): Gulf fact', 'S24')
    add('W90 man suffix', 'B', 'W90 preceded by a closer', 'adjacency', L['prv'](cl, T(90)), '', 'S24')
    add('W90 man suffix', 'B', 'W90 preceded by a numeral (plain person never counted: low)', 'adjacency', L['prv'](num, T(90, 91)), '', 'S98')
    add('W91 twins', 'B', 'W91 adjacent to a numeral', 'adjacency', L['cooc'](T(91), num, 1), '28% of texts abroad (S274); trade objects (S316): Gulf/object facts', 'S293')
    add('W595 pre-closer', 'A function', 'W595 text also carries the jar at any distance (avoidance: low)', 'distance', L['cooc'](T(595), jar, 1), '', 'S288')
    add('W595 pre-closer', 'A function', 'W595 followed by W820', 'adjacency', L['nxt'](T(595), T(820)), '', 'S288')
    add('W595 pre-closer', 'A function', 'W595 adjacent to W820 in either order: share with 595 first', 'adjacency', L['before'](T(595), T(820), 1), '', 'S288')
    add('W595 pre-closer', 'A function', 'W595 final (suffix stripped)', 'position', L['final'](T(595)), '', 'S288')
    # ---- closers (C entries, closed set)
    for w, nm, g in [(520, 'W520 arrow closer', 'C'), (151, 'W151 carrier closer', 'C-'), (156, 'W156 jar-carrier closer', 'C'), (527, 'W527 hatched box closer', 'C'),
                     (226, 'W226 fish-4 closer', 'C'), (617, 'W617 closer', 'C'), (154, 'W154/158 closer', 'C'), (700, 'W700 tablet unit', 'B')]:
        X = T(w) if w != 154 else T(154, 158)
        add(nm, g, f'{nm.split()[0]} final (suffix stripped)', 'position', L['final'](X), 'W617/W700 transcription-fragile (S-DARK-24.3); W700 Harappa tablets only' if w in (617, 700) else '', 'S289')
        add(nm, g, f'{nm.split()[0]} text also carries the jar at any distance (low)', 'distance', L['cooc'](X, jar, 1), '', 'S289')
    add('W520 arrow closer', 'C', 'W520 preceded by tall 3 or a fish', 'adjacency', L['prv'](T(33) | fish, T(520)), '', 'S303')
    add('W156 jar-carrier closer', 'C', 'W156 preceded by short 3', 'adjacency', L['prv'](T(3), T(156)), '', 'S303')
    add('W527 hatched box closer', 'C', 'W527 preceded by pincer 550/555', 'adjacency', L['prv'](T(550, 555), T(527)), '', 'S303')
    add('W226 fish-4 closer', 'C', 'W226 preceded by tall 2', 'adjacency', L['prv'](T(32), T(226)), '', 'S303')
    add('W617 closer', 'C', 'W617 preceded by W142', 'adjacency', L['prv'](T(142), T(617)), '', 'S303')
    add('W154/158 closer', 'C', 'W154/158 preceded by leaf-tree 803/806', 'adjacency', L['prv'](T(803, 806), T(154, 158)), '', 'S303')
    # ---- 3. titles before the jar
    add('W176 seated man', 'C', 'W176 followed by the jar or W100', 'adjacency', L['nxt'](T(176), T(740, 100)), '38 copies of 176-740-400 (whole-text copies)', 'S301')
    add('W176 seated man', 'C', 'W176 before W100 at any distance (order; 13:0)', 'distance', L['before'](T(176), T(100), 1), '', 'S298,S302')
    add('W176 seated man', 'C', 'texts with 100-jar also carry 176 earlier, at distance >= 2 from 100', 'distance', L['cooc'](T(100), T(176), 2), '', 'S302')
    add('W176 seated man', 'C', 'run 176-100-740 per 1000 texts', 'trigram', L['trigram'](T(176), T(100), jar), '', 'S302')
    add('W100 three-headed man', 'C', 'W100 preceded by another person sign (176/140)', 'adjacency', L['prv'](T(176, 140), T(100)), '', 'S298')
    add('W760, W923, W690, W482, W752, W48 pre-jar titles', 'B function', 'followed by the jar', 'adjacency', L['nxt'](T(760, 923, 690, 482, 752, 48), jar), '', 'S229,S341')
    add('W923 title', 'C', 'W923 preceded by tall 3 (33-923 fixed)', 'adjacency', L['prv'](T(33), T(923)), '', 'S197')
    # ---- 4. counted things
    add('W390/W405 tree', 'B counted / C word', 'tree preceded by a numeral', 'adjacency', L['prv'](num, tree), 'Umma bulla (S345) outside find', 'S70')
    add('W390/W405 tree', 'B counted / C word', 'distinct numerals before the tree', 'adjacency', L['numvar'](tree), '', 'S70')
    add('W390/W405 tree', 'B counted / C word', 'tree final (suffix stripped; self-closing)', 'position', L['final'](tree), '', 'S350')
    add('W390/W405 tree', 'B counted / C word', 'numeral-tree pair preceded by W2 (opener-of-N-tree seal type), i.e. run 2-NUM-tree per 1000', 'trigram', L['trigram'](T(2), num, tree), '31 texts, 8 sites (S336)', 'S336')
    add('W220 plain fish', 'C', 'W220 preceded by a numeral', 'adjacency', L['prv'](num, T(220)), 'on seals 85% vs 70% (S292): object fact', 'S292')
    add('W900 bracket', 'C', 'W900 preceded by a numeral', 'adjacency', L['prv'](num, T(900)), '', 'S341')
    add('W900 bracket', 'C', 'W900 followed by the jar', 'adjacency', L['nxt'](T(900), jar), '', 'S341')
    add('W575 seven-X', 'B fixed / C meaning', 'modal numeral share before W575 (always 7)', 'adjacency', L['numfix'](T(575)), '', 'S197')
    add('W585 seven-X', 'B', 'modal numeral share before W585', 'adjacency', L['numfix'](T(585)), '', 'S197')
    add('W632 two-X', 'B', 'modal numeral share before W632', 'adjacency', L['numfix'](T(632)), '', 'S197')
    add('W632 two-X', 'B', 'W632 preceded by a numeral at all', 'adjacency', L['prv'](num, T(632)), '', 'S197')
    # ---- 5. fish family and name signs
    add('fish family W220/240/235/233/231', 'B', 'two modified fish adjacent: predicted order share (hat>whiskers>bar>stroke)', 'adjacency', L['before'](T(235), T(240, 233, 231), 1), '', 'S296-S297')
    add('fish family W220/240/235/233/231', 'B', 'two modified fish NON-adjacent (distance >= 2): predicted order share 235 first', 'distance', L['before'](T(235), T(240, 233, 231), 2), '', 'S297')
    add('fish family W220/240/235/233/231', 'B', '240 before 233/231 at distance >= 2', 'distance', L['before'](T(240), T(233, 231), 2), '', 'S297')
    add('fish family W220/240/235/233/231', 'B', 'share of texts with two or more distinct fish signs (stacking)', 'distance', L['two_of'](fish), '', 'S296')
    add('fish family W220/240/235/233/231', 'B', 'modified fish stacked adjacently: run 235-240', 'adjacency', L['nxt'](T(235), T(240)), '', 'S296')
    add('arrow phrase fish (+705-33) + W520', 'B', 'W520 preceded by a fish word', 'adjacency', L['prv'](fish, T(520)), '', 'S299')
    add('arrow phrase fish (+705-33) + W520', 'B', 'W520 with a fish exactly two before and a non-fish between (fish + 1 sign + arrow)', 'distance', L['at2'](fish, T(520)), '', 'S299')
    add('arrow phrase fish (+705-33) + W520', 'B', 'run 705/706-33-520 per 1000', 'trigram', L['trigram'](T(705, 706), T(33), T(520)), '', 'S300')
    add('arrow phrase fish (+705-33) + W520', 'B', 'W705/706 preceded by a fish word', 'adjacency', L['prv'](fish, T(705, 706)), '', 'S300')
    add('arrow phrase fish (+705-33) + W520', 'B', 'texts with 520 also carrying a fish at distance >= 2', 'distance', L['cooc'](T(520), fish, 2), '', 'S299')
    add('W415+W803 Gulf formula', 'C', 'run 415-803-1 per 1000', 'trigram', L['trigram'](T(415), T(803, 806), T(1)), 'Ur and Dilmun finds (S99-S104): Gulf fact', 'S99')
    add('W140 woman', 'C', 'W140 initial', 'position', L['init'](T(140)), 'round seals at home (S106): object fact', 'S106')
    # ---- 8. name elements (S311)
    ni = T(692, 575, 125, 416, 413, 920, 495); nf = T(840, 460, 435, 440, 717, 70, 35, 690)
    add('name-initial elements W692/575/125/416/413/920/495', 'B', 'share of tokens standing first in the middle (after opener+marker)', 'position', L['midinit'](ni), 'held-out sites P = 0.0005 (S313)', 'S311')
    add('name-initial elements W692/575/125/416/413/920/495', 'B', 'share of tokens text-initial', 'position', L['init'](ni), '', 'S311')
    add('name-final elements W840/460/435/440/717/70/35/690', 'B', 'share of tokens standing last before the closer', 'position', L['midfinal'](nf), 'held-out sites (S313)', 'S311')
    add('name-final elements W840/460/435/440/717/70/35/690', 'B', 'followed by a closer', 'adjacency', L['nxt'](nf, cl), '', 'S311')
    # ---- 10. functions of further frequent signs
    add('W255-W435(-W690) unit', 'B function', '255 followed by 435', 'adjacency', L['nxt'](T(255), T(435)), '', 'S224,S341')
    add('W255-W435(-W690) unit', 'B function', 'run 255-435-690 per 1000 texts', 'trigram', L['trigram'](T(255), T(435), T(690)), '', 'S224')
    add('W255-W435(-W690) unit', 'B function', '255 before 690 at distance >= 2 (share 255 first)', 'distance', L['before'](T(255), T(690), 2), '', 'S224')
    add('W705/706-W33 unit', 'B function', '705/706 followed by tall 3', 'adjacency', L['nxt'](T(705, 706), T(33)), '', 'S341')
    add('W590-W390/405 unit', 'B function', '590 followed by the tree', 'adjacency', L['nxt'](T(590), tree), '', 'S229,S341')
    add('W590-W390/405 unit', 'B function', 'run 590-tree-740 per 1000', 'trigram', L['trigram'](T(590), tree, jar), 'Kish and Gonur finds (S326): outside', 'S326')
    add('W845-W407 copper-tablet caption', 'B function', '845 followed by 407', 'adjacency', L['nxt'](T(845), T(407)), 'copper tablets (S306); IM77 blind M169 final (S307)', 'S306')
    add('W368, W892 name endings', 'B function', 'final (suffix stripped)', 'position', L['final'](T(368, 892)), '', 'S341')
    add('W503, W413, W824, W255, W550 opening elements', 'B function', 'text-initial share', 'position', L['init'](T(503, 413, 824, 255, 550)), '', 'S341')
    add('W61 medial word', 'B function', '61 followed by the jar or W142', 'adjacency', L['nxt'](T(61), T(740, 142)), '', 'S341')
    add('W798 pincer', 'B function', '798 followed by the jar or W415', 'adjacency', L['nxt'](T(798), T(740, 415)), '', 'S341')
    add('W550 pincer variant', 'B function', '550 followed by 527 or 60', 'adjacency', L['nxt'](T(550), T(527, 60)), '', 'S341')
    add('W920+W60(+741) opener unit', 'A function (S331)', 'W920 initial', 'position', L['init'](T(920)), '', 'S331')
    add('W920+W60(+741) opener unit', 'A function (S331)', 'run 920-60-741 per 1000', 'trigram', L['trigram'](T(920), T(60), T(741, 742, 745)), '', 'S334')
    add('W692+W60 opener unit', 'A function (S331)', 'W692 initial', 'position', L['init'](T(692)), '', 'S331')
    # ---- 7. pot and bangle signs (object facts; chain fitted within type reproduces by construction)
    add('W34 tall 4 alone on pots', 'C', 'W34 as a one-sign text per 1000 W34 texts', 'corpus', lambda TT: 1000 * sum(1 for _, s in TT if s == tuple(T(34))) / max(1, sum(1 for _, s in TT if set(s) & T(34))), 'pot-enriched p = 1e-8: object fact', 'S294')
    # ---- 11. credential model (S347) and S366 mechanisms
    add('credential model (sec. 11)', 'B-', 'nesting: share of distinct 3-5-sign texts found whole inside a longer text', 'corpus', nest_share, 'killed vs Markov-1 in S-DARK-41.3', 'S347')
    # ---- GRAMMAR statements
    add('GRAMMAR frame: opener first', 'grammar', 'share of texts (>=2) beginning with 817/861/820', 'position', L['frame_opener_first'], '', 'GRAMMAR')
    add('GRAMMAR frame: connective never initial', 'grammar', 'share of connective tokens that are text-initial', 'position', L['frame_conn_initial'], '', 'GRAMMAR')
    add('GRAMMAR frame: closer last', 'grammar', 'share of texts (>=2) ending (suffix stripped) in a closer', 'position', L['frame_closer_last'], '', 'GRAMMAR,S289')
    add('GRAMMAR frame: suffix after closer', 'grammar', 'share of non-initial suffix tokens preceded by a closer or suffix', 'adjacency', L['frame_suffix_after_closer'], '', 'S24')
    add('GRAMMAR frame: slot order opener < closer', 'grammar', 'opener before closer at distance >= 2 (share)', 'distance', L['before'](T(817, 861, 820), cl, 2), '', 'S36-S38')
    add('GRAMMAR marker rules', 'grammar', 'W2/W60 twice per 1000 texts with a marker (low)', 'distance', L['repeat'](mark), '', 'S-DARK-15')
    add('GRAMMAR marker rules', 'grammar', 'marker in second position', 'position', L['second'](mark), '', 'GRAMMAR')
    add('GRAMMAR numeral series: tall before 520/700/590/226/632', 'grammar', 'share tall among numerals before 520/700/590/226/632', 'adjacency', L['prv'](tall, T(520, 700, 590, 226, 632)), '', 'S204')
    add('GRAMMAR numeral series: short before tree/900/585/575', 'grammar', 'share short among numeral-preceded tree/900/585/575 tokens', 'adjacency', L['prv'](short, tree | T(900, 585, 575)), '', 'S204')
    add('GRAMMAR minimum lot (counts before goods >= 3)', 'C (demoted)', 'share of goods (tree/520/900) preceded by short 1 or 2 or tall 1 (low)', 'adjacency', L['prv'](T(1, 2, 31), tree | T(520, 900)), 'IM77-only 3/19 (S-DARK-21.1)', 'S-DARK-15.3')
    add('GRAMMAR partial order (S-DARK-19)', 'grammar', 'fixed-pair share (BH, any distance)', 'corpus', L['fixed_pair_share'], '', 'S-DARK-19.2')
    add('GRAMMAR partial order (S-DARK-19)', 'grammar', 'fixed-pair share among pairs at distance >= 2 only', 'corpus', L['fixed_pair_share_nonadj'], '', 'S-DARK-19.2')
    add('GRAMMAR sequence rigidity (S-DARK-19.1)', 'grammar', 'anagram share among copy pairs of the same multiset', 'corpus', L['anagram_share'], '', 'S-DARK-19.1')
    add('GRAMMAR no-repeat rule (S88)', 'grammar', 'share of texts (>=3) repeating a sign non-adjacently (low)', 'distance', L['norepeat'], '', 'S88,S366')
    add('GRAMMAR two-closer exclusion', 'grammar', 'share of texts with two distinct closers', 'distance', L['two_of'](cl), '', 'S289')
    add('GRAMMAR two numerals (S366)', 'grammar', 'share of texts with two non-adjacent numeral tokens', 'distance', L['two_tokens_nonadj'](num), '', 'S366')
    add('GRAMMAR long-range attraction (S51, S67, S366)', 'grammar', 'count of sign pairs attracting at distance >= 2 (z >= 3, E >= 3)', 'corpus', L['longrange_pairs'], '', 'S366')
    add('GRAMMAR length distribution (S366)', 'grammar', 'sd of text length', 'corpus', L['length_sd'], '', 'S366')
    return C

def nest_share(T):
    S = set(s for _, s in T); subs = set()
    for t in S:
        if len(t) >= 4:
            for n in (3, 4, 5):
                if n < len(t):
                    for i in range(len(t) - n + 1): subs.add(t[i:i + n])
    small = [t for t in S if 3 <= len(t) <= 5]
    return sum(t in subs for t in small) / len(small) if small else float('nan')
