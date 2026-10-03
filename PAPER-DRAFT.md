# Indus seal inscriptions as credentials, not names: calibrated and pre-registered tests, with the Umma sealing as an exported standard formula

*Draft, 2 October 2026 (adds the pot comparison and the 'no language to decipher' argument; Umma sealing corrected to the Ashmolean). Authors: P. Garg (and co-authors to be agreed). The code and data for every number are in this repository; the strategy IDs (S###) point to the log in STRATEGIES.md.*

## Abstract
Most attempts to decipher the Indus script assume that seal inscriptions record owners' names, spelled phonetically. Mukhopadhyay (2023) argued instead that the inscriptions contain no proper nouns and served licensing, taxation and access control. We test this "no proper nouns" model quantitatively for the first time, on the merged Wells/ICIT corpus (3,599 distinct texts), with Mahadevan's 1977 transcription (IM77) as a transcription check.
(1) **Calibrated name test.** Real seal-owner names from Mesopotamia (6,240 names, one legend per seal, CDLI) repeat far more than a bigram generator trained on them predicts: the uniqueness ratio is 0.62. The Indus seal 'middles' do not repeat any more than such a generator predicts: the ratio is 0.97.
(2) **Pre-registered credential test.** We model a seal text as a credential: an introductory 'shield' sign, then holdings or dealings, then an office word. We froze two predictions before testing and checked them on 239 texts from sites absent from IM77. Short texts recur whole inside longer ones at 4.1× the within-text-shuffled rate, and 40% of the held-out texts reuse a ≥3-sign run known from Mohenjo-daro or Harappa (5×).
(3) **Nameless quantity seals.** Thirty-one seals from eight sites carry only 'shield · of · N · TREE', with N = 3–6, the same counts as the Harappa voucher tablets. The Indus sealing from Umma (southern Iraq) carries this same formula, '… of 4 TREE', which appears as the complete text of seven home seals.
(4) **Seal–sealing flow.** All 18 clay sealings that match a seal text were found at a different site from that seal, mostly in the direction Mohenjo-daro → Lothal and northern towns.
(5) **Pots and other objects.** Pots carry one-sign labels from the same symbol set (half are a single sign: 'jar', a numeral, 'tree'), either stamped at the workshop or scratched on by users; no object class carries running text (mean 3.3 signs over 5,369 objects; maximum 17).
Together these results support reading Indus seal texts as standardised credentials rather than personal names, and the sign system as a structured business code rather than a record of speech. **The Indus script has resisted decipherment because there is probably no written language in it to recover**: what survives is a coded symbol system for trade and administration, with a slot grammar, counts and qualifiers, comparable to hallmarks or shipping codes, not to sentences. It can be decoded by function, not read aloud. We also propose a testable anchor: the tree sign as a timber unit, since timber (gesh ab-ba Meluhha) is the Meluhhan good most often recorded at Umma.

## 1. Background
- **Structure.** Indus seal texts have fixed beginners and enders, a fixed direction and correlated sign order (Yadav et al. 2009; Rao et al. 2009). Long texts divide into recurring segments (Yadav, Vahia, Mahadevan & Joglekar, 'Segmentation of Indus texts').
- **Function.** Mukhopadhyay (2023, *Humanities and Social Sciences Communications*) argued, largely from archaeological context and qualitative script-internal evidence, that the inscriptions encode no proper nouns and served taxation, trade and craft licensing and access control. Earlier readings (Mahadevan, Parpola) took the texts to be names plus titles.
- **Decipherment by statistics.** Statistical decipherment can produce convincing but wrong readings: planted keys 'read' the same symbols as English, Sanskrit or Tamil (arXiv 2608.02999). Our own planted-language controls agree (S107–S137).
- **What is missing, and what this paper adds.** No study has tested the no-proper-noun claim against real names with a calibrated null model, or tested a structural model of the texts on held-out sites with pre-registered predictions. This paper does both.

## 2. Data
- **Main corpus.** The merged Wells/ICIT corpus (`data/derived/merged-corpus-canonical.json`): 5,369 inscribed objects, 3,599 distinct texts after deduplication by site and text, with object type, emblem, find-spot and dimensions (`data/raw/inscriptions.csv`).
- **Transcription check.** IM77 (Mahadevan 1977), used for transcription-robustness checks. About 70% of its texts are the same objects as in the main corpus (S312), so agreement with IM77 is not independent replication.
- **Real names.** Mesopotamian seal legends from CDLI (ATF export).
- **Meluhha attestations.** The CDLI Meluhha attestations (214 texts).
- **Held-out set.** Texts from sites absent from IM77 (239 texts; S349).

## 3. Results
### 3.1 The middles are not a stock of personal names (S321–S322)
**Method.** For each text we remove the frame: the opener with its connective, and the closer with its suffix. We then compare the share of unique 'middles' with a bigram model trained on the same middles. The ratio of observed uniqueness to model uniqueness is the statistic. We apply the same procedure to the Mesopotamian seal-owner names (line 1 of each legend, one legend per seal), matching the sample size (1,748) over 50 draws.

**Results.**
- Real names: 42.5% unique, against 68.6% under the bigram model (ratio 0.62). Common names recur: ur-Baba 111 times, lu2-dingir-ra 78, ur-Lamma 76.
- Indus middles: 84.6% unique, against 87.4% under the model (ratio 0.97).

A population's names repeat because people share names; the Indus middles do not repeat in this way. Caveat: the Mesopotamian names are written syllabically, so the unit differs, but the ratio is measured against each corpus's own null model.

### 3.2 A credential model, tested on held-out sites (S347–S349)
**Model.** A seal text consists of:
- a shield or opener (leaf, diamond, wheel, bracket or X) plus a connective;
- holdings or dealings (counts of goods; fish-class words);
- an office word from a closed set of about ten mutually exclusive closers, each with its own qualifiers (S289–S303).

**In-sample results.**
- Short texts recur inside longer ones 4× more than when signs are shuffled within each text: 15.2% against 3.7%.
- The longer and shorter versions of a credential show no ordering in time (p = 0.40), so they are parallel grades rather than promotions.

**Pre-registered test.** Predictions were frozen in `prereg/preregistration-4-credential.json` before the test, and checked on the held-out sites.

| Prediction | Held-out result | Shuffled control | Ratio |
|---|---|---|---|
| P1: nesting | 13.6% | 3.3% | 4.1× |
| P2: reuse of big-city runs | 40.2% | 7.9% | 5.1× |

Both predictions pass.

### 3.3 Nameless quantity seals and the Umma sealing (S335–S346)
- **Quantity seals.** Thirty-one texts consist only of 'opener · 2 · N · item', mostly with the item W390 (the tree or branch sign). They come from 8 sites and include two Kalibangan sealings, so the seals were used. Within Mohenjo-daro they are not clustered in one quarter, and seal size does not rise with N (ρ = −0.38, p = 0.14).
- **Counts.** On seals, the numbers attached to the tree sign peak at 3 and 4. The Harappa voucher counts (tall 2–4 + W700) peak at the same values.
- **The Umma sealing.** The Umma sealing (Ashmolean Museum, Oxford, 1931.120; published by Scheil 1925; Parpola 1994 no. 25) reads 127-705 · 2 · 4 · 390. Its ending, '2-4-390', is the complete text of seven home seals (Mohenjo-daro M-103, M-278, M-984, M-1750, M-1844; Harappa H-2003 and H-55; Dholavira; Surkotada). We are not aware of this equivalence being noted before.
- **Candidate anchor.** In CDLI texts about Meluhha, the most frequently named good is a wood, {gesz}ab-ba me-luh-ha, used for furniture at Umma (P107404, P249043), Ur, Girsu and Isin. We therefore propose W390 = a timber unit. This is a hypothesis with a stated test: sealing backs (log or rope impressions), and Mesopotamian wood accounts with counts of 3–6 units. A point against it: the back of the Umma bulla carries the impression of textile, so it was attached to a cloth-covered bale (Parpola 1994), which fits bundled goods better than logs. The find-spot 'Umma' rests only on a dealer's report that Scheil himself doubted (Scheil 1925), so the object is best rated unprovenanced; the bulla carries no cuneiform.

### 3.4 Where sealed goods went (S324–S330)
- **All matches are cross-site, but mostly by seal-count.** Eighteen sealings match a seal text exactly. All 18 were found at another site: Mohenjo-daro → Lothal 13, and → Harappa, Kalibangan, Hulas and Rupar; one match runs Harappa → Lothal. A null that permutes seal site labels (keeping each site's seal count) already predicts 93% cross-site matches and 8.8 Mohenjo-daro → Lothal (observed 13, p = 0.23), because Mohenjo-daro holds 63% of the long seals. The residual is that 8 of 9 distinct matched texts come from Mohenjo-daro against 56% expected (p = 0.012, uncorrected). The direction of flow is therefore suggestive, not established.
- **Lothal.** The ten Lothal bale sealings (L-161–170) carry 817-2-48-740 with an **elephant**. The only seal with this text, Mohenjo-daro M-895, has a **unicorn**, and Lothal has no elephant seals at all (0 of 43). So the same credential was held under different emblems, and the bales were sealed away from Lothal.
- **Kish.** A second Kish seal reproduces the full text of Mohenjo-daro seal M-94 (3-220-590-390-740), preceded by a unit, 416-840, that also occurs at Mohenjo-daro, Dholavira and Harappa. An identical Kish–Mohenjo-daro pair (Mackay 1925 / M-228) has been noted before; this is a second such case.
- **Harappa tablets.** Harappa tablets reproduce seal texts from Mohenjo-daro, Lothal and Dholavira, e.g. 415-220-520, which occurs on 15 Harappa tablets and 1 Mohenjo-daro seal (S344).

## 4. Discussion
The results agree with Mukhopadhyay's (2023) thesis and add calibrated, pre-registered and held-out tests. The emblem is independent of the text: emblem × title association P = 0.38 (S329); the same text appears under different animals (S325). The script behaves as a word-sign labelling system with a fixed slot grammar, not as recorded speech: repetition is formula-like, there are no spelling traces, and phonetic keys fail their controls. On this view, 'decipherment' means identifying what the credential terms referred to (goods, counts, offices) through outside matches, not recovering a spoken language.

**Why no decipherment has succeeded.** Every decipherment since 1925 has assumed a script that writes a language and sought its sound values. Our evidence points the other way: (a) no object class carries running text (pots and bangles mostly one or two signs; seals 4.2 on average; the longest text 17 signs); (b) the same small set of symbols is used in full on seals, sealings and tablets and as single labels on pots; (c) the middles do not behave like a population's names; (d) phonetic keys 'read' planted controls equally well, so their successes carry no information. If the signs encode business categories (authority, goods, counts, makers) rather than words, there is no language to recover, and the long run of failed readings is the expected result, not a puzzle. Three cautions. First, statistics alone cannot settle the question either way: Raghavendra (2026) built an emblem system with meaning but no sound values that matched the Indus signs on every reproducible published test. Our argument therefore rests on positive features of an accounting code (counted goods, a closed set of office words, the same formulas sealed across cities, one-sign labels on pots, no long text on any object), not on entropy or Zipf statistics. Second, this does not show that the Harappans lacked spoken language, or that they never wrote on perishable materials (cloth, palm leaf, wood) that have not survived. Third, the claim is falsifiable: one long running Indus text, of the kind found in every literate neighbour, would overturn it. The non-linguistic view was first argued by Farmer, Sproat & Witzel (2004) and disputed by Rao et al. (2009); the tests here add calibrated, pre-registered and held-out evidence to that debate.

## 5. Limitations
- The corpus is a small sample of all seals ever made: Chao1 estimates about 950 signs, of which about 240 are unseen (S309). Most texts come from Mohenjo-daro and Harappa (76%).
- Sign identities depend on the transcription. Allograph merges were run at three levels (strict, strong, all), and the results hold at each.
- The Mesopotamian comparison mixes a syllabic script with a word-sign script. The within-corpus null model mitigates this, but does not remove it.
- The timber reading of W390 is a hypothesis, not a demonstration, and the textile impression on the back of the Umma bulla argues against it. The Umma provenance is a dealer's report.
- A literature check (web; Mukhopadhyay 2023; Yadav et al.; Laursen 2010; arXiv 2608.02999) found no prior report of results 3.1, 3.2 (pre-registered form), 3.3 or 3.4 (cross-site statistics). A specialist check of Parpola's CISI commentary is still pending.

## 6. Reproducibility
The scripts are in `tools/`. Each is named after its STRATEGIES row:
- `strat_namecalib.py`, `strat_namecalib2.py` (S321–S322)
- `strat_credential.py`, `strat_credential_heldout.py` (S347, S349)
- `strat_batch1.py` (S336–S338)
- `strat_sealflow.py` (S324)
- `strat_branches.py`, `strat_590title.py`, `strat_abroadsuffix.py` (S326–S328)
- `parse_all.py` (S310)
- `strat_potcompare.py` (S354, S355)
- `strat_counted.py`, `strat_countedcloser.py` (S350, S351)

## References
- Farmer, S., Sproat, R. & Witzel, M. 2004. The collapse of the Indus-script thesis: the myth of a literate Harappan civilization. *Electronic Journal of Vedic Studies* 11(2): 19–57.
- Mukhopadhyay, B. A. 2023. Semantic scope of Indus inscriptions comprising taxation, trade and craft licensing, commodity control and access control. *Humanities and Social Sciences Communications* 10. https://www.nature.com/articles/s41599-023-02320-7
- Yadav, N., Joglekar, H., Rao, R. P. N., Vahia, M. N., Mahadevan, I., Adhikari, R. 2009/2010. Statistical analysis of the Indus script using n-grams. arXiv 0901.3017; *PLoS ONE* 5(3): e9506.
- Yadav, N., Vahia, M. N., Mahadevan, I., Joglekar, H. Segmentation of Indus texts. https://www.harappa.com/sites/default/files/pdf/indus-texts.pdf
- Rao, R. P. N. et al. 2009. A Markov model of the Indus script. *PNAS* 106.
- Laursen, S. T. 2010. The westward transmission of Indus Valley sealing technology. *Arabian Archaeology and Epigraphy* 21: 96–134.
- Parpola, A. 1994. Harappan inscriptions: an analytical catalogue of the Indus inscriptions from the Near East. In F. Højlund & H. H. Andersen, *Qala'at al-Bahrain* 1: 304–315. Aarhus.
- Scheil, V. 1925. Un nouveau sceau hindou pseudo-sumérien. *Revue d'Assyriologie* 22: 55–56.
- Mahadevan, I. 1977. *The Indus Script: Texts, Concordance and Tables.*
- Shah, S. G. M. & Parpola, A. (eds) 1987–1991; Parpola et al. 2010. *Corpus of Indus Seals and Inscriptions.*
- Raghavendra 2026. On the non-specificity of statistical measures used in script decipherment. arXiv 2608.02999.
- CDLI, Cuneiform Digital Library Initiative. https://cdli.earth
