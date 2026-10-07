# Three undeciphered scripts: where things stand (state of 6-7 Oct 2026)

Proto-Elamite, Linear A and the Voynich manuscript, attacked with data only. This file is the current, cleaned status. Each script's FINDINGS.md holds the details. Those files grew by appending, so an early section can show a grade that a later loop changed. **Where two FINDINGS sections disagree, the later test wins, and this file gives the later grade.** STRATEGIES.md in each folder has one row per strategy. LITERATURE.md lists related papers.

**Bottom line: none of the three scripts is cracked.** No sign, word or passage in any of them has been given a meaning that passes calibrated controls and predicts something outside the corpus. What we have is structure (how the documents are built, how the numbers work) and many results that rule ideas out.

Context: the Indus work (a structured business code, not speech) is finished and lives outside other-scripts/; Proto-Elamite and Linear A show a comparable accounting skeleton.

## How to read the grades

| Term | Meaning |
|---|---|
| **A** | Established. It passes its controls and is unlikely to change. |
| **B** | Likely. It passes its controls, but the evidence is thinner or rests on one method. |
| **C** | A guess with a stated test. The "kill line" is the result that would refute it. C+ = a guess that survived its kill test; C- = a guess that was weakened. |
| **negative** | A test that found nothing, although a planted or known effect of the same size *was* found. A negative without such a control says nothing. |
| **control / planted** | The same test run where the answer is known: Ur III cuneiform, Linear B, Latin, or a fake signal inserted on purpose. |
| **null / decoy** | The same test on shuffled data, or on frequency-matched random signs or words. A result must beat these. |
| **held out** | Found on one part of the data, then checked on a part not used to find it. |
| **kill sweep** | A loop that re-tested every C guess against its own kill line: pe66, la67, v77. |
| **audit** | A loop that checked whether results rest on bad transcriptions or editorial restorations: pe73, pe74, la71, v83. |
| **crack attempt** | A loop that put all survivors into one frozen reading and tested it on held-out documents: pe59/pe72, la60/la70/la72, v72. |

Sampling rule: each corpus is a small fraction of what was written. "0 of n" means "fewer than about 3/n", not "never".

---

## 1. Proto-Elamite

Data: CDLI, 1,585 transliterated tablets, 95% from Susa, so nearly every result is a Susa result. Since pe74 the default loader is restoration-aware: signs and numerals restored by the editor are dropped, and damaged ones are marked.

### 1.1 What is established now (A or B)

**How a tablet is built**

| Result | Loop | Grade now |
|---|---|---|
| Entries follow a slot pattern: optional prefix, then a middle, then a final class sign, then the number. M288 ends 237 of 250 entries. | test a | A |
| The final class sign predicts the number system: some signs are measured in capacity units (M297, M002, M036, M243), others only counted (M263, M264, M346, M003); M376 takes fractions. | test b, pe51 | A (M297 and M376 as number carriers: B) |
| But the tablet decides the number system more than the sign does; decimal or sexagesimal is chosen per tablet (p 0.0005). | attack 2, pe6, pe8 | B |
| Header slot: M157 opens 37% of headers. The [M327+M342] header slot holds on held-out halves (decoys 0/40). | test d, pe66 | A (M157), B ([M327+M342]) |
| There is no dedicated total sign (142 of 330 total lines carry no sign). | pe38, pe61 | A |
| The top-edge mark '1(N34)' is a constant tag of M157-headed tablets and never equals the entry sum. | pe23 | B |
| Reverses and totals are in the same hand as the obverse. | pe25 | B |
| Sealed tablets are a distinct short document type (M157 header, M288 lines, few capacity lines). A recurring seal marks a template series (PES0329, PES0334). | pe70 | B |
| \|M153+X\| and \|M153+M342\| close the tablet on its last, unnumbered line (24 of 34 vs 13.2 by chance; held out p 0.0017). | pe71, kept by pe74 | B |
| 8 dossiers of tablets written from one template (42 tablets), including the serial dossier P008718-P008731. | pe63, pe56 | B |
| Plateau tablets use a different sign mix from Susa; there is a weak plateau writing fingerprint. | pe48 (A), pe17 (B) | A / B |
| PE reads as shared institutions with no detectable centre; outposts cannot be predicted from Susa. | pe65 | B |

**Numbers and arithmetic**

| Result | Loop | Grade now |
|---|---|---|
| Counting units 1 / 10 / 100 / 300 (N01, N14, N45, N34). N14 = 10 is firmly pinned. | attack 2 | B |
| Capacity ratios N30C : N24 : N39B : N01 : N14 = 1 : 3 : 6 : 30 : 180, pinned from per-head rates, not from totals. With N39C = 1/4 N30C this is the ladder frozen in pe59 (N39C 1, N30D 2, N30C 4, N24 12, N39B 24, N01 120, N14 720). | pe27, pe59 | B- |
| Five tablets that fail as counts close exactly as capacity texts with N14 = 6 N01 (after the pe73 photo check: 6 of 36, p 0.0002). | pe42, pe73, pe74 | B |
| M288 is a per-head allotment: 60 N39C = 2(N39B) 1(N24) per unit of the line above. It holds on bare 'M288 n' lines after person-class lines (33/49 after pe73; 21/30 on read-only tokens). There is one rung only: no ration ladder and no grade, sex or age split. The '1 N01 per unit' variant was killed (pe59); 48 and 75 per unit were killed outside one dossier (pe66). | pe27, pe28, pe59, pe64, pe69, pe74 | B (the rate), B- (its scope) |
| The 14 signs before the standard allotment form one closed class. | pe32 | B |
| M106 is one unit per head of the herd in MDP 17,085/097. Herd counts are coupled like a flock's, with M362 as the reference count. | pe27, pe20 | B |
| No corpus-wide ration rate; no hidden network of rates; no variant marker counts against a total ("red ink"); no category is left out of totals by rule. | pe27, pe28, pe24, pe30 | A (negative, calibrated) |
| PE counts look counted and capacity looks measured; no sign carries estimated or target numbers. | pe23 | B |

**Names and vocabulary**

| Result | Loop | Grade now |
|---|---|---|
| Middles are as varied as free combination: this fits names or spelled words, not a small fixed vocabulary. They are not slot codes or weighted descriptions. | test e, pe46, pe52 | B |
| Each tablet draws its name signs from its own freely reused pool (a used sign becomes 2-3x more likely to recur there). | pe7, pe9, pe45 | B |
| Sign census: about 750 base signs (551 seen); a closed set of about 425 simple signs. Hashed predictions of the post-2000 tablets passed. | pe22 | B |
| The information is in which signs share a tablet, not in their order (no ordered entry constraint, 0 of 48). | pe46, pe51, pe56 | B |
| Variants (~a, ~b) are used like their base sign. | pe31, pe43 | A / B |

**Audits.** pe73 checked the readings behind every B result against CDLI photos and line art, with planted wrong values to measure the checker's own accuracy (0.84-0.875). 76 of 79 decidable numerals and all 57 sign, seal and header checks agree. pe74 found the editor restored 1.5% of signs and 0.6% of numerals, and re-ran every B result on all tokens, read + damaged, and read only. No B headline changed; four sub-claims were demoted (see 1.2).

**Crack attempts.** pe59 froze all survivors into one reading and ran it as a generator and decoder on held-out tablets; pe72 updated it. The accounting skeleton beats simpler models and shuffled twins (pe72: 104 of 564 held-out tablets fully explained, against 78-86 for twins; 94 on read-only tokens). But the same pipeline run with the *true* proto-cuneiform readings gains no more (pe59: 0.001 bits/entry), so this test cannot tell a right meaning from a wrong one. Not cracked; no sign's meaning is read.

### 1.2 Surviving C-grade leads

| Lead | Loop | Grade | Kill line |
|---|---|---|---|
| Yahya has its own small lexicon (M056, M044, M219, M136) | pe48, pe66 | C+ | 40 or more new Yahya tablets with none of M219, M056, M136, M044 |
| M005~a is a header-slot form (shift larger than 42/42 variant decoys) | pe43, pe66 | C+ | |
| Grain-office signs M081 M266 M296 M112 M286 M248 (M265 demoted to C-, pe74) | pe15, pe66 | C+ | under 30% capacity numerals on new tablets |
| The M153 closing line names an office or a closing act | pe71 | C | 1 or fewer of the next 5 new tokens on the last signed line |
| Sealed = signed-for M288 disbursements | pe70 | C | 40 or more new sealed tablets with M157 headers and capacity lines at unsealed rates |
| Independent offices sharing one script, not a Susa-centred state | pe65 | C | a new outpost archive predicted from Susa with gain >= 0.2 |
| \|X+M342\| makes a sign unnumbered or header-like | pe34, pe66 | C | new \|X+M342\| numbered as often as X |
| Entries are combinatorial designations, not running language | x4, pe66 | C | language-like word-order arrows in a larger corpus |
| Unit scale: N39C about 0.6-0.8 l (or 5-7 l on an alias) | pe16, pe55, pe57 | C | pe57 retired the vessel window as evidence; needs PE-period vessel capacities |
| A 48-per-unit grade local to one dossier (P008764-P008810) | pe69 | C | |
| pe69 scope clause ('signs + M288' lines do not follow the rule) and pe70 'M288-final' separator, both demoted from B/B- by pe74 | pe69, pe70, pe74 | C | |
| Tablets with M297 have their own clay format (taller for width, thinner than the text predicts; frozen model held out +25 mbit, p 0.006) | pe75 | C+ | frozen model gain <= 0 on >= 30 newly measured M297 tablets |
| PE tablet heights step at ~11 mm (both museums); no ~17 mm finger (B negative) | pe75 | C | ripple lost on independent re-measurement |

About 30 further guesses cannot be tested on the existing corpus (`proto-elamite/loops/pe66_final.txt`, F28-F29).

### 1.3 Killed or demoted (do not re-run)

- Linear Elamite values do not make the middles read as Elamite names (p 0.65 against value shuffles).
- The joint arithmetic fit missed its pre-set 70% bar. Failing totals are not single slips (pe42); "off by one entry" is now C- (pe66).
- No seasonal order (pe11), no duty rota (pe40), no lamb-season clock (pe53), no flock histories (pe54).
- Names are not built from a used-up set of tokens (pe9); no kin groups are visible (pe7, pe45; but the method is blind to real Ur III families too).
- Bones cannot name the herd signs (pe37); physiology cannot read activities (pe55); seal pictures show no sign link at this sample size (pe33, pe35; the caprid-herd and M139 leads were killed); excavated materials do not label outpost signs (pe67).
- Type inference (pe38), unsupervised translation to proto-cuneiform (pe39), simulated societies (pe41), the form tree as a dictionary (pe43), clerk habit groups (pe49) and the deletion game (pe51) all fail their own controls.
- pe52/pe58 sign weights: B -> C- (they fail on held-out tablets, pe72).
- pe68 "team rule" (M288 = 60 x the sum of the run): killed by pe69 (1 of 38 decisive runs).
- pe62 "'M056 n' is a hidden M288 line": killed by pe69.
- pe70 "M153 compounds name the sealing party": killed by pe71.
- pe66 kills: M388 tablets read in capacity; 48 and 75 N39C per unit; M297 running totals; \|M036+1(N30D)\| = 5 per unit; the pe61 commodity candidates (0/11); the name-frame elements; the pe43 twig families; closed-frame sign pairs; rare-number tablet pairs; the companion-line ratio; several "sticky sign" claims.
- No master table behind the tablets (pe47); writing sessions cannot be rebuilt (pe29); clay colour cannot separate site from photo batch (pe26).
- Our own photo reading cannot replace an expert (sign recall 3%, pe60).

### 1.4 Frozen outside-corpus predictions

None has been killed or supported yet: each is below its frozen sample size on the 5 newly transliterated tablets that exist (pe60). Hashes are as recorded in the loop files.

| File (in `other-scripts/proto-elamite/`) | What it predicts | Recorded hash |
|---|---|---|
| `data/pe72_frozen_predictions.json` (reading `data/pe72_reading_frozen.json`) | Q1-Q13: unit size, Yahya set, sealed type, per-head rule, header slot, decode rates | b54af26b4af0ce43 (reading 9aec01ce38e72d42) |
| `data/pe59_frozen_predictions.json` | P2-P10: class -> system, totals, edge tag, outpost rates, the 89 Tehran tablets | 599fd3459b9311a5 |
| `data/pe71_frozen_predictions.json` | the M153 closing line on new tokens | d90726c10d1208ad |
| `data/pe69_frozen_predictions.json` | 18 lost M288 values, 4 lost counts | 2130b6cd6146fc16 |
| `data/pe68_frozen_broken_predictions.json`, `data/pe68_frozen_join_predictions.json` | what broken tablets lost; join candidates | 398c02e608b7fc18, fca3466e3cea930b |
| `data/pe64_ladder_frozen.json` | L1: 60 per unit on new M054/M388/M124 -> M288 pairs; L2: no 30 or 20 rung | 4126d5628cc5c485 |
| `data/pe48_frozen_forecasts.json` | Yahya signs on new Yahya tablets | 1946e391ffe27be8 |
| `data/pe22_outofcorpus_predictions.json` | 9 [4-15] new base signs on the 89 Tehran tablets | hashed in `loops/pe22_final.txt` |
| `data/pe17_frozen_ranking.json`, `data/pe26_frozen_clay_ranking_month.json` | which Susa tablets are chemically foreign | 78afe813...e828, f9fbbc12... |
| `data/pe46_frozen_predicted_codes.json`, `data/pe67_frozen_predictions.json` | name codes; outpost materials | 1e7141c52aabf617, 9725a2bf8d6882b1 |
| `loops/pe57_final.txt` | vessel test (at least 3 capacity peaks) | in file |

### 1.5 What new data would help most (`docs/HUMAN-ACTIONS.md`)

- **Section N**: unseen tablets (Hameeuw 2025, four tablets; the Hermitage tablet; an expert transliteration of the 89 Tehran Susa tablets; Louvre photos). This is the only route to an outside test of pe59/pe72.
- **Section O**: six readings to check on the tablets (P009286, P009220, P009296, P009237, P008784), from pe73.
- **Section L**: capacities of Proto-Elamite-period vessels, for the unit-size guess.
- Also sections F (hXRF sample list), H (P008294), I and J (seal catalogues; Dahl 2005 and 2019).

---

## 2. Linear A

Data: lineara.xyz (1,721 inscriptions, 65% from Hagia Triada); Linear B from DAMOS (about 94% complete). Since la71 the default corpus keeps break marks and erasures; 68% of tokens are cleanly read.

### 2.1 What is established now (A or B)

| Result | Loop | Grade now |
|---|---|---|
| KU-RO = total: 9 of 30 sections add up exactly against 0.5 by chance; 8 of 29 without erased lines; 5 of 11 on cleanly read tokens (P < 0.0001). | test 2a, la71 | A |
| KI-RO is a residual amount or a list heading. | test 2a | B |
| The fraction signs have a strict writing order (0 of 500 random relabellings match it). | attack 1 | A |
| Which fraction sign appears depends on the commodity (P 0.0002). | test 2b, la79 | C (la79: fails digging-order time travel) |
| Commodity logograms are written in a fixed order by commodity, not by size (held out 0.715 against 0.50). | la20 | A |
| An ordered basket after SA-RA₂ (GRA/CYP, then NI, then VIN) is the only relation besides KU-RO that forgers cannot fake. | la46, kept by la71 | B |
| Numbers on one tablet share a scale. | la14 | A |
| D is a counted part (written 2-4 times); K, L2, JE, F, H are never repeated, so they act as single denominations. | la27 | B |
| 12 words of 3+ signs shared with Linear B, against 2.0 by chance. | test 2e | A (as a statistic) |
| Affixes are real (prefixes and suffixes z about 6); -TE, -ME and JA- belong to objects, not tablets. They do not predict new forms. | test 2f, attack 2, la79 | A for affixing as a whole (la79 time travel); single prefixes I-, A-, KI- C- |
| With the Linear B-derived values, Linear A obeys language-independent sound rules better than 99.996% of relabelings, as strongly as Linear B's own values at the same size. Rows supported: pure vowels, M, T (held out), R, K. | la38 | B |
| Blind consonant rows agree with the Linear B-derived series. | la21 | B |
| No Linear B-like doublets among the common signs; one-sign variants are weakly sound-structured and local to a site and hand. | la37, la32, la13 | B |
| The single signs on sealings form one heading class (*301, KU, KA, SI, RO, ZE ...). On tablets, *301, RO and TE are counted, not headings (la70). | la45, la49, la71 | B |
| Receipts (roundels, nodules) and ledgers are separate layers. | la28 | B |
| Lines are run-on: 14% of line breaks split a word or number. Totals sit at the bottom on their own line. | la47 | A / B |
| Word census: at least ~4,500 word types in use (925 seen); about 55% of entry words on a new Hagia Triada tablet should be new. The hashed prediction of the post-1985 documents passed for new words. | la15, la23 | A / B |
| Sites act as separate administrations; the six Hagia Triada deposits share words at 2.5x shuffles; Khania is an island. | la58, la26 | B |
| Hands are partly recoverable from sign shapes and from shared words. | la34, la16 | B |
| Linear A is not context-deleted notes; its poor cross-site transfer is normal at this pool size. | la69 | B |

**Audit (la71).** lineara.xyz has no per-sign GORILA apparatus, and our old corpus counted 15 erased tokens and 31 words joined across breaks as text. The rebuilt corpus changed no A result. Grades moved down: the la45 counted-commodity class B -> C; HT 127b running total C -> C-; NI with houses and consonant-repeat avoidance C+ -> C.

**Time-travel audit (la78, la79).** Hypotheses that fit the published corpus best predicted later finds worst, and random-document cross-validation rewards overfitting; a truth planted only at Hagia Triada passes 73-80% of random splits but 0 of 31 digging orders. Re-graded on 62 digging orders and the 1950/1976/1988 cuts: affixing B -> A; commodity order A -> B; fraction-by-commodity B -> C; prefixes I-, A-, KI- and SI- -> C-; shared first sign C+ -> C-; consonant-repeat avoidance C -> C- (vowel position). Kept: -ME (B), Linear B shared words (A), fraction order (A). Frozen: data/la78_frozen_ranking.json, data/la79_frozen_predictions.json.

**Crack attempts (la60, la70, la72).** The frozen reading (V2c) beats its own role shuffles by 3.3% of held-out documents (2.3% on read-only tokens). All of the gain is KU-RO arithmetic, the *308 fraction rule and commodity order. A true Linear B partial reading gains the same, and so does a copy with a third of its items replaced by random words, so this test cannot grade word readings. The ledger is parsed (B); the language is unread (69% of word tokens, every entry name).

### 2.2 Surviving C-grade leads

| Lead | Loop | Grade | Kill line |
|---|---|---|---|
| LA and LB share a staple order: grain, olive good, figs, wine | la50, la67, la72 | C+ (its edge over la60's order was killed on read-only tokens, la72) | olive goods after VIN on 3+ new baskets |
| QA behaves like a pure vowel | la21, la67 | C+ | QA at shuffle level in a larger run |
| Consecutive entries share their first sign outside Hagia Triada | la14, la67, la79 | C- (la79) | |
| Room links: TE, A-DU, QA2 (Villa magazines / houses) C+; NI C | la51, la67, la71 | C+ / C | new finds in the other kind of room |
| KI before a number marks a reduced amount (rests mostly on HT 118) | la66, la70, la72 | C (weaker after la72) | KI amounts >= the entry above in half of 10 new cases |
| *308 is a fraction-bound commodity sign | la63, la67 | C | *308 with a bare integer on a new tablet |
| D = 1/5 and B = 1/3 | la27 | C | HT 115a re-read, or five Ds in one amount |
| HT 9b scales three of HT 9a's recipients by 4/5 (only works if J = 1/2) | la24 | C | a re-reading of the J signs on HT 9a |
| PA-DE is a specially marked entry | la53 | C | PA-DE as an ordinary crowded entry on a new tablet |
| *28B-NU-MA-RE and SI-PI-KI belong to the one Zakros wine template | la65 | C | either word on a tablet of another template or commodity |
| SI- as an alternating prefix | la18, la79 | C- (la79) | q > 0.1 within Hagia Triada alone |
| Consonant repeats avoided inside words (Hagia Triada only) | la10, la71, la79 | C- (la79: the effect is vowel position) | |
| Also C: TA-I with AROM; HT 95b a grain document; no standing quotas; layout profiles; the easier sailing leg; separate site administrations; the la45 commodity class | various | C | see FINDINGS |
| C-: final alternations SI~TI, TI~TE, RE~ME; entry words returning with their commodity; Greek-type suffixing lean; syllabic commodity words; HT 127b running total; a jar-sized wine unit | | C- | |

21 guesses need new data to test (`linear-a/loops/la67_final.txt`).

### 2.3 Killed or demoted (do not re-run)

- The fraction values cannot be solved from the totals; the "J < 1/2" argument rests on two doubtful J+J readings (photo check).
- The Linear B anchors are too few to separate places from persons.
- No cross-tablet ledger sums (la12); no dialect sound rules (la13); no sign equations from scribal spellings (la16); no epidemic source or adoption order (la17); no multilingual mixture (la19).
- No restoration anticipates SigLA's re-readings (la22); persons cannot be separated blindly from other entry words (la23).
- No apportionment rule (la24), fire month (la25), weight ladder (la27), hidden currency (la30), number marking on words (la35), harvest calendar (la50), urban scaling (la55), ration units (la64) or sealing abbreviations (la68).
- Type inference (la40), machine dissection (la49), phonetic maps from human confusion data (la59), word typing (la61) and the notes fingerprint (la69) fail their own controls.
- la67 kills: *318 as a commodity marker; OLE+RI fading; KU-RE, U, A, DA, KA-NA as headings; OLE+U/MI/DI as entries; SI/NI/TA2 commodity links; the KA/JA, KU~WA and MA~ME rules; *307 top rank; KA/KU/SI/I unnumbered; grain lists as rations; figs/olives avoidance; CYP/O/*86 in houses; Casa Room 9.
- la72 withdrew *307 as a heading, OLE before NI (it rested on an erased NI) and the Zakros site default.
- la60's outside test: no gain on the 11 post-1985 documents; the Knossos commodity profile failed (la62).

### 2.4 Frozen outside-corpus predictions

| File (in `other-scripts/linear-a/`) | What it predicts | Recorded hash |
|---|---|---|
| `loops/la72_predictions.txt` (current) | the 37 la70 predictions: 20 unchanged, 15 corrected, 2 withdrawn; for RILA-S1 and any new text | 6c493bb3... (of the JSON block; check command in the file) |
| `loops/la70_predictions.txt` (record only) | superseded | 2d7c6c9b... |
| `data/la72_ckpt/frozen_V2c.json` (git-ignored) | the frozen reading | 7d6df161... |
| `data/la22/la22_predictions.json` | restorations for 974 break edges | 37303998... (`la22_predictions.sha256`) |
| `loops/la15_final.txt`, `loops/la23_final.txt` | new-word rates on new tablets (~55% new on a new HT tablet) | 2a97d00c... (la15) |
| `data/la66_frozen_factors.json` | word quantity factors | 3127699d... |

### 2.5 What new data would help most

- **Section M** of `docs/HUMAN-ACTIONS.md`: RILA-S1 (Del Freo & Zurbach 2024, 107 post-1985 items), the Anetaki II drawings of KN Zg 58 (six fraction signs in order), Notti 2023. This is the only outside test of the frozen reading.
- Section A item 6: specialist re-reading of PH 9b, PH 22a, ZA 8 and HT 104 (settles J). Section K (Hagia Triada room records); sections G and J (Petruso weights, Salgarella sign tables).

---

## 3. Voynich manuscript

Data: ZL3b and IT2a transcriptions (39,019 words), GC2a as a third. Since v83 the default parser flags every doubtful word (17.4% of ZL words carry some doubt). Working assumption (user): the text carries meaning; the hoax, gibberish and generator readings are not pursued.

### 3.1 What is established now (A or B)

| Result | Loop | Grade now |
|---|---|---|
| On letter statistics the text equals a third-order glyph Markov chain. A fixed letter-to-glyph-group cipher of Latin or Italian is ruled out. | test a, V1 | A |
| The last glyph of a word predicts the next word's first glyph within a line, and this drops to zero at every line break, even between ordinary words. (Replicates arXiv 2604.19762.) | V2, V3, §9, v61 | A |
| Lines have opener and closer vocabularies; about one m/g glyph per line; paragraph-first lines are a spelling regime of their own (more gallows, p/f). | test b, N1, v49 | A / B |
| The q-type and a-type word classes are mostly a property of section and page. | §9 | B |
| Words repeat next to each other 2.8x chance. But a token recurs one word later as often as two or three words later: there is no neighbour ban, unlike every language tested. | test e, v78 | A / B |
| Glyphs that share strokes behave alike, more than in any invented or cipher script tested. | v25, v28, v35 | A |
| k~t and ch~sh act as variants of fewer units (merging them collapses word types). Which form is written depends on line position and a short hand habit, not on the word; this does not hold book-wide (v77). | v39, v40, v77 | B (variants); A measurement / B reading (position) |
| Currier A and B are two conventions of one system at dialect distance (a regular ending shift). | v30 | B |
| No hidden message in word lengths, spaces or line counts (v15, v16); nothing sorted (v19); no quotas (v20); no written calendar dating (v32); not a spell (v31); no forger passes, and no page topic (v21); no source-text shape (v58); no language island of 1,200+ tokens (v43); not verse (v62); not tables (v80); not template dossiers (v76); no Lullian alphabet (v64). | listed | A or B (negative, calibrated) |
| The payload of the crack-attempt model has no page topic, no recurring phrases and no lexical gaps; most of its word-to-word information is the last-glyph pairing. | v72 | B (negative) |
| Line-bounded onset persistence: the next word tends to start like this one, never across a line break. A fitted meaningless generator reproduces it (v82). | v81, v82 | B (measurement) |
| Page vocabulary is real and of real-text size (0.06-0.10 bits/word), but the one statistic that separates real page content from fitted generators puts the Voynich on the generator side. | v84 | A (measurement); C- as evidence of meaning |
| No scribe has a vocabulary beyond its own glyph grammar (real texts x1.4-2.2). Hands share words by section, not by hand, and a section glyph mood without a dictionary reproduces this. | v85 | A (measurement); B (kill of the shared-dictionary reading) |
| Squeezed words (line ends, words against drawings) are not compressed recoverably: random abbreviation rules that find planted medieval abbreviation find nothing. The one recoverable change is length-neutral: line-final r/l is written m, also where space is not short. | v86 | B (negative); C (m as a line-final form of r/l) |
| Generator-like statistics do not by themselves mean "no meaning": known list languages (Ur III, proto-cuneiform) are also voted "generator" (71-95%). | x4 | B |

**Audit (v83).** Every B result holds on unflagged words in both transcriptions, against matched random thinning. Some numbers shrink, and these replace the earlier ones: v72 sequence excess 0.057 -> 0.032-0.034 under thinning; v74 short-word gap gradient 0.39 -> 0.26; v79 line-start table z 24 -> 5-6 on agreed words; v68 line-initial pool 0.114 -> 0.081. Old parser bugs fixed: `<~>` glued 6 word pairs, 190 GC2a glyphs were deleted, and IT2a carries none of ZL's uncertainty markup.

**Crack attempt (v72).** A two-layer model (message tokens under a surface of twins, benches, e-padding, optional q-, edge pairing and sealed lines) gives back planted Italian, Latin and German messages (purity 0.96-1.00). On the Voynich the message layer carries no language-like message. Not cracked.

### 3.2 Surviving C-grade leads

| Lead | Loop | Grade | Kill line |
|---|---|---|---|
| Hand 1 drifts only between quires (survives v77 beyond the generators) | v41, v77 | C | |
| Words on one pen load are slightly more alike | v18, v77 | C+ (underpowered) | z < 1 on new pages or in a known-language manuscript |
| Within a line, ch-initial words lean early and gallows words late | v80 | C | a line-aware generator with a glyph-position gradient reaching +0.016 bits |
| The payload behaves like a notation symbol or list value, glossary- or index-like, not a word of running language | v75, v78 | C | frozen Q1/Q2 (3.4) |
| Two-word blocks repeat with free re-spelling in Currier B (f75-f86, f103-f113) | v78 | C | |
| The physically documented quire-centre bifolios share the most vocabulary | v65 | C | |
| Self-contained lines; quire M (hand 2) as the repetitive extreme; pages as independent records | v22, v43, v56, v77 | C (survive v77, but generators do as well or better) | |
| C-: the short designation code (v72); the picture-text link in hand 1 (v38); p~f and benched pairs as variants (v39); doubling as a mark (v75); page vocabulary as content (v84) | | C- | |
| Open, untestable now: an undated calendar (v32), sparse sky codes (v55), word-level path messages (v50), spelling-only line marking (v73), complexion per page (v60) | | C | their planted controls were not recovered at this size |

### 3.3 Killed or demoted (do not re-run)

- The line-initial glyph chain (v6/v10; C+ in v77) is a scribal habit, "a mark unlike the one above, from a short table of successors" (v79), not a counter, calendar, acrostic or index.
- v81's front-of-word link as meaning: killed by a fitted generator (v82). "Only the initials carry content": killed (v81).
- v74 tested v72's frozen predictions: P1 (narrow spaces at or/s/r + aiin) killed by measurement on 56 pages; P2 killed narrowly on the GC2a transcription; P3 untestable.
- v77 kills: index-like within-line direction; endings re-encoded between A and B; payload/padding slots; stem+l -> stem+y; the Sanskrit-like line-end lean; the ink sawtooth (a Latin manuscript has it too); concepts as whole words; the fixed-spacing chain; op- evenness; edges as fields; ch...sh one apart; k~t / ch~sh by position book-wide; bench sharing; the marker sign; reversed compression; ch/sh runs needing a state; A/B symmetry.
- Also killed: sandhi in both directions (v61); an A -> B transducer (v59); featural sound change A -> B (v36); the published "ched- = hot, qok- = dry" reading (v60); slip-and-rewrite copying (v63); the real sky as key (v55); the genre classifier (v71); ink alphabets (v70, bounded by scan resolution); speech-model listening (v51); the evolved encoders' read-back (v53); line consensus (v67); one marked word per line (v73); the mind map (v66); dice, interleaving, a hidden map, an exemplar width (v14, v12, v11, v17).
- The Codex Seraphinianus resemblance (v42) is background only, by the working assumption.

### 3.4 Frozen outside-corpus predictions

| File (in `other-scripts/voynich/`) | What it predicts | Recorded hash | Status |
|---|---|---|---|
| `loops/v72_final.txt` | P1 space widths; P2/P3 values on another transcription; extraction rule E1c | text sha256 c8b21509...; E1c prefix 2803cbeb0deb51bf | P1 killed, P2 killed narrowly, P3 untestable (v74) |
| `loops/v75_final.txt` | Q1: adjacent-repeat rate of payload tokens >= 0.015 on any further transcription; Q2: the glossary/index family vote beats its generators | 83c8d0fa... | open |
| `loops/v24_final.txt` | 50+ erasure or overwrite pairs from the vellum would decide copy-like vs language-like repair | not hashed | open |

### 3.5 What new data would help most

`docs/HUMAN-ACTIONS.md` has no Voynich section yet. What would help: a further independent transcription (tests Q1/Q2); full-resolution Beinecke TIFFs (v70 needs a Latin control that recovers its letters); 50 or more readable erasure or overwrite pairs from the vellum (v24); a hand-transcribed second long asemic book (v42).

---

## 4. Method lessons across the three scripts

**Tests that fail their own controls at small corpus size.** Many ideas were tested properly and turned out to be blind at our corpus sizes, so their negatives say nothing about the script. Examples: Ur III seal-title links are found 8/8 at ~480 tablets but 2/25 at Proto-Elamite size (pe33, pe35); type inference recovers 0-4% of known proto-cuneiform roles (pe38) and nothing in Linear B at Linear A size (la40); phonetic maps from human confusion data miss planted syllabaries at 1-30x Linear A's data (la59); coded herbals do not recover their own complexion slot (v60). The crack attempts share this limit: proto-cuneiform with its *true* readings gains no more than PE (pe59), and a one-third-corrupted Linear B reading gains as much as the true one (la70), so whole-document decoding cannot grade a reading. The kill sweeps showed how weak C guesses are: pe66 killed 12 of 28 testable guesses and only 4 survivors clearly beat their decoys; la67 killed 14 of 35, with 5 clean survivors; v77 killed 17 of 32, with only 2 surviving beyond fitted generators. Lesson: run every new test on a planted signal and on a real known-answer corpus *at the same size* first, and put every guess against frequency-matched decoys before grading it.

**The list-language confound.** Administrative lists and designation systems also look "generator-like". Ur III and proto-cuneiform are voted "generator" by 71-95% of classifiers and lack word-order arrows; once list references are added, the Voynich's generator vote falls to 41-57% (x4). A genre classifier puts generator texts into a genre 18 times out of 18 (v71). In the controls, words homed at one site are commodities and office terms, not names (pe48). Outside bureaucracies do not even share a layout for totals (la57). Lesson: "generator-like", "no word order" and "site-specific" do not separate meaning from no meaning, or names from goods, without list-type controls.

**Restoration and transcription parsing.** All three corpora had parser problems, found and fixed by the audits. The Proto-Elamite parser counted the editor's restorations as read (1.5% of signs, 0.6% of numerals; pe73, pe74). The Linear A corpus counted erased tokens and words joined across breaks as text (la71). The Voynich parser glued words, a converter deleted GC2a glyphs, and IT2a lacks ZL's uncertainty markup (v83). The audits changed few grades, but each demoted some sub-claims, and several earlier results hung on one restored or erased token (P009296's N14, HT 127b's erased line, the erased NI of HT 96b). Lesson: load status-aware corpora by default (`common.load()`, `la71_parse.load('rd')`, `vlib.load_voynich`), report results on read + damaged tokens, and check them on read-only tokens against random thinning.

**Other recurring points.** Arithmetic is the strongest tool in the two accounting scripts, but with 2-3 tablets almost any total can be matched by chance (la12), and a failing total does not say which line is wrong (PE attack 2 photo check). Short random programs (r2, ~8.9 M programs) explain no script beyond Markov baselines. Transfer learning between corpora teaches format, not sign identity (x3).

## 5. What the next real progress needs

1. **New, genuinely unseen texts** scored against the frozen predictions: RILA-S1 and Anetaki II for Linear A (HUMAN-ACTIONS M); Hameeuw 2025, the Hermitage tablet and the 89 Tehran tablets for Proto-Elamite (N).
2. **Expert checks of named readings**: Proto-Elamite section O; Linear A PH 9b, PH 22a, ZA 8 and HT 104 (section A item 6).
3. **Outside physical data**: Proto-Elamite-period vessel capacities (L); for the Voynich, high-resolution images and vellum corrections.

Possibly new on the Voynich side (not found in the literature checked): the m/g one-per-line quota, the ch/e trade-off by line, the split of word-class variance across section, page, paragraph and line, and the verbose-cipher merge test with encrypted-language controls. Most other Voynich measurements replicate published work (arXiv 2608.17096, 2604.19762; Vogt 2012).
