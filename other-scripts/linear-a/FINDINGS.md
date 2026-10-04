# Linear A: where data analysis can bite

Rule: we use others' data and the conventional sign values, not their interpretations. Every result has a control.

## 1. Data
- **Linear A:** `mwenge/lineara.xyz` `LinearAInscriptions.js`, saved in `data/`. `tools/build_corpus.py` turns it into `data/corpus.json`. Only the Unicode signs and the transliteration are used. The site's "translatedWords" are not used. Fraction signs are decoded from Unicode names as letters (J, E, D, B, K, JE, L2, F, A, H, L…).
- **Counts:**
  - Records: 1,721. Nodules 890, tablets 435, roundels 151. Hagia Triada 1,110 (65%).
  - Words: 2,972 tokens and 1,084 types. With 2+ signs: 1,332 tokens and 952 types.
  - Quantities: 1,566, of which 414 have fractions.
  - Logograms: 707. Main ones on tablets: OLE 130, GRA 113, CYP 88, VIN 58, VIR 47, OLIV 28.
  - Fraction letters: J 136, E 76, D 42, B 41, K 34, JE 31, L2 30, F 19, A 18, H 16.
- **Linear B:** DAMOS (CC BY-NC-SA), fetched by `tools/fetch_damos.py` into `data/damos_items.jsonl`. This gives 5,932 documents (KN 4,227, PY 986, TH 427, others 292), 4,832 word types and 12,382 tokens. **Partial:** 368 item IDs return HTTP 500 every time. Only spellings and site prefixes are used.

## 2. Tests

**2a. KU-RO = total** (`tools/totals_test.py` → `data/totals_test.out`)
- Method: entries are the quantities since the previous KU-RO, KI-RO or the top of the tablet.
- Result: 9/30 exact (HT 9b, 11b, 85a, 88, 89, 94b, 104, 117a, 127b). 13/30 within 1. 16/30 within 5%.
- Controls: totals shuffled across tablets give mean 0.5, max 5, P<0.00005. A random subset of entries matches the total 2.2% of the time. The "any run of entries" rule gives 10/30 vs 2.0.
- Misses: two-commodity tablets (HT 27a, 94a, 100, 123a), damaged lists, and slips of 1 or 10 (HT 94a, 102, 119).
- HT 122b PO-TO-KU-RO 97 = 31 (side a) + KU-DA 1 + 65 (side b).
- HT 118: the KI entries 10+4+1 = the "KI 15" after KU-RO.
- **Verdict: confirmed.**

**KI-RO** (16 cases)
- 6 have no number and act as a heading. 4 of those head lists of "1" entries, which a KU-RO then counts exactly (HT 88: 6, HT 94b: 5, HT 117a: 10).
- When it has a number, the number is smaller than the entry just before it in 8/9 cases. For KU-RO this happens in 1/30.
- Its number never equals a sum of entries.
- **Verdict:** a residual amount or a list heading. "Deficit" fits but n=10.

**2b. Fractions**
- **Totals:** only 7 testable sections. The site values (J=1/2, E=1/4, F=1/8, K=1/16, D=1/5, B=1/3, A=H≈1/6, JE=3/4) balance 1/7 (HT 104). Permuting those values among the letters gives a mean of 0.17 (P=0.17). A grid search over 50,625 value sets also tops out at 1/7, with 3,825 ties. HT 13 cannot balance under any value of J. HT 123a needs H = A + 1/4 if E=1/4 and JE=3/4.
- **Writing order:** in the 61 compound fractions, 45/45 pair-instances follow one direction. Shuffling the letters gives 7.6 reversals on average (P(0)<0.0002). The order has no cycles: JE, L, E, J, Y, A, F, H, B, K, L6, L2, L4. If the rule is larger-first, the site values break it in 14/42 pairs: B comes after J, E and A; E comes before J once; L comes before E.
- **By commodity:** chi² 321 vs a permutation mean of 120, P=0.0002 (n=317). L appears only with HIDE. B appears with CYP, NI or OLE 30 times, with GRA once and with VIN never.
- **Verdict:** a real ordered system, but not one universal set of values. The conventional values are unsupported.

**2c. Commodity ties** (`tools/words_tests.py` → `data/words_tests.out`)
- Only 16 words occur on 3+ tablets that carry commodity signs.
- 38% of them occur with 4+ commodities, vs 72% when commodity sets are shuffled across tablets (P≈0.01).
- Tied to one commodity: KU-NI-SU and DA-ME, both with GRA (P=0.18 overall).
- Spread over many commodities: KU-RO, KI-RO, SA-RA₂ (directly before GRA 8x and CYP 5x), A-DU, KA-PA, KU-PA.
- **Verdict:** the method works, but n is small.

**2d. Name-like repetition** (unique share vs a sign-bigram model, 200 runs)

| set | ratio obs/model |
|---|---|
| tablets | 0.61/0.73 = 0.84 |
| Hagia Triada tablets | 0.78 |
| all objects | 0.80 |
| non-administrative objects | 0.87 |
| words with quantity "1" | 0.89/0.83 = 1.07 |

- **Verdict:** a mix of frequent terms and a long tail. This is not a pure name population (~0.6). Words with "1" are almost all different individuals. Uniqueness alone does not prove names.

**2e. Linear B overlap** (`tools/linearb_overlap.py` → `data/linearb_overlap.out`, `data/la_lb_matches.json`)
- 769 Linear A types give 69 exact matches in Linear B. Shuffling signs across Linear A types gives 45.8 (max 70, P=0.001).
- 2 signs: 57 vs 43.8 (P=0.011). 3 signs: 9 vs 2.0. 4 signs: 3 vs 0.02. Both P<0.001.
- Matches of 3+ signs: PA-I-TO (Linear B: KN 46), SE-TO-I-JA (KN 15), SU-KI-RI-TA (KN 5), I-TA-JA, DA-I-PI-TA, KI-DA-RO, PA-RA-NE, TA-NA-TI (KN), and A-RO-TE, DA-MA-TE, I-JA-TE, MA-TE-RE (PY).
- Knossos share of the matched words' Linear B tokens: 0.73, vs 0.47 for random Linear B types of the same lengths (P=0.006). Without the two most frequent matches it is 0.64. Counted by type it is not significant (40/69 vs 37.7, P=0.32).
- **Verdict:** real shared vocabulary, mostly on Crete. The matches are likely place or personal names, but that is not tested.

**2f. Affixes** (769 types; control shuffles first, middle and last signs separately, 300 runs)

| pattern | observed | control | z |
|---|---|---|---|
| first-sign swap | 42 | 29.5±5.8 | 2.2 |
| last-sign swap | 66 | 44.8±8.5 | 2.5 |
| prefix added | 83 | 45.5±6.0 | 6.2 |
| suffix added | 111 | 63.7±7.7 | 6.1 |

- Clearest individual cases:
  - I-: 8 vs 2.9 (e.g. I-DA-MA-TE / DA-MA-TE)
  - SI-: 5 vs 1.3
  - A-: 11 vs 5.5
  - KI-: 5 vs 1.4
  - -ME: 6 vs 1.2 (A-SA-SA-RA / -ME, SA-RA / SA-RA-ME)
  - A-/JA- swap: 3 vs 0.5 (A-/JA-SA-SA-RA-ME)
- -JA, -RE and -TE are not significant on their own.
- **Verdict:** real affixes. No claim about which language.

## 3. Chinks (ranked)
1. **Arithmetic.** Use KU-RO, KI-RO, PO-TO-KU-RO and column totals to fix list structure, damaged numbers and commodity columns. Test: a full parse of every Hagia Triada tablet into commodity columns. Success: well over 9/30 sections balance, with the controls still near 0.5.
2. **Fractions.** Solve the values jointly from the writing order, the totals and each commodity's units. Constraints already found: B < A, J, E; H = A + 1/4 given E and JE. Success: one value set, unique per commodity, that balances most of the 7 sections and keeps larger-first order. Then use it to predict damaged totals.
3. **Linear B anchors.** The 12 shared words of 3+ signs are phonetic fixed points. Test: do they sit in Linear A slots where a place or person fits (list entries, Hagia Triada vs Khania by geography)? Success: they behave like Linear B places or people in their position and their site spread.
4. **Affix paradigms.** I-, SI-, A-, KI-, -ME and A-/JA- in the libation formula and on tablets. Test: is each affix tied to a document type, a position or a commodity? Success: a stable paradigm table that predicts unseen forms in new finds.
5. **Commodity-tied words.** KU-NI-SU and DA-ME go with GRA. SA-RA₂ is a general term. Test: more tablets from non-Hagia Triada sites. Success: a word tied to one commodity at two or more sites.

Caveats: 65% of the data is from Hagia Triada. Absences are upper bounds only. The Linear B side is about 94% complete.

## Attack 1: solving the fraction values jointly (`tools/attack_fractions.py`; report in `data/attack_fractions_report.txt`)
- **The writing order is real.** Random relabelling of letters gives 0 of 500 fully ordered runs.
- **Direction is open.** Both larger-first and smaller-first allow systems with no order violations and no compound reaching 1.
- **J must be below 1/2.** J+J is written twice (PH 9b, PH 22a), so 2J < 1. Under larger-first the order also gives J < E < L < 1/2.
  - **The conventional J = 1/2 survives only if both J+J readings and one E-before-J case (ZA 8) are misreadings.**
- **Conventional values:** they break larger-first in 4 of 20 pair instances, and 3 compounds reach or exceed 1 (HT 27a JE+B = 13/12, and both J+J).
- **The totals add nothing.** Shuffled letters balance as well as the real ones (P = 0.45–0.92).
  - Five sections cannot balance under any values. Three of those are section-cutting errors.
  - Best clean score: 1 of 8 exact, or 3 of 5 allowing a one-unit slip.
  - Commodity-specific values (36 parameters) predict 0 of 8 held-out sections, so there is no gain.
- **Best larger-first systems.** One binary example:
  - L 7/16 > E 3/8 > J 5/16
  - A = F = H = JE 1/4
  - B = D = K = Y 1/8
  - L2 = L6 1/16 > L4 1/32

  No letter is pinned exactly. Stable across solutions: J ≈ 3/10–5/16, E ≈ 3/8–2/5, and 2J + E ≈ 1. No compound reaches 1 (0 of 61).
- **New compared with the conventional values:** E and L are larger than J, JE is not forced to 3/4, and B is small (at most 1/4).
- **Predictions to check against photographs:**
  - **HT 9a:** the entries sum to 30 JE. Is the total "31" really 30?
  - **HT 104:** the entries give 94 5/8 against a total of 95. One "J" should be a larger sign worth 11/16. If both are clearly J, then J = 1/2 after all and the J+J readings are wrong.
  - **HT 13:** the total's fraction should be worth 2J, not J.
  - **HT 123+124b:** "*188-*308 11" is probably a heading.

**Verdict.** The values could not be solved: there are too few clean totals. One hard constraint emerges: either J < 1/2, or three published readings (PH 9b, PH 22a, ZA 8) are wrong. Checking those three tablets and HT 104 on photographs would settle J.

## Photo check of the J+J readings (3 Oct 2026; GORILA photos and drawings via lineara.xyz, © École française d'Athènes, viewed only, not committed)
- **PH 9b.** The corpus reads `*412-VS-VIN 1 J J`. The drawing shows the sign group, then **one vertical stroke and one L-shaped fraction sign**, with a dot beside the L. I see **one** J, not two. The stroke is the unit "1".
- **PH 22a.** A fragment. The drawing shows **two L-shaped signs**, then a dotted (restored) sign. Read left to right, the "fractions" come **before** the commodity sign, which is unusual for a quantity. The commodity sign is restored, not seen.
- **Consequence.** Both J+J cases are weak:
  - PH 9b looks like a single J in the drawing.
  - PH 22a is a damaged fragment with an odd order.

  The constraint "J < 1/2" (attack 1) rests on these two readings. **It is not secure.**

**Verdict (my visual read, grade C).** The conventional J = 1/2 cannot be ruled out. The fraction order stays real (0 of 500 random). To settle J, the next step is a specialist re-reading of PH 9b and PH 22a in GORILA vol. 1 (pp. 334, 346), and of HT 104, where the drawing is too faint to tell J from a larger sign.

## Attack 2: Linear B anchors and affix paradigms (`tools/attack_anchors.py`; report in `data/attack_anchors_report.txt`)
**A. Anchors as place or person probes: no usable result.**
- The anchors were classed from Linear B (DAMOS) data alone:
  - **place-like:** PA-I-TO, SE-TO-I-JA, SU-KI-RI-TA;
  - **person-like:** I-TA-JA, PA-RA-NE;
  - **unclear:** 7, each with only one Linear B document.
- Linear A then has 4 place tokens and 3 person tokens. No slot feature separates them (all p ≥ 0.24).
- With 5 word types the best possible p is 0.10, so this test could not succeed.
- One small observation: in Linear A, PA-I-TO is a list entry with a quantity on Hagia Triada tablets, not a heading.

**B. Affixes**
- **Attach beyond chance** (control: a random sign added to the same stems):
  - A-: 11 stems against 3.4.
  - I-: 8 against 3.4 (p = 0.02).
  - -JA: 7 against 3.1 (p = 0.03).
- **Not beyond chance:** SI-, KI-, -RE, and JA- on its own.
- **Possibly new: -TE, -ME and JA- belong to objects** (stone and metal objects, vessels), not tablets. 55–80% of their types are on objects, against 17–27% for the control (p ≈ 0.01).
  - So the A-/JA- alternation is not free variation: JA- is the object form.
- **SI- forms always carry a number:** 6 of 6 types (p = 0.006).
- **Prediction fails.** Fitted on Hagia Triada, 94 predicted forms gave 1 hit elsewhere (A-MI-TA, Zakros), against 0.43 for the control (p = 0.35).
- About 99 comparisons were run. The -RE→"1" and KI-→commodity ties (p ≈ 0.02–0.04) are probably noise.

**Verdict.** Real but small structure: the affixes split by document type. The Linear B anchors are too few to classify Linear A words.

## Evolved grammar machine (la8)
- The best evolved machine is a small flat form with 4 classes in one loop: entry word or commodity, number, optional fraction, then the next entry. Planted flat and recursive languages were recovered (agreement 0.999 and 0.97). Shuffled Linear A collapses.
- Linear A has 2 open classes and no closed formula words. Linear B, matched for size, has 7 classes, including closed formula words. Proto-Elamite has 4.
- In Linear A a number follows a word directly (0.52 vs 0.37 shuffled), as in Proto-Elamite. Linear B words avoid numbers.
- No recursion was found, but the test is weak because Linear B shows none either.
- Grades: the ENTRY class (KU-RO, KI-RO, SA-RA₂, A-DU, KA-PA, TE, SI) is B. Commodity logograms are A. NI, *304 and *308 as commodities are B.

## Blind bijection compression contest (la11)
- About 21,200 annealing searches over 39 languages from 18 families found no language that takes up Linear A better than unrelated real words, under any of 3 spelling schemes.
- The Linear B control failed. Ancient Greek ranked between 17th and 33rd of 35. The method recognises a corpus, not a language family, so this null says nothing about Linear A's family.
- An early Afro-Asiatic lead (Arabic, Coptic) was half random-string bias and vanished under the third spelling scheme (p = 0.51). Dropped.
- Linear A's sign-to-sign structure is real but about 70% as strong as Linear B's, consistent with la5.

## Tablets as jigsaw pieces (la12)
- Searched about 1.7e8 three-tablet combinations per run for Hagia Triada totals that equal sums written on other tablets, then assembled 'ledgers' by simulated annealing. Nulls kept each tablet's magnitude; controls cut real tablets into pieces.
- No cross-tablet closure beats chance. Joined tablets are not more often the same hand, room or vocabulary than chance joins. No word gets a total, deficit or transfer role.
- The method cannot reassemble even real cut HT tablets: with 2-3 tablets almost any total is reachable in hundreds to thousands of exact ways (grade A). Arithmetic joins need physical evidence first.
- Leads, grade C: HT 116a GRA lines sum to 109 (its own KU-RO says 100) = HT 1 KU-PA3-NU 109; HT 27a VIR 140 = all numbers of HT 27b (probably coincidence).
