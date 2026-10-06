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

## Bred languages (la10)
- 16 genetic-algorithm runs scored about 110,000 rule sets, or about 331,000 artificial vocabularies. Planted languages were recovered. Linear B passed in part after the model was fixed: its suffix-heavy morphology came back, but its cluster spelling did not. Shuffled Linear A gave no rules.
- Linear A profile, which holds outside Hagia Triada:
  - Grade B: both prefixes and suffixes, no consonant clusters, no vowel harmony.
  - Grade B: mostly a, i and u, with o nearly absent. This depends on Linear B-derived values, so it carries that caveat.
  - Grade C: repeated consonants avoided inside roots, and consonant-final stems.
  - Killed: vowel dissimilation, because it also appeared on the shuffled control.
- WALS and Grambank data put this profile weakly closer to languages with both prefixes and suffixes than to suffix-only Hurrian, Luwian or Greek.
- The spelling convention cannot be tested, because the Linear B control fits a spelling other than its own better.

## Linear A as its own dialects (la13)
- Each site group (HT, Khania, Zakros, Phaistos, Knossos, others) was treated as a language. All cross-site word pairs differing by one sign were searched for recurring substitutions (sound laws or spelling conventions), then site sign-contexts were aligned by annealed sign maps, and a sign grid was built from the substitution graph. Scribes were tried as idiolects.
- No rule beats label-shuffle or Markov nulls. Nothing recurs at all on words of 3+ signs. The best rule (HT KA vs others JA, 2-sign words) is matched by random splits of LA: C, demoted.
- Power is the limit: sites share only 43 word types (scribes 26-30), so even a planted exceptionless substitution is not recovered (0/18). Absence of dialect rules is an upper bound only.
- No sign grid emerges from LA substitutions (held-out no better than shuffled pairs). Full Linear B does yield partial rows; LB cut to LA size does not.
- Grade B side result: words differing by one sign cluster in the same site and the same hand, beyond the document level (P <= 0.003 in 3 groupings). Variant families are local (archive/hand), not regional.

## Counting the words like wildlife (la15)
- Field-ecology census, no readings: richness estimators, community bootstraps, ~80,000 random Zipf communities (ABC), curveball nulls, capture-recapture, species-area, and a hashed prediction of the post-GORILA documents. Controls: planted communities, Linear B, shuffled word-site links.
- Grade A: the corpus is a thin sample. 925 word types seen, at least ~4,500 in use (Chao1/ACE are lower bounds here: on planted Zipf vocabularies they recover only 20-90%); ABC 19,000 [5,200-82,000]. Sample coverage 0.39. The same tools recover Linear B Knossos's out-of-sample type count (1,597 true; 1,552 predicted).
- Grade B: syllabic signs. 139 seen, ~170 (Chao1/ACE) to ~200 [153-265] (ABC) in use; the method recovers LB's 89 signs from an LA-sized sample.
- Grade A: words are site-endemic (0.63 of recurrent words at one site vs 0.21 null) and the site x word matrix is anti-nested (NODF 4 vs 12): small sites are not subsets of HT. Both match Linear B at the same size and vanish when word-site links are shuffled. No word is migratory beyond its frequency; the widest-spread words are the libation-formula words.
- Grade B: KI-RO and SA-RA₂ are HT-only terms (absence elsewhere has P < 0.001 if they were in use there). Would kill: a secure non-HT KI-RO or SA-RA₂.
- Demoted: a 'keystone guild' (DA-ME, DI-DE-RU, MI-NU-TE, KU-NI-SU, SA-RU) is one list repeated on HT 86 and HT 95.
- Out-of-corpus test (66 post-1985 documents, predicted before opening): new word types 33 [27-38] predicted (habitat model) vs 32 true; new syllabic signs 1 [0-3] vs 0; reappearing known words AUC 0.76, 4 of the top 20 (chance 0.2). New findspots: 1 [0-3] predicted, 6 found (excavation targets new sites). The same pipeline on Linear B underpredicts Thebes unless site novelty is modelled.
- Hands: ~105-115 scribal hands in the attributed set (calibrated on Knossos hands). Findspots: at least ~75.
- Use: plan on most of the vocabulary being unseen; any proposed lexicon should expect ~3 of every 4 words on a new tablet to be new. Not a decipherment.

## Private spellings as a Rosetta stone (la16)
- Idea: if one scribe writes X-B-C where another writes Y-B-C, X and Y are interchangeable signs. Tested with hand-contrastive one-sign pairs (2,000 hand permutations), annealed sign-equivalence partitions, hand inference from spelling conflicts, and 10,000 random equation sets per split re-tested on held-out documents.
- The positive control fails: in Linear B (KN+PY, DAMOS hands), known variant spellings (A/A3, O/WO, RA/RA3 ...) are written within one hand as often as across hands, and no stage recovers them. Planted private spellings are not recovered at LA size (0/19).
- No Linear A sign equation survives. KU~SA, I~SA, MA~RA2, RA~RO/TA fail held-out re-test (killed). KU~WA and MA~ME (site groups, 2-sign words) stay C.
- Grade B: LA scribal hands are recoverable from shared words (leave-one-out 0.37 vs 0.16 site-majority baseline, 0.06 permuted), not from spellings (conflicts carry no hand information, as in LB). 52 unattributed documents get grade-C hand assignments (held-out accuracy 0.31).

## Words that avoid each other (la18)
- Idea: alternative forms of one word fill one slot, so they should never meet on a tablet. Tested on every form-related pair (w / w+X, X+w, first- or last-sign swap) against a site-stratified curveball null (word frequencies and tablet sizes fixed), pooled by alternation class, at the level of endings (case harmony, annealed ending partitions with equal-effort null anneals) and at the level of the entry. Controls: Linear B with Greek inflection labels, LA-sized LB draws, shuffled and relabelled LA, planted paradigms (avoiding, attracting, case-harmonic, entry-exclusive).
- Linear B turns the idea around: forms of one lemma ATTRACT on a tablet (3.1x; 9/10 LA-sized draws) and AVOID the same entry (0.27x). The class test finds real Greek endings (+JO, RA~RO, +O, +DE) from co-presence alone, even at LA size (precision 0.73). LB tablets are written 'in one case' (ending homogeneity z 13.8). Method grade B.
- In LA, absence carries no information: a perfectly avoiding planted paradigm is never detected (0/48). Attraction and case harmony would be detected (planted 4/4 each).
- Grade B (calibrated negative): LA one-sign alternants are not listed together on tablets (0/27 classes; pooled 1.29x vs 1.7-5.4x in every LA-sized LB draw), and LA tablets are not case-harmonic (z 1.8, the same as relabelled endings). Either LA lists do not name a lemma in several forms, or most one-sign alternants are different words.
- Killed: the one significant LA ending partition is the libation formula (I-PI-NA-MA / SI-RU-TE / A-TA-I-*301-WA-JA), and it changes with every subset.
- Grade C: SI- as an alternating prefix (SI-DA-RE/DA-RE, SI-TE-TU/TE-TU, SI-KI-RA/KI-RA, ...): the pairs share contexts beyond site- and document-type-matched random pairs (P 0.0025, q 0.065). Would support: SI-X and X in the same entry slot on new texts; would kill: q > 0.1 within Hagia Triada alone.


## Random programs (r2, 4 Oct 2026; `voynich/loops/r2_final.txt`)
- ~8.9 M random/evolved small programs across the three scripts, scored by held-out description length over Kneser-Ney; planted ledger (commodity cycling by line) recovered blind.
- Linear A: the best program is 'copy the sign in the same slot of the word two back' (the previous entry, skipping its number): +114 bits on held-out tablets with line ends given, but only +28 (p 0.14) against a KN + document-cache baseline. Grade B, known in kind: entries repeat logograms and word shapes down a tablet. No generating procedure.

## Forgery contest (la14)
- 24 forgers and about 16,000 forger/discriminator pairings were scored on 292 tablets. A forger is caught at AUC 0.73–0.78 even against its own output (it was retrained on half the data), so survivors were scored against each forger's own corpus.
- What real tablets have and forgers lack is accounting structure, not grammar:
  - Grade A: numbers on one tablet share a scale (P = 0.002 against a site and commodity null).
  - Grade A: tablets repeat their line pattern, words and values more than any forger does.
  - Grade B: on Hagia Triada tablets, commodity sections run from large to small amounts (0.601, P = 0.0005).
  - Grade C: outside Hagia Triada, consecutive entries share their first sign (P = 0.001).
- Not detected: top-to-bottom agreement, links between particular words and particular numbers, or line order.

## Words spread like epidemics (la17)
- Idea: treat each word (and each sign type) as a pathogen and each site as a host population; fit SI/SIR outbreaks on a distance-and-size contact network, with unknown source, adoption order and (cycle 2) deposit time, by likelihood-free inference (~390,000 simulated outbreaks; rejection ABC and ABC random forests). Dates are never inputs; the inferred order is scored afterwards against the deposit phase of each site (PH MM II ... HT/KH/ZA LM IB).
- The method fails its controls: a planted Phaistos-source outbreak is recovered 1/20 and 0/20; outbreaks planted by a different (lexicon-copying) generator are at chance (source 0.05, order rho -0.01); shuffled incidence gives the same "source" as real data (HT for signs, KN in Linear B). In Linear B, KN comes first only when shuffled data say the same, and the deposit-time model puts KN 3rd.
- Linear A: the source posterior is flat (best 0.15 vs 0.077); no order matches the dates (rho -0.18 to 0.28, P >= 0.14). Phaistos, the only MM II site, is always placed late. No source or order is claimed.
- Grade C: sites far apart in date share fewer words than distance predicts (LA -0.21 z per phase, P 0.25; LB the same sign, P 0.16); PH-HT, 3 km apart but MM II vs LM IB, share less than other near pairs (z -1.6 vs 0.0). Would support: P < 0.05 with more MM II-III material; would kill: a zero or positive coefficient on a larger dated set.
- Grade B (known in kind): the strongest cross-site vocabulary cluster is the sanctuary group IO-SY-PK (z 4-4.7 against document-label permutation), i.e. the libation formula: genre, not contagion.
- Cause of failure: 63 % of words are site-endemic and sites share only 43 types (la13, la15), so a symmetric sharing matrix carries no direction, and size (document mass) dominates every summary.

## The palace spoke many tongues (la19)
- Idea: if Linear A words (especially entry names) come from several languages, as in a trading port, a mixture of sign-level phonotactic models should need K >= 2 'languages', and those should split by site, document type, commodity or position. About 31,000 collapsed-Gibbs fits on 65 corpora (sign identities only; no sound values).
- The K test is uninformative by itself (grade A, method): 33 of 36 single real languages at Linear A size also choose K >= 2 by held-out likelihood. Planted two-language mixtures are found (8/10) and give STABLE components; resampled one-language lists give K = 1.
- Grade B (calibrated negative): Linear A chooses K = 4, but its gain (0.079 nats/word) is below the single-language median (0.114) and its partitions are less stable (ARI 0.18 vs 0.38). The components track no site, support, commodity, number size, position or hand (0 of 24 after Holm), do not carry over from Hagia Triada to other sites, and cluster on tablets only by site.
- Grade B: an outlier test (per-sign surprisal under one pooled model) recovers planted 20% minorities (z 10-19) and Knossos's non-Greek-looking Linear B names (z 2.6; label z 6.8). In Linear A, Phaistos-only and roundel words look foreign (z 3.3, 4.2), but this is killed when words with rare or unnamed signs are dropped (z 1.4, -0.5). It is an effect of the sign inventory, not of phonotactics.
- Limit: the Linear B positive controls pass only in part (components separate Greek-looking names in about half the fits), so the component tests are moderate negatives. The outlier test is the strongest test, and it finds nothing.

## Universals build the grid (la21)
- Idea: assign the 65 commonest signs blindly to a consonant x vowel grid by annealing (~7,200 restarts), scoring only universals: consonant and vowel tiers independent, consonant OCP, pure vowels word-initial, small cells. No values as input; LB values only to score controls and as an outside check.
- Control: Linear B consonant rows are recovered at LA size (AUC 0.79; shuffles 0.68 / 0.52) and replicate Knossos -> Pylos. Vowel columns are not recovered beyond word co-membership, so columns are void.
- Grade B: LA's blind rows agree with the LB-derived consonant series (AUC 0.72-0.74; shuffles 0.62 / 0.51; z 6-10; all 5 grid sizes). Independent of the la5 consonant-avoidance test, which used values.
- Grade B: pure-vowel row A, U, I. Grade C: QA behaves like a pure vowel (vowel row 0.70; in both site halves). Would kill: QA at its shuffle level in a larger run.
- Killed: specific new sign groupings: LA row pairs do not replicate across HT / non-HT halves (0.045 vs base 0.049), though LB's do.


## Pecking order (la20, 4 Oct 2026; `loops/la20_final.txt`)
- Each pair of items in one list was treated as a contest won by the item written first. Rankers (Bradley-Terry, Plackett-Luce, Elo, Bayesian BT, mixtures of 1-4 orders) were fitted to all lists and scored on how well they predict the pair order on unseen tablets. Nulls: within-list shuffles, with the family-wise maximum taken over every configuration tried.
- Grade A (existence): commodity logograms are written in a consistent order. Held-out accuracy is 0.715 against 0.50 (FWER 0.005). It beats a null that writes large amounts first (0.548, P 0.002), it is more transitive than chance, and it holds on Hagia Triada alone and when tablet sides are split. The order is CYP, GRA (and *307) before VIR and OLE, before OLIV, VIN and QA2. Rank does not follow typical quantity (rho 0.31, P 0.17). Grain-first sequences on HT tablets are known in kind.
- Grade B: the Hagia Triada order predicts the other sites (0.655, P 0.045; the reverse direction gives P 0.1).
- Grade B (calibrated negative): words, entry words and first signs show no pecking order (0.51-0.52). Linear B Pylos words at Linear A size give 0.61 in 10 of 10 draws. Planted hierarchies of moderate strength that most lists follow are found. Limit: the Pylos town order is recovered only when some lists write it out in full, so a canonical tour that Linear A scribes only ever wrote in pieces would not be seen.
- Grade C: *307 is a top-rank (grain-class) commodity. Would support: *307 placed before GRA or CYP on new tablets. Would kill: *307 written after OLE or VIN.

## Grow the script in a box (r3, 4 Oct 2026; `voynich/loops/r3_final.txt`)
- Simulated societies + ABC put LA in a named, numbered ledger world with a ~100-200-sign syllabary, written dividers, no secrecy (C; agrees with known structure). Held-out 3/4.
- 'One commodity carried down each tablet' killed: it predicts line-final purity 0.56, real 0.28.

## The fire set the clock (la25, 4 Oct 2026; `loops/la25_final.txt`)
- Idea: each LM IB archive was baked by its destruction fire, so its commodity mix (GRA, VIN, OLE, OLIV, other; one count per document and commodity) is a snapshot of one month. About 1.5x10^7 simulated archive sets (random agronomic crop calendars x destruction month per archive x seasonal strength x site noise x base rates), rejection ABC.
- Killed (grade A, calibrated negative): no destruction month for HT, KH, ZA or TY. Posteriors are flat (max 0.10-0.12 vs 0.083) and the best month changes with each setting. Planted archives at LA size are not recovered (true-month mass 0.087-0.094), and not at 64x LA size either. With base rates known (oracle) the clock reaches only about a season (MAP within one month 0.5-0.6).
- Cause: what a site normally records (base rate) and when it burned (season) are confounded; more tablets do not help.
- The Linear B control fails: KN and PY, argued to have burned in spring, get 0.32-0.50 of their posterior in Feb-May (prior 0.33); every LB archive is pulled to Jun-Jul. Shuffled commodity labels are as informative as the real ones; agronomic calendars fit no better than random ones.
- One fire or several: untestable (P(same month) 0.49; planted AUC 0.52-0.61). Two random halves of Hagia Triada are not recognised as burning together.
- Outside check: no published season for any LM IB fire was found. Grade C hint: the bowl of preserved olives at Zakros (autumn-winter if fresh). Would support: fresh-olive stores in other LM IB fire layers; would kill: the olives shown to be cured.
- Lesson: a fire clock needs an independent anchor (month names, dated intake records, or seasonal plant remains in the same burnt room).

## Count the people (la23)
- Idea: treat small-number entry words on the 205 Hagia Triada tablets as individuals and count the population the administration dealt with by capture-recapture (Chao1, Chapman Villa vs Casa, log-linear models, Bayesian Mh, ~110,000 simulated archives for ABC with homonyms and variant spellings). Controls: Linear B KN and PY at LA size (whole-archive distinct-name counts as truth), planted populations, shuffles.
- Killed: blind person/non-person separation at LA size. LB-trained classifiers transfer across sites at AUC 0.59 (controls 0.50); a blind mixture is at chance; on LA, transaction words (KI-RO, SA-RA2) come out most 'person-like'. The operational rule used instead (entry, a number <= 5, on <= 5 tablets; 177 HT types) has LB precision 0.66 (KN) / 0.36 (PY), and includes LB place names such as PA-I-TO.
- Grade B (method): from an LA-sized LB sample, the ABC predicts how many distinct named people the whole archive shows (5/6 intervals cover KN 798 / PY 963; ratio 0.92), and new names on unseen LB tablets within 2 %. Planted coverage 95 %.
- Grade B: Hagia Triada used ~1,000 [510-2,430] distinct small-number entry words (raw estimators 370-1,000). The figure rests on recurrence frequency only: shuffling names across tablets or permuting findspots leaves it unchanged (no locality signal). The p >= .5 set gives 4,580, the open vocabulary of la15.
- Grade B (pre-registered prediction): on a newly found HT tablet, ~55 % of its small-number entry words will be unseen (mean 1.7 new per list tablet, 90 % 0-5). Would kill: < 25 % or > 85 % new across 5+ new tablets.
- Grade C: if 36-66 % of the set are persons, ~370-670 named people [~180-1,600], consistent with a town of ~1.8 ha excavated (~300-740 residents at Minoan densities) plus dependants, and an order of magnitude below Knossos/Pylos. Would support: a new HT tablet's names falling inside this pool at the predicted rate; would kill: a secure person list showing mostly recurring names. Casa Room 9 lists entry words met nowhere else (P 0.045, Holm 0.18): C.

## The clerk ran an apportionment algorithm (la24, 4 Oct 2026; `loops/la24_final.txt`)
- Idea: amounts in a list were made by dividing a total among recipients by share classes and rounding (largest remainder, D'Hondt-, Sainte-Lague- or Adams-like divisors, or exact division with fractions absorbing the remainder). About 90 M (list, rule, share alphabet) fits by MDL against an adaptive baseline, then baseline-free fingerprints and a cross-tablet test. Controls: value-band and shuffle nulls, planted apportionments, Ur III ration lists (CDLI), Linear B (PY Es).
- Grade B (method): the MDL apportionment search has no power at Linear A list lengths (median 4-5 entries). Planted apportionments are found 2/40 and Ur III rations 1/100, and every dataset compresses less than its nulls. No Linear A tablet is explained by an apportionment rule; no share classes or recipient ranks can be read.
- Grade B (calibrated negative): Linear A lists are not built on one unit share (4/144 vs 3.2 shuffled, P 0.38), while Linear B lists are (P 0.006). Upper bound: a few lists could be and go unseen.
- Grade B (negative): fraction-absorption fits favour the conventional values over random value sets (P 0.011) only because those values are simple; simple-value sets (P 0.31) and scrambled letters (P 0.36) do as well. These fits do not test the fraction readings.
- Grade C: HT 9b repeats three of HT 9a's recipients at exactly 4/5 (*306-TU 10 -> 8, *324-DI-RA 2 J -> 2, TA-I 2 J -> 2; pair P 0.016, 0.25 after 15 pairs). This works only if J = 1/2. Would support: another pair scaled by one factor whose fractions only work at J = 1/2. Would kill: a re-reading of the J signs on HT 9a. Cross-tablet proportionality overall is weak (summed k P 0.044 against within-side permutation; nothing against value replacement), although the test passes on PY Es and on planted pairs.

## The balance pans know the numbers (la27, 4 Oct 2026; `loops/la27_final.txt`)
- Idea: excavated balance weights give the physical ladder (unit c. 61 g, mina 8 units, talent c. 29 kg, about one oxhide ingot). Linear A amounts of weighed goods should heap on it, and the fraction signs should fall on its sub-units. Data: 86 published masses (`data/la27_weights.json`; Michailidou 1990, 2001, 2006; de Zwarte 1998-99, which reproduces Petruso's masses). Petruso's full catalogue is a lending-only scan ([archive.org](https://archive.org/details/ayiairinibalance0000petr), login needed).
- Grade B (calibrated negative): the 70 Neopalatial masses show no quantum (61 g: P 0.45; best 17.1 g, P 0.59) and no 8-unit mina step. Planted lattices with errors up to 6 % are found (0.69-0.85), so mixed standards or larger errors are likely. The ladder taken forward is the literature's 1:8:60 (an ingot is 475 units, or 59 minas).
- Grade B (calibrated negative): Linear A integers do not heap on 1:8:60 (P 0.37 overall; 0.56-0.996 per commodity) or on any of 10^4 random ladders beyond decimal counting (the best does not replicate across document halves, P 0.75; jittered numbers do as well). Planted 5 % heaping is found 8/8. Weighed-goods signs cannot be found this way.
- Control (A): the Linear B ceiling search recovers M:N = 4 and LANA:M = 3 exactly from the numbers alone (a sub-unit is written at most r-1 times). The search gives only a lower bound when n is small (L:M >= 27, n 8). The fraction-versus-weight test has no power (planted 0-0.41).
- Grade B: the same ceiling rule applied to the fraction signs. D is a counted part: it is written 2-4 times in 15 of 37 amounts (9 of 31 without the separate 'DD' sign) at HT, PH and ZA, so r >= 3. B is written twice in 5 of 36. K, L2, JE, F and H are never repeated (0 of 16-34; against D, P 6e-4 to 0.02), so they act as single denominations.
- Grade C: D = 1/5 (DDDD at HT 115a gives r = 5) and B = 1/3, as in the conventional values. A binary D or B (attack-1's 1/8) is disfavoured. Physical pieces exist near 1/5 of the unit (five of 11.5-12.6 g) and 1/3 (two discs of 20.2 g). No piece is near 1/2 (30.5 g). Would support: a sound DDD or DDDD on another tablet. Would kill: HT 115a re-read and five Ds in one amount anywhere.

## The receipts fed the ledgers (la28, 4 Oct 2026; `loops/la28_final.txt`)
- Idea: roundels and inscribed nodules are receipts; if clerks copied them into ledgers, their terms, quantities and counts should reappear on tablets of the same site and deposit. Data: 141 inscribed roundels, 876 nodules, 7 sealings; Linear B sealings and nodules (KN Ws/Wn/Wm, PY Wr, TH Wu, MY Wt, MI Wv) as the control.
- Grade B (calibrated negative): 41 of 131 receipt types occur on their own site's tablets vs 35.9 under a site swap (z 1.7, P 0.064). Vessels and other documents show the same regional excess (z 2.4), and tablets share far more with each other (z 6.3). The Linear B control passes (z 4.2, P 0.0005), and planted copying of 10 % of receipt types into ledgers is found 0.97 of the time. The HT Room 13 tablets share less with the Room 13 nodules than random HT tablets do (1 vs 3.0). The Khania roundel classes (*411-VS, *164, O, *301+*311, *86+*188, VIR+KA) occur on none of the 104 Khania tablets.
- Not testable: arithmetic reconciliation. Only 14 receipt quantities survive (4 integers). Receipt-class counts match KU-RO totals at the jitter rate (P 0.39), and the test only sees 4 or more planted totals. No tablet looks like a summary of receipts.
- Grade A (counts): 19 % of receipt types use a sign that no tablet uses (other documents 6.5 %, tablet hold-out 5.6 %; P 0.005). 9 of 14 receipt quantities are fraction-only, against 17 % of tablet quantities (P 0.0001).
- Grade B: receipts and ledgers are separate administrative layers. The Khania roundels record goods of the vessel class (*4xx-VS) and others that no surviving ledger lists.
- Grade C: KA, KU, SI and I, the signs on HT nodules, appear in HT ledgers more often without a number (0.66 vs 0.77, P 0.001; P 0.007 without TE), as markers or headings rather than counted entries. Would kill: the effect disappearing under a stricter heading/entry parse.

## Predict what the fire destroyed (la22, 4 Oct 2026; `loops/la22_final.txt`)
- Idea: restore lost signs, numbers and commodity signs with random ensembles (thousands of random predictors over n-gram, lexicon, tablet, site, scribe, accounting and neural experts), calibrate on known items hidden at the real damage pattern (the edition's break mark sits at line ends, so words are cut at one end), freeze predictions for every real break (sha256 in `data/la22/la22_predictions.sha256`), and score them against SigLA's later independent re-reading. Controls: frequency and Markov baselines, Linear B LA-sized sample, shuffled-tablet model.
- Grade B (method): sign restoration is calibrated (top-1 0.28, top-5 0.52, ECE <= 0.09; frequency 0.06, Markov 0.22; Linear B 0.45 under the same conditions). It rests on word-internal regularities: shuffling words across tablets leaves it unchanged, unlike Linear B, where repeated words on a tablet help.
- Grade B: the accounting skeleton (la14) restores quantities and commodity signs: numbers top-5 0.61 vs 0.56 (gain gone when tablets are shuffled; KU-RO arithmetic 0.31 vs 0.20 top-1 inside total-closed sections); commodity signs 0.19 vs 0.09 (shuffled 0.13), carried by the tablet's other commodities. Language does not reach the numbers, and accounting does not reach the spellings.
- Grade B (negative, out of corpus): SigLA and lineara.xyz agree on 98.8 % of 2,789 aligned signs. Of 19 clean re-readings, the frozen top 5 holds SigLA's sign once (the old reading 5 times, P 0.024): the restorer stays anchored to the corpus's own spellings and does not anticipate corrections.
- Grade C: break-edge restorations: 2/13 SigLA extra signs at top-1 (P 0.03), but 1/4 in the clean same-word subset (HT 3 MA-[DI]); low restoration probability flags changed readings (AUC 0.65), mostly through sign rarity (frequency alone 0.60). Would support: >= 5 of 10 frozen high-confidence edges confirmed by new photographs or joins. Would kill: 0 of 10.
- Frozen (arithmetic, not new): missing KU-RO entries HT 11a 4, HT 27a 13, HT 39 72, HT 46a 42, HT 100 4, HT 109 121, HT 122a 9.

## The rings know who talked (la26, 4 Oct 2026; `loops/la26_final.txt`)
- Built the seal-sharing network as data (7 same-seal groups across HT, KN, ZA, GO, Thera and Sklavokambos; Khania only a look-alike; `data/la26_ring_network.json`) and asked whether ring-linked sites share more words, sign profiles, ligature variants or entry structures than distance and size predict, and whether multi-site words sit on ring paths. Nulls: doc-perms within support, QAP, seal-graph rewiring, distance gravity, all graphs of equal size; planted controls; Linear B calibration.
- Not supported (calibrated upper bound): no measure passes at site level (best P 0.058). A planted ring vocabulary is found only at ~96 shared types, about 4x all real cross-site sharing, so a real ring effect could be invisible.
- The word-level 'ring path' signal (doc-perm P 0.0005) is killed: the same null passes 37.5% of random graphs on Linear B, and against all graphs the ring graph ranks P 0.16-0.66. HT shares 9 admin words with Zakros (3 rings) and 9 with Khania (no ring), 1 with Knossos.
- Grade B: Khania is an island in sign profile and entry structure, not just vocabulary (z -11 to -17 vs HT and ZA beyond size and support).
- Grade C: ring pairs have slightly more alike sign profiles (P 0.06). HT-ZA shared words (incl. KU-RO) as ring-office terms, demoted. Same seal, same incised sign (RO on 3 of 4 S68 nodules), P 0.088.

## The winds carried the words (la31, 4 Oct 2026; `loops/la31_final.txt`)
- Built travel-time matrices for 29 find-sites (`data/la31/travel.json`): walking (Tobler on a 0.01 deg DEM), square-sail sailing with oars over daily ERA5 winds 2014-2018 (1.48 M simulated voyages, by month and season, asymmetric), combined walk + sail, and calm rowing. Tested whether site-pair sharing (words, rarer signs, logograms, entry structure; z against support-stratified document permutations) follows sailing time better than km, walking or size, and whether types sit downwind. Controls: rewired networks, planted wind / km / seasonal / downwind spread on the real geography, a corpus-size power curve, Linear B (7 sites).
- Grade A (method): symmetrised, every travel model ranks the Linear A site pairs like straight-line km (Spearman 0.96-0.999). Wind only adds direction (etesian Kea -> Knossos 38 h, back 137 h), and inside Crete the asymmetry is small (median 4 %).
- Grade B: inside Crete, nearer sites share more rare signs and logograms (ALL r +0.24, P 0.023), but km, rowing, walking and every wind season tie. There is no distance decay over all 29 sites or among the 12 big archives.
- Not supported (calibrated, low power): wind beyond km (partial P >= 0.14; planted wind spread is never told apart from km, 0/12), downwind flow (words Delta -0.86, upwind side, P 0.91; test AUC vs symmetric spread 0.68 at LA size, 0.79 at 4x, 0.74 at 16x), any month or season (max-null P >= 0.31; planted summer vs winter spread are indistinguishable), slower or faster leg.
- Grade C: among the big archives, words follow the easier leg (P 0.097). Would support: P < 0.05 with more island material; would kill: r <= 0 without HT. Linear B words fall off with distance (P 0.03) and lean to sea-route rowing time over km (P 0.04-0.06); wind adds nothing there either.

## The land wrote the ledger (la29, 4 Oct 2026; `loops/la29_final.txt`)
- Idea (CONTEXT #6, terroir): each site's hinterland shapes what it stored. 53 per-site land variables from open data (DEM, WorldClim, CLC 2018, SoilGrids; 2/5/10 km catchments; `data/la29_land.json`) used to predict a held-out site's commodity mix, and as a decoder for signs whose site frequency tracks land. Nulls: 10^4 random smooth fields, size alone, shuffled site labels; controls: planted land effects, CLC land cover as a 'modern ledger', Linear B.
- Not supported (B, calibrated negative): out-of-site skill ~0. Best vine_5 +0.031 nats/entry (P 0.18 against fields); shuffled labels score higher. Planted effects of x2.7 per SD are found 9-19/20; the modern ledger 5/10 at LA counts. The Linear B control (4 archives) fails. 10^4 random land indices chosen on half the documents do no better on the other half than chosen random fields, and do not transfer to the Linear B archives.
- Decoder killed: in administrative documents no sign tracks land (3/66 vs 2.9 shuffled; 0/42 without HT). With all documents the libation-formula signs (JA, WA, I, NA, *301) rise with upland, non-arable ground because the sanctuaries are there (A, known in kind: genre, not produce). No commodity class assigned to any sign.
- Grade A (method): Gaussian random-field nulls are far too lenient for skewed site-level variables (shuffles 24 signs vs 3.6 expected); rank-mapped fields are calibrated.
- Grade C: vine-suitable land as the weak best predictor. Would support: P < 0.05 with new commodity tablets; would kill: skill <= 0 on new non-HT, non-KH commodity documents.

## The slips tell how they wrote (la32, 4 Oct 2026; `loops/la32_final.txt`)
- Idea: dictation produces slips between signs that sound alike, copying produces slips between signs that look alike. One-sign word pairs (163 of 3+ signs, 2,745 of 2+ signs) were scored against shape similarity (4 image metrics on the Noto Sans Linear A/B glyphs) and against the blind la21 consonant rows. Null: partners shuffled with each sign's confusion count kept. Controls: Linear B (DAMOS KN+PY), 29 modern SigLA re-readings of LA signs, planted dictation and copying slips (generator different from scorer), English misspellings vs OCR errors, ~9,000 random production hypotheses re-tested on held-out pairs, and a distributional-similarity circularity control.
- Grade A (control): Linear B one-sign variants follow sound (same consonant z 17.6) and never shape (z <= 0.3). Most are grammar or spelling variants, not slips.
- Grade B (method): the shape score sees real visual misreadings of LA signs (SigLA vs lineara.xyz z 2.6 without *118) and planted copying slips down to 20-30 % of LA's pairs; planted dictation slips are found down to 10 %. The English pair: misspellings sound z 35 vs shape 4.5; OCR shape 21 vs sound 20 (vowels look and sound alike), so the shape channel is the noisier one.
- Grade B (calibrated negative): no slip mode is visible in Linear A. Shape z 0.3-0.8, below the planted 20 % copying level; sound below the planted 10 % dictation level. The idea that LA tablets were written from dictation gets no support from their variants; nor does copying from documents.
- Grade B: LA one-sign alternations are weakly sound-structured under the blind rows (z 3.95 for all pairs; 3.65 within distributional-similarity deciles; LB 13.7 under the same control), about a quarter of LB's strength. The LB-value consonant check is weak (z 1.6).
- Grade C (near demotion): variants on one tablet look alike (z 3.1-3.4 against the degree-kept null, but P 0.08-0.18 against random within-site pairs). Would support: >= 10 new same-tablet pairs with mean shape rank > 0.6. Would kill: P > 0.1 on a larger set.
- Killed: pair-level consonant or vowel guesses (14 of 347 pairs at P < 0.05, against ~17 by chance).

## Queues at the storeroom (la33, 4 Oct 2026; `loops/la33_final.txt`)
- Idea: scheduled rations to a fixed staff and demand-driven transactions leave different queue statistics (list length and its dispersion, recipient recurrence, amount uniformity and consistency, tails). About 12,000 simulated archives (scheduled / gamma-Poisson demand / mixed) train classifiers. Controls: planted archives; Ur III rations and regular offerings vs deliveries, expenditures and receipts (CDLI, kind fixed by keyword); Linear B series of known kind; shuffles; size matching; totals as an outside prediction.
- Grade B (method): at archive level the statistics rank Ur III ration groups above delivery groups (AUC 0.87 at 150 texts, 0.66 at 30; label shuffle 0.52) and Linear B personnel/census series above disbursement series (0.80, P 0.011). The scale is not calibrated across corpora.
- Grade A (calibrated negative): a tablet-level model that recovers planted regimes (0.96) fails on real Ur III tablets (0.42-0.55). Linear A tablets and words therefore cannot be labelled scheduled or demand-driven. Word cohesion (P 0.0095) was circular; a non-circular Ur III-trained score gives P 0.29. Killed: KU-NI-SU, DA-ME, DI-DE-RU as ration recipients; MI-NU-TE, SA-RU, DA-ME as 'switching' offices.
- Grade B (negative premises): in Ur III, ration texts have longer and more overdispersed lists than deliveries (var/mean 13.6 vs 5.6), and both regimes name mostly persons ({ki} places 2.1 % vs 2.7 %). Totals do not mark the regime in Ur III (1.2 % vs 1.0 %) or in LA (P 0.25).
- Grade B: grain (GRA) lists hold amounts of similar size (within-list CV 0.36 vs 0.52 expected, P 0.012, site x length stratified); OLE and CYP lists do not. Linear A list lengths are far more regular than Ur III lists of either kind (dispersion 2.2-3.3 vs 8-78), probably the small tablet format.
- Grade C: every Linear A group (site, scribe, commodity) sits at the demand end of the scale (w 0.01-0.32 vs Ur III 0.29-0.52). Would support: Linear B multi-entry ration lists in LA format scoring above every LA group; would kill: known ration lists scoring as low as LA. Grade C: grain lists are rations. Would support: new grain lists with near-equal amounts for recurring entry words; would kill: new grain lists as varied as the OLE/CYP lists. Khania's low score (0.01 vs HT 0.11) is within Ur III same-kind site spread and is not claimed.


## The totals are in a hidden currency (la30, 4 Oct 2026; `loops/la30_final.txt`)
- Idea (CONTEXT #1): KU-RO totals that miss the plain sum would balance if each commodity converted into one unit of value at a fixed rate. About 8x10^7 random rate vectors (p/q <= 12) with free fraction values, plus annealing, fitted on half the tablets and tested on the other half. Unlike la6, only written totals constrain the rates.
- Grade A (method): the same search recovers a real ancient currency from totals alone. Ur III merchant accounts (CDLI) with the per-line silver hidden: 7.4 of 29 held-out totals are balanced, against 1.6 with labels shuffled and 2.5 with random totals. At Linear A size (32 sections) it is 3.8 of 16 against 0.5 and 1.2. Bulk-good prices come back within about 20 % (barley 0.0028 vs the written 0.0033 gin2 per sila3, oil 0.065 vs 0.078). Even Ur III prices scatter ±20-50 % between texts.
- Grade B (calibrated negative): no rate vector balances held-out Linear A totals beyond the nulls. With the Occam reset, the exact criterion gives -0.30 against -0.22 for random totals (P 0.6), and the rounded criterion gives -1.43. A planted currency at Linear A size is found in 11 of 12 runs, including on mixed sections. The misses on HT 27a, 94a and 100 are not conversions of VIR+X, *86, *305 or TI+A. There are no implied commodity values to check against Bronze Age prices.
- Fractions: D and B occur in none of the 32 KU-RO sections (0/32), so totals cannot test D = 1/5 or B = 1/3. Free fraction values fitted to totals do not transfer to held-out tablets.
- Not testable: hidden totals without KU-RO (212 tablets). The planted control fails, because chance coincidences among small numbers swamp a planted currency. A P 0.05 edge in the strict setting comes from parallel tablet sides (HT 9a/b, ZA 6a/b). Not supported.

## The number bends the word (la35, 4 Oct 2026; `loops/la35_final.txt`)
- Idea: in many languages a counted noun changes with its number (singular, dual, plural, counting form), so a stem written before 1 should end differently from the same stem before 2 or 10. Tested on 1,000 word + number entries: 117 stem-definition x number-class cells (39 stem definitions, 30 of them random cuts), a held-out search over 1,500 random ending rules per split x 20 splits, and bare W vs extended W+X pairs.
- Grade B (calibrated negative): no ending tracks number class once the total words are removed (family-wise P 0.16-0.61). A planted ending rule on 15% of entries is recovered 3/3. Linear B count entries at Linear A size show their real number marking (-U before 1; -WE, -DE, -RA before >= 2) against the site null in 5/6 subsamples, but against the strict within-tablet null in only 2/6. So effects the size of Linear B's are excluded only loosely.
- Grade A (known, accounting): the one ending that tracks number is final -RO in KU-RO / KI-RO / PO-TO-KU-RO (48 of 53 entries >= 3). This is the totals, not grammar.
- Killed: "the longer form W+X goes with 1" (P 0.003 against shuffled numbers). Length-matched fake pairs do the same (P 0.31), because 3-sign words head 1-entries more often (0.35 vs 0.22). Linear B passes this control (W+X before larger numbers, P 0.004; 3/6 at LA size).
- Grade C: an LB-sized plural suffix cannot be excluded with 37 LA stem pairs (90% interval -0.11 to +0.18 in share of '1'). Would support: new W / W+X pairs with the longer form before >= 2. Would kill: a pooled excess >= 0 over 80+ pairs.

## Find the doublets (la37, 4 Oct 2026; `loops/la37_final.txt`)
- Idea: Linear B has doublet signs (a/a2, a/a3, ra/ra2, ro/ro2, pu/pu2) used in the same words. Every pair of the 64 commonest LA signs was scored blind for interchangeability (context cosine, the Voynich v35 split-half index, one-slot swaps), and separately by the held-out gain of a class-bigram model that merges the pair (plus 20,000 random merge hypotheses re-tested on held-out tablets). Search correction by replication across random document halves. Controls: Linear B KN+PY blind, LB at LA size and length-matched, planted LA doublets of six kinds.
- Grade A (method): LB doublets and same-consonant pairs rank at the top (AUC 0.78; top 20 70 % related vs 23.5 %; 0.69 at LA size); merge gain is doublet-specific (0.79 vs same C 0.45). Planted free-variation doublets are found down to a 15-25 % use rate (4-5/6) and writer-split ones 5/6; doublets fixed per word or confined to word-initial position are not (0/6 each). A within-word shuffle null removes planted doublets too (0/24), so it cannot serve as the test.
- Grade B (calibrated negative): Linear A has no free-variation or writer-split doublet among its 64 commonest signs (0 replicated pairs; < 3/6 at use rate >= 0.25 could hide), and fewer freely alternating pairs than every LB sample of its size (0 vs 2-8, 20/20 draws, also length-matched). In LB those pairs are mostly vowel alternations of one consonant (inflection), so LA lacks LB-like inflectional alternation among common signs (agrees with la18).
- Grade B (outside check, depends on LB values): which LA signs are near-interchangeable follows LB-derived consonants (AUC 0.67, P 0.0001; merge gain 0.61, P 0.0001), as strongly as true consonants do in LB at LA size. The blind la21 rows agree (Spearman 0.32, LB 0.39), but rows are themselves context-derived: consistency, not independent proof.
- Grade C: SI~TI, TI~TE and RE~ME alternate word-finally (TA-NA-TI/TA-NA-TE, NU-TI/NU-TE, DA-RE/DA-ME) and pass both detectors. Would support: both forms of one stem on one tablet or in one hand; would kill: the same pairs alternating word-medially as often. KU~KI is the KU-RO/KI-RO pair.
- Demoted: A~DA, A~PA, A~KU (word-initial slot effect; they fail the merge test). The writer-vs-position split test fails its LB control (LB doublets show no split either), so spelling-habit vs sound-rule cannot be decided. Rare optional doublets of the LB kind (a2, a3, ra2) are below detection at LA size even in LB.

## The hand in the strokes (la34, 4 Oct 2026; `loops/la34_final.txt`)
- Idea: recover scribal hands blind from SigLA's drawing of every sign occurrence (779 documents, 5,105 occurrences; images kept outside the repo; per-occurrence shape scalars in `data/la34/la34_occ_features.json`), score them against published hands held out of the clustering (within site, label-permutation nulls), then use them. Controls: the two sides of one tablet, planted affine hands, a 3,000-hypothesis random search re-tested on unseen hands against permuted-label searches. No Linear B per-occurrence drawings are openly available, so there is no LB control. Prior work: Castellan et al. (IWCP 2021 abstract) announced ML on SigLA for hands, without published scores.
- Grade B: hands are partly recoverable from sign shapes. Comparing the same sign across documents gives nearest-neighbour hand 0.37 vs 0.19 (0.49 vs 0.28 with >= 8 drawings), ARI 0.28-0.31 vs 0.16-0.21 (P <= 0.006); unseen hands AUC 0.58 vs 0.49 (P < 0.017); sides of one tablet AUC 0.65. The hand shows in per-sign width and diagonal steepness and in line pitch and word gap; slant averaged over all signs carries little. Planted hands are found only at about +/-10 deg slant + 20 % stretch.
- Grade B: without any labels, documents drawn alike share vocabulary (residual rho 0.083, P 0.007), so the shape route agrees with la16's text route to hands.
- Not supported: commodity specialisation by hand (P 0.14-0.26; upper bound). Cross-site pairs are farther apart than different writers at one site (site or drawing campaign dominates), so a travelling hand cannot be shown.
- Grade C: one-sign variant pairs are written by hands no closer than unrelated documents (P 0.24), so variants are not one writer's alternation; tablets vs sealings/roundels differ in drawn form (P 0.02, few pairs); cross-site look-alikes HT 63-ZA 9, KH 16-KN 32, HT 127-KN 32; blind writer clusters in `data/la34/la34_blind_hands.json`. Would support: hand attributions from photographs reproducing the clusters. Would kill: equal recovery when hands are permuted within find-group (a draughtsman or drawing-campaign signal).

## Are the borrowed values better than chance? (la38, 4 Oct 2026; `loops/la38_final.txt`)
- Idea: give LA signs their LB-derived values and ask whether the resulting sound strings obey language-independent phonotactics (consonant OCP, place OCP, onset principle, phoneme-stream compressibility) better than 10^5 relabelings that keep the grid's shape (vowels permuted inside rows, consonants inside columns) or not (all values permuted). Measure sets fixed on full Linear B first. Unlike Packard 1974 (LB-like alternations) and Davis 2024 (cross-script overlap), no Linear B words or alternations are used.
- Grade A (method): LB true values beat their relabelings 10/10 at LA size (consonant set z 3.7-5.8, vowel set 3.2-6.6); 100 wrong-value maps sit in the middle (2-11 % at P < 0.05); the Cypriot Idalion tablet passes the vowel test (z 3.9), the consonant test only at P 0.06 (1,010 signs). Relabeling whole rows (R1) is killed: it tests row sizes, not values (wrong maps pass 47-74 %).
- Grade B: the LB-derived values are not arbitrary for Linear A. Consonant set z +4.1 (P 4e-5), vowel set +3.8 (P 4e-4), inside LB's true-value range at the same size; replicated for consonants in Hagia Triada and non-HT halves (z 2.8, 3.8). Consonant OCP works at the word level (survives within-word shuffling, as in LB).
- Grade B (rows): pure vowels A E I O U, M-, T- beat chance in the full corpus and in both halves; R- and K- in the full corpus. Grade C: D-, S- marginal. No verdict on J-, N-, P-, Q-, Z-, W- or the *2 signs (true LB rows also fail about half the time at this size); vowel columns cannot be tested one by one, in LB or LA.
- Grade C: the vowel fit is weak at Hagia Triada (z 1.2, below all LB half-size samples) but holds elsewhere (3.0). Would support: the same deficit on new HT documents; would kill: HT >= 2 on a larger sample. Sign flags (support test AUC only ~0.68): TO (vowel), PI (consonant), WA/WI order.

## Type inference as a compiler does it (la40, 4 Oct 2026; `loops/la40_final.txt`)
- Idea: give every word type one role (person/group, place, commodity/qualifier, heading/transaction, total, number-word) and make every tablet type-check (totals close, headings before lists, logogram contact, one entry role per list); sample role assignments for all types at once (tempered Gibbs, ~1e9 updates in all), read off 'forced' roles. Controls: word-shuffle null through the same search; Linear B KN+PY at Linear A size with known classes; planted roles; 1,200 random learned rule systems re-tested Knossos -> Pylos against a label-permuted search; disjoint-halves stability.
- Grade A (method, calibrated negative): the type checker cannot name word roles from structure. In Linear B at Linear A size forced roles match known classes no better than shuffled words or the majority class (TO-SO is never typed as a total, even on all of KN+PY), and rule systems learned on Knossos score at chance on Pylos (0.176 vs permuted-label search 0.174). No Linear A word class is established this way. Its 27 'forced' non-default roles (KU-RO, PO-TO-KU-RO total; TE, SA, *307 qualifier; A, A-DU, MA, JA, KU-RE, U heading) echo the rules, and KU-RO keeps 'total' in only 3 of 12 half-corpora.
- Grade B (calibrated, relative): role partitions are reproducible across disjoint halves in Pylos (ARI 0.22-0.33 at matched size) and much less in Linear A (0.10-0.14; shuffled ~0). LA word types carry position/role identity at about 0.4x Linear B strength, in line with la8's flat two-open-class grammar.
- Grade C: KU-RE, U, A, DA, KA-NA are heading-type words (first on the tablet, no count; same role in 3-5 of 4-6 held-out halves). Would support: these words opening new tablets without a number; would kill: them as counted entry heads on new tablets.

## la45 adversarial self-play (5 Oct 2026)
Proposer and critic machines argued over value-free meanings for about 1.4 billion candidates. The controls (planted system, Ur III, Linear B) passed.
- **B:** VIN, OLE, OLIV, VIR, GRA, NI and the single signs KI, DI, RE, TI and MA settle into one counted-commodity meaning. This comes from the type of document they appear in, not from word order.
- **B:** single-sign nodule words (*301, KA, KU, SI, RO, ZE) form a heading class.
- **B (negative):** Linear A tablets carry at most about 0.2 bits of role signal in word order, against at least 1 bit in Linear B and Ur III. Meanings learned at Hagia Triada predict other sites worse than having no meanings.
- **C:** OLE+U, OLE+MI and OLE+DI act as entries rather than commodities. Would support: they appear with personal-name-like words in new finds. Would kill: they appear in commodity slots after a word.

Open question: if word order carries so little, where does Linear A put its structure? la47 tests the physical layout.

## The layout is the grammar? (la47, 5 Oct 2026; `loops/la47_final.txt`)
- Idea: if word order carries almost no role signal, the structure may sit in space (line, line start, side, size, gaps). Tablets were rebuilt as 2-D pages: physical lines from the GORILA line layout in lineara.xyz (287 LA sides), sign boxes from SigLA (118 sides), Linear B pages from DAMOS. About 100,000 random spatial grammars (rules over layout features only) were scored on held-out tablets against equal searches over word order. Nulls: identities permuted within each page; text re-flowed onto another page's line breaks. Controls: Linear B, a planted line-start rule (found at 0.13-0.21 bits, order 0), a planted size shift (5 % detectable).
- **B (negative):** layout carries about 0.045 bits per word about which word sits there. Order carries 0.049, and layout adds only about 0.002 bits beyond order. In Linear B, order carries 0.32-0.38 bits and layout 0.06-0.10. Linear A does not hide its structure in space.
- **A (data):** Linear A lines are run-on. 14 % of line breaks split a word or a number (none in DAMOS Linear B). Breaks fall at entry boundaries 1.4 times chance (Linear B 2.2 times). This holds at Haghia Triada (1.55 times) but not at Khania (0.9 times, P 0.72).
- **B (calibrated negative):** commodities, header signs and totals are not written larger or smaller (a 5 % shift would be seen), and the gaps between words mark nothing robust.
- **B:** layout agrees with la45's classes. Totals sit at the bottom on a short line of their own (z 5-6), header signs start lines (z 2.9), and commodities sit inside lines next to their numbers (z 4.4; in Linear B, z 22.6).
- **C:** profiles of where each word is placed separate la45's classes (balanced accuracy 0.44 against a null of 0.33, P 0.035) where order profiles do not (0.28). In Linear B the reverse holds. Would support: P < 0.01 once more SigLA-aligned tablets exist. Would kill: P > 0.2. Nothing learned at Haghia Triada carries over to other sites.

## la48 the physics of goods (5 Oct 2026)
About 230 million flow-direction hypotheses were tested. No flow network can be read.
- **A (method):** conservation of goods recovers a planted economy only when at least half the documents survive. It gives only relative direction inside a chain of linked documents.
- **B (the control fails):** Ur III Puzrish-Dagan shows no balance beyond chance.
- **B (negative):** Linear A scores 16 against 10.1 ± 4.4 shuffled (P 0.14). Its "hubs" are the largest numbers and KU-RO.
- **B (negative, low power):** the amounts attached to a word repeat across documents no more than chance.
- **C:** Linear A entry words are not standing accounts with fixed quotas. Would kill: a word found with the same quota on new tablets.
- **C:** PE Wy 5 and PE Zg 5 record one transfer (*307 OLE 1 + J) twice. Would support: matching seal or clay. Would kill: different hands or find spots incompatible with one transaction.

## Hire a forger and watch where he fails (la46, 5 Oct 2026; `loops/la46_final.txt`)
- Idea: la14 asked whether forgers can be caught. la46 asks where. Every specific relation (sign pair, word -> number scale, word -> logogram, entry -> next entry, words sharing a tablet, site x word, totals) was counted on held-out real tablets and on ~3 M forged ones. The forgers came from 200-300 random architectures per corpus, including topic forgers that cluster tablets.
- Grade A (known): the KU-RO running total is the one relation that beats every forger group.
- Grade B: the only other relation the forgers cannot fake is an ordered commodity basket after SA-RA₂ (GRA or CYP, then NI, then VIN, sometimes OLE and *23M; 11 HT tablets). Residues found on half the tablets fire 3.0x as often as forgeries on the other half (Linear B 2.3-2.4x, forger worlds 0.2-1.4x). This fits la20's commodity order. Would support: a new HT tablet with SA-RA₂ in this order. Would kill: SA-RA₂ with a different order on 3+ tablets.
- Grade B (method): the calibrated rule gives 0 residues in forger worlds and shuffles, recovers 3/6 + 3/6 planted ordered constraints with no false passes, and finds the girls-then-boys order (ko-wa -> ko-wo) in both Linear B draws. It cannot see adjacent pairings (commodity-unit, word-number scale, sign pairs), because any forger copies adjacent statistics. Once forgers can cluster tablets it also cannot see co-occurrence, in Linear B as well.
- Killed: Linear A 'casts' of words that share tablets (cycle 1, 0.50 vs 0.20). They came from the two sides of one tablet being counted as two documents, and from forgers too weak to cluster tablets.

## The farming year is the answer key? (la50, 5 Oct 2026; `loops/la50_final.txt`)
- Idea: the fixed order of goods (la20, la46) follows the Cretan farming year (grain Jun, figs Aug, wine Sep, olives Nov, oil Dec; calendar from the FAO crop-calendar API for TN, MA and JO, plus Hesiod, Varro, Columella, Pliny and Cretan harvest dates). It was scored against 10^5 random calendars, all permutations, a northern-European calendar and shuffles, with Linear B KN/PY and planted calendars as controls.
- Killed (B, calibrated negative): LA order fits the Aegean year no better than random calendars (P 0.55; Hagia Triada 0.51; SA-RA2 basket 0.35), and the northern calendar fits as well. Co-occurrence shows no season (P 0.51). Blind search over random calendars selects non-Aegean calendars (agreement 0.47 vs 0.50). Planted farming years are found (13/20 at half the baskets, 10/10 at 80 %; blind agreement 0.79). The Linear B control fails too (P 0.19): LB writes goods in an administrative order and groups them by office.
- Limit: with 5 identified goods the calendar test cannot go below P ~ 0.04 (24 cyclic orders).
- Not graded: the seasons given to CYP (Mar), MI (Mar), QA2 (Dec), *401 (Dec), SI (Aug) and *304 (Oct) only restate list rank under an alignment that fails.
- Grade C: LA and LB share a staple order: grain, then the olive good, then figs, then wine (LA concordance with the LB order rank 1/24, P 0.04, olive goods merged after seeing the data). The harvest calendar puts olives last. Would support: the same rank on new LA baskets. Would kill: olive goods written after VIN on 3+ new baskets.
- Grade C: figs and olives share fewer LA tablets than chance (4 vs 9.2 expected). Would kill: chance level once 10+ more tablets with either are found.

## la51 find contexts (5 Oct 2026)
**B (negative, calibrated):** Linear A words do not track the rooms they were found in beyond chance. This used 1,541 Hagia Triada documents in 28 deposits and about 85,000 random links per run (P 0.48 and 0.81 against shuffles; held-out prediction P 0.71). Planted links were recovered, and the Linear B control passed: oil texts track storerooms (z 8.7), wheel texts the Knossos Arsenal (z 10.0), cloth texts the West Magazines. B (known pattern): libation-vessel words track sanctuary rooms. C: QA2 and TE go with the Villa magazines; NI, CYP, O, *86 and A-DU go with houses. Would kill: new finds of these in the other kind of room. Limit: room-level object records for Hagia Triada (Montecchi, Militello) are not open access; see docs/HUMAN-ACTIONS.md K.

## Resurrect the ancestor by simulating history (la44, 5 Oct 2026; `loops/la44_final.txt`)
- Idea: instead of comparing LA with real languages, simulate 190,000 random proto-languages (6 morphology classes, syllable structure, inventories), evolve each through random regular sound changes and borrowing, write the descendant in a CV syllabary under random spelling conventions, and read the ancestor's typology off an ABC posterior (random-forest projection + rejection).
- Grade B (method, calibrated negative): from a written type list of LA's size the typology posterior is driven by word length and sign inventory, which a sign shuffle keeps; shuffled LA gets the same confident 'isolating' call as LA (0.79-0.98) under every panel, and the shuffle control is never flat. The Linear B control passes at 583 types (Greek-like inflection 4-5/5) but fails at 107 types, as does the Cypriot Idalion control (agglutinative 0.79-0.86). Syllable structure, inventory size, borrowing and the kinds of sound change are not identifiable (R^2 <= 0.05, same for LB).
- Grade B (diagnostic): LA and LB both avoid same-series neighbours and doubled signs far beyond any simulated history and have lower adjacent-sign predictability, so these constraints mark a real spelled language, not a typology.
- Grade C: given that LA affixes, the lean is suffixing with Greek-type final alternation (0.57), then agglutinating suffixes (0.23); 6/60 shuffles reach 0.57 (P 0.10). LA's Kober-type alternation is a third of Greek's. Would support: P < 0.05 on a larger or held-out site sample. Would kill: P >= 0.2 with more shuffles.

## Clay is attention, attention is status? (la53, 5 Oct 2026; `loops/la53_final.txt`)
- Idea: rank words by the layout a clerk spent on them (own line, lines, first place, extra words, length) beyond what quantity, commodity and document size predict; 2,000-6,000 random specs; nulls permute attention within commodity x size strata and shuffle quantities.
- Grade A (method, calibrated negative): planted hierarchies are found (AUC 0.65-0.81), but real status is not, in either control: Linear B gods / wanax / officials AUC 0.56 (p 0.24), Ur III gods / kings / temples 0.46 (p 0.78). Recipes chosen on one administration fail on the other (p 0.64, 0.31). Status words come late, not first, in both. Layout surplus cannot rank Linear A referents by status.
- Grade B: Linear A's high-surplus words (A-PA-RA-NE, PA-DE, MA-KA-RI-TE, A-DU, *516, KU-NI-SU, SA-RA₂) are heading or transaction words on tablets, mostly HT-only, with larger amounts than usual; none is on a libation table. 1-2 types at p < 0.01 vs 1.8 expected.
- Grade C: PA-DE is a specially marked entry (own lines, extra words, small amounts; p 0.01-0.02, n 3). Would kill: PA-DE as an ordinary crowded entry on a new tablet.

## la52 iterated learning (5 Oct 2026)
About 7,000 learner chains and 30,000 rewritten corpora were run. **A (method):** how fast a feature dies under copying depends on how far apart its parts sit, not on whether it carries meaning. Totals, entry order and co-occurrence die after about one copy, while neighbouring pairs survive. The Linear B meaning check therefore failed (commodity-unit pairs persist). **B (negative):** Linear A word rankings do not replicate across tablet halves (rho 0.03), while Linear B and Ur III do. C: OLE+RI dies faster than its frequency predicts (z −3.4, full corpus only). Would support: z below −2 in both halves with more tablets. Would kill: z above −1 with a new seed.

## la54 reading what was not written (5 Oct 2026)
About 20,000 random classifiers tried to infer the unwritten commodity on 166 documents that have numbers but no commodity sign. The controls passed (Linear B 0.53–0.60, Ur III 0.64–0.68). **B (negative):** on Linear A the inference mostly returns each site's commonest commodity (72–75%: Khania CYP, Zakros/Arkhalokhori GRA). Numbers do not carry the commodity, wine/oil/people/olives cannot be recovered, and no word-commodity link survives (12 of 259 at p<0.05, 13 expected). C: about 0.16 bits per document beyond site (P 1/9); would kill: P>0.2 with more shuffles. C: HT95b is a grain document; would kill: a non-grain sign there.

## The scribes' mistakes show how they thought? (la56, 5 Oct 2026; `loops/la56_final.txt`)
About 2,000 random inclusion rules were scored per corpus run, over about 100 real, null and planted runs: entries of a given word, sign, commodity or role left out, subtracted or doubled; fraction signs swapped; sub-totals or the other side added. Every 1–3-change reading of each total that does not close was also tried, and so was every contiguous run of quantities on the tablet. Fraction values were treated as a nuisance. **B (calibrated negative):** no exclusion habit, fraction interchange or misplaced section cut beats the nulls (totals reassigned, entries shuffled, noise on balanced totals). Planted excluded words rank first in 21/22 runs, and Ur III gives back a real excluded commodity (kin-gur-ra lines with their own szunigin). The Linear A mismatches are explained as often as noise on balanced totals (0.48 vs 0.50 at ≤ 2 changes; Ur III 0.60 vs 0.43). **C:** HT 127b's second KU-RO 292 is a running total: the earlier KU-RO 156 plus the next five lines, including the erased [[KI+MU]] 14 (278 without it). This would be a double-count corrected on the face but not in the sum. Would kill: a collation that changes 292, 14 or the erasure. **C (post hoc, n = 4):** KU-RO adds a logogram written after a name (HT 9b TA-I AROM, HT 89 VIR+*313a) but totals commodity columns separately (HT 123+124a OLIV vs *308, PH 31a per animal).

## Words grow like cities? (la55, 5 Oct 2026; `loops/la55_final.txt`)
The idea: archives might grow like cities, with function words saturating, names growing in proportion and specialised goods appearing only above a size threshold. Scaling exponents were fitted for every word and sign across 35 Linear A deposits, 103 Linear B find areas and 277 Ur III site-year archives, and compared with documents re-dealt within genre and within genre x length (300-1,000 times each).
- **A (method, calibrated negative):** the expected classes are absent in the controls. Function words do not saturate relative to names in Linear B or Ur III (p >= 0.18). Names-linear holds only in full Linear B under one null (p 0.037), not at Linear A size or in Ur III. Scaling-based predictions of a held-out archive's contents, and estimates of an archive's size from a fragment, fail in Linear B. Planted saturating and linear classes are recovered; superlinear words cannot be told from threshold words with Linear A's two large archives.
- **B (negative):** Linear A scaling exponents predict nothing held-out. Linear A archives show no lexical closure, while Linear B find areas re-use their own names and goods.
- **B (descriptive):** GRA, OLIV, HIDE, KI, TI, *188, *131B, *86-RO and TE-RI saturate (small archives hold them at higher rates). CYP, MI, QA2, *411-VS, *164, *306, O and DI-NA-U are 'threshold'; the nodule signs among them occur only at Khania and HT R13, so this is locality, not size. KU-RO, VIN and SA-RA2 change class with the null. Sign saturation was document length (killed).
- **C:** la45 commodities lean to saturation and headers to threshold (p 0.018, 0.10 under the length null). Would kill: commodities spread proportionally in new small deposits.
- **C:** larger towns (Whitelaw's hectares; 6 sites) left more documents and logogram types, sublinearly (exponent about 0.4-0.5; logogram types rho 0.81, p 0.042). Richness is explained by document count. Would kill: a large town with a small, poor archive.

## Dissect a machine that learned Linear A (la49, 5 Oct 2026; `loops/la49_final.txt`)
- Idea: decipher a model, not the script. 115 tiny masked-token transformers and GRUs (random sizes and seeds) were trained on whole documents and dissected (attention, head ablation, number interventions, word transplants, ICA feature matching, geometry, direction ablation). Controls: shuffled Linear A, Linear B and Ur III at Linear A size and 4x, planted totals and bindings.
- **A (method, calibrated negative):** at 5,000-20,000 tokens no model of any corpus learns to add, binds a number to its word through a recurring head, or gives words a seed-independent 'function'. The planted totals, to-so and szunigin are missed. The lack of such circuits in Linear A models says nothing about Linear A.
- **B:** word geometry beyond simple profiles and co-occurrence recurs across seeds (0.22 vs shuffled 0.07-0.09) and across two independent populations (r 0.70 vs 0.35). It recovers Linear B commodity, person and place groups and Ur III commodity, place, office and verb groups.
- **B:** one Linear A group is a true circuit: the single signs of sealings (*301, KU, KA, SI, RO, TE, ZE, I, TA, A, O, JA, DA, *164, *411-VS). Removing its direction costs 1.31 nats on its own held-out tokens and none elsewhere, in 12/12 models. This independently confirms la45's heading class.
- **C:** a commodity group (NI, *304, *306, *308, E, SU with CYP, OLE, VIN) and a libation-formula group (A-TA-I-*301-WA-JA, JA-SA-SA-RA-ME, SI-RU-TE, I-PI-NA-MA with PA, NA, TU, NE, QE, DI) are stable but not causally specific. Would kill: the groups dissolving when duplicate tablets and stone vessels are held out.

## The crack attempt: one reading, frozen, decoding held-out documents (la60, 6 Oct 2026; `loops/la60_final.txt`)
- Idea: stop testing pieces. Every B result (and the best C) was assembled into one machine-checkable reading of the administrative documents (`tools/la60_common.py`, `prior_reading`): KU-RO/PO-TO-KU-RO total (running total allowed), KI-RO residue, commodity class (logograms, NI KI DI RE TI MA, *304 *308 *306 E SU *307), heading signs (*301 KA KU SI RO ZE TE A I ...), heading/transaction words (SA-RA₂, PA-DE, A-DU, KA-PA, KU-PA, ...), OLE+U/MI/DI as entries, the commodity order, site defaults and the libation register. A train-only INDUCED version comes from fixed corpus-agnostic rules. Both drive a generative document grammar (role trigram, identity within role, numbers in context, KU-RO arithmetic, order), then a frozen decoder that glosses held-out documents and checks coverage, role steps, totals and order. Controls: roles shuffled among the reading's types, random readings, the empty reading. Calibration: Linear B KN+PY at Linear A size with the same rules and with a true partial LB role reading (to-so, o-pe-ro, transaction words).
- Grade A (method): the bare kind grammar (logogram / single sign / word / number) beats a trigram on held-out documents by 1,486 bits per half (LB 367).
- Grade B (calibrated): the reading is not arbitrary (shuffled roles cost 159 bits per held-out half, 10/10 splits), but it adds nothing beyond the kind grammar (+98 bits, worse in 9/10) except the KU-RO arithmetic (36 bits per half). A true LB partial reading gains 211 bits over the empty grammar.
- Grade B: frozen on training documents, the reading explains 22-26 % of held-out administrative documents in full (empty reading 21 %). Over shuffled readings it gains 5 'strict' documents per 191 (2.7 %; closing totals 5.2 vs 0.3; order agreement 0.70 vs 0.50). A true LB partial role reading gains the same 2.9 % at the same size. The role layer is about as right as a true role layer, and a role layer is far from a reading.
- Not supported (outside, n = 11): reading frozen on GORILA documents gives no gain on the 11 administrative documents published after 1985; KH103 writes OLE > NI > CYP > VIN, half against the order.
- Grade C: entry words return with the same commodity on other tablets (held-out 0.47 vs site default 0.16 vs shuffled 0.10; 0.41 without the HT86/95 group; LB 0.71), carried by KU-PA₃-NU, MA-DI (DI), SA-RU (GRA), SA-RO (VIN), DA-RE (VIR), TA-I (AROM). Would support: >= 3 of these with their commodity on new tablets. Would kill: < 1 in 3 right on 10+ new uses.
- Frozen predictions (sha256 538ad6fa64360eb8178f65a54ea78aa15d9dcee3532eb5181a704ec3063c50fb; `data/la60_ckpt/c3_frozen_predictions.json`, git-ignored; key items in `loops/la60_cycle3.txt`): per-site KU-RO rate and closure (HT 0.18, closes ~0.46; KH none, < 0.038), commodity shares, order pairs (e.g. CYP before NI 0.90, GRA before OLE 0.87, NI before VIN 0.89), first commodity after SA-RA₂ GRA/CYP (14 of 19).
- **Verdict: not cracked.** The ledger skeleton is parsed (B); no word has a meaning, 69 % of administrative word tokens (every entry name) are unread, fractions are unvalued, and the only outside test gives no gain. Biggest gaps: referents for entry words; single signs between lines (RA, PA, PA₃, TU, ME, *318) that fit neither heading nor count; 15 non-closing totals.
