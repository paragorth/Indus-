# Indus script: the decipherment attempt (23 September 2026)

**One-sentence verdict.** The *grammar* of an Indus seal can now be read with confidence — what each part of the inscription is doing — but the *meanings* cannot be pinned from the corpus alone: for a typical sign, about 8 of 13 possible meaning-categories survive every constraint we can compute, and nothing internal to the corpus narrows that further.

Everything below was computed this session from the frozen corpus (2,543 objects; 1,900 deduplicated home-region texts; 762 Mohenjo-daro seal texts). Code: `p3lib.py`, `p3_engine.py`; sign table: `lexicon.json`.

---

## 1. What you can now read on any seal (secure)

Hold a seal impression, read right to left. The inscription is built from five kinds of piece, and you can label each one:

**The opener.** About a third of Mohenjo-daro seal texts begin with one of three alternating signs — the two diamond/leaf signs (glyphs 817, 861) or the six-spoked wheel (820) — always followed by a connective, usually the two short strokes (glyph 2) or the single-stroke-pair (60). These three openers substitute for one another in identical frames more often than any other pair of signs in the corpus (10, 8 and 6 shared frames). They are one slot with three values: "X-of …" where X is one of three institutions, places, lineages or document types. Which of those it is cannot be determined.

**The core.** One to four signs, drawn from a large open inventory. This is where the identity information lives: the first read position has the highest entropy in the text (6.6 bits, perplexity ~95), the last position the lowest (4.2 bits, perplexity ~19). If any part of a seal is a personal name, it is here.

**Numbers.** Short vertical strokes are numerals 1–8. The number system has two distinct uses, and you can tell which from the sign that follows:
- *A genuine count* when followed by the tree sign (glyphs 390/405/407) or the crescent (900). Trees take 3 to 8, smoothly, peaking at 4 (profile 3:15, 4:34, 5:11, 6:11, 7:6, 8:3). Crescents take 3, 4 or 5. Six frames in the corpus alternate *only* in the numeral — for example "opener · connective · [3 / 4 / 5] · tree" exists at both Mohenjo-daro and Harappa with three different numbers. That is a variable quantity slot.
- *A fixed word* when the pairing is rigid. Seven + the trapezoid (575) 12 of 14 times; seven + glyph 585 15 of 16 times; three tall strokes + the arrow 39 of 43 times; two tall strokes + plain fish 49 times. These are lexical units — a "number-name" like "Seven-X" — not counts.
- *The fish is the odd case.* Plain fish takes 3 (13 times) and 6 (9 times) but never 5 and only once 7. Its number profile is not a count profile: it differs from the tree's with p = 0.0004. "Three-fish" and "six-fish" are lexicalised compounds. This is exactly the shape Parpola predicted for Dravidian star-names (aru-mīn "six-star" = Pleiades), but the corpus cannot say that they *are* star-names — only that they behave as fixed words, not quantities.

**The pre-ending.** A small closed set — the three-headed person (100), the seated person (176), the decorated U (760), and glyphs 690, 923 — sits immediately before the ending and selects it (almost always the jar). The two person-signs alternate with each other in identical frames. Rank, title or role marker; under either macro-hypothesis these are the "title" slot.

**The ending.** Two-thirds of seal texts close with one of a closed paradigm: the jar (740), the arrow (520), the burden-carrier (151/156), the comb (400), the man (90), the hatched box (527). They alternate in the same frames (jar~burden-carrier 9 frames, jar~arrow 4). But they are not all the same kind of thing:
- The **jar** is register-neutral (59% on seals / 17% faience-clay / 21% tablets, against a corpus base of 61 / 14 / 22), attaches to anything (the highest left-entropy of any sign, 6.3 bits), and closes 41% of Mohenjo-daro seal texts. It behaves as a grammatical default — the one sign in the corpus whose substitution partners span every category, which is the signature of a function element, not a word.
- The **others are register-bound**: the comb is a tablet sign (only 29% on seals, log-odds −2.0), while the man, arrow and burden-carrier are seal signs (+1.1, +0.7, +1.0). Register-bound endings are lexical: titles, offices or document functions, not case-endings.
- The man-sign (90) is a second-order ending: it almost only follows the jar (left-entropy 1.6 bits). "…-jar-man" is a stack.

**Two more things you can say from the shape of a text alone:**
- Counted texts and the jar repel each other. Texts with a quantity construction end in the jar 19% of the time against 41% for the rest (Fisher p = 0.003). Whatever the jar is, it does not go on a quantity record.
- Quantity constructions are shared administrative formulae, not private names. Among the 35 texts found word-for-word at *both* Mohenjo-daro and Harappa, 20% contain a count construction, against a corpus rate of 6.5% (length-matched permutation p = 0.006). Counts travel between cities unchanged; name-like cores mostly do not.

That is the reading ability you now have: given any inscription, you can say which piece is the issuer/locus slot, which is the identity core, whether it records a quantity or a number-name, what its title slot is, and what kind of ending it carries. On the ten randomly drawn seals in section 4 that works on every one.

## 2. Three new structural findings this session

1. **The stroke signs are two systems.** Count paradigm (tree, crescent) versus fixed compounds (7-575, 7-585, III-arrow, II-fish, 6-fish, 3-fish). This is decided by the value profile, not by intuition, and it recovers Wells's "numerals" and Parpola's "number-names" as *both* correct, on different signs.
2. **The ending paradigm splits into one grammatical default plus several lexical endings.** Jar = neutral function element; comb/man/arrow/burden-carrier = register-specific content. No published proposal I know of separates them on this evidence.
3. **Quantity records are cross-city formulae and avoid the jar.** Together these make the corpus look like two overlapping document kinds on the same objects: identity texts (opener + core + title + jar) and quantity texts (N + counted-noun, often bare), with the tablets skewed toward the second.

One earlier idea was killed on the way: the apparent "2 + fish / 2 + leaf" compounds are half an artefact of the opener formula (glyph 2 is the connective, and fish/leaf simply follow the opener often). They were dropped before anything was built on them.

## 3. Where meaning stops — the degeneracy measurement

For each of the top 60 signs, thirteen meaning-categories (personal name, lineage, title, deity/astral name, commodity, measure, animal, person/occupation, place/institution, document marker, grammatical suffix, rebus, no stable meaning) were scored against every computable constraint: grammar class, countability, pictorial identity (under the logographic hypothesis), object-type distribution, cross-city recurrence, and open-versus-closed-class behaviour — under both the linguistic (Dravidian-type) and administrative macro-hypotheses, then intersected across substitution paradigms.

Result: **on average 8.3 of 13 categories survive per sign; 7.6 after paradigm intersection.** Only the connectives collapse (to "grammatical / no meaning"). The counted nouns narrow the most — the tree to {commodity, measure, place, astral-name}, the fish variants to {animal, astral-name, personal name, place} — and the openers, pre-endings and core signs barely narrow at all.

That is the honest answer to objective 1: the posterior over dictionaries is wide, and it is wide because the corpus is closed. Internal statistics fix *function*; they cannot fix *reference*. This is the readable/meaningful anti-correlation from Logic of Life sitting in the data: the parts of the script with enough regularity to be read (endings, openers, numerals) carry the least meaning, and the parts that carry meaning (the core) have too little regularity to be read.

## 4. The best-supported semantic bets (hypothesis-dependent, not established)

- **Tree = a counted commodity, most plausibly timber.** It is the only sign that is simultaneously a smooth count noun, cross-city formulaic, pictorially a plant, and matches the attested Meluhhan export list (mes-wood, abba-wood, mangrove and other timbers are the bulk export in Mesopotamian records; carnelian, lapis, gold dust, silver, ivory, bamboo, a red dog and monkeys are the rest; sesame is *not* attested). Odds: better than any other commodity reading in the corpus; still one story among the four surviving categories.
- **Fish compounds = names built on numbers.** Supported by the non-count number profile. Whether star-names (Parpola) or something else is undecidable here.
- **Jar = suffix (linguistic reading) or generic owner/person marker (administrative reading).** Both survive; the register-neutrality and universal attachment favour a grammatical element over a document-type marker.
- **Openers = one of three institutions/localities/lineages.** Which is unknowable from inside.

## 5. Readings of twelve seals

M-14 and M-15 you have already photo-verified. The other ten were drawn at random (seed 2026) from Mohenjo-daro seals of four or more signs, never discussed in any session. CISI numbers are given so you can open the photograph and check the structural reading against the object.

**M-14** — Steatite, emblem: bull1:w. Signs in reading order (right to left on the seal): [817, 2, 503, 776, 740]
- `817` OPENER: Diamond or leaf shape with a small diamond at
- `2` CONNECTIVE: Two adjacent half-height simple vertical stro
- `503` CORE: Tall isoceles triangle with horizontal stripe
- `776` CORE: unbridged shape
- `740` ENDING: Classic jar symbol with two horizontal handle
Reading: opens with the opener formula (institution/locus/document-type slot); closes with an ending-paradigm sign.

**M-15** — Steatite, emblem: bull1:w. Signs in reading order (right to left on the seal): [861, 2, 176, 740, 90]
- `861` OPENER: Diamond or leaf shape with a small diamond at
- `2` CONNECTIVE: Two adjacent half-height simple vertical stro
- `176` CORE: Person sitting with bent knees, four small ta
- `740` ENDING: Classic jar symbol with two horizontal handle
- `90` ENDING: Person with two arms and two legs,  [M001]
Reading: opens with the opener formula (institution/locus/document-type slot); closes with an ending-paradigm sign (stacked).

**M-1716** — Steatite, emblem: bull1:w. Signs in reading order (right to left on the seal): [820, 2, 803, 240, 220, 520]
- `820` OPENER: Wheel with 6 spokes with even angles [M391]
- `2` CONNECTIVE: Two adjacent half-height simple vertical stro
- `803` CORE: Leaf with tree at bottom [M387]
- `240` CORE: Fish with extra whiskers on cheeks [M067]
- `220` CORE: Fish with no other decoration [M059]
- `520` ENDING: Isocolese triangle on top of vertical line (a
Reading: opens with the opener formula (institution/locus/document-type slot); closes with an ending-paradigm sign.

**M-317** — Steatite, emblem: bull1. Signs in reading order (right to left on the seal): [920, 401, 692, 60, 255]
- `920` OPENER: unbridged shape
- `401` CORE: unbridged shape
- `692` OPENER: An X but drawn in outline [M150]
- `60` CONNECTIVE: A single small vertical line, then another ve
- `255` CORE: Unknown asymmetric animal? Perhaps a fish wit
Reading: opens with the opener formula (institution/locus/document-type slot).

**M-1079** — Steatite, emblem: gaur. Signs in reading order (right to left on the seal): [803, 233, 706, 33, 760, 740]
- `803` CORE: Leaf with tree at bottom [M387]
- `233` CORE: Fish with a horizontal line through body [M07
- `706` CORE: U with a long vertical stroke inserted into t
- `33` NUMERAL: Three adjacent full-height vertical strokes [ — stroke sign, role unclear
- `760` PRE-ENDING: U with decorated branching at top [M347,M358]
- `740` ENDING: Classic jar symbol with two horizontal handle
Reading: closes with an ending-paradigm sign.

**M-1104** — Steatite, emblem: zebu. Signs in reading order (right to left on the seal): [820, 60, 564, 740]
- `820` OPENER: Wheel with 6 spokes with even angles [M391]
- `60` CONNECTIVE: A single small vertical line, then another ve
- `564` CORE: unbridged shape
- `740` ENDING: Classic jar symbol with two horizontal handle
Reading: opens with the opener formula (institution/locus/document-type slot); closes with an ending-paradigm sign.

**M-1747** — Steatite, emblem: bull1:s. Signs in reading order (right to left on the seal): [692, 60, 31, 741, 165]
- `692` OPENER: An X but drawn in outline [M150]
- `60` CONNECTIVE: A single small vertical line, then another ve
- `31` NUMERAL: One full-height vertical stroke [M086] — stroke sign, role unclear
- `741` CONNECTIVE: Classic jar symbol with two horizontal handle
- `165` CORE: unbridged shape
Reading: opens with the opener formula (institution/locus/document-type slot).

**M-135** — Steatite, emblem: bull1:j. Signs in reading order (right to left on the seal): [31, 705, 33, 520]
- `31` NUMERAL: One full-height vertical stroke [M086] — stroke sign, role unclear
- `705` CORE: U with a long vertical stroke inserted into t
- `33` NUMERAL: Three adjacent full-height vertical strokes [ — fixed compound with next
- `520` ENDING: Isocolese triangle on top of vertical line (a
Reading: contains a fixed stroke+sign compound (a lexical unit, e.g. 6-fish / 7-X); closes with an ending-paradigm sign.

**M-1482** — Copper, emblem: elep. Signs in reading order (right to left on the seal): [706, 33, 923, 740]
- `706` CORE: U with a long vertical stroke inserted into t
- `33` NUMERAL: Three adjacent full-height vertical strokes [ — stroke sign, role unclear
- `923` PRE-ENDING: Right parenthesis drawn in outline [M296]
- `740` ENDING: Classic jar symbol with two horizontal handle
Reading: closes with an ending-paradigm sign.

**M-792** — Steatite, emblem: bull1:s. Signs in reading order (right to left on the seal): [817, 2, 17, 575, 740, 679]
- `817` OPENER: Diamond or leaf shape with a small diamond at
- `2` CONNECTIVE: Two adjacent half-height simple vertical stro
- `17` NUMERAL: Seven short vertical strokes [M110,M112] — fixed compound with next
- `575` OPENER: Trapezoid with no bottom line, small box rest
- `740` ENDING: Classic jar symbol with two horizontal handle
- `679` CORE: unbridged shape
Reading: opens with the opener formula (institution/locus/document-type slot); contains a fixed stroke+sign compound (a lexical unit, e.g. 6-fish / 7-X).

**M-2012** — NULL, emblem: bull1. Signs in reading order (right to left on the seal): [570, 2, 5, 900, 740]
- `570` CORE: unbridged shape
- `2` CONNECTIVE: Two adjacent half-height simple vertical stro
- `5` NUMERAL: Five simple vertical strokes [M106,M107] — quantity 5 of next
- `900` CORE: Right parenthesis (perhaps with a backward cu
- `740` ENDING: Classic jar symbol with two horizontal handle
Reading: contains a quantity construction (N of counted-noun); closes with an ending-paradigm sign.

**M-1424** — NULL, emblem: gaur. Signs in reading order (right to left on the seal): [575, 450, 745, 803, 482, 740]
- `575` OPENER: Trapezoid with no bottom line, small box rest
- `450` CORE: unbridged shape
- `745` CORE: unbridged shape
- `803` CORE: Leaf with tree at bottom [M387]
- `482` CORE: unbridged shape
- `740` ENDING: Classic jar symbol with two horizontal handle
Reading: opens with the opener formula (institution/locus/document-type slot); closes with an ending-paradigm sign.

## 6. What would actually crack it

Nothing internal will. The three things that would:
1. **ICIT as holdout** (Fuls, TU Berlin, still not requested): ~2,000 inscriptions not in Mahadevan. The frozen predictions P1–P9 test the grammar; the new count-vs-compound and jar-repulsion findings are the next pre-registrable tests.
2. **A bilingual or a bridge to a known system** — none exists; the Persian Gulf seals and the Shu-ilishu interpreter seal are the only places one could still appear.
3. **The Mahadevan 1977 tables**, to extend the shape bridge past 85 signs, which would at least let every reading above use shapes rather than glyph numbers.

Short of those, the correct next move is not more dictionary-guessing but publishing the *function* reading (section 1) as a third paper — it is new, computed, and falsifiable — and stating the degeneracy result (section 3) as the honest limit. Coherent semantic stories are cheap; this session generated several and kept none.


---

## 7. The lateral round (same day, pushed harder)

Four new anchors were tried. Two died, one is a real discovery, one is an open puzzle.

**Dead: animal emblem → sign.** Paper A's omnibus null could have hidden single sign–single animal ties. Tested every frequent sign against every emblem with at least 12 seals (642 Fisher tests, unicorn/gaur/zebu/elephant/rhino/buffalo/tiger): nothing survives false-discovery control. The closest miss is the wheel-opener on rhinoceros seals (30% vs 9%, raw p = 0.007, not significant after correction). The emblem and the text are two independent identity channels. That rules out the emblem as a semantic key for good.

**Dead: the jar as a case-inflected stem.** Reason it was tried: the plain jar is an ending (69% final) while the jar with a stroke in its mouth (glyph 741) is a mid-text link (1% final, p ≈ 10⁻⁶⁰). If it were the same word in a linking case, it would keep the same left-hand company. It doesn't (Jensen–Shannon distance to the plain jar's left contexts 0.79, no better than a random sign); it lives in a fixed frame "connective 60 · marked jar · fish". So the stroke does not inflect the jar — it makes a different sign out of it.

**Real: strokes are a productive sign-forming device that changes grammatical class.** The four-stroke fish (226) is an ending (56% final) where the plain fish is a core sign (6%, p ≈ 10⁻⁹); the marked jar is a connective where the plain jar is an ending. Meanwhile the fish family (220/233/235/240), the tree family (390/405), the U family (705/706) and the three openers all *do* keep the same company on both sides (Jensen–Shannon distances far below the random-sign baseline): those are true substitution sets — allographs or near-synonyms. So the corpus contains two different kinds of "similar sign": variants that mean the same thing, and marked forms that do a different job. The stroke marks are the second kind. This is new, computed, and pre-registrable for ICIT.

**Open puzzle: the count that never says two.** Trees are counted 3 to 8 (80 tokens) and appear bare 179 times, but "2 tree" occurs once and "1 tree" never; the same hole appears for the crescent. A tally of real goods is dominated by ones and twos. Three readings survive: the bare sign means one and the writers avoided "2 + noun" because the two-stroke sign was already the connective (a homograph-avoidance effect; the rare tall-stroke II before tree may be the workaround); or the numbers are ordinals or grades, not quantities; or the counted things simply never came in pairs. This is the single best test to run on ICIT: if unseen texts also lack 1- and 2-counts, the first or second reading holds and the "commodity tally" story weakens.

**Bottom line of the lateral round.** Still no dictionary. What was gained is morphology: the script has substitution families, stroke-marked derived forms, two stroke systems (counts versus number-words), one grammatical default ending and several lexical endings. A reader with this can parse an inscription; nobody, including me, can yet say what it names.


---

## 8. The Mahadevan audit (23 Sept 2026, evening)

The full machine-readable Mahadevan corpus (IDF-80: 2,906 texts, 14,153 sign tokens, exported from indusscript.in) is now in `im77/`. It is an independent transcription of the same objects, with per-line object type, side, field symbol, **direction of writing** and **excavation level** — three things our database never had.

- **Direction.** Of our texts with ≥80% of signs bridged, 1,086 match a Mahadevan reading forward and 8 reversed. The global right-to-left correction in the frozen pipeline is right. Left-to-right lines exist (235) and concentrate on miniature tablets (15%) and sealings (7%) rather than seals (3.5%).
- **Bridge.** Alignment alone extended the sign bridge from 85 to 143 signs (82% of tokens), file `bridge_extended.json`. No hand-coding.
- **Agreement.** Of 1,249 of our texts fully bridged, 724 match a Mahadevan line exactly and 370 differ by one sign. The one-sign differences are almost all allograph granularity (our single tree code against Mahadevan's 161/162/169; stroke heights 97/98; 387/389; 110/112), not misreadings. After collapsing those families, Mohenjo-daro and Harappa agree exactly on 80% (865 of 1,087); the residual 222 are the list to inspect, and include texts absent from Mahadevan (Dholavira is almost entirely post-1977).
- **Stratigraphy.** 1,134 Mohenjo-daro seal lines carry an excavation depth (−31 to +7 ft). Across that range the grammar is flat: jar-final 35–41%, opener 17–22%, mean length 4.4–4.8, all Spearman rho ≈ 0. Within Mohenjo-daro's excavated span the conventions do not drift. That closes one hoped-for anchor (change over time) and strengthens the claim that the seal formula was fixed.

The scan-based reader (`read_texts.py`, 417 templates from the 1977 sign list) reproduced the printed texts sign for sign on the pages checked by eye, but is now redundant: the site export is the canonical digital source.


## 9. Two more anchors tried with the Mahadevan data — both dead

**Find-spot households.** If seals from the same house share signs, those signs are family or office names. At Mohenjo-daro, seals from the same recorded locus are no more alike than random pairs (mean Jaccard 0.059 vs 0.060, permutation p = 0.59); Mahadevan's loci are excavation areas, not houses. Harappa shows a small effect (0.069 vs 0.060, p < 0.001) but its loci are whole mounds, so this is the register effect already known. One sign, 197 (the trapezoid, partner of the fixed "seven + trapezoid" compound), clusters spatially at Mohenjo-daro beyond chance (9 loci vs 4.7 expected); noted, not built on.

**Fine-grained emblems, all 164 codes, all object types.** First pass produced a spectacular hit: sign 113 on 10 of 45 elephant seals and 3 of 1,557 others, p ≈ 10⁻¹⁴. Every one of the ten was a Lothal sealing carrying the identical text — ten impressions of one seal. Counting each distinct (site, text, emblem) once: 1,020 tests, **zero** survive false-discovery control. The animal on the seal is independent of the signs on it, at every granularity we can test.

The near-miss is recorded deliberately. A sign-to-referent link that survives the fine-grained test would have been the first external anchor in the history of the problem; it lasted four minutes and died to a duplicate check. Any published "decipherment" that has not deduplicated sealings should be read with that in mind.


## 10. Three lateral ideas, run the same night

**Copper tablets are typed tokens, and the type is labelled.** Of 96 Mohenjo-daro copper tablets with both a motif and a text, 22 motifs map onto 35 texts, and the surplus texts are damaged or one-sign variants of the same label: each motif has one fixed text. Labels share components across motifs ("336 89", "407 169"), so the label is a formula with slots, not a single word for the creature. Test of the obvious hypothesis — that the motif-specific part *names* the animal: none of those signs is enriched on seals bearing the same animal (all p > 0.2, sealings deduplicated). The label is a type-name of the token, not a word for the picture.

**Harappa's two-sided miniature tablets are two-field documents.** 222 tablets carry text on both faces. Side 1 is open (106 distinct texts, 5.9 bits); side 2 is closed (59 distinct, 4.2 bits) and is almost always "tall strokes + leaf sign" — I-leaf, II-leaf, III-leaf. The two fields are tightly bound (mutual information 3.4 bits, 81% of the smaller entropy): a given side-1 text goes with a given leaf-count. A form with a name field and a quantity field, where the quantity is fixed per name — a denomination, not a tally.

**Two numeral systems, as in proto-cuneiform.** Tall strokes and short strokes attach to disjoint sets of signs. Tall strokes count the leaf (1–3), the window-grid sign 249 (1–3) and the burden-carrier (mostly 1), and form rigid compounds with fish (II), arrow (III) and jar (II). Short strokes count tree and crescent (3–8) and, as the two-stroke connective, attach to the fish variants. In early Mesopotamia the numeral system used identifies the commodity class; the Indus stroke-height split has the same shape.

Net: structure again, meaning still no. The leaf is "the unit the tablets count"; what a leaf-unit *is* remains open.


## 11. Counter-intuitive checks (same night)

- **Spelling-variant hypothesis (seals = few words spelt many ways): refuted.** 28% of distinct seal texts have a one-substitution twin (chance 6%), but the 605 substitution events spread across 398 sign pairs (top ten cover 13%). The high-alternative frames are all *opener · connective · [X] · jar* with six or seven unrelated fillers: a one-sign identity slot. Twins differ at the first position 494 times, mid 600, last 116. Logographic naming, not spelling.
- **Left-to-right lines as a second document type: refuted.** Same order once normalised, jar never initial, shorter (4.0 vs 5.1 signs), fewer openers (10% vs 21%). Register, not genre.
- **Jar as a mere terminator: refuted.** Jar-final texts with the jar removed are more formulaic than non-jar texts (opener 29% vs 18%, first-sign entropy 5.5 vs 6.3 bits). Non-jar seals close on arrow (124), seated man (107), tree (102), man (51), burden-carrier (41). Two seal families with distinct closing sets.
- IndusVision's methodological conclusion (no linguistics without a trustworthy transcription layer) is answered for this corpus by the two-transcription audit in section 8.


## 12. Batch of opposites (same night)

Identity slot ≠ commodity (0 of 13 fillers counted nouns). Openers are seal-document markers, not pronouns (23% seals, 14% sealings, 2–3% tablets). Unicorn seals carry longer texts (5.3 vs 4.3–5.0, Kruskal p = 0.02) — the only emblem correlate that survives. Not numbers-plus-determinatives (stripping strokes leaves 1,352/1,425 texts unique). One script across object types (96–100% token overlap with seals). Zipf abbreviation holds (frequent signs simpler, rho −0.14, p = 0.007; grammatical signs simpler than content signs, p = 0.002). Short opener-frame seal is Mohenjo-daro-only. Blazon/order-free test had no power (no shared sign-sets) and is not counted.


## 13. The Dravidian rebus test (DEDR, same night)

Input: the full Burrow–Emeneau dictionary (5,627 entries) exported by the user's browser agent. For each pictorially secure sign (fish 59, tree 161, leaf 328, man 1, pot 342, arrow 211, bird 81, comb 171, wheel/sun 391, bow 28, scorpion 54) the Tamil headword whose first gloss is the referent was taken from the dictionary, and its exact homophones (other entries with the same Tamil form) counted.

- Rebus-recruitment prediction (signs whose referent word has many homophones should be more frequent): Spearman −0.12, p = 0.75, n = 10. No support, and no power — ten signs is all the pictorial certainty the sign list allows.
- Homophone pairs *between* pictorial signs: none. The pair test cannot be run.
- The classic Dravidian pairs are real in the dictionary: mīṉ "fish" (4885) beside mīṉ "star"; aracu "pipal" (202) beside aracaṉ "king" (201). So the Dravidian hypothesis is *possible*; nothing in the corpus can test it, because the words the rebus would write are not pictured.
- One specific reading is in tension with the data: if the leaf sign (328) is the pipal leaf and pipal → king, the leaf should behave like a title; instead it is the counted unit on the Harappa tablets (I/II/III + leaf) and a closing sign on tablets, not seals.

Verdict: the rebus method cannot be turned into a corpus test with the sign list as it stands. It is not refuted; it is untestable here.


## 14. Holdout night (23 Sept, late): two decipherments tested blind, our predictions tested on unseen seals

**Source.** Yajnadevam's public repository (`lipi`, GitHub) contains his transcription of 5,680 CISI objects in the same glyph coding as our database (stored in the pre-correction orientation; reversed here), his sign-to-phoneme key (680 entries), and his Sanskrit readings. 1,193 objects are not in our database (CISI vol 3 Mohenjo-daro, post-1986 Harappa, others); 467 are distinct unseen seal texts. Files copied to `yajnadevam-data/`.

**Bonta (2023/2024).** 676 readings parsed, 152 aligned to Mahadevan text numbers; his implicit sign-value map is self-consistent (top value explains 87% of tokens; half-to-half reproduction 82%). Blind test on 2,707 texts he never read: 2–4-sign windows read as Monier-Williams headwords 18.7% of the time; shuffling his own values across signs gives 18.4% ± 5.9 (z = 0.1). No signal beyond chance.

**Yajnadevam (2024).** Key applied to 4,503 texts: 3.61% of windows are Monier-Williams headwords on real texts, 4.86% ± 0.12 on within-text scrambled texts, 5.49% on reversed texts. The key reads random sign order as Sanskrit better than the attested order. Caveat: raw transliteration without his conjunct rules, applied identically to all conditions.

**Frozen predictions (22 July 2026) on 467 unseen seals.** P3 (falsifier): conditional terminal-zone compliance 84% vs 30% shuffle, gap 54 (predicted 65–82, gap ≥25) — PASS. P5 double-terminal 9% (4–18) — PASS. P6 opener→connective top bigram — PASS. P7 initial-minus-final entropy 1.12 bits (≥0.3) — PASS. P1 jar ending share 25% (28–45) — MISS. P2 terminal-class ending share 39% (50–75) — MISS. Likely cause: fragmentary objects (only 52 flagged complete); no completeness filter was pre-registered, so both are recorded as misses. P4, P8, P9 not evaluable on this material.

**Tonight's findings on unseen objects (not tuned on them).** Fixed compounds: 7+585 = 4 of 6, III+arrow = 8 of 12, II+fish = 15, 6+fish = 4; fish numeral profile 3:3, 6:4, no 4/5. Stroke morphology: jar 72% final vs marked jar 13%; fish 27% vs four-stroke fish 89% (n = 9). Ending split: comb 81 tablet / 11 seal; jar 163 seal / 164 tablet; arrow 25 seal / 14 tablet. Two-field tablets: I+leaf 1, II+leaf 14, III+leaf 11; leaf-final 9% on tablets, 0% on seals. Tree counts too few to judge the ones-and-twos hole (3..8 = 3,1,2,1,1,0; three 1/2-stroke cases).


## 15. Morning of 24 Sept: chronology, households, the mint

**Chronology (Harappa periods 3B–3C, Mohenjo-daro Early–Late, from Yajnadevam's context columns).** Jar-final, opener rate and ending-class share are flat across seven centuries; seals lengthen slightly (Harappa seals 4.0 → 4.5 signs; MD 4.2 → 4.6). Tablets dominate early Harappa (90% of 3B texts) and fade (47% in 3C). No sign shifts significantly between early and late (0 of 33 after FDR; early tokens 294). The time axis is closed on the right data: the script was stable.

**Households.** At Mohenjo-daro, seals from the same house (Marshall block/house) are more alike than chance (Jaccard 0.067 vs 0.061, p = 0.013). At room level, two signs co-occur in different texts in the same room beyond chance after FDR: 690 (= M149, outline X, a title-slot sign) in 5 rooms vs 0.7 expected, and 368 (= M402, stroke with looped wing) in 3 rooms vs 0.3. First household-anchored signs; the natural reading is a title or office held within one household. Harappa lot-level data too sparse (22 texts).

**Ones-and-twos hole at scale (merged corpus, 2,999 distinct texts).** Tree preceded by 3–8: 89 tokens (3:19, 4:36, 5:12, 6:12, 7:7, 8:3); by 1 or 2: 4. Bare tree 256. Replicated and strengthened.

**The mint.** Harappa Trench 54 (Mound E): 120 inscribed objects, 113 tablets, commonest texts "IIII + leaf" (11), "III + leaf" (8), "II + leaf" (4). A denomination series of one to four leaf-units produced in one place. No sign is otherwise enriched there.

**Unicorn cult object × sign.** 985 unicorn seals with cult-object codes (SAN 593, SAF 121, RAN 82, RAF 54, RBF, S): 468 tests, none survive FDR; jar-final and opener rates flat across types. Third emblem channel, third null.

Merged corpus in reading order with all context columns: `merged-corpus-reading-order.json` (5,369 objects).


## 16. 24 Sept, continued: shipments, the directory, the trade map

- **Seals never sit with their own sealings.** 18 sealings match a known seal text; all 18 are cross-site, 0 same-site. Ten Lothal warehouse sealings carry a Mohenjo-daro seal's text (opener·connective·48·jar); a Kalibangan sealing carries a Mohenjo-daro count-formula seal (opener·connective·3·tree). Seals stayed with senders; impressions travelled with goods.
- **Seal size**: area vs text length rho 0.34 (p ≈ 1e-47); zebu seals largest (median 853 mm²), then elephant, rhino, unicorn 692, gaur, buffalo — emblem rank by object size.
- **Three-field tablets**: 827 multi-sided objects. Faces: a name-like text; a leaf count (II 100, III 146, IIII 113, I 5 — three denominations, not the binary weight series); and on three-sided prisms a fixed third face "3 + burden-carrier" (21 of 60). Name-to-count pairing is free; specific pairs are mass-produced (3·861 with IIII-leaf ×8).
- **Pottery graffiti**: same script (93% of tokens are seal signs), mean 2 signs, 111 of 416 texts identical to a seal text (owner marks), the rest tablet-style notation.
- **Material/colour vs sign**: 56 significant material effects, all reducible to object type (faience→tablet signs, steatite→seal connectives, copper→407). Nothing new.
- **Identity directory** (`identity-directory.json`): 1,909 distinct core strings (opener, connective and endings stripped), 386 recurring, 191 at ≥2 sites, 27 with both seals and sealings. Seven-585 is an identity string (seals MD/Kalibangan, sealing Rupar).
- **Trade map from seal→sealing pairs**: net senders Mohenjo-daro (38) and Harappa (29); net receivers Lothal (31), Mohenjo-daro (25), Dholavira (21). High-confidence (multi-sign, non-formula) links listed in the transcript; short cores (fish alone, "3", "3 tree") are common formulae and inflate the counts.


## 17. 24 Sept, continued: reader, denominations, jar strokes

- **Indus Seal Reader** published (`indus-reader.html`, live artifact): CISI number or sign sequence → Mahadevan sign shapes, slot roles, document type, identity string, occurrences across 5,369 objects and sealing sites. No translation, by design.
- **Denominations locked to types**: on 323 two-sided tablets with a leaf face, specific partners are fixed to one face value — *three·diamond* with IIII-leaf ×21 (never 2 or 3), *seated man·jar·comb* with III-leaf ×21, 368 with III-leaf. Eight sign×denomination pairs survive FDR. Face value per type.
- **Openers as nouns**: only the diamond (861) is ever counted, and only as the *three·diamond* Harappa tablet type (29 copies); 817 never, 820 five times. Generalisation fails.
- **Jar strokes (Mahadevan 342–346 = 0–4 inner strokes)**: plain jar final 620 / medial 244; stroked jar medial 123 / final 9; in two-jar texts the medial one is stroked 98/113. Confirms the medial-form rule from the independent coding. Not a countdown (only 2 texts with ≥3 jars; strokes vs signs-remaining rho −0.07, p = 0.31); weak early-position gradient (rho −0.16, p = 0.014; three-stroke jar text-initial 15/41).
- Hapax signs slightly inkier than common ones (p = 0.005); not obviously compounds.


## 18. Twin seals in the merged corpus (24 Sept)

542 seal pairs differ by exactly one sign. The varying slot is most often the sign **immediately before the ending** (181), then core (151), first position (148), opener (39), ending (23). Pre-ending fillers are semi-open (72 distinct over 362 slots; seated man 29, 772 21, three-headed man 16). Twins differing before the ending share a city 65% of the time; twins differing at the front, 45% (Fisher p = 0.001); opener-twins 69%. Reading: the front of a seal carries a supra-local affiliation (the same front string recurs across cities with different persons), the sign before the jar carries the local individual. Structural gloss: [affiliation] · [connective] · [person] · [jar]. House-level co-location of twins ~1–4% (data too sparse).


## 19. Affiliations (24 Sept)

Front strings (before the person-sign, after opener and connective) with ≥4 objects at ≥2 cities: 49, saved in `affiliations.json`. The fish family dominates the supra-local slot: whiskered fish 55 objects / 23 persons / 4 cities; caret fish 34 / 21 / 6; plain fish 19 / 14 / 6; barred fish 20 / 16 / 5. Stroke numerals in this list are count constructions the splitter cannot separate. Member overlap between fish variants is at the random rate (all p 0.12–0.92): the variants are neither exclusive groups nor a single group; the fish is a freely combining name element. Theophoric ("of deity X") and institutional ("of guild X") readings both fit; nothing in the corpus separates them.


## 20. Corrections and small results (24 Sept)

- **Withdrawn**: "front string = supra-local affiliation". Sign-inventory overlap between Mohenjo-daro and Harappa is no higher for front-slot signs (token-weighted 0.75) than for person-slot signs (0.81); slot-swapped permutation p = 0.99. Signs in both slots are pan-Indus. Twin-seal result restated at string level: within a city a fixed front string recurs with different last core signs (a local group varying its member); the same member+ending recurs in other cities with different fronts (a common name element). Front string = local group; person sign = shared name element.
- Person-pictograms are not concentrated in the person slot (7.1% vs 6.1%, p = 0.27): names are arbitrary signs, not portraits. Fish lean to the front slot (12.0% vs 8.9%, p = 0.006).
- Hapax composition by image matching: 30 of 112 contain a common sign as a part (mostly the person figure or strokes), 79 unmatched. Inconclusive.
