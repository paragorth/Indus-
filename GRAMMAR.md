# The Indus text frame: a data-only grammar (25 Sept 2026)

This grammar is built only from sign sequences, find-spots and object types (IM77 and Wells/Yajnadevam transcriptions). No readings or published interpretations are used. Each claim cites the strategy in `STRATEGIES.md` that tests it against a null.

## Relation to earlier work in this repository (read this first)

Much of this frame was **already described** in `docs/INDUS-DECIPHERMENT-ATTEMPT.md` (23 Sept 2026, Wells corpus):
- the three alternating openers + connective;
- the open "core";
- the two stroke uses (counts on trees and crescents vs fixed number-names such as 7-575, 7-585, III-arrow, II-fish);
- the pre-ending title slot (100, 176, 760, 690, 923);
- the ending paradigm, with the jar as default, the comb as a tablet sign and "…-jar-man" as a stack;
- cross-city quantity formulae.

This push re-derived those pieces without looking at that document. They were found again from the IM77 transcription and are *independent confirmations* (S18, S23, S24, S36–S38, S70–S72, S77).

What is **new** here:
- the controls: unsupervised paradigm finder vs planted syllabic text, Ur III legends and Proto-Elamite (S37, S87, S89, S90); non-local constraints beyond a bigram model (S50, S51, S67); out-of-sample sites (S44); repetition calibration (S16, S88);
- opener vs suffix as seal vs impression/tablet sub-systems within each city (S29, S66, S86);
- closers clustering by house and quarter (S62, S63, S63b);
- slot inventory sizes (S85);
- the round-seal tradition (S46, S74, S98b, S99–S104);
- voucher behaviour of two-sided tablets (S93, S94; the README's "two-field reading of the tablets" may overlap);
- line breaks (S27), doubling (S57, S58), copper tablets (S56), time trends (S45, S49), and the typological filter (S43).

## The template

```
[opener 267-99 | 391-99 | 293-123-343]  [M65|M86 ...] MIDDLE  [closer X-342 | X-162 | X-169 | X-15 | X-254 | X-12]  [suffix M176 | M1]
   optional, seals (S18, S29)   middle-initial, optional (S24, S38)  unique      titles/offices, shared across cities (S17, S18)      optional, tablets (S24, S29)
```

Reading order is right-to-left on the object, first-read-first here. Coverage: 59% of seal texts show at least an opener or a closer (S108).

| Slot | Evidence | Behaviour |
|---|---|---|
| Middle-initial M65, M86 | S24, S38 | Deletable (17% and 13% of uses vs 4% baseline); they follow opener+marker (M65 after M99 41:1), so they open the middle rather than the text |
| Opener 267-99 (also 391-99, 293-123-343) | S18, S29, S32, S33 | Starts 143 long seal texts. M99 and M123 are never initial and follow 81 and 29 different hosts respectively (post-positioned markers). Used on seals owned; almost absent from impressions and tablets |
| Middle | S17, S19, S26, S32 | Unique to one object and one city. Texts of 6 or more signs never recur across cities. No productive name heads at the start; mild concentration at the end. Combines with any closer at chance rate |
| Closer X + jar (or 162/169/15/254/12) | S17, S18, S25, S30, S40, S62, S63 | Short 1–2 sign texts that appear on their own on seals in several cities, more often than chance; not city-specific, but clustered by quarter and house *within* cities (Mohenjo-daro P = 0.025 and specific to the closer; Harappa P = 0.011 but middle vocabulary also clusters there, S63b). A closed class. Replicated in the Wells transcription |
| Suffix M176, M1 after the jar | S24, S29 | Deletable; 5× more frequent on sealings, moulded tablets and rods than on seals, within each city. Belongs to the tablet/token sub-system |

**Order (S36–S38).** Pairwise order between slot classes holds 85–100%: opener → marker 99/123 → 65/86 → {162,169} / {8,347} → closer {342,211,15,12} → suffix {176,1}. Unsupervised complementary-distribution analysis recovers the same classes (S37).

## Other structure

- **Two numeral systems (S70).** Short strokes (3–8) count trees, the crescent and the "Seven-X" signs. Tall strokes (1–4) count the arrow, the jar, W590, W240 and W384. The item classes are almost disjoint. But most tall-stroke uses have a fixed value per item (arrow = 3, jar = 2, fish = 2; S71), so they are lexicalised number-compounds rather than counts. Genuine counting shows mainly with short strokes on trees. Counted items sit mostly in the closing part of the text (S68).
- **Numbers.** Some signs are genuinely counted (trees 3–8, smooth). Others take one fixed number as part of a name: "7" with W585/W575, then the jar ("Seven-X" closers); the fish with 3, 4 and 6 (S23, Q4).
- **Tablets as vouchers (S93, S94).** Two-sided Harappa tablets put a count (N + leaf) on one face and a text on the other, almost never two counts (4 vs 34 expected). A given text tends to keep the same count, even on hand-incised tablets (P = 0.04).
- **Animal image vs text.** They are independent: no sign, and hardly any aspect of form, depends on the animal (S15, S28).
- **Line breaks.** They fall at weak junctions, so lines are units (S27).
- **Repetition.** Indus avoids repeating a sign within a text, like formulaic legend systems (Ur III seal legends, proto-cuneiform) and unlike running text (S9, S16). This is transcription-sensitive: adjacent doubles must be handled the same way in every corpus.
- **Abroad and on foreign-format seals.** The person sign moves from text-final (home) to text-initial (abroad), and foreign texts lack the home opener (S20, S42; ANCHORS.md). Round (Gulf-type) and cylinder seals *found at home* also lack the home frame (S46, robust at known sites, S46b). Their person-sign enrichment is shown only for round seals found abroad or of unknown provenance (S46b, S74). Square seals and pots abroad keep the home grammar (S74). That tradition favours the short-horned bull ("gaur") emblem (50% vs 5% of square seals, S65), but square gaur seals follow the home grammar (S65b). The format, not the animal, goes with the grammar.
- **Time.** The grammar is stable through Mohenjo-daro's sequence and holds at post-1977 sites out of sample (S44). It erodes at Dholavira's late stage 6 (S45).

## Controls that make this a grammar rather than an artefact

- **Unsupervised paradigm finder** (S37, S87, S89, S90):
  - finds slot classes in Indus (both transcriptions);
  - finds none in a planted syllabic corpus with the same text shapes, and none in Ur III seal legends;
  - finds a few in Proto-Elamite (one hub sign).
- **Non-local constraints** (S50, S51, S67): marker exclusivity and suffix/opener exclusion hold beyond what a bigram model of the same corpus reproduces, in both transcriptions.
- **Out of sample** (S44): the order predictions hold at 248 texts from sites excavated after 1977.
- **Repetition** (S88): language-like text, whether syllabic or logographic, repeats signs at 0.7–0.8 of chance; Indus at 0.1–0.3, like Ur III seal legends. Long texts are stacked formulae (S91b).
- **Inventory sizes** (S85): about 200 closers (Chao1) vs an effectively unbounded set of middles.

## What this means

The Indus seal text looks like an **owner legend**: an individual element (middle) set in a small inventory of shared titles/markers, like an Ur III seal legend (name, title, "son of"). There is one difference: no "son of" element is found (S12). The markers are mostly suffixed (S33).

The frame gives **no sound values**, and statistical key-fitting cannot supply them: in controls it "passes" without recovering the planted readings (STRATEGIES S107, S113–S126). It does tell a future decipherer where names should be (the middles) and which signs are grammatical (99, 123, 342, 176, and the closer set). Any proposed reading must treat those signs as markers or titles, not as syllables of words.

## What would test it

These tests are frozen predictions R1–R7 (`prereg/preregistration-3-frozen.json`). In addition, any newly found seal should:

- have its 6+-sign middle unattested elsewhere;
- place M99/M123 only in second position;
- be an owned seal that takes the opener more often than the tablet suffix.

## Added 27 Sept 2026 (replicated in both transcriptions)

- **Bar-seal genre (S157, S157b, S175, S184).** Long script-only bar seals (Harappa Period 3C, Dholavira Stage VI) keep the opener but drop the jar closer and the person sign: Harappa 3C 31% vs 47% jar-final among square seals; Dholavira 10% vs 38%; IM77 script-only seals 37% vs 44% (P = 0.007). The Dholavira excavator independently dates these seals to the late phase, made in kaolinite after the steatite supply stopped. The sign inventory does not shrink (S181); the change is in the formula.
- **Shared number + item terms (S173, S177, S184b).** Stroke-numeral + item pairs found on tablets recur on seals more than other tablet pairs (Wells 59% vs 39%; IM77 78% vs 57%). On seals they fill the title slot before the jar. The number + leaf counts stay on tablets. So the tablets' unit vocabulary doubles as seal titles, while the tablets' counts are a separate system.
- **Law of abbreviation (S164, S164c, S184c).** Frequent signs are graphically simpler: ρ ≈ −0.3 among well-attested signs, in the range of cuneiform measured the same way (−0.07 to −0.24). Consistent with a script shaped by use; not by itself proof of language.
- **Negative results.** Seal size (CISI size groups) does not predict the opener or the choice of closer once text length is controlled (S179, S179b). The apparent "name-final elements" were the known closer and marker slots (S182c).

## Added 27 Sept 2026 (later): two uses of stroke numerals (S197–S199; replicated in IM77, S201)
- **Fixed terms.** Some signs always take the same numeral before them: 3 + comb (W422–426), 17 + W585 and 17 + W575 (M197), 32 + W877 (M284), 33 + W923 (M296), 33 + W520 (M211). These pairs sit in the title slot, either just before the jar or text-final (33 + W520 is final in 44 of 49 cases). Read each pair as one term. The comb's tooth count is a sign variant, not a number.
- **Counts** (glyphs: docs/counted-vs-fixed.png). A small set of signs take varying numerals: jar W740 (M342), W220 (M59), W700 (M328), W390 (M161–169), W900 (M287), W405. The numeral choice is the same on seals and tablets and in both big cities (S199, P = 0.92). Numeral + W700 occurs only off seals.
- Caution: W2 (M99) is also the opener's marker, and W31 (M86) is a middle-initial marker; '2 + W235' is the frame sequence M99 → M65, not a numeral term. About half of the "2 + X" pairs sit right after an opener sign, so do not read them as counts.
- **Two numeral series** (S204, both transcriptions; glyphs docs/numerals.png). Short strokes (1–10, one row or two tiers) and tall strokes (2–9). Each sign takes mainly one series: tall with W520/M211, W700/M328, W590/M249, W226/M60, W877/M284, W632/M244; short with W390/W405 (M162/169), W900/M287, W585, W575/M197, W407, comb, W156/M15, W142/M25, W140/M17. Jar and fish take both. Caveat (S208): the form also depends on the value (1 and 7–9 short, 2 tall); the sign preference is shown cleanly at value 3, drawn tall before W520/M211, W700/M328, W590/M249, the jar, and short before W390/405, W900, the fish, the comb, W840/M403.
- **Short-3 terms stand at the front, tall-3 terms inside** (S210, both transcriptions). S211: this is mostly because different signs follow each form; with the sign held fixed only a small extra front-placement remains (P = 0.01).
- **Restatement (S216–S217):** tall strokes = the Harappa tablet voucher count ('tall 2–4 + W700') plus a few frozen seal terms (33+W520, 32+W877, 32+W632, 32+W226, tall-2 + jar/fish). On seals, every other count uses short strokes. Read S204's 'two series' in this narrower sense.
- The 'tall 2–4 + W700' voucher is Harappa-specific: 417 of 1,765 Harappa tablets, 0 of 139 Mohenjo-daro tablets (S218).
- **Frozen pre-jar titles** (S229, both transcriptions): 590-390/405 (M249-M162/169), 435-690 (M130-M149), 840-‖ (M403-M87), 17-585 (M112-M194) stand before the jar as fixed pairs; W100 (M8) and W760 (M347) combine freely.
- **The single stroke W1 (M97/98) is a marker, not 'one'** (S234, both transcriptions): it precedes other numerals and rarely precedes counted signs.
- **One motif-linked label** (S223, S253, both transcriptions): the stand-alone sign W930/M393, usually on its own line, goes with zebu seals (3 of ~50 vs 1–2 of ~1,700). The only text–animal link found.


**Opener caveat (S256, S260):** W817/861/820 (M267/391) opens 465 texts but closes 25, mostly square seals with two lines. Both transcriptions agree, and most are not space wrap-arounds. After a jar-closed unit a second unit can end with the same sign, so 'opener' names its usual position, not a fixed one.

**Medium variants (S262–S264):** one sign carved differently on seals vs moulded or bas-relief tablets: W390~W405, W154~W158, W320~W318, W527~W525/526. Merge them before counting. After the jar, tablets end in W400 while seals use other endings (W679, W565, W621). The script-only seals carry a second closing unit, A-n + W806 + W154.

**12 as a fixed term (S273):** W55 (12) is about 50× commoner than a smooth number distribution predicts, and there is no 11. It stands before the jar (title slot) or before the pan-Indus title 255-435-690, not before counted goods. Treat 12 (and 24) as named numbers, like the fixed numeral terms.

**Opener as closer (S286):** in the 25 texts where W817/861/820 stands last, the jar closer is usually absent (16% vs 46%, p = 0.003). The opener sign can take the closer slot.


### Closer alternative: W595 (M252) (S288)
W595 comes before a final opener 3 of 25 times (P = 0.0001), is never before the jar, and avoids texts with the jar (0.24× expected). It stands next to W820 in either order (820-595 ×5, 595-820 ×4). Treat W595 as a closer-slot sign that alternates with the jar ending; this replicates in IM77 (M252 before M391, 4 of 10).

### Closer paradigm (S289)
The jar ending (X-342) is one of about ten mutually exclusive closing units. Each is a final sign with a fixed left partner: 806-154/158, 550/555-527, 220/33/240-520, 3-156, 32-226, 142-617, plus W151, W236 and W700. Each is ≥ 40% final and occurs with the jar ≤ 0.5× expected; that is 10 signs against a shuffled-null maximum of 5.
Replicated in IM77 with a prior prediction (S291): M15, M254, M12, M211 and M60 are the only signs that pass the same cuts (P = 1.4 × 10⁻⁷).

### Fish qualifier order (S296–S297)
Modified fish signs form one word class in both corpora. When two stand together they keep a fixed, transitive order: W235 (hat) → W240 (whiskers) → W233 (bar) → W231 (stroke); e.g. 36:5 and 22:3. The plain fish W220 has no fixed place.

### The arrow phrase (S299–S300)
Template, in both corpora: fish word(s) in the order hat → whiskers → bar → stroke → plain, then optionally W705/706 + tall 3 (M336-M89), then the arrow closer W520 (M211). 'U-stroke 3' occurs only after a fish word (64% vs 33% base) and never between a fish and a directly following arrow (1 of 86).

### The seated-person phrase (S301–S302)
W176/M48 (seated person) (+ W100/M8, three-headed person) + jar W740/M342. M8 comes after M48 and never before it (Wells 13:0, IM77 10:0). Texts with M8-jar carry M48 earlier far above chance in both corpora.

### Whole-corpus parse and name structure (S310–S311)
The grammar labels 45.5% of held-out sign tokens (shuffled 25%). The residue is a 0–3-sign NAME in 71% of texts. Names have their own positions, replicated in IM77: name-initial W692/M150, W575/M197, W125/M28, W416/M173 (+ W413, W920, W495); name-final W840/M403, W460/M230, W435/M130, W440/M127, W717/M341 (+ W70, W35, W690).

### Opener class extended (S331)
Besides W817/W861/W820 + W2, two more opener units start texts: W920 + W60 (text-initial 61 of 62) and W692 + W60 (33 of 39). The connective after the opener is W2 for the leaf/diamond/wheel and W60 for these two (and sometimes for the wheel: 820-60, 26 of 26 initial).

### Two different connectives (S333)
W60 (M123) is followed by the marked-jar family W741/742/745 (M343–346) in about a quarter of cases; opener + W2 (M99) almost never is (0 of 324 Wells; 1 of 275 IM77). W2 introduces fish, number and leaf-tree words. CORRECTION (S334): all W60 → marked-jar cases follow W920; elsewhere W60 behaves like W2 (cosine 0.90). The fixed unit is 920-60-741, an opening phrase of its own.

## Added 3 Oct 2026: the minimum-lot rule (S-DARK-15.3; all three merge levels; held-out sites agree)
- **Counts before a good start at 3.** For every counted item (tree 390/405/407, arrow 520, bracket 900, and the fixed terms 845, 923, 550) the numeral before it is 3 or more. The apparent 'no 1 tree, no 2 tree' hole is not about trees: the **short-stroke 1, short-stroke 2 and tall 1 glyphs are grammar markers** (W1, W2 = connective; tall 1 = a fixed-term element), not counts, so they never appear before a good.
- **1 and 2 as true counts are written only with TALL strokes and only inside fixed terms**: '2 tree' exists 13 times at home as tall 2 (32-390 ×4, 32-407 in the seven-copy tablet 231-17-585-95-520-32-407). Short-stroke 2 before a tree occurs only as the W2 marker after an opener (M-1851; K-78, K-431, Ns-67).
- **Persons (S98 refined):** figure signs take fixed numerals; only the plain person W90/W91 is never counted (2 of 196 at home, same at held-out sites).
- **Frozen prediction for new finds** (prereg/preregistration-5-minimumlot.json): no new text will show a short-stroke 1 or 2 directly before a counted good; counts before goods will be 3–8. Would kill: three or more new texts with short 1 or 2 before a tree, arrow or bracket.
- **DEMOTED the same day (S-DARK-21.1).** On the 210 IM77 texts that Wells never transcribed, 3 of 19 counted goods take a short single stroke directly before them (a pot, a seal 391-99-98-296-342, and a 'miscellaneous' object), plus one tall-1 + arrow; counts 3–8 hold for 14 of 19; depletion of 1–2 is only marginal (3 vs 6.5 expected, P = 0.056). That meets the frozen kill rate (P5.2: > 5% of new counted goods with a short 1/2; observed 16%), on 19 goods rather than the 200 assumed. **Status: a strong tendency on seals at Mohenjo-daro and Harappa, not a writing rule.** Retest when Rakhigarhi/Binjor material is read. The marker reading of short 1/2 and tall 1 stands where it was tested (the home corpus); the claim that it is exceptionless does not.

## Added 3 Oct 2026: a text is a sequence, ordered below the slot level (S-DARK-19.1–19.3; all three merge levels; held-out sites agree)
Among texts of three or more signs, the same multiset recurs in another order in 24 of 2,811 copy pairs (0.9%), against 46% if order were free inside slots and 92% for an unordered set; seals 1 of about 200, held-out sites 0 of about 150. Half of all testable sign pairs have a fixed order; 74% of those follow the frame and 26% order signs inside one slot (84–98 such pairs, frame-only null at most 27–31); none contradicts the frame. The fixed pairs form one consistent partial order with a longest chain of 16–19 signs: 320 → 920 → 60 → 741 → 2 → 803 → 32 → 350 → 798 → 415 → 220 → 233 → 705 → 255 → 435 → 690 → 740 → 400. So the middle is not a bag of signs: marked jar 741 and leaf-tree 803 come first, then the tall numerals, then 350-798-415, then the fish words, then 705-33 or the 255-435-690 title, then the closer. The free pairs (8%) are the numeral signs 1/31/32/2 among themselves, plain fish 220 against modified fish 240/235, and the two uses of W2; they co-occur at chance (O/E 1.2) and are independent signs, not alternatives. The one loose construction is the Harappa two-sign voucher 'count + W700', reversed on 15% of tablets (22% of moulded, 8% of incised); a photo check of tablets reading 700-33 would settle whether these are mirrored moulds. Calibration: planted free order 0% fixed, planted strict 100%, Ur III seal legends 73%, Ur III names 95%; Indus sits with the whole Ur III legend, not with its name part.

### Generative adequacy of the frame (S366)
A slot grammar fitted on Mohenjo-daro + Harappa (opener, marker, order-1 middle chain, closer, qualifiers, suffix) and run as a generator fails 36 of 50 held-back statistics; a plain order-2 chain with no slots fails only 25. The frame becomes adequate (23 of 50, at the test's calibration floor of 19) only after three additions, all predicted by the credential reading: whole texts are copied within a city (not middles recombined with closers: 20% of repeated middles change closer, a lexicon model gives 80%), the closer depends on the middle's last sign, and the object type sets the frame. What no local mechanism reproduces: long-range attraction and avoidance between signs across slots (S51, S67), the near-ban on repeating a sign (S88), and the tight length distribution. A text is composed as a whole, like a checklist, not written sign by sign.

## Added 3 Oct 2026: the frame is as old as the dated record (S-DARK-43.1–43.4; all three merge levels)
Across nine chronology series (HARP phases, Vats strata, seven Mohenjo-daro levels, depth bands, Dholavira, Kalibangan, Lothal) the earliest well-sampled seals already show the frame at its all-period rates: opener-first 0.26 against 0.24 overall, closer-last 0.68 against 0.67, W2 in slot two 0.69, repeat rate 0.04, order agreement with the all-period partial order 0.97. Inventory, Chao1 total, closer menu and repeat ban are flat across phases (0–2 of 90 trend arrows significant). A simulation in which the rules switch on progressively shows that a system 70–80% formed at the earliest phase would have been detected in 26–40 of 40 runs, so growth of more than about 10% is excluded; growth below that is undetectable. The pre-Mature Kot Diji pottery marks at Harappa (39 tokens) consist of numerals, W700 and the jar, with W700 and tall-1 over-represented (P = 0.002 and 0.001): the counting and unit layer is older than the seal frame, and openers are absent on pots in every period, so their absence early is a medium effect. Caveat: the dated record rests on Harappa and Mohenjo-daro, and the pre-Mature evidence on Harappa alone.

## Added 3 Oct 2026: how much of the script this grammar explains (S-DARK-59.1–59.3; all three merge levels within 0.1 bit)
Scored as the probability of the next sign given the text so far, the object class and the site, fitted on Mohenjo-daro + Harappa and tested on 424 held-out-site texts and the 324 IM77-only texts: a unigram model costs 6.2–6.5 bits per sign, the best structural model (frame, rules, whole-text caches, Kneser–Ney order 2–3) 4.9–5.2, so the rules explain 17–24% of the unigram entropy (43–47% against a flat inventory). By slot: markers 89% predictable, suffixes 57%, text end 38–46%, openers 34%, closers 32%, qualifiers 26%, counts 7–12%, the middle 8% (19% with site-local text caches). The hand grammar performs exactly like a second-order Markov chain (5.44 vs 5.37 bits), and the residual contains no new rule beyond the frozen pairs already listed here. Of about 27 bits in an average seal, 18.7 (69%) are irreducible middle content at 7.1–7.5 bits per sign, the frame carries 4.3, the qualifiers 1.5 and the counts 2.7. Through identical code, Ur III seal legends give titles 26–48% predictable and names −18% to 34%, with names 65–71% of the bits; Linear B words 50%. The Indus split matches a name-plus-title legend. This is the accounting of what the grammar covers: the frame is read, the designation is not.

S-DARK-57 (3 Oct 2026) tested this directly with 985 pre-Mature pottery marks transcribed from Mehrgarh IV–VII (Quivron 1980), Sarai Khola, Harappa Ravi/Kot Diji (Kenoyer 2006) and Kot Diji (Khan 1965): the founder set of 26 signs (538 tokens) is stroke numerals (47%) plus simple middle shapes (X, the V-cup W700, arcs, bowtie, chevron, tree); frame signs are 0 of 25 (< 0.12) against 0.18–0.22 under frequency- and complexity-matched nulls (P ≤ 0.001), and the jar, W520, W60, W861, W817 and W920 never occur pre-seal. No founder sign was recruited into the frame. Held-out Mature pots still draw 0.56 of their tokens from the founder set against 0.35 for seals. So the seal frame was introduced with seals; only the counting layer is inherited. Transcriptions are single-pass from drawings and shape identity is not sign identity.

## Added 3 Oct 2026: one system for all cities (S-DARK-48.1–48.3; all three merge levels)
With one text per distinct text per site × type and equal-size draws, the divergence between Mohenjo-daro and Harappa seals is ×1.07 of the divergence between two random halves of one city (IM77 ×1.05; seq_strong/seq_all ×1.03; no component significant), the same as a synthetic single generator sampled twice (×1.04) and as Latin inscriptions from Lazio and Rome (×1.04). Known multi-city systems run through identical code are far more local: Ur III legends by city ×1.5–2.1, Linear B by palace ×1.7–2.6, proto-cuneiform ×1.9; a half-local synthetic gives ×1.9 and a fully local one ×3.6. By component, the designation slot (×1.01), numerals (×0.99) and the inventory (×1.04) are identical across cities; the opener and closer menus differ by ×1.4–1.5 but carry under 0.01 bits; whole seal texts are shared more across cities than within (×0.75). Locality appears only when tablets are pooled (reuse ×5.6, P = 0.002). The held-out sites fall inside the Mohenjo-daro–Harappa band on 95–100% of tests. One menu and one inventory served the seal texts everywhere; the tablets are local. **Correction (S-DARK-68, 4 Oct 2026):** this uniformity does not require central issuance. Calibrated on real name pools about 600 km apart at the same noise level, US given names give ×1.03–1.05 and US surnames ×1.05, the same as Indus (×1.04–1.10), while a central code with a geographic field (radio call signs with a district digit) gives ×1.9 and Latin cognomina by province ×1.24–1.32. So the Indus figure is what one naming population looks like when sampled in two cities. The one feature that points beyond that: in every name pool whole names repeat more within a place than between places (×1.05–3.4; 0 of 225 pairs below 0.88), whereas Indus seal texts repeat more across the two cities (×0.77–0.88), a weak hint (2–3% of texts, not significant) of a shared stock of copied texts.

S-DARK-39 (3 Oct 2026) fitted the complementary non-chain model: a form of 25 menus (none obligatory, no forbidden pairs) filled in a strict linear order, with site-local reuse. It brings every residual of the chain model inside the band (long-range attraction and avoidance, repeat rate, two-numeral share, start entropy) but loses the chain's local statistics (conditional bigram entropy, bigram and trigram hapax shares, same-middle-different-closer); its controls (independent fields, random field order, no reuse) are all worse. About 1,400 field subsets cover 90% of texts; the 324 IM77-only texts are valid form instances at 0.58–0.60 against 0.26 for shuffles, but so are Markov-2 clones (0.75–0.77). So the texts need both descriptions, a menu-and-order form and local sign-to-sign statistics, and the fitted form is this frame, not a code beyond it.

## Added 4 Oct 2026: line breaks fall between the grammar's words (S-DARK-70.1–70.3; all three merge levels)
When a text wraps onto a second line, the scribe almost never splits a unit learned from single-line texts (qualifier + head, the opener phrase, numeral + item, the fish words, the frozen pairs): 1 of 77 Wells breaks falls inside a unit against 8–13 expected under random, ink-midpoint and balanced-cut nulls, and 3 of 87 in IM77 against 15–19 (P < 0.001 each); Ur III word-internal breaks give 0.04 of chance and planted unit-avoiding breaks 0.12. Both editors place the break identically on all 19 objects they both record as two-line. About 60% of breaks fall right after a closer or suffix (3x chance), so line 1 is usually a complete legend. Inside the designation, a break between two middle signs is as likely as at a frame edge, so no multi-sign words are detected there; strong cohesion would have been found 95% of the time, but up to about 45% weakly bound pairs cannot be excluded. Only 4 texts come from sites outside IM77, so this is transcription-robust, not replicated.
