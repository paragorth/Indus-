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
