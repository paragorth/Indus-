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

The frame gives **no sound values**. It does tell a future decipherer where names should be (the middles) and which signs are grammatical (99, 123, 342, 176, and the closer set). Any proposed reading must treat those signs as markers or titles, not as syllables of words.

## What would test it

These tests are frozen predictions R1–R7 (`prereg/preregistration-3-frozen.json`). In addition, any newly found seal should:

- have its 6+-sign middle unattested elsewhere;
- place M99/M123 only in second position;
- be an owned seal that takes the opener more often than the tablet suffix.
