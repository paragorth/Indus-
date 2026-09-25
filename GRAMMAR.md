# The Indus text frame: a data-only grammar (25 Sept 2026)

This grammar is built only from sign sequences, find-spots and object types (IM77 and Wells/Yajnadevam transcriptions). No readings or published interpretations are used. Each claim cites the strategy in `STRATEGIES.md` that tests it against a null.

## The template

```
[opener 267-99 | 391-99 | 293-123-343]  [M65|M86 ...] MIDDLE  [closer X-342 | X-162 | X-169 | X-15 | X-254 | X-12]  [suffix M176 | M1]
   optional, seals (S18, S29)   middle-initial, optional (S24, S38)  unique      titles/offices, shared across cities (S17, S18)      optional, tablets (S24, S29)
```

Reading order is right-to-left on the object, first-read-first here.

| Slot | Evidence | Behaviour |
|---|---|---|
| Middle-initial M65, M86 | S24, S38 | Deletable (17% and 13% of uses vs 4% baseline); they follow opener+marker (M65 after M99 41:1), so they open the middle rather than the text |
| Opener 267-99 (also 391-99, 293-123-343) | S18, S29, S32, S33 | Starts 143 long seal texts. M99 and M123 are never initial and follow 81 and 29 different hosts respectively (post-positioned markers). Used on seals owned; almost absent from impressions and tablets |
| Middle | S17, S19, S26, S32 | Unique to one object and one city. Texts of 6 or more signs never recur across cities. No productive name heads at the start; mild concentration at the end. Combines with any closer at chance rate |
| Closer X + jar (or 162/169/15/254/12) | S17, S18, S25, S30, S40, S62, S63 | Short 1–2 sign texts that appear on their own on seals in several cities, more often than chance; not city-specific, but clustered by quarter and house *within* cities (Mohenjo-daro P = 0.025, Harappa P = 0.011). A closed class. Replicated in the Wells transcription |
| Suffix M176, M1 after the jar | S24, S29 | Deletable; 5× more frequent on sealings, moulded tablets and rods than on seals, within each city. Belongs to the tablet/token sub-system |

**Order (S36–S38).** Pairwise order between slot classes holds 85–100%: opener → marker 99/123 → 65/86 → {162,169} / {8,347} → closer {342,211,15,12} → suffix {176,1}. Unsupervised complementary-distribution analysis recovers the same classes (S37).

## Other structure

- **Numbers.** Some signs are genuinely counted (trees 3–8, smooth). Others take one fixed number as part of a name: "7" with W585/W575, then the jar ("Seven-X" closers); the fish with 3, 4 and 6 (S23, Q4).
- **Animal image vs text.** They are independent: no sign, and hardly any aspect of form, depends on the animal (S15, S28).
- **Line breaks.** They fall at weak junctions, so lines are units (S27).
- **Repetition.** Indus avoids repeating a sign within a text, like formulaic legend systems (Ur III seal legends, proto-cuneiform) and unlike running text (S9, S16). This is transcription-sensitive: adjacent doubles must be handled the same way in every corpus.
- **Abroad and on foreign-format seals.** The person sign moves from text-final (home) to text-initial (abroad), and foreign texts lack the home opener (S20, S42; ANCHORS.md). The same holds for round (Gulf-type) and cylinder seals *found at home* (S46): the grammar follows the seal format, i.e. a community of users, not the find-spot.
- **Time.** The grammar is stable through Mohenjo-daro's sequence and holds at post-1977 sites out of sample (S44). It erodes at Dholavira's late stage 6 (S45).

## What this means

The Indus seal text looks like an **owner legend**: an individual element (middle) set in a small inventory of shared titles/markers, like an Ur III seal legend (name, title, "son of"). There is one difference: no "son of" element is found (S12). The markers are mostly suffixed (S33).

The frame gives **no sound values**. It does tell a future decipherer where names should be (the middles) and which signs are grammatical (99, 123, 342, 176, and the closer set). Any proposed reading must treat those signs as markers or titles, not as syllables of words.

## What would test it

These tests are frozen predictions R1–R7 (`prereg/preregistration-3-frozen.json`). In addition, any newly found seal should:

- have its 6+-sign middle unattested elsewhere;
- place M99/M123 only in second position;
- be an owned seal that takes the opener more often than the tablet suffix.
