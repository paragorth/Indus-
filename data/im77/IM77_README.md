# IM77 corpus export — handover note

Pulled 23 Sept 2026 from indusscript.in ("The Indus Script: Texts, Concordance and Tables", Mahadevan 1977 / Input Data File 1980), run by the Roja Muthiah Research Library. Exported from the site's own database while signed in; nothing retyped by hand.

## The one-line version
This is the full machine-readable Mahadevan corpus with every line of text tied to its site, object type, side, line, writing direction, field symbol (the animal/motif), and excavation level and locus. It can replace or cross-check the 2,543-inscription corpus in the existing pipeline.

## Files
- im77_corpus_lines.csv — one row per line of text (3,916 rows, 2,906 distinct text numbers, 14,153 sign tokens).
- im77_field_symbols.csv — all 164 detailed field-symbol codes with plain descriptions (e.g. 13 = "Unicorn to R. facing cult object").
- im77_sign_frequencies.csv — token count for each of the 417 signs, plus how many of those readings are marked doubtful.
- IM77_README.md — this file.

## Columns in im77_corpus_lines.csv
- text_no — Mahadevan text number (1001–9905). Several rows share a text number when an object has more than one side or line.
- site_code / site / source — derived from the text number (table below).
- object_code / object_type — 1 seal, 2 sealing, 3 miniature tablet, 4 pottery graffito, 5 copper tablet, 6 bronze implement, 7 ivory/bone rod, 9 miscellaneous.
- side — 0 only side, 1–6 first to sixth side.
- line — 0 only line on that side, 1–3 first to third line, 9 side has a field symbol but no text.
- direction_code / direction — 1 right-to-left, 2 left-to-right, 3 single sign, 4 symmetrical, 5 doubtful, 9 no text. Code 0 appears on 343 rows and is not defined in the site's tables — treat as unrecorded.
- fs80 / fs_category / fs_description — field symbol (Input Data File 1980 code). 0 = none, 999 = damaged/illegible. Categories: 11–352 animals, 361–431 reptiles/fish/birds, 441–460 trees and leaves, 470–599 anthropomorphic, 600–810 scenes, 821–988 symbols/motifs/geometric.
- level, locus — excavation level and locus as recorded.
- n_signs — number of sign positions in the line.
- signs_reading_order — the signs as stored, space-separated, in reading order (first sign read first). A leading * means Mahadevan read that sign doubtfully.
- signs_clean — same sequence with the * removed and leading zeros stripped.
- doubtful_positions — which positions (1 = first read) carried the *.

## Conventions that matter for analysis
- Sign 0 is not a sign. It marks an illegible or lost passage (781 occurrences). Drop it or treat it as a break before counting pairs, openers or closers.
- Sequences are stored in reading order, so position 1 is the opener and the last position is the closer, whatever the physical direction. (On the site, text 1028 ends in sign 342, the jar, displayed at the left end of a right-to-left seal.)
- 144 sign readings are marked doubtful. Worth running key results with and without them.

## Site table (from text number)
- 1001–1905 Mohenjodaro, published in MIC vol III
- 2002–2952 Mohenjodaro, published in FEM vol II
- 3001–3513 Mohenjodaro, other texts
- 4001–4905 Harappa, published in EH vol II
- 5001–5601 Harappa, other texts
- 6104–6306 Chanhudaro, published in CE
- 6402–6405 Chanhudaro, other texts
- 7001–7301 Lothal
- 8001–8302 Kalibangan
- 9001–9701 Other sites
- 9801–9905 West Asian finds

## Things to check before trusting it
- 2,906 distinct text numbers here. The existing pipeline has 2,543 inscriptions. Reconcile the two: which texts are missing from which, and whether the difference is side/line splitting or genuinely different coverage.
- Direction code 0 on 343 rows is undocumented.
- The site also offers pair and triplet tables (2,730 sign pairs). These are derivable from this file, so not exported separately — useful as a cross-check of any bigram code.

## Not exported
- Sign images: the site holds a drawing of every sign at https://indusscript.in/assets/images/<sign>.png (0–417).
- Photographs of objects.
- Mahadevan's introduction PDF (the link on the site is dead).
