# Gulf Type seals (Laursen 2010) as an outside test of the slot grammar

**Verdict: the grammar does not treat the western Gulf Type seals as the same kind of text as Indus Valley seals.** They never start with the home opener, they rarely show the home frame, they start with a person or "twins" sign about 80× more often than home seals do, and they fit the home grammar far worse than home seals of the same lengths. But the same misfit appears on Gulf Type seals found *inside* the Indus Valley, while home square seals with the same bull emblem fit normally. So what the grammar separates is the round-seal community and its formula. Whether that community wrote a different *language* cannot be settled by this test.

## Numbers

Gulf/western register = inscribed Gulf Type seals found outside the Indus Valley, with a matched reading (n = 19). Comparison = Indus Valley square seals, deduplicated by site + text (n = 1,574). "Length-matched" = 10,000 random sets of home seals with the same text lengths as the 19.

| Feature | Gulf, abroad (n=19) | Gulf Type found in Indus Valley (n=5) | Home square seals, length-matched | P |
|---|---|---|---|---|
| Starts with a home opener (W817/861/820 = M267/391) | **0%** | 20% | 21% | 0.009 |
| Home frame (opener first, or jar last / jar + suffix) | **16%** | 20% | 53% | 0.0006 |
| Starts with a person sign (W90/91/93) | **32%** | 0% | 0.4% | < 0.0001 |
| Contains "twins" (W91, or the two-man pair coded W121-W99) | **32%** | 0% | 0.3% | — |
| Ends jar → person (W740-W90) | 0% | 0% | 2.9% | 0.57 (not informative) |
| Grammar fit: mean log-probability per sign transition, bigram model trained on half the home square seals | **−5.62** | −5.65 | −4.38 (95% range −4.77 to −3.98) | < 0.0001 |

Emblem control ("postage" question: is the animal a separate stamp, or does it change the text?). I read your "postage model" as: the animal is an emblem added to the seal and does not decide what the text says. If you meant something else, tell me.
- Emblems on the 19 western seals: short-horned bull ("gaur") 13, none/illegible 4, scene 1, bird 1. No unicorn. Home square seals: unicorn 72%, gaur 5%.
- **Home square seals with the gaur emblem fit the home grammar normally**: fit −4.32 (unicorn seals −4.25), frame 43%, person-first 0%, twins 0% (n = 79).
- So the animal does not carry the difference; the seal format and community do. That fits the emblem-as-stamp reading: the western seals chose the gaur, and their texts differ, but a gaur on a home seal changes nothing in the text.

### Where the openers and the twins sit
- **Openers:** no western text begins with a home opener. The only opener anywhere is W817 in second position on seal 13, a Failaka seal that is technically Dilmun Type, with an uncertain reading.
- **Twins:** first sign on seals 8, 11 and 15; second on 24; third on 9; on 10 the two-man pair comes near the end (fifth of seven). Where it occurs it takes the slot the home opener would hold, or stands beside a stroke group. A person sign sits next to a stroke group in 9 of 19 western texts.
- **Jar then person at the end:** never seen abroad. It is also rare on home seals (2.9%); the grammar puts that ending on sealings and tablets (GRAMMAR.md). So its absence does not discriminate.

### Frozen predictions (read only, not changed)
The western seals agree with every prediction in `prereg/preregistration-3-frozen.json` that concerns them:
- R2, person-led: 32% (≥ 15% required).
- R3, person + stroke block: 47% (≥ 10%).
- R4, no home opener: 0% (≤ 5%).
- R9, frame ≤ 25% and person ≥ 15%: 16% and 32%.

**This is not an out-of-sample confirmation.** Laursen's Fig. 11 is listed in that preregistration's own basis, and 24 of these seals are already in the corpus the rules were drawn from. It only checks consistency. The true holdouts are the post-1977 Bahrain finds (seals 9, 10, 11, 56). They are too few to test on their own.

## How the readings were obtained
- Laursen's PDF text gives each seal's source (Figs 8–9 captions) and Parpola's note on seal 10. Fig. 11 (the inscriptions as seen on the impression) was rendered at 220–600% and each drawing compared by eye, sign by sign, with the matching corpus text drawn in the same orientation.
- All 24 matched readings were found in the merged corpus (lipi/CISI Wells coding). Three also matched Mahadevan's 1977 West Asian finds: no. 21 = IM77 9846, no. 22 = 9852, no. 23 = 9901.
- Confidence: 11 confirmed (drawing and corpus agree sign for sign), 10 probable (one extra or damaged sign, or a reversed direction), 3 uncertain, 5 missing.
- **Missing, listed not guessed:**
  - no. 6 (Fig. 11, Bahrain; fragment, only the twins legible);
  - nos. 17, 20 and 25 (Fig. 11, Mesopotamia column; no corpus match);
  - no. 28 (Linear Elamite, not Indus).
- **Conflicts found:**
  - Seal 9: the stroke block is drawn as 2×4 (8) but coded W55 (12) in the corpus.
  - Seal 10: the last block is drawn as about 3×3 (9) but coded W55 (12), and the "twins" is coded as two separate man signs (W121 + W99).
  - Seals 7 and 12: the reading direction is reversed between the drawing and the corpus.
- **Seal 10 and Kalibangan:** Parpola (Laursen n. 2) reads the end of seal 10 as twins + a stroke sign, the same pair that opens the Kalibangan tablets K-69 to K-75. On the CISI photos the Kalibangan block is 3 rows of about 8 (≈ 24). On the seal 10 drawing the blocks are about 12 and about 9. The only published photo of seal 10 is worn. So the shared *pattern* (twins + a stroke block) holds, but an identical text is not confirmed.

Files: `data/gulf_seals.csv` (seal_no, site, concordance, sequence_glyph_coding, match_source, confidence, plus note, motif, found_abroad); `tools/gulf_seals.py`, `tools/gulf_seals_test.py`; `results/gulf_seals_rates.json`.

## Caveats
- n = 19; readings come from drawings, not photos. The bigram fit is the crudest part: a formula change alone (a different opener) lowers it without implying another language.
- "Same language or not" is beyond what position statistics can decide: a different formula in the same language would look the same here.
