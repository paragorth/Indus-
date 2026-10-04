# Proto-Elamite: where the script is attackable

Goal: find the weak points ("chinks") of Proto-Elamite (PE) that data analysis can exploit.
Rules: data and sign lists from anyone; nobody's readings used as evidence. Every test has a control.

## Data

- Source: CDLI bulk data, github.com/cdli-gh/data, commit d66b12b (11 Oct 2023).
  Files `cdliatf_unblocked.atf` (86.9 MB) and `cdli_cat.csv` (155 MB).
  These are Git LFS files. raw.githubusercontent.com returns only the 133-byte LFS pointer.
  The full files came from a Git LFS clone made earlier in this project.
- Filter: catalogue `period` starts with "Proto-Elamite". CDLI lists PE `language` as "undetermined". In the ATF the tag is `lang qpc` (proto-cuneiform), not `qpe`.
- `data/pe_raw.atf`: the PE subset, verbatim. `data/pe_corpus.json`: parsed version. For each line it gives the surface, the M-signs and the numerals as (count, N-code), kept separate.
- Counts:
  - 1,729 PE catalogue entries. 1,585 of them have a transliteration.
  - Sites: Susa 1,502 (95%), Tepe Yahya 27, Malyan 22, Sialk 12, Sofalin 11, others 11.
  - 18,074 non-numerical sign tokens: 15,372 M-signs or compounds, plus 2,640 `x` (unreadable).
  - 351 distinct base M-numbers. That rises to 1,641 forms if variants (~a, ~b) and compounds count separately.
  - 11,850 numeral groups.
  - Entries ("sign-string , numeral"): 6,166 have a readable final sign. 4,816 are fully clean (no `x`, no break).
- Caution: 95% of the corpus is from Susa, so every result below is a Susa result. The other sites have too few texts for held-out replication of most tests.
- Rebuild: `tools/run_all.sh <atf> <csv>`. Results go to `data/res_*.json`.

## Tests

### a. Entry structure (tools/test_a_slots.py)
- Method: entry = a line with signs and a numeral (not the header, not a lone reverse total). Count how often each sign is first or last in multi-sign entries.
- Control: 1,000 shuffles of sign order within each entry.
- Numbers:
  - Lengths: 1 sign 62.6%, 2 signs 19.1%, 3 signs 8.6%, 4 or more 9.7%.
  - Of 62 signs with n >= 20: 11 are initial-biased and 14 are final-biased (z >= 3). No sign is both.
  - Final slot: M288 237/250 final, against 86 expected (z = 20). Also M297 (z = 9), M263 (8.9), M346 (7.9), M264, M072, M003, M354, M371, M096, M376, M036.
  - Initial slot: M387 185/290 initial, against 107 expected (z = 10.5). Also M157, M370, M124, M305, M038, M111.
  - 45% of multi-sign entries end in a final-biased sign.
- Verdict: **strong slot grammar.** Each entry is [optional prefix] + [middle] + [final class sign], followed by the numeral. The final sign behaves like a category or commodity marker.

### b. Counted goods vs number system (tools/test_b_goods.py)
- Method: each numeral is classed by its diagnostic N-codes:
  - C = capacity or grain-type codes (N39B, N30C, N24, N30D, N39C)
  - C* = hatched or modified capacity codes
  - B = N51 or N54
  - S-frac = N08 or N02 fractions
  - N23 = a separate system
  - SDB = plain N01 / N14 / N34 / N45 numerals

  Then compare these classes against the sign that touches the numeral.
- Control: permute the numerals 1,000 times, once across all entries and once only within each tablet. The within-tablet version is the strict control, because whole tablets tend to be about one commodity.
- Numbers:
  - Mutual information between final sign and system: observed 0.309 bits.
  - Global permutation: mean 0.065, max 0.077. Within-tablet permutation: mean 0.178, max 0.190. Both are beaten (p < 0.001).
  - Baseline C share is 20%.
  - Tied to capacity: M297 73% C (n = 253), M002 84%, M036 66%, M243 64%, M075 60%, M106 58%, M010 52%. Also M379 and M050 >= 90%.
  - Never capacity (<= 2% C, with n >= 30): M346, M263, M376, M003, M032, M373, M264, M102, M362, M317, M149, M046.
  - The key detail: M263 is C in 3/184 entries, while 44% of the other entries on the same tablets are C. M264 is C in 0/51 against 34% on its tablets. So these are sign effects, not tablet effects.
  - M376 takes fractional numerals (N08 etc.) in 59/158 entries, against ~2% overall.
  - M288 is mixed (34% C). It is a general head sign, not a grain marker.
- Verdict: **strong.** The final-slot signs split into "measured in capacity units" (M297, M002, M036, M243...) and "counted, never measured" (M263, M346, M264, M003...). M376 is a "takes fractions" class. This is the determinative layer of the script.

### c. Totals (tools/test_c_totals.py, tools/test_c2_rations.py)
- Method: take tablets whose only numeric line off the obverse is a total. Sum the obverse entries and compare.
  - Relaxed set: 77 tablets with no breaks and no `n` (damage marks allowed).
  - Strict set: 44 tablets with no damage marks either.
  - Unit ratios come from a grid search, scored by how many tablets add up.
- Control: give each tablet another tablet's total.
- Numbers:
  - Non-capacity tablets: values N14 = 10*N01 and N34 = 60*N01 fit 15/39 (38%; strict set 10/21 = 48%). The shuffled-total control gives 2-4%.
  - Tablets using only N01/N14: 14/29 add up.
  - Tablets that use N34/N45: 2/13 add up under any tested value set. The grid cannot even fix whether N45 is 100, 600 or 1000.
  - Capacity tablets: the best ratio set fits 5/22 (23%) against a 0.9% control. Ratios are underdetermined; many grids tie.
  - Only 6 of 77 tablets add up code by code with no carrying.
  - Conversion tablets: 37 tablets have plain-count entries and a capacity total. The total line is almost always headed by M297 or M243, the two capacity-tied signs from test b. Test c2 asked whether total / head-count gives a round per-head ration. Best grid 5/14, shuffled control mean 4.8: **no signal.**
- Verdict: **mixed.** Small sums work well above chance. About half the totals still do not add up, from scribal error, transcription, or a wrong model of what the total covers. Higher units and capacity ratios are poorly pinned by clean data. This is a data-quality hole, and also an attack point.

### d. Header signs (tools/test_d_headers.py)
- Method: 744 tablets have a first line with signs and no numeral (the header). Compare the header's first sign with its rate among entry tokens and among entry-initial signs.
- Control: 1,000 draws of the same number of random entry-initial signs.
- Numbers:
  - M157 opens 277/744 headers (37%). It is 86x its entry rate, z = 153.
  - |M327+M342| 34 headers (176x). M327 22 (18x). |M327+X| 14. |M377+M320+M377| 11. M136, M005, M305, M247 are all 3-9x, with permutation p < 0.001.
  - 200 distinct header strings, 142 of them used once. Top 3 cover 45%.
  - M136 headers lean toward capacity entries (47 C / 65 SDB). M157 tablets are mostly SDB (809 vs 227 C).
- Verdict: **strong "opener" slot.** M157 is a general opener. The M327 family is a second one. Header type partly predicts the commodity.

### e. Do entry strings repeat like names? (tools/test_e_names.py)
- Method: unique share = types seen once / tokens. Compare with a bigram model trained on the same strings, 200 runs.
- Planted control: a fixed lexicon of model-generated strings used with Zipf frequencies.
- Numbers:
  - Full strings of 2+ signs: n = 1,803. Unique share 0.792 against a model 0.737 (sd 0.013). Ratio 1.08.
  - Strings minus the final class sign (3+ signs): ratio 1.06.
  - Planted lexicon: ratio 0.21-0.29. So the test can see a fixed vocabulary when one exists.
  - 123 strings repeat, 100 of them on 2 or more tablets. The top repeats are prefix + class pairs: M387+M263 (18), M056+M288 (13), M387+M297 (12).
  - Strings from non-Susa sites: 4/43 are also found at Susa, against 13.1 expected under the bigram model.
- Verdict: the long strings are **not a small fixed lexicon** (ratio is not below 1). They are at least as diverse as free combination. This fits one-off personal names or spelled words, and does not fit formulaic category terms. Per standing rules, uniqueness alone does not prove names.

### f. Linear Elamite foothold (tools/test_f_le_matches.py, data/le_pe_graphic_matches.json)
- Source: Desset et al., *Linear Elamite Inscriptions* vol. 1 (2026), Plates III-V. It is free at orientlab.net, linked from hatamti-elam.uliege.be. The plates print a "PE signs" column (Dahl 2006 numbers) beside each Linear Elamite (LE) syllable. I read 63 base PE signs from it, for example:
  - M218 -> a, M096 -> e, M263 -> ha, M387/M002 -> na
  - M288 -> pu2, M297/M298 -> ri2, M371 -> u2/w, M009 -> zu, M032 -> še, M219 -> ta, M057 -> u
  - M157a/M175 -> undeciphered US5
- Their 2022 ZA paper (doi 10.1515/za-2022-0003) is paywalled and was not read.
- These are leads, not evidence. The authors may have picked PE matches with knowledge of PE usage, so the sign set is not blind.
- Test: if these PE signs were already phonetic in PE, they should sit inside long strings rather than standing alone as one-sign entries.
- Control: 2,000 random sign sets matched on frequency decile.
- Numbers:
  - All 63 signs:
    - share of their tokens in 3+ sign strings: 0.533 against 0.443 (p = 0.02)
    - share as one-sign entries: 0.269 against 0.342 (p = 0.04)
    - adjacent pairs where both signs are in the set: 0.59 against 0.34 (p < 0.001)
  - Without the 10 frame signs (M288, M387, M297, M263, M371, M218, M346, M009, M032, M157), 55 signs remain:
    - non-final position in long strings: 0.510 against 0.328 (p < 0.001)
    - one-sign entries: 0.222 against 0.342 (p = 0.005)
    - adjacency: no effect (0.265 against 0.236, p = 0.28). The adjacency result came from the frame signs.
- Verdict: **positive but not independent.** The proposed LE-ancestor signs do concentrate in the name-like middles of long strings. A blind replication needs a match list made without looking at PE distributions.

## Chinks (ranked)

1. **Determinative layer (tests a+b).** The final slot holds about 14 class signs. Number systems split them into "measured" (M297, M002, M036, M243) and "counted" (M263, M264, M346, M003), and M376 takes fractions.
   - Attack: cluster all signs by their numeral-system profile and slot. Then predict the system of numerals that are broken or `x` from the final sign alone.
   - Success: held-out accuracy well above the within-tablet baseline (MI > 0.19 bits). The same classes should hold on non-Susa tablets (Malyan, Yahya, Sialk).
2. **Arithmetic as a decoder (test c).** Small sums work (38-48% against 2-4% control). Two things remain unknown: the high units (N34/N45) and the capacity ratios.
   - Attack: run a joint integer-programme fit over all totals and sub-totals, including tablets with several reverse lines. Allow one scribal error per tablet.
   - Success: a single value set making >70% of clean totals add up, stable across random halves of the corpus. The conversion tablets (count -> M297 capacity total, n = 37) should then yield consistent per-head rates.
3. **Name-like middles + the LE bridge (tests e+f).** Middles are as diverse as free combination, and the LE-ancestor signs cluster there (p < 0.001 after removing frame signs).
   - Attack: take the 20-30 LE-matched signs with fixed CV values. Transcribe every middle that is >= 70% LE-matched signs, then check whether known Elamite name elements appear more often than in shuffled-value controls. Name elements must come from a list compiled from cuneiform Elamite, fixed before the test.
   - Success: more hits than 95% of value-shuffles, plus a prediction of an unmatched sign's value that a later LE or cuneiform form confirms.
4. **Opener slot (test d).** M157 heads 37% of headers, and the M327 family is a second opener.
   - Attack: correlate header type with entry systems, tablet totals and seal impressions (82 seal notes in the ATF) to see whether openers mark office, transaction type or commodity.
   - Success: header predicts the dominant system or commodity class of the tablet's entries better than a header-shuffle control.
5. **Prefix slot M387 / M370 / M124 (test a).** These are initial-biased and combine with class signs (M387+M263, +M297, +M036, +M069).
   - Attack: test whether the prefix changes the numeral profile of the class sign. For example, does M387+M297 take smaller capacity amounts than bare M297? A "qualifier" (male/female, young/old) should shift amounts.
   - Success: a consistent amount shift with permutation p < 0.01 across at least 3 class signs.
6. **Weak spot to fix first: data quality.** About half the totals fail, there are 2,640 `x` tokens, and Susa is 95% of the data.
   - Fix: check failing totals against photos and line art (CDLI has images for many MDP tablets) before using arithmetic as evidence. Report all results as Susa results.

## Attack 1: do the name-like middles read as Elamite names with Linear Elamite values? (`tools/attack_names.py`; report in `data/attack_names_report.txt`)
- **Name list, frozen before matching** (`data/elamite_name_list_frozen.json`): 2,448 names and 448 elements.
  - Sources: Zadok 1984 OCR (noisy), CDLI Elamite texts, and royal and divine names.
  - Achaemenid-era forms (mostly Iranian) were left out.
  - 1,661 names and 322 elements of 2+ syllables were used.
- **Control list:** 5,887 Akkadian and Sumerian names from CDLI.
- **Spelling rules, fixed in advance:** e=i, b=p, d=t, g=k, z=s, double letters merged. A strict variant also merges š and s.

| strings | hits | value shuffle (1,000×) | Akkadian/Sumerian list |
|---|---|---|---|
| full strings (318) | 107 | mean 91, p = 0.24 | mean 77, p = 0.055 |
| name-like middles (224) | 53 | mean 61, **p = 0.65** | p = 0.19 |

- With strict matching, value-shuffle p ranges from 0.11 to 0.85.
- **No hit of 5+ letters** (shuffled values give about 2). Every hit is a short pair (aha, api, ata…).
- The weak full-string signal comes from fixed opener and closer signs. For example, M387+M263 reads "na-ha" on 11 tablets.
- Best single match: a-ha-ru-ra on 3 tablets ("ahar", as in Tepti-Ahar). That is within chance.
- **Inšušinak:** never found. The value set has no plain "n" sign, and šu + še/ši occurs on only 3 tablets.

**Verdict: null.** The Linear Elamite values do not make the Proto-Elamite middles read as Elamite names any better than shuffled values. There is no support for a phonetic layer from this test, but this does not rule one out.

Caveats:
- 95% of the tablets are from Susa.
- The values were read by eye from the Desset plates.
- The Elamite-vs-control comparison is partly circular, because the Linear Elamite values were themselves derived from Elamite words.
- The varied middles (test e) still fit one-off names, but this attack found no route to reading them.

## Attack 2: arithmetic as a decoder (`tools/attack_arith.py`; report in `data/attack_arith_report.txt`)
**Data clean-up.** The earlier totals test scored some top-edge marks as totals. That partly explains why only 2 of 13 N34/N45 tablets added up. Broken or unnumbered tablets and 21 note lines were dropped. The clean set is 32 count tablets and 17 capacity tablets.
- Multi-line reverses are mostly further entries, not totals: 1 of 19 adds up.

**Joint fit.**
- **Counting system:** the best set is N14 = 10, N45 = 100, N34 = 300, with 17 of 32 tablets adding up (control 0.6%).
  - **N14 = 10 is firmly pinned.**
  - N45 and N34 win by only one tablet over the sexagesimal set and the "N34 = 1000" set (16 of 32 each).
- **Independent check from how the numerals are written:**
  - N45 is written before N34, so N45 is the smaller unit.
  - N14 repeats up to 9 times.
  - N45 almost never appears more than twice (151 of 154 lines), which fits N34 = 3 × N45.
  - **Grade B: 1 / 10 / 100 / 300.**
- **Capacity system:** the totals cannot pin it (29,282 value sets tie). A set taken from how the signs are written, fixed before checking any totals, matches the best fit at 5 of 17 (control 0):
  - N39C 1, N30D 2, N30C 4, N24 12, N39B 24, N01 120, N14 720.
- **One-unit slips:** allowing one rescues 0 count tablets and 2 capacity tablets. Most failures are not one-unit errors.

**Out of sample.** No split of the 50 reached the pre-set 70% bar. **The pre-registered test fails.**

**Decimal or sexagesimal: the choice goes by tablet, not by commodity sign.**
- Mixed tablets: 21 against 39.7 expected (p = 0.0005).
- No effect of the commodity sign within a tablet (p = 0.12).
- Conversion tablets show no constant ration per head (p = 0.38).

**Tablets to check on photographs** (predicted readings):
- P008210: total should read 5(N14) 4(N01).
- P009107: total should read 5(N14) 1(N01).
- P008784: obverse line 2 should read 8(N01).
- P008803: the damaged total should read 6(N14).
- P008011: line 2 should read 5(N1@b).

Photos: https://cdli.earth/artifacts/<number>.

### Photo check (CDLI photo P008210, viewed 3 Oct 2026, not committed)
- **P008210 reverse:** 5 round impressions (N14) and 1 horizontal wedge (N01), i.e. **5(N14) 1(N01) = 51**, exactly as transliterated.
- **The prediction "total should read 54" fails on the total.** If there is an error, it lies in the obverse entries (transliterated 20 + 11 + 10 + 10 + 3 = 54) or with the scribe. Several obverse wedges are shallow on the photo.
- **P009107** (predicted total 51, read 55): the reverse is cracked through the numeral area and could not be counted reliably by eye.
- **Lesson:** "fix the total" predictions should also test "fix one entry". The arithmetic model cannot tell which line is wrong.

## Economy simulation (pe6, 4 Oct 2026; `loops/pe6_final.txt`)
An agent-based economy fitted by approximate Bayesian computation (38 parameters, 43 statistics, 20,000 runs) matches the corpus on 40 of 43 statistics. Writing habits are pinned (one general opener on about a third of headers; 83% of tablets list named units; totals on 53% of tablets, 63% of them not equal to the sum); institution size is not (offices and workers per office fail the recovery control). The model's one systematic failure is informative: the final class sign predicts the number system far less than the model expects (0.31 vs 0.95 bits), so **the number system follows the tablet, not the good**. Role grades calibrated against a shuffled corpus: counted-goods A M072, M346; capacity A M002, M297 (restates test b); name-spelling A M099, M136, M005, M246, M251, M254, M304, M262, M340, M390, |M106+M288|; B M418, M206, M050, M379, M243, M036, M319, M328, M352, M384.

## Name phylogenetics (pe7)
- Proto-Elamite middles show no detectable kinship structure. Tree-likeness is weak (retention index z 3.6, about the Linear B level). Father and son samples are no more tree-like than unrelated names. Planted lineages were recovered 8 of 8, but the method failed on real Ur III father and son pairs, so the kinship test is weak.
- Names on the same tablet share rare signs: 0.30 of pairs against 0.125 by chance (2.4×). Kin pairs in Ur III and Old Babylonian give only 1.1–1.25×. The effect survives removal of common signs and is not due to repeated spellings or adjacent lines. Grade B: each tablet draws on a local pool of name signs.
- Herd tablets show similar names through |M362+X| compounds. This is a herd-naming pattern, not a lineage. Grade C.
- No inherited name element reaches grade A or B. Grade C only: M388, M124, M370, M054, |M362+X|.

## Filled-in forms? (pe8, 4 Oct 2026; `loops/pe8_final.txt`)
Each tablet was reduced to a skeleton of sign roles and number systems, and the skeletons were clustered. The method recovers 6 planted forms (5-7). The real corpus gives 5 loose clusters, which is exactly what a corpus with no forms (layout drawn per entry) gives. A graded dependence tree fits the real skeletons better than any set of discrete forms. The layout predicts nothing about the number system (0.89 bits, the same as the marginal; the final sign gives 0.78; the other entries on the tablet give 0.55). Museum or publication neighbours share headers but rarely whole forms, and rare skeletons are not mixtures of common ones. **Verdict: no small set of fixed forms. The number system belongs to the individual tablet but is invisible in its layout.**

## Running out of clay (pe10)
- Proto-Elamite scribes did not shorten entries near the end of a crowded face. Last entries on crowded faces are only 0.13 signs shorter (z −0.9). A planted squeeze was caught (z −2.4). Grade B: no shortening larger than 0.4 signs.
- The search over about 234,000 long and short string pairs found no abbreviation pairs. The method recovers Ur III year-name abbreviations (5 of 5), but Proto-Elamite repeats too few multi-sign strings to test this way. The only hint, bare M387 = M387 M263, fails: the two take different number systems.
- No sign switching or merging appears near the end of a face (z 0.03; a planted substitute was found at z 7.9).
