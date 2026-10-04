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

## Rebuilding the clock from the animals (pe11, 4 Oct 2026; `loops/pe11_final.txt`)
- Undated tablets were treated as shuffled frames of a film. A search placed each of 638 tablets in one of 12 months under a closed-loop (yearly) model and an open-drift model with the same parameters, and compared the loop excess against copula, shuffle and random-walk surrogates.
- Calibration: in 8,255 Ur III Drehem livestock tablets with known months, the month explains only 1.0% of the feature variance. The search orders them barely above chance (0.29-0.32 within one month against 0.28-0.30), and the label-free statistic does not see the season at all. Pure synthetic rings show the statistic needs the season to carry about 40% of the variance.
- Proto-Elamite: ALL and HERD feature sets close a slightly better loop than surrogates (z +1.4, +2.4). A timeless mixture of three commodity types does the same (z +2.0); real seasonal Drehem data do not (z -2.0). Of 1,500 random feature subsets, none survives the family-wise null (p 0.56) or held-out re-test (0 of 15). The inferred "months" are a tour of tablet types (capacity tablets, then large counted tablets, then M218/M371 small-count lists).
- No sign switches at inferred season boundaries. The test finds a planted season header at z 18 when the order is right. Grade C hint only: M338 and header M388 (strict-null FWER p 0.06-0.09).
- **Verdict: null.** Grade B: no strong yearly cycle in tablet composition. Any time-based attack on PE needs dated anchors; more search will not help.

## Names drawn from a bag? (pe9, 4 Oct 2026; `loops/pe9_final.txt`)
- Idea tested: name strings were assembled from a finite set of counters or stamps on the desk, used up within a tablet or session. Held-out likelihood race, 1,000,000-corpus ABC, planted-bag recovery, Ur III and Linear B controls, 52 reserved tablets.
- **Killed in its strong form.** No sign is used up: after a sign has been used in one name on a tablet, it is 2-3x as likely to appear in another name there, and more after a third use (fit set 2.6, 2.9, 4.6; reserved tablets 3.1, 2.2, 14). A planted two-copy bag shows the collapse (8.6 then 0.5-0.7); PE never does. 0 of 90 signs are depleted. One- and two-copy bags are excluded (ABC posterior 0.00 / 0.01).
- Best model: a per-tablet topic/cache (0.42 bits/sign better than a language model; 0.40 on reserved tablets, shuffled max 0.13). If called a bag: refilled per tablet, ~10-50 sign types, ~40% of signs, each inexhaustible (grade C).
- Grade B: names on a tablet draw on a freely reused tablet pool (out-of-sample pass). The Linear B control also prefers a pool, so this does not show that PE middles are not spoken names.
- Tablet-sticky signs: B M370, M376, M124, M388, M001, M054; C M145, M203, M136, M210, |M036+1(N30D)|, M002 (did not replicate at 52 tablets).
- No model reproduces PE's surplus of one-off signs (hapax share 0.35 vs 0.22-0.29 simulated).

## Names drawn from a bag? (pe9)
- Names were not built from a stock of tokens that gets used up. Once a sign is used on a tablet it becomes more likely to recur there. Repeat chance is 2.6× after one use, 2.9× after two and 4.6× after three or more. A planted two-copy bag drops to 0.5–0.7×. This held on 52 held-out tablets. Grade B.
- Held-out likelihood: the tablet-topic (rich-get-richer) model wins at 8.43 bits/sign. A use-up bag and a plain language model tie at 8.85. ABC over 1,000,000 simulated corpora gives the topic model 0.45 and the use-up bag 0.09.
- Signs reused on a tablet beyond chance, grade B: M370, M376, M124, M388, M001, M054. None of 90 signs behaves like a used-up token.
- Linear B names also prefer a tablet pool, so this does not show the strings are not spoken names.

## Random programs (r2, 4 Oct 2026; `voynich/loops/r2_final.txt`)
- ~8.9 M random/evolved small programs across the three scripts, scored by held-out description length over Kneser-Ney; planted PE-sized ledger recovered blind (+96 mbit/token).
- Proto-Elamite: the winning programs copy entry-initial signs (M346, M075, M388, M297, M036) from the line(s) above: +492 bits on held-out tablets with line ends given, +20 (p 0.28) against KN + document cache. Grade B, known in kind (within-tablet repetition). No generating procedure.

## The scribe's memory has a shape (pe13, 4 Oct 2026; `loops/pe13_final.txt`)
- Question: is the tablet effect (pe7, pe9) a flat topic, copying from a source list, or the writer's own fading memory?
- Method: about 1,200 random memory kernels per corpus, fitted on 4/5 of tablets and scored on the rest, against within-tablet shuffles and topic-model surrogates. The method recovers 11 planted mechanisms (topic, priming, dip, reverse reset, copying, mixture, alternation, canonical order). Controls: Ur III, Linear B, proto-cuneiform admin and lexical lists.
- **No fading memory (grade B).** The best non-flat kernel gains only 3.6 millibits per sign over a flat topic, the smallest of the five corpora. No sign carries priming (0, against 1 by chance and 85 in a planted priming corpus).
- **A dip at the next entry (grade B).** Adjacent entries share about 10% fewer signs than chance. Entries two apart share a little more. This holds without headers, without class signs and on the obverse alone. Every control corpus instead peaks at the next line. M288 and M346 carry most of it: they skip the next entry and recur every second one. Some herd tablets (P008295) cycle through a fixed run of entries.
- **The reverse does not reset (grade B).** Reverse entries draw on obverse signs 1.7-2.3 times as much as on their own face. This fits sub-totals or summaries.
- **Not copied (grade B).** Ordered pairs of entries recur across tablets at chance (0.173 vs 0.171). A copied lexical list gives 0.51 vs 0.17.
- **A weak shared order (grade B).** An order of signs learned on half the tablets predicts the other half at 0.555 against 0.527 (z 3.6). That is proto-cuneiform admin strength, far below Ur III (0.637). No fixed multi-sign sequence was found, but that search was underpowered.
- Grade C: entries were written one per record from a per-tablet set (tallies or a pool of names), laid out by type with alternation and a loose habitual order. Would support: the same dip on non-Susa or newly published tablets. Would kill: a decaying kernel in a larger corpus.

## Check digits (pe12)
- Entries carry no check marks. About 130,000 candidate rules over 5 slots found nothing beyond the null (best z +0.4 after size correction). A planted modulo-3 check sign was found at z +9.7, and the Ur III weight-unit marker at z +43. Grade B negative.
- No verification tick and no check sign on totals (56 legible tablets, FWER p 0.80). Mismatched totals are not single-sign slips.
- Side finding, grade B: about 20 lists are fixed two-line records, such as MDP 31, 004. There, each "name M288, 2" line is followed by "M376, 1/4". This explains the pe13 dip at the next entry and the peak two entries later (lag 1 z −5.5, lag 2 z +4.6).

## Woven tablets (pe14)
- Grade A: a period of 2 is real, but it is confined to about 12–20 tablets of fixed two-line records, such as P009343, P009000, P008689 and P008020. The class sign gives z 3.5 on the search half and z 6.2 on the re-test half, and it survives a per-tablet adjacency null.
- Grade B: beyond those tablets nothing repeats. All 50 feature × period (2–6) tests give z ≤ 2.7. There are no 3–6-line records, and no fixed sign pair survives a 1,160-pair search.
- Grade B: the records are fixed recipes. The companion line's number is often constant, and where it varies it follows the main line's number (Spearman 0.23 vs 0.11, p 0.001). M288 and M346 are each the repeating companion, on different tablets.
- Grade C: each record is one person's two-part allotment.

## Grow the script in a box (r3, 4 Oct 2026; `voynich/loops/r3_final.txt`)
- Simulated societies + ABC put PE in a world of counted lines that mostly carry no personal name (good + number; nameless share 0.73-0.88), with 100+ logograms and unwritten dividers (C). Held-out 3/4 (tablet purity predicted 0.32, real 0.29). PE line ends are less marked than any simulated world. The shuffled corpus fits as well, so sign order within lines is invisible to the box.

## The administration as a food web (pe15, 4 Oct 2026; `loops/pe15_final.txt`)
- Idea: treat name strings or name signs as consumers and goods (final class sign with number system) or tablet headers as resources. Analyse the webs with ecology tools: NODF, Barber modularity, H2', a degree-corrected stochastic block model fitted by MCMC, link hold-out and new-name prediction, and 3,000 random office partitions tested on reserved tablets. Controls: Ur III (CDLI, 7 sites; Drehem with its receiving official), Linear B, planted modular and planted no-module webs, curveball and within-tablet nulls. The block model recovers planted offices (NMI 0.94) and known Ur III and Linear B institutions above chance (NMI 0.15-0.38).
- Grade B: PE webs are nested (NODF 40-65), but only as far as their degree sequences force (z about 0). There are no binary compartments in the entry webs (Q z +0.2 to +2). Ur III and Linear B are anti-nested and modular beyond their degrees. In PE the structure is in how often a consumer takes a good (H2' z +14 to +16 even within a tablet), not in which links exist.
- Grade B: three sign "offices" replicate on reserved tablets (+0.135 bits per entry over degree). No within-tablet-shuffled corpus reaches this, and none of 3,000 random partitions does better. The offices are GRAIN (M010, M106, M002, M243, M075, M081, M265, M266, M296, M112: capacity), CLASS (M387, M388, M218, M124, M009, M066, M057: entries ending in class signs) and BARE-COUNT (M054, M367, M001, M370, M032, |M036+1(N30D)|, M206, M269, M059, M102: bare counts; overlaps the herd run).
- Grade C, new grain-office signs M081, M265, M266, M296, M112, M286, M248. Would support: capacity numerals with these signs on new tablets. Would kill: less than 30% capacity.
- Grade C: two header families, {M157, |M327+M342|, M388, |M377+M320+M377|} against {M327, M217 M247, M377, M136}. They do not predict held-out tablets.
- Grade B: the signs of a never-seen name predict its goods and number class out of sample (+0.175 bits; a frequency-matched scramble gives -0.199). This is half the Ur III strength and 4x the Linear B strength. Hidden links are predicted by popularity alone in every corpus (PE AUC 0.80 for the block model against 0.84 for degree).
- Nothing is read; nothing is cracked.

## Predict the clay before looking (pe17, 4 Oct 2026; `loops/pe17_final.txt`)
- Idea: rank Susa tablets by how "plateau-like" their writing is with 2,000 random classifiers trained on 67 plateau tablets, hash and freeze the ranking, and only then read the hXRF clay provenance of Yeganeh et al. 2025 (JAS Reports 61:104973; CC BY manuscript on ORA). Labels in `data/pe17_xrf_labels.json`.
- Grade B: a weak plateau writing fingerprint exists. Selected on Malyan and Yahya held out, it re-tests at AUC 0.69 on 20 other plateau tablets (10 label-shuffled pipelines: max 0.63). Hidden real plateau tablets rank at AUC 0.75 among Susa, and a planted shared tic is found perfectly. It is carried by which signs are used (M157, M136, M072, M158, M379, M264 ...), not by variant choice, numeral system or bigrams. It is confounded with early, short Susa formats: the most plateau-like Susa tablets are CahDAFI 1, EMSP and TCL 32 tablets.
- Grade C, not supported and not killed: scored against the five foreign tablets the manuscript identifies by P-number. ST-11 (P009157, MDP 26, 469, Yahya clay) ranks in the top 10% (p 0.12 within the 476 National Museum of Iran Susa tablets). The Susa-clay tablets at Yahya, YT-01 = Yahya 12 and YT-07 = Yahya 13, look Susa-like (ranks 3 and 8 of 27; p 0.07). The Malyan-clay tablets, YT-02 = Yahya 21 and YT-06 = Yahya 16, do not look clearly Malyan-like (p 0.28). Fisher p 0.14. Power is about 12% for five imports and 34% for twenty (control 5%), so a null here says little.
- Itinerant scribe: not seen. ST-11 looks most like Sofalin and Sialk writing and least like Yahya (p 0.91). Yahya and Sofalin do have their own hands (beyond a label permutation). Susa affinity falls with distance (Sialk > Sofalin > Malyan > Yahya), but this is grade C because it has 4 points and is confounded with site size and date.
- What the imports record: common signs, N01/N14 counts and the N39B-N24-N30 series. YT-07 is a 15-line list of two-sign entries with 1-2(N01) each. ST-11 is a fragment with M265 (a pe15 "grain office" sign). No sign or numeral occurs only on imports (0 of 5).
- Not cracked. Table S2 and the full ST/MT/YT to museum-number map are not public (`docs/HUMAN-ACTIONS.md`).

## Hunger as the ruler (pe16, 4 Oct 2026; `loops/pe16_final.txt`)
- Method: allotment = biological need (litres/day by recipient class) x period / unit size, with class and period latent. Positive control, Ur III: blind 0.83 l per sila (truth about 1 l); daily wages and fodder are told apart from monthly rations on 97-100% of tablets. Limits: month and year alias 1 time in 3 at PE size, and random nonsense rulers fit as well as biology, so fit cannot confirm the ruler.
- Grade C: on the two-line records, N39C = about 3 adult-man-days = 4.6 l (90% 1.5-14), on a lattice x{1/30, 1, 12}. The scale is set by which units the scribes used (a same-code random-count null gives the same value). All capacity entries give 0.6-1.4 l per N39C.
- Grade C (weak): only the /30 point puts a PE unit in the measured bevelled-rim-bowl range (N30C 0.62 l; then N24 = 1.85 l = one adult day). A chance hit is 43% likely. Would support: PE-period vessels near 0.6 and 1.9 l in a 1:3 ratio. Would kill: PE ration vessels clustering far from 0.6 l.
- Grade C: in capacity records the companion line is 0.2-0.67 of the main line (P009000 1(N30C) after 1(N24); P008403; P009123): one adult-sized and one dependant-sized ration. Children and small stock cannot be told apart by amount.


## The scribe knew the length? (pe21, 4 Oct 2026; `loops/pe21_final.txt`)
- Idea: a scribe copying from a finished source (tokens, a bulla, an earlier tablet) knows the list length and sizes the clay to fit; one writing as events happen does not. Statistic: elasticity of clay area (CDLI h x w) on text amount, and a mixture with an area-proportional-to-text line. Nulls: texts re-dealt across tablets of the same kind, labels permuted within text-size bins. Planted corpora recovered (planned share error <= 0.06, AUC 0.86-0.93).
- Ur III control: single-day receipts fit their clay best (0.91); multi-day running logs (0.70) and grand-total summaries (0.72) do not separate. The size test measures "one known transaction vs a list", not "copied vs as-it-happened".
- Grade B: PE clay grows with about the square root of the text (0.40). This is a floor effect plus loose sizing: 0.56 at 5+ lines and 0.65 at 8+ lines. Proto-cuneiform shows the same (0.44-0.49), so this is archaic practice, not specific to PE.
- Grade B: about 42% of PE lists with 5+ lines sit on the sized-to-fit line (re-dealt texts 2-7%). Ur III running logs reach 61%, summaries 76% and receipts 95%. The sized-to-fit PE tablets are an ordinary cross-section: no system, office, seal, total or site is enriched (|z| < 2.1). Grade C: the largest herd lists (P008295, P008759, P008322) were sized for a known length.
- Grade B: totals sit in a reserved place. When entries end on the obverse, the total goes to the reverse although the obverse still had room for about 2.6 more lines (upper estimate).
- Token hypothesis: piles of tokens written as they lay (unexchanged) are killed. N01 > 9 occurs in 4 of 5,048 counts, giving an upper bound of 1% (a planted 1% is found 92% of the time); proto-cuneiform is the same. There is no hand-counting cap at 5: counts 6-9 are common. Grade B. Tokens already exchanged into denominations are not excluded. Grade C side note: N01 = 8 is over-represented against 7.
- Tablet size does not predict count size once text amount is fixed (z -1.8).
- Nothing is read; nothing is cracked.

## The script aged? Blind seriation vs stratigraphy (pe19, 4 Oct 2026; `loops/pe19_final.txt`)
- Idea: order all 1,585 tablets blind by evolving traits (signs, variants, numerals, format) with CA, spectral, TSP, a unimodal latent-trait model, 3,000 random trait subsets and partial CA; hash-freeze; only then score against the CDLI stratigraphic levels (17 Susa Acropole I tablets 17A-16 to 14B, 18 Malyan level 3 vs 2, 12 Sofalin + Ozbaki).
- Calibration fails: a planted drift of 40 born-and-dying traits is invisible (abs rho <= 0.04); only 200 planted traits (a third of all) are recovered. Uruk IV vs III is not seriated in the right direction (best unoriented AUC 0.69, oriented by the fixed rule 0.31). Grade B (method): PE content / office structure swamps any plausible drift.
- Grade C: frozen orders agree with the hidden levels above random (CA: Susa rho +0.47, early-site AUC 0.76, Malyan AUC 0.92), mostly via tablet size; entry length alone does nearly as well. After sign count, CA keeps partial rho 0.43 (Susa) and 0.49 (Malyan), n 17-18.
- No relative chronology of signs or variants can be claimed; the CA ends are format (early) and one content cluster (late).
- Prediction (C): CahDAFI 8 Acropole tablets, earliest to latest, P009433, P009435, P009436, P009432, P009434; their published levels would test it. Not cracked.

## The flock does the arithmetic (pe20, 4 Oct 2026; `loops/pe20_final.txt`)
- Idea: if herd signs name animal categories, their counts must obey flock demography. Eight herd signs were assigned blind to categories by 200,000 random draws plus Gibbs sampling. Each assignment was scored by a herd-biology likelihood with pastoral rate ranges. The search used the 10 herd blocks of MDP 17,096+325+380; 35 records from 22 other tablets were held out.
- **Grade B: the herd counts are coupled like a flock's.** Search p 0.005 against all-entry and within-sign shuffles. Held-out +3.6 log units (p 0.008). A model fitted on the other blocks predicts hidden counts at +0.86 bits per cell (shuffles <= 0.09; planted herds 0.80).
- **Grade B: M362 (the herd header count) and M362~a are the reference counts** that the other numbers scale with. This survives a wether-aware model and the held-out test.
- **Grade B-: the numbers alone split the signs into the scribe's plain and ~a series** (P 0.92; random column splits p 0.033, within-sign shuffles p 0.08). The herd lists MDP 17,085/17,097 track the plain series only (M362 rho 0.75, M362~a -0.52). Grade C: two kinds of animal (for example sheep and goats).
- **The labels are not calibrated.** On Ur III Girsu/Umma herds with known categories, the same search prefers rams/wethers over ewes as "breeding females" and reads kids as males (1.8/7 exact, chance 0.8). 35% of 300 random biologies fit PE as well as the real one. Grade C: M006 is a small male group (0.07 per M362). M367 as young is demoted: it is too few (0.22 per M362) and its posterior spreads over all classes in the wether model. M346 fits no category.
- Predictions to check by collating P008294: rev 6.c M006 = 1 (80%: 0-4); obv 5.a M367 = 2 (0-6). Not cracked.
- **Collation check (`loops/pe20_check.txt`): the predictions cannot be checked.** In the CDLI ATF both cells are fully restored (obv 5.a `[M367] , [...]`; rev 6.c `[M006?] , [...]`). CDLI's photograph shows only edges and uninscribed fragment faces. Scheil's hand copy was aligned with the ATF through the undamaged neighbouring counts, all of which it confirms (obv 2-5, rev 6.a-b, 6.d-g). In that copy both slots fall in breaks of the surface (obv: the broken top-right corner after the line-5 header's 1 ten; rev: the break at the top of the column, before 6.d's 6 tens), and no impressions are drawn. For rev 6.c, the copy also cannot rule out that no M006 entry was written; four other herd lines lack one. No later reading was found; the CDLI text, which cites Dahl 2005 (SMEA 47, no. 100), still gives [...]. Status: open, untestable without new photographs or autopsy of Sb 22286+22480+22534.

## Forensic accounting: counted, measured, estimated or allocated? (pe23, 4 Oct 2026; `loops/pe23_final.txt`)
- Idea: auditors tell counted numbers from estimated, targeted or invented ones by digit fingerprints (round-number heaping, last-digit preference, leading digit, exact repeats). Six fingerprints were computed on each value's canonical notation against a smooth null that keeps the notation, then for every system, sign, office, header, site, position and tablet size.
- Ur III control (77,435 CDLI lines of known type): counted, measured, rule-allocated and computed-estimate numbers separate at 0.91-0.97 (label-shuffled 0.29-0.38; size-matched 0.77-0.98). The textbook rule fails there: computed harvest estimates are less round than measured grain (0.15 vs 0.36 one-denomination). The round, repeated numbers are rule numbers: rations 0.80, yield rates 0.43.
- Grade B: PE counts look counted and PE capacity looks measured. PE counts are about 3x rounder than Ur III counts, but no PE group looks like an Ur III computed estimate.
- Grade B: no PE sign, office (GRAIN, CLASS, BARE), site or final sign carries rounder or more repeated numbers than other numbers of the same system and size (1,004 tests, calibration 0 flags; within tablets 0/624). The scan only sees signs with 60+ numbers of 10 or more (planted 50% rounding 5/5; rarer signs 0/43), so this is an upper bound for about 8 signs.
- Grade B-: mismatched totals are not rounded sums (19 clean tablets, p 0.52; planted rounding found 16/20).
- Grade B: the top-edge numeral (117 tablets: '1(N34)' 91, '2(N01)' 19) is always one denomination, sits on M157-headed tablets 3.8x more often (21% vs 7%, p 4e-10) and never equals the entry sum. It is a constant tag of the tablet type, not a count, measure or target. It explains most of a "rounder numbers on M157 tablets" signal (p 0.002, held-out 17/20).
- Grade C: M157 entries are slightly rounder even without edge marks (p 0.03-0.07). Would support: replication on new M157 tablets. Would kill: p > 0.1 there. The M136 hint was demoted (held-out 0/20).
- Nothing is read; nothing is cracked.

## Counting the unseen signs and predicting the next tablets (pe22, 4 Oct 2026; `loops/pe22_final.txt`)
- Method: the la15 ecology census (Chao1, ACE, Chao2, jack2, Zipf-Mandelbrot ABC, community bootstrap, habitat novelty). Training: tablets published up to 1999. Hashed predictions for 156 later tablets (TCL 32 2019, unpublished Susa and Malyan entries, Sofalin, Ozbaki). Controls: random and contiguous-batch hold-outs, planted repertoires, proto-cuneiform older to newer, Linear A.
- Grade B: about 750 [680-850] base-sign types including compounds; 551 seen. The simple signs form a closed set: about 425 [393-484], with 353 seen. Compounds are the open class. About 2,600 variant graphs and about 8,500 sign pairs. At equal size, PE uses 0.5-0.6 of proto-cuneiform's repertoire.
- Grade A: the entry vocabulary is mostly unseen (1,568 seen, at least 13,900; coverage 0.22).
- Grade B: the predictions passed. New signs: 8 [3-14] predicted, 6 genuine (plus 8 transliteration artefacts). New strings: 39 [33-43] vs 37. New graphs: 38 [27-50] vs 28. Known-sign ranking: AUC 0.92, chance 0.5. The recent tablets behave like a random batch of the old archive (old batches give AUC 0.90-0.91).
- Failed predictions: sign pairs (123 predicted, 96 true) and the known-type count (205 vs 173). Old batches show the same bias (p 0.09-0.74). The 'risky absence' list was killed by a frequency-matched null.
- Grade B: no recurrent site-endemic sign is found outside Susa (0 of 6 non-Susa signs). Sofalin and Ozbaki add no genuine new sign.
- Calibration: Linear A reproduces la15 (32 vs 32). For proto-cuneiform, a new archive (CUSAS) has about twice as many new signs as predicted.
- Out-of-corpus prediction, hashed: 89 untransliterated Tehran Susa tablets will carry 9 [4-15] new base signs (`data/pe22_outofcorpus_predictions.json`).
- Grade C: MDP 26 and the post-1950 batches add fewer new signs than random order would (p 0.003 and 0.05; not length-matched).

## Red ink? (pe24, 4 Oct 2026; `loops/pe24_final.txt`)
- Idea: variant markers (~a, ~b, @g ...) or particular signs flag another accounting state (owed, dead, not delivered), so their entries count against the total (-1), not at all (0) or at a fixed ratio. Every assignment was searched on 49 clean tablets with a written total (23 close with all entries +1): exhaustive over the 10 commonest markers (3^10) and with ratios (5^7), annealing over 61 signs, fitted on half and re-tested on the other half. Nulls: markers re-dealt among signs, totals replaced by random numbers of the same size.
- Controls pass. On 21 Ur III balanced accounts (CDLI) the blind search returns credit -1 and subtotal 0 as the unique best (held-out +2.5 tablets, p 0.03). Planted negative markers are recovered as the unique best (held-out +2.9 to +4.0).
- **Grade A (negative, calibrated): no variant marker is red ink.** No assignment closes a single extra PE total (held-out gain 0.00; held-out closure 47% = baseline), while random totals gain 2-6 tablets from the same search freedom. Grade B (negative, power-limited for rare signs): no plain or variant sign counts against the total either.
- So the ~a series of the herd accounts (pe20: M362 vs M362~a) is a parallel group that is added like any other, not a debit or a shortfall. That inference is grade C, because the herd tablets have no clean written total.
- Grade C: failing totals are off by about one entry. Omitting 1-3 entries rescues 6/26 (random totals 13%, p 0.01), but the omitted entries are unmarked and of no common kind. Negating entries rescues only 1/26.
- Nothing is read; nothing is cracked.
