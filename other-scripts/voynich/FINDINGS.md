# Voynich MS 408: where the armour is thinnest (data-only tests)

Other people's transcriptions are used only as data. Their readings and theories are not used.

## 1. Data

**Sources:**
- Main: ZL v3b (13 May 2025) by Zandbergen, in EVA (IVTFF 2.0). From https://www.voynich.nu/data/ZL3b-n.txt, saved as `data/ZL3b-n.txt`.
- Second reading: Takahashi IT2a, from the same site, saved as `data/IT2a-n.txt`.
- Both files transcribe the same manuscript, so agreement between them shows the result does not depend on one transcriber's reading; it is not replication.

**Parsing (`tools/parse_ivtff.py`):**
- `.` and `,` (uncertain space) both count as word breaks.
- `<->` (a break around a drawing) also counts as a word break.
- For `[a:b]` alternatives, the first reading is kept.
- Rare glyphs `@nnn;` become `?`. Words containing `?` are dropped from the statistics.
- Glyph units: `ch`, `sh`, `cth`, `ckh`, `cph` and `cfh` each count as one sign.

**ZL3b counts:**
- 227 pages, 5,384 lines, 39,019 words.
- By line type: paragraph 35,069; circle 2,398; label 1,198; radial 354.
- 740 paragraphs.
- Currier language: A 11,620; B 24,063; untagged 3,336.
- Hands 1–5: 11,065 / 11,167 / 11,770 / 3,627 / 929.
- Sections (by illustration type): stars-recipes 11,646; herbal 11,582; bio 6,387; pharma 2,589; text-only 2,362; cosmo 2,254; zodiac 1,316; astro 883.
- Clean paragraph text: 34,857 words, 7,025 distinct words. 308 words had an uncertain glyph.
- IT2a: 37,918 words on 225 pages.

**Comparison texts (Project Gutenberg, first 20,000 words each):**
- Latin: Caesar (pg218) and Descartes (pg23306).
- Italian: Manzoni (pg45334) and Dante (pg1000, verse).
- German: Kafka (pg22367).
- Spanish: Cervantes (pg2000).

**Download problems:**
- The gutendex catalogue API timed out, so book IDs were chosen by hand and checked against each file's `Language:` header.
- No medieval Latin herbal was included.

## 2. Tests

**(a) Is it language-like?**

Method: word statistics and letter-entropy statistics on 20,000-word samples. The controls are:
- a character shuffle;
- character bigram and trigram models;
- a Rugg-style table generator;
- a crude copy-and-modify generator.

| text | Zipf | TTR | hapax share | word length (mean ± SD) | h1 | h2 | h3 | adjacent-word MI excess |
|---|---|---|---|---|---|---|---|---|
| Voynich ZL (glyph units) | −1.06 | .23 | .68 | 4.29 ± 1.60 | 3.86 | 2.25 | 2.01 | .16 |
| Voynich ZL (EVA letters) | – | – | – | 4.85 ± 1.76 | – | 2.11 | – | – |
| Voynich IT | −1.05 | .23 | .68 | 4.37 ± 1.55 | – | 2.25 | – | .15 |
| Currier A / B | – | – | – | – | – | 2.33 / 2.08 | – | .09 / .17 |
| six languages | −0.86 to −1.07 | .17–.28 | .60–.64 | SD 2.2–2.9 | 4.01–4.13 | 3.08–3.31 | 2.42–2.64 | .27–.44 |
| character shuffle | −0.66 | .78 | – | – | – | 3.78 | – | 0 |
| bigram model | −1.08 | .33 | .82 | ± 2.97 | – | 2.23 | – | 0 |
| trigram model | −1.08 | .25 | .73 | ± 1.98 | – | 2.22 | 2.00 | 0 |
| table generator | −0.57 | .53 | – | – | – | 2.92 | – | 0 |
| copy-and-modify | −0.39 | .37 | – | – | – | 3.44 | – | .06 |

Verdict:
- **Matches language:** the Zipf slope, the vocabulary size (TTR), and the presence of real word-to-word dependency.
- **Fails:**
  - h2 is far too low.
  - Word lengths are too uniform.
  - Word-to-word dependency is half the language level.
  - Too many words occur only once.
- A trigram model copies every letter-level statistic but has no word-to-word dependency.
- The copy-and-modify generator is crude. It is not a fair test of the published algorithm.

**(b) Line and paragraph position**

Method: the null shuffles word order within each line. The effect is measured as the mutual information between position and glyph, minus the mean of 20 shuffles. Enriched words are tested with a z-test and BH FDR at 5%. Modern prose lines (arbitrary breaks) and Dante's verse lines are the controls.

| text | first-glyph excess | same, paragraph-first lines excluded | last-glyph excess | words enriched line-initial / line-final |
|---|---|---|---|---|
| Voynich ZL | 0.106 | 0.092 | 0.053 | 94 / 56 (of 265 tested) |
| Voynich IT | 0.109 | 0.094 | 0.055 | 87 / 51 |
| prose | 0.001–0.003 | 0.001–0.003 | 0.002–0.004 | 1–15 / 0–4 |
| Dante (verse) | 0.047 | 0.046 | 0.084 | 78 / 121 |
| generators | ≈0 | – | ≈0 | – |

Specific rules:
- **Paragraph starts:** 83% of paragraphs begin with a gallows glyph, against 9% of other lines. 50% begin with `p`/`f`, against 1.4%. 81% of all `p`/`f` glyphs sit on paragraph-first lines, which hold 21% of the glyphs.
- **Line-initial glyphs** (non-paragraph lines): `y` 3.1×, `s` 3.1×, `t` 2.7× and `d` 1.7× are favoured. `ch` 0.23× and `a` 0.09× are avoided.
- **Line-initial words:** `aiin` once against 47.5 expected; `chedy` 7 against 52 expected.
- **Line-final:** `m` 5.9×, `g` 6.0×; `am`, `dam`, `qokam` 5–7.5×.
- **Word length:** line-initial words are longer, but prose lines show the same wrap effect, so this one is not diagnostic.
- **"Extra glyph" check:** dropping the first glyph leaves a common word in 77% of line-initial words, against 61% mid-line. Within each first glyph the gap almost vanishes (for `s`: 0.84 against 0.82). So the gap mostly reflects which glyphs start lines.

Verdict: a strong chink. The line has its own start and end rules, stronger than Dante's verse lines. This holds in A (0.086) and B (0.129) and in both transcriptions.

**(c) Word structure (slot grammar)**

Two measures:
- **Order rigidity:** how often pairs of glyphs keep the same order inside a word (1.0 = always the same order).
- **One-order slot grammar:** a single glyph order is learned on half the text, then tested on the other half.

| text | rigidity | held-out words covered | held-out distinct words covered | unseen words covered |
|---|---|---|---|---|
| Voynich ZL | .878 | .58 | .20 | .12 |
| Voynich IT | .876 | .57 | .21 | .13 |
| six languages | .64–.69 | .18–.46 | .025–.073 | .009–.023 |
| character shuffle | .51 | .13 | .06 | .04 |
| trigram model | .873 | .62 | .21 | .12 |
| table generator | .833 | .43 | .26 | .17 |

The learned order runs roughly `q` → `o` → gallows/`ch` → `e` → `d`/`s` → `a` → `i` → `n`/`l`/`r` → `m`/`g` → `y`.

Verdict:
- The grammar covers 3–8× more held-out distinct words than in any language.
- A shuffle removes the structure entirely.
- A trigram model reproduces it fully, so this is the same fact as the low h2 in (a).

**(d) Sections, Currier A vs B, labels**

Section-specific words:
- 323 of 1,186 tests are significant after FDR. With section labels shuffled across pages, the mean is 16 and the maximum 39 (20 runs).
- Examples: bio `qol` 4.1×; pharma `okeol` 8.2× and `qokeol` 7.6×; astro `am` 5.9×.

Currier A against B, measured as divergence between word or letter-pair distributions (JSD; 0 = identical):

| comparison | word divergence | letter-pair divergence |
|---|---|---|
| A vs B | .53 | .094 |
| A split / B split (pages) | .30 / .28 | .006 / .003 |
| herbal-A vs herbal-B | .61 | – |
| herbal-A split | .38 | – |
| one author split | .29–.36 | – |
| Caesar vs Descartes | .63 | .034 |
| Manzoni vs Dante | .46 | – |
| Latin vs Italian | .90 | .17 |

- 289 of 485 word tests separate A from B.
- At word level, A and B look like two authors in one language.
- At letter-pair level they differ about 3× more than two Latin authors, about halfway to two languages.
- The difference survives inside the herbal section alone.

Labels (1,163 label words, 836 distinct):
- Only 60% occur in running text, against 80% for running-text words of the same length.
- Labels are longer: 4.98 glyphs against 4.46.
- 54% start with `o`, against 21% of text words.

Verdict: sections and A/B are real, strong divisions, and labels are a separate vocabulary.

**(e) Repetition**

Method: adjacent word pairs inside a line, compared with a whole-text word shuffle (10 runs).

| text | identical pairs per 1000 | vs chance | near-identical per 1000 | vs chance | near-pairs differing inside the word |
|---|---|---|---|---|---|
| Voynich ZL | 9.4 | 2.8× | 43 | 2.0× | 32% |
| Voynich IT | 9.6 | 2.8× | 42 | 2.0× | 35% |
| Currier A / B | – | 2.0× / 2.1× | – | – | – |
| six languages | 0–3.3 | 0.00–0.53× | – | 0.5–0.85× | 1–8% |
| trigram model | – | 0.9× | – | – | – |
| table generator | – | 1.3× | – | – | – |
| copy-and-modify | – | 135× (overshoots) | – | – | – |

The most common repeats are `chol chol` (24), `qokeedy qokeedy` (18) and `qokedy qokedy` (16).

Verdict: real languages avoid repeating a word. Voynich seeks it out, and no letter-level model produces this.

**(f) Cipher class**

Method: compare a profile (h1, h2, h3, word length, alphabet size, share of the 5 commonest signs) with candidate encodings built from the six language samples. Distance is in units of the spread between the languages; lower is better.

| rank | class | distance | why it fails |
|---|---|---|---|
| 1 | verbose cipher with re-cut word spacing | 2.6 | h2 2.90 |
| 2 | verbose cipher | 3.0 | h2 2.47, but words 6.6 ± 3.8 long |
| 3 | position-dependent verbose cipher | 3.3 | words too long |
| 4 | simple substitution | 3.7 | h2 3.08 |
| 5 | abjad (vowels dropped) | 4.1 | h2 too high |
| 6 | simple substitution with re-cut spacing | 4.9 | h2 3.5 |
| 7 | homophonic | 10.1 | h1 4.9 |
| 8 | syllabary | 14.1 | hundreds of signs |

Verdict:
- Relabelling letters cannot change h2, which rules out simple substitution of these languages.
- Homophonic substitution and a syllabary are ruled out by h1 and alphabet size.
- A verbose cipher ranks best, but no version tested gets low h2 and short, uniform words at the same time. This is a ranking of hypotheses, not a solution.

## 3. Caveats

- Word breaks follow the transcribers' spacing, and uncertain spaces counted as breaks.
- The comparison lines are modern typesetting, plus a single verse text.
- The copy-and-modify generator is crude.
- All comparison languages are European, alphabetic and later than the manuscript.
- ZL and IT readings overlap heavily, so they are not independent.

## 4. Chinks (ranked)

1. **The line is a coded unit.**
   - Attack: find rewrite rules for line-edge words, using a within-glyph control.
   - Success: 5 or fewer rules bring line-edge vocabulary overlap up to the mid-line level (75%) on held-out pages, in A and B, and in both transcriptions.
2. **h2 shortfall and rigid word order.**
   - Attack: search groupings of the glyph stream into 20–35 units.
   - Success: h2 of 3.0–3.3 together with a word-length SD of 2.2 or more and a language-like Zipf slope.
3. **Repetition.**
   - Attack: fit one copy-and-modify generator, then test it unchanged on Zipf, h2, the line rules and the section effects.
   - Success: it matches all of them, which would argue for a mechanical process. If it fails, the repeats are more likely cipher artefacts.
4. **Labels.**
   - Attack: look for consistent label structure on parallel zodiac/star pages, against a page-shuffle control.
   - Success: it predicts labels on pages not used for fitting.
5. **A vs B.**
   - Attack: find a glyph rewrite that maps A onto B.
   - Success: letter-pair divergence of 0.035 or less, and herbal word divergence near the split level (0.38).
6. **Word-to-word dependency.**
   - Attack: class-based next-word prediction on held-out pages, against a shuffle.
   - Success: word classes predict their neighbours better than the shuffle does.

## 5. Files

- `data/ZL3b-n.txt`, `data/IT2a-n.txt` (transcriptions)
- `data/pg*.txt` (comparison texts)
- `data/derived/*_lines.json` (parsed lines)
- `data/results/*.json` (all numbers)
- `tools/`: `parse_ivtff.py`, `vlib.py`, `gen.py`, `test_a_language.py` … `test_f_cipher.py`
- `test_b_position.py extra` runs the extra position checks.

## 6. Attacks

### V1. Verbose-cipher merge attack on the h2 gap (`tools/attack_h2_bpe.py`, `data/results/attack_h2_bpe.json`)
**Question.** If each plain letter is written as a fixed group of glyphs (a verbose cipher), merging the commonest glyph pairs should restore language values at a plausible alphabet size.

**Method.** Merge the most frequent adjacent glyph pair inside words, repeatedly (byte-pair merging). Record h2 and the mean and SD of word length (in units) as the alphabet grows. Each text is 20,000 words.

**Controls.**
- Positive: Latin (Caesar) and Italian (Manzoni), encrypted with a fixed verbose codebook (each letter becomes 1–3 symbols from 12).
- Negative: an order-3 glyph Markov text trained on the Voynich (same letter statistics, no hidden language), and the self-citation generator.

| text | start h2 | h2 at 25 / 35 / 50 units | word length at 35 units (mean ± SD) |
|---|---|---|---|
| Latin, verbose-encrypted | 2.34 | **3.25** / 3.55 / 3.77 | 5.2 ± 2.7 (plain Latin: h2 3.30, 6.0 ± 2.9) |
| Italian, verbose-encrypted | 2.40 | **3.19** / 3.39 / 3.65 | 4.5 ± 2.7 (plain: h2 3.25, 4.6 ± 2.8) |
| **Voynich** | 2.25 (31 units) | – / 2.48 / 3.12 | **3.6 ± 1.4** |
| Markov-3 from Voynich (no language) | 2.24 | – / 2.47 / 3.12 | 3.8 ± 1.5 |
| self-citation generator | 3.43 | – / 3.77 / 4.10 | 3.9 ± 1.0 |

**Result.**
- The method works on the positive controls. Within the first merges, the encrypted Latin and Italian return to their real h2 (3.2–3.3) and real word lengths (SD 2.7–3.1).
- The Voynich does not. Its trajectory is the same as the no-language Markov text to two decimals (2.48 vs 2.47 at 35 units; 3.12 vs 3.12 at 50).
- Its words shrink to 2.6 units with SD 1.1 before h2 reaches language level. A language never looks like that.

**Verdict: chink 2 is closed for the simple case.** A fixed letter-to-glyph-group cipher of Latin or Italian, with the spaces as real word breaks, is ruled out. On letter statistics the Voynich is indistinguishable from a third-order glyph Markov text.

**What stays open:**
- **The spaces may not be word breaks.** Their word lengths are far too uniform for any language.
- **Some features no letter model makes:** the word-to-word dependency (0.16 bits; Markov texts score 0), the doubled words (2.8× chance) and the line rules.

If there is a hidden language, the evidence for it sits at word and line level, not inside words. Next attacks: chink 1 (line rules) and chink 6 (word-to-word prediction).

### V2. Where the word-to-word link comes from (`tools/attack_wordlink.py`, `data/results/attack_wordlink.json`)
**Method.** For adjacent words inside a line, measure the information between the *last glyph* of one word and the *first glyph* of the next (the junction). Also measure it between whole words. Both are reported as excess over a within-line word shuffle (20×). Each text is 20,000 words.

| text | junction MI | without doubled words | without line edges | whole-word MI |
|---|---|---|---|---|
| **Voynich** | **0.178** | 0.179 | 0.178 | −0.134 |
| Latin (Caesar) | 0.036 | 0.035 | 0.036 | 0.181 |
| Italian (Manzoni) | 0.080 | 0.080 | 0.077 | 0.111 |
| German (Kafka) | 0.046 | 0.046 | 0.046 | 0.328 |
| Italian verse (Dante) | 0.173 | 0.172 | 0.184 | 0.095 |
| Markov-3 (no language) | 0.000 | 0.000 | 0.001 | −0.001 |
| self-citation generator | 0.005 | 0.005 | 0.006 | 0.006 |

**Results.**
- **The Voynich junction coupling is 2–5× prose and equal to Dante's verse.** Dante's comes from Italian vowel endings and metre.
- **Strongest pairs:** `…o r…` 7.0× expected, `…s a…` 5.9×, `…r a…` 3.2×, `…y q…` 1.8× (1,884 cases).
- **Avoided pairs:** `…y a…` 0.14×, `…r q…` 0.30×, `…n q…` 0.46×.
- **Not from repeats or line edges.** It is unchanged with doubled words removed and with line edges removed.
- **No generator produces it.** The letter-level and self-citation generators score zero.
- **Whole-word dependency is negative.** Inside a line, specific word pairs are *rarer* than in a shuffle. The earlier positive figure (0.16, test a) used a whole-text shuffle, so it measured line and section effects, not neighbour links.

**Verdict.** The link between neighbouring words lives at the junction: the end of one word constrains the start of the next. Two things look like this:
- the sound-joining rules of a real spoken or sung language, which verse shows strongly;
- a writing procedure that picks each word's first glyph from the last glyph of the word before.

This is still the strongest feature that no published generator reproduces.

**Next test.** Does the junction rule also hold across line breaks (last word of a line → first word of the next)?
- If the text is continuous language wrapped into lines, it should.
- If each line is a self-contained coded unit (chink 1), it should not.

### V3. Does the junction rule cross line breaks? (inline check, same code as V2)
**Method.** Junction MI (last glyph → next first glyph) for word pairs that straddle a line break inside a paragraph. This is compared with a sample of within-line pairs of the same size, each against 200 shuffles.

| text | across a line break | within a line (same n) |
|---|---|---|
| **Voynich ZL** (n = 3,390) | **0.002 (p = 0.24, nothing)** | 0.174 |
| **Voynich IT** (n = 3,345) | **0.002 (p = 0.27)** | 0.208 |
| Italian prose (Manzoni) | 0.054 (p < 0.005) | 0.086 |
| German prose (Kafka) | 0.036 (p < 0.005) | 0.036 |
| Italian verse (Dante) | 0.024 (p < 0.005) | 0.145 |
| Latin (Caesar, n = 1,916) | −0.004 (n.s.) | 0.045 |

**Result.** In real texts the link between neighbouring words carries on across a line break, weaker in verse but still clear. In the Voynich it **drops to zero at every line break**, in both transcriptions, although the same paragraph continues.

**Verdict: strong support for chink 1, "each line is a sealed unit".** The text is not continuous language that has been wrapped into lines. Whatever process joins neighbouring words restarts at the start of every line. Together with the line-start and line-end glyph rules (test b), this points to a line-by-line writing procedure: a cipher, a table or a notation applied one line at a time.

**Novelty.** The Voynich community knows the idea of the "line as a functional unit". This cross-line junction reset, with prose and verse controls, may not have been measured in this form. Check the literature before claiming it.

**Next.** Model one line as a unit. Do line-start glyph, line length and line-end glyph predict each other? Can a small line grammar generate held-out lines better than a Markov model?

### Control for V2–V3: uncertain spaces (`tools/attack_junction_spaces.py`)
The junction coupling could have come from where transcribers placed word breaks. Junction MI, ZL:

| space type | junctions | junction MI |
|---|---|---|
| certain spaces (`.`) | 28,150 | 0.164 (0.171 on a sample of 2,434) |
| uncertain spaces (`,`) | 2,434 | 0.510 |

IT shows the same on certain spaces: 0.181. The coupling is real at certain word breaks. The uncertain spaces mostly mark places where one word may have been split in two.

## 7. Tests the literature has not run (as far as we know)

### N1. Line quotas and line clumping (`tools/attack_quota.py`, `tools/attack_quota2.py`)
**Question.** Is each line filled to a budget of some glyph class, like a tally, a checksum or a metre? Or do lines differ in make-up?

**Method.** Within each paragraph, the words are dealt back into the lines at random, keeping each line's word count (200×). The statistic is the dispersion of a class's count per line, corrected for line length, observed against the null. Above 1 means the class clumps into some lines. Below 1 means it is spread evenly, like a quota.

| class | Voynich | prose and verse letters (range) |
|---|---|---|
| gallows | **1.12** (p < 0.005) | — |
| e | **1.14** (p < 0.005) | — |
| d | **1.10** (p < 0.005) | — |
| a | **1.11** (p < 0.005) | — |
| r, n, y, l | 1.05–1.09 (p ≤ 0.025) | — |
| m/g | **0.90 (spread evenly, p < 0.005)** | — |
| vowels | — | 0.89–0.97 |
| consonants | — | 0.88–1.08; only Latin s and m reach 1.07–1.08 |

**Result.**
- **In languages, letters are spread across lines about as evenly as chance, or more evenly.**
- **In the Voynich, most glyph classes clump: lines have their own make-up.**
- The one quota-like class is m/g, the line-final glyphs: about one per line, a line-end marker.

### N2. Line bracketing (`tools/attack_linebracket.py`)
**Question.** Does the first glyph of a line predict its last glyph, or the line's length?

**Result.** No. First vs last: −0.003, p = 0.77. First glyph vs length: p = 0.75. Prose and verse are also null.

**Verdict.** Lines are not opener–closer frames. The line-start and line-end rules work separately.

### N3. Two line modes inside paragraphs (`tools/attack_lineflavour.py`)
**Method.** Correlate per-line residuals between glyph classes, against the same re-dealing null (100×).

**Result.** Two kinds of line exist inside the same paragraph:
- a **q-type** (q, gallows, e, d, y travel together: q~d z = +10.8, gallows~d +11.2, q~y +6.9; "qokedy"-like);
- an **a-type** (a, i, r, n: "aiin/ar"-like).

They trade off against each other (q~a z = −5.8, q~i −4.7, q~r −4.7). The strongest single trade-off is **ch against e (z = −10.7)**: a line rich in `ch` is poor in `e`, and vice versa.

### N4. Do line modes alternate, persist or vary freely? (`tools/attack_linemode.py`)
**Method.** Line score = share of q-initial words minus share of words with `ai` or `ar`, demeaned within each paragraph. Lag-1 and lag-2 correlation against line-order shuffles inside the paragraph (1000×).

**Result.**
- All lines: lag-1 −0.098 against a null of −0.151 (z = +2.9); lag-2 z = +2.9.
- Currier A: z = 1.6. Currier B: z = 2.4.
- Lines drift slowly. There is no alternation, so no fixed key schedule such as ABAB.

### What N1–N4 and V1–V3 say together
Each Voynich line:
1. is a **sealed unit**: the word-to-word coupling resets at every line break (V3);
2. has **its own glyph make-up**, in one of two modes, q-type or a-type (N1, N3);
3. has **start and end rules** but no link between its start and its end (test b, N2);
4. ends in about **one m/g glyph**, an end marker (N1);
5. changes mode **slowly**, not on a schedule, from line to line (N4).

No language sample shows points 1, 2 or 4. The best working model is a **line-by-line procedure with a state that resets each line and drifts slowly across a paragraph**: something like a table or wheel setting chosen per line, or each line copied and modified from a seed. A fixed per-line key schedule is ruled out (N4). Simple verbose substitution is ruled out (V1).

**Next.** Fit the two modes as two "tables": can q-type lines be mapped to a-type lines by a small set of glyph swaps (ch↔e, q-…↔…-a-), so that the two modes become one vocabulary? If a few swaps merge them, the modes are two settings of one device. That would be a real crack in the mechanism.

### N5. Can glyph swaps merge the two line modes? (`tools/attack_modemerge.py`)
**Method.**
- Split lines into a q-type third and an a-type third by the N4 score.
- Search greedily for up to 6 glyph rewrites (single glyphs and common pairs, deletion allowed), applied to the a-type lines, that make their word distribution closest to the q-type lines (Jensen-Shannon divergence, JSD).
- Fit the rewrites on half the lines and score them on the other half.
- Controls: an Italian text and a Latin text split by the same recipe.

| text | JSD between modes (held-out) | after rewrites | same-mode baseline | gap closed |
|---|---|---|---|---|
| **Voynich** | 0.414 | 0.412 | 0.266 | **2%** |
| Italian (Manzoni) | 0.385 | 0.383 | 0.305 | 1% |
| Latin (Caesar) | 0.489 | 0.489 | 0.389 | 0% |

**Result: null.** No small set of glyph swaps turns a-type lines into q-type lines. The two modes are different word stocks, not one vocabulary under two substitution settings.

The mode gap is larger in the Voynich (0.414 against 0.266) than in the prose splits (0.385 against 0.305). So the modes are real word classes that cluster by line, as if each line drew mainly from one of two word lists.

**Verdict.** The "two settings of one device" idea, in its simple substitution form, is ruled out.

**Model now:**
- Each line draws its words mainly from one of two word classes: q-words (`qokedy`, `qokeey`…) or a-words (`aiin`, `ar`, `dain`…).
- Word junctions are coupled within the line and reset at the line break.
- The line ends with an m/g glyph.

This looks like a structured record per line. For example, each line could be one entry, with the class marking the entry type, like the Indus slot classes. It does not look like running prose.

**Next tests (not yet run):**
- Do the line classes follow the illustration on the page (plant, star, bath)?
- Do the a-words and q-words take different positions inside the line, like a two-column table?

## 8. Novelty check (web, 3 Oct 2026)
- **V3 (junction coupling resets at line breaks): not new.**
  - "Evidence of Layered Positional and Directional Constraints in the Voynich Manuscript" (arXiv 2604.19762) reports that cross-word suffix dependency works within lines and collapses across line boundaries. Example: before k-initial words, -l is 71.3% within lines and 22.2% across lines. It also finds the left-to-right boundary dependency absent in English, French, Hebrew and Arabic.
  - The "line as a functional unit" idea goes back to Vogt (2012).
  - arXiv 2608.17096 ("A Glyph Is Not a Letter, a Token Is Not a Word, a Space Is Not a Space") covers related ground.
  - Our V2/V3 is therefore an **independent replication**, with prose and verse controls and both transcriptions.
- **N3 (two line modes), partly known.**
  - Two dominant word communities at paragraph level: arXiv 1806.08467.
  - qokaiin-type and daiin-type word clusters: Viridis Green's part-of-speech posts.
- **May be new, but needs a deeper literature check before any claim:**
  - Mode clustering *within* paragraphs against a word re-dealing null (N1, N3).
  - The ch↔e trade-off by line.
  - m/g as a one-per-line quota.
  - The failure of glyph swaps to merge the modes (N5).
  - The verbose-cipher merge test, with encrypted-language positive controls (V1).

## 9. Line-type structure (`tools/attack_linetype.py`, `tools/attack_linetype_extra.py`; full report `data/results/attack_linetype_report.txt`)
Both transcriptions were used. Agreement between them is transcription-robust, not replication.

1. **Mode follows the page. This corrects N3/N4.**
   - The illustration section explains 12% of the variance in line mode (null 1%, z = 18).
   - The page within its section explains another 13% (null 4%).
   - The paragraph adds about 4%.
   - Bio pages and quire M are q-heavy. The effect sits in Currier B (0.15 of variance) and hardly in A (0.03).
   - Scribe hand adds little.
   - With page means removed, the two halves of a line barely share a mode (r = 0.06–0.07; Italian 0.08).
   - **So "each line picks one of two word classes" overstates it.** Mode is mainly a page and section property. The within-paragraph clumping (N1, N3) is real but small.
2. **Order within a line: real but small.** In mixed lines, q-words sit about 5% of the line length earlier than a-words (z = −7 in both transcriptions). That is not a two-column table.
3. **Line templates fail.** Once first and last words are fixed, the order of word classes is nearly free: 0.02–0.03 bits below a shuffle. Dante's verse, as a control, gives 0.075.
4. **The first and last words of a line come from special vocabularies** (z ≈ 30), at 2–4× the prose level.
   - Openers: dshedy, sho, shor.
   - Closers: dy, oly, dal.
   - Lines rarely end in a q-word (8.7% against 16%).
   - The word before a final -m word is an a-word 33–35% of the time (shuffle 24%, z = 6–7).
5. **The self-citation generator reproduces none of this.**

### Does the line-break reset survive once edge vocabulary is accounted for? (`tools/attack_reset_edges.py`)
Junction MI (last glyph → next word's first glyph), n = 1,500 per set, against 200 shuffles:

| pair type | ZL | IT |
|---|---|---|
| across a line break | 0.005 (p = 0.24) | 0.004 (p = 0.28) |
| across a line break, **ordinary words only** (neither in the 60 most edge-enriched openers or closers) | **0.007 (p = 0.14)** | **−0.010 (p = 0.91)** |
| within a line, touching the first or last word | **0.197** | **0.180** |
| within a line, inner pairs | 0.173 | 0.165 |

**Verdict.** The reset is not an artefact of edge vocabulary:
- Inside a line, pairs that involve the opener or the closer are coupled as strongly as inner pairs.
- Across the break, coupling is zero even between ordinary words.

The line is a real production unit (this replicates arXiv 2604.19762 with stronger controls).

**Updated model.** Each line is written as a unit:
- It opens with a word from an opener vocabulary and closes with a word from a closer vocabulary (often an m/g glyph).
- Word junctions are coupled inside the line and not across the break.
- The mix of word classes is set mainly by the page and section, not by the line.
- The order of words in the middle is nearly free.

### Novelty check, part 2: arXiv 2608.17096 (abstract read 3 Oct 2026)
"A Glyph Is Not a Letter, a Token Is Not a Word, a Space Is Not a Space" uses the same ZL transliteration, with prose, cipher and pseudo-text controls. It already reports:
- **low glyph entropy** (2.7 bits against about 3.5), too strong for one-to-one substitution, resolving onto multi-symbol units (cf. our V1);
- **edge-glyph coupling of about 0.2 bits** between tokens, more than any prose control (cf. our V2);
- **token-to-token prediction below every control** (cf. our negative whole-word MI);
- **uncertain spaces that behave like word-internal junctures** (cf. our space control);
- **Voynich-imitating ciphers and self-citation generators that fail** on the edge coupling and the open vocabulary.

**So V1, V2, V3 and the space control are replications of published work.** They agree with it independently.

**Not covered in that abstract or the other papers found so far, so possibly new:**
- the m/g one-per-line quota (N1);
- the ch↔e line trade-off (N3);
- the variance split of line mode across section, page, paragraph and line (§9);
- the edge-vocabulary-controlled reset (§9);
- the verbose-cipher merge with encrypted-language positive controls, as a method.

Our Voynich work mostly confirms the current state of the art. It has not cracked anything.

## 10. The line-initial chain (v6, 4 Oct 2026; `loops/v6_final.txt`)
No reading order beats left-to-right rows: columns, diagonals, spirals and boustrophedon carry 0.002 bits or less of junction information against 0.17–0.19 along rows, while Latin written in those orders is found at z 79–265. One new effect: **the first glyph of each line predicts the first glyph of the next line** (0.113 bits, z 19, in both the ZL and IT transcriptions; an encoded Latin acrostic gives 0.152, plain prose 0). It holds out across folio halves (z ≈ 10), in Currier A and B and in hands 1–3, is one glyph wide (absent in the rest of the opening word, other columns and the right edge), avoids repeating the glyph above (8.5% vs 14.8% by chance) and prefers certain successions (q→ch/sh, o→q, d→q). It does not follow Voynichese glyph rules, so it reads as a margin rule rather than vertical text. Grade C: a line-marker sequence. Novelty: vertical similarity of nearby words is known (Timm 2014) and 'vertical keys' were proposed but not tested; no published test of line-to-line initial transitions was found.
**Follow-up (v10, 4 Oct 2026).** The chain is a marker, not a hidden channel. About 10 glyph types carry it (d y o q s t sh ch l p) at 3.1 bits per line, only 0.09 bits of which is predictable from the line above; memory lasts 2–3 lines and it restarts at every paragraph. Killed: a counter or rotating key (best fixed order 23% of steps vs 20% chance; a planted counter scores z 127), a key stream acting on the line it opens (36 rules, all z ≤ 1.7; planted keys z 107–160), and a Latin, Italian or German acrostic (beats shuffles but not first-order surrogates; planted acrostics beat surrogates at z ≈ 9). No second-order memory beyond the paragraph's glyph mix (z 1.2). Grade C stands: a line-marker sequence with first-order succession rules.

## Text against drawing (v13)
- Drawing complexity does not predict text length or vocabulary on the same leaf (r ≈ 0). A planted link of r ≥ 0.5 was caught 6 times out of 6.
- Busier drawings go with fewer ch/sh (r −0.39), but this is drift: the next page's drawing gives −0.35.
- Only one link survives: busier root-zone linework goes with fewer benched gallows. The reverse side predicts it more strongly than the page's own text, so it is probably ink showing through. Grade C.
- Links weaker than r ≈ 0.35 cannot be excluded.

## Hidden-map walk (v11)
- A low-dimensional hidden map does not explain the text. The Voynich map fit (0.57) lies in the range of Latin, Italian, Spanish and a random graph. A planted grid scored 1.9–2.7 and was recovered.
- No walk signature appears. Words two steps apart are no closer than in shuffled lines, and a one-way route is ruled out.
- The line-initial glyph chain held up on unseen pages (z 13–17).
- A small table of (last glyph of a word, first glyph of the next word) explains about 85% of what any 2-D word model learns. For Latin the figure is 5%. This junction effect is the strongest structure found so far between words. Grade A as a statistic; its cause is open.

## Lengths as the message (v15, 4 Oct 2026; `loops/v15_final.txt`)
- **No hidden stream in the lengths (Grade A, negative).** Word lengths (glyph units, EVA characters, pen strokes), uncertain-space gaps, words per line, line totals mod 20–30 and length differences were searched with ~10 M random codebooks and ~16,600 annealing chains. The search used 8 schemes and letter models for Latin, Italian, German, Old Czech, Hebrew, Occitan and Greek, with held-out halves and the same search on shuffled nulls. Best Voynich result: +0.03 bits/letter over the best null. Caesar planted in the lengths of a Voynich-like text scores +0.3 to +1.3, and is readable when the code is lossless. The word lengths of ordinary Latin and Italian prose score +0.1 to +0.4.
- **Lengths carry no structure of their own.** Repeated length runs are at the shuffle level (z ≤ 1.4; planted messages z 82–410). There is no length link across line breaks (z −1.1; planted +56). The first glyph of a word carries as much word-order structure as its length (0.018 vs 0.015 bits). In a text whose message is in the lengths, the glyphs would carry nothing.
- **Side results.** Uncertain spaces (`,`) bunch together and become more common toward line ends (6% at gaps 1–4, 16–17% at gaps 10 and later), which looks like crowding (Grade B). A length rhythm at period 8 matches the line length, i.e. layout (Grade C).
- **Method caveat.** A homophonic key can bend any language model onto a structured stream: planted Latin also "decodes" as German, Czech or Greek at nearly the same margin. Length-code searches therefore detect "message-like", not "which language", and must be calibrated on natural-language length streams.

## Are the spaces lies? (v16, 4 Oct 2026; `loops/v16_final.txt`)
- **No hidden spacing.** 80,000 random mechanical re-spacing rules and free annealing (MDL with an adaptive, bigram-spelled lexicon; held-out lines) never beat the written Voynich spaces by more than they beat the spaces of a Markov-2 resynthesis (-0.08 to -0.17 bits/glyph against -0.12 to -0.17). Planted mechanical spacings on Latin and Italian are beaten 2-3x harder and the search moves toward the real words. The best rule is not transcription-robust (ZL merges words, IT splits them). Counting rules ("every k glyphs") are killed. Grade B, negative.
- **Voynich words do not pay for their lexicon.** On held-out lines the written words cost 0.04-0.07 bits/glyph more than no spacing at all (a glyph-bigram code), while Latin and Italian words save 0.4-0.65. A Markov-2 chain's "words" behave like the Voynich's. Spaces are better predicted from 2 glyphs each side than Latin or Italian word breaks (residual 0.19-0.22 of their entropy vs 0.37), exactly as well as Markov-2 spaces. Grade B.
- **What survives re-spacing.** The junction effect keeps ~70% of its size (0.19 -> 0.13 bits, still 2.5-3x Latin/Italian), so it is not an artefact of where spaces were drawn. The line-initial chain lives in the first glyph and is untouched. Annealed units straddle a written space 10% of the time (Latin 1.5%), at r|a, l|ch, s|a, l|k, but the Markov-2 null proposes the same edits.
- **Prior art.** Unsupervised (BPE) re-segmentation recovering ~90% of the written word edges has been reported on voynich.ninja (thread 5049), and rule-placed spaces have been suggested there. The calibrated rule search with planted and Markov nulls, and the "words do not pay for their lexicon" MDL comparison, were not found.


## Interleaved texts (v12)
- Voynich lines are not two or more texts woven together. In woven text the word two back predicts 2.8–6.6× more than the previous word. Single texts give 0.09–0.35, and Voynich gives 0.16–0.18 in both transcriptions. All planted weaves were found. A hidden-switch weave model loses to a plain previous-word model (−0.14 to −0.40 bits). Grade B.
- The enriched neighbour swaps are l/r, k/t, l/o and ch/cth, and they sit inside one stream. a/o is not enriched.
- New lead, grade C: ch and sh words occur one word apart (shedy … chedy) but never adjacent. The pattern holds against a position-preserving null (z 4.4 in ZL, 6.0 in IT), usually with a q- or o- word between them. It is small, about 30 extra pairs.

## Words rolled with dice? (v14)
- No dice device fits the slot frequencies better than chance. Eight devices were tried, from one die to two astragali. Percentiles ran from 0.14 to 0.95. Planted dice corpora were identified, 7 of 8 exactly.
- The slots within a word are dependent (mutual information 0.12–0.30 bits, all 60 tests p ≤ 0.005). Rolled words could not be: planted dice give 0.000. A mostly-rolled text is ruled out. Grade A.
- Lead, grade C: a 16-class model leaves less residual slot dependence in the Voynich (0.045 bits) than in verbose Latin or Italian (0.10–0.12).

## Ghost of an exemplar (v17)
- No lost exemplar line length is detectable in any section. About 16,800 widths and phases and an exact flexible-width search all gave corrected p = 0.54–0.98. Planted Latin dittography was found at width 34 (true 33.2). The text was composed on the page, or copied line for line.
- A junction resynthesis draws each next word from real words that follow a word with the same last glyph. It reproduces the mid-line rates of line-initial-looking words (8.05% vs 8.07%), line-final-looking words and their pairing exactly. Only immediate word repeats are in excess (1.15% vs 0.68%). Grade A.

## Is anything sorted? (v19, 4 Oct 2026; `loops/v19_final.txt`)
- **No glossary, index or name list in glyph order (Grade A negative for runs of >= ~25 entries; B for 12-24).** Every page, 5-line window, line-initial, line-final and paragraph-initial run, label set, margin column (f66r, rings, pharma) and ~23,000 reading-order windows across page breaks were searched for the glyph order that sorts them best (first-glyph, two-glyph, full dictionary and reverse-dictionary keys; random orders, iterated local search, GA check). Nothing beats shuffled copies searched the same way, or unsorted Latin prose. The same machine finds planted sorted pages (z 12-38) and glossaries (z 7.6-8.0 at 30 headwords) and gives back their orders, and finds the alphabetical headwords of Isidore, Etymologiae X under an unknown substitution (pooled z 17.7) with most of the Latin alphabet recovered from local windows.
- **No writer's alphabet can be read off the text**, since nothing is sorted.
- **Side result, q-early / a-late gradient (Grade B, description).** One glyph order fits all pages weakly (z 4-10, held-out pages predicted, both transcriptions, absent in Latin). It is not an alphabet: q-initial words sit higher and a-/l-initial words lower, within pages and within paragraphs, after removing paragraph-first and -last lines and holding line length fixed (q z -3.0 to -5.9; a +3.4 to +6.1). Lines also get shorter down the page (r = -0.18). Cause open. Would kill: the gradient vanishes within single sections on held-out pages.
- **Prior art.** Eyeball remarks that the pharma labels are not alphabetical exist (voynich.ninja); no systematic order search with planted and real-alphabetical controls was found.

## Written to a quota? (v20, 4 Oct 2026; `loops/v20_final.txt`)
- **No quota per page, paragraph or bifolio (Grade A negative for hard quotas).** About 2,150 features per corpus (every glyph, bigram, word class, top word, 1,200 random glyph sets, 500 random word sets) were tested for under-dispersion against a redeal null that keeps unit sizes and word slots, with family-wise correction. Planted caps and quotas are found (z -8 to -10); Voynich gives 0 hits above the line level in ZL and IT. Voynich pages are 2-5x more uneven than natural texts (median dispersion ratio 3.6 per page, 6.8 per bifolio; Latin, Italian and English 1.1-1.6 and 1.2-2.2). A soft quota weaker than this unevenness cannot be seen (power limit, a half-strength planted quota was missed).
- **No conserved linear combination per page.** A generalized-eigen search and 20,000 integer combinations recover a planted balanced pair exactly, but find nothing per page in the Voynich or the references.
- **Per line, Voynich lacks the evenness of natural lines (Grade B).** Natural lines are under-dispersed (fixed width, function words spread by syntax: 43-551 hits); Voynich lines are not, and their glyph totals vary more than chance (ratio 1.23 vs 0.69-1.01).
- **One line-level conserved balance (Grade B).** It is the count of y-final words against the light glyphs (y, i, o, e, a, d, l), net of the glyph total. On unseen pages it varies only 0.72 (ZL) / 0.76 (IT) as much as chance (p < 0.007). Its strength matches Latin prose (0.69) and Gadsby (0.76). Line-flavour mixtures (K = 1-8) and filling lines to a glyph width do not reproduce it; a word-bigram chain does (0.76). It is a by-product of word-to-word dependence (the junction rule), not a quota.
- **Lead (Grade C).** op- initial words are the least over-dispersed word class across pages in both transcriptions. Would kill: the evenness vanishes without paragraph-first lines.
- **Prior art.** FINDINGS N1 tested 8 glyph classes per line within paragraphs; no published multi-level under-dispersion or conserved-quantity search with planted quotas was found (web search, 4 Oct 2026).


## Random programs (r2, 4 Oct 2026; `loops/r2_final.txt`)
- ~8.9 M random and evolved small programs (copy, counters, lookup tables, L-systems) scored by held-out description length over Kneser-Ney. A planted table-and-grille text is recovered (+24.7 mbit/token, the program reads the grille row from the line number); Linear B and, once line ends are given, Latin are not explained.
- Voynich: no program beats KN-3 on held-out folios beyond what the same search gets from Markov-resynthesised text (real <= +1.2 mbit/token, layout given -0.8; Markov null +1.4). Grade A (negative, for programs of <= 15 operations over these primitives): the glyph stream is not the output of a short generator of this kind.

## The pen remembers? (v18, 4 Oct 2026; `loops/v18_final.txt`)
- **No reset at pen dips (Grade B, negative).** Ink darkness was measured for 11,599 words on 25 text pages (Yale IIIF images, scratch only) and dip proxies found under 6,048 settings. Word choice, length, glyph junction and repeats flow across the darkenings as in 12th-15th c. Latin manuscripts (HTR-United CREMMA-Lat, 9,583 words). A planted generator that restarts at every dip is found 3/3 (at half the dips 2/3). Primary reset score +0.19, p 0.16.
- **Dip spacing.** A darkening every ~13 words (IQR 9-20) in both scripts, not concentrated at line starts. The intervals are no more regular than random marks in words, glyphs, ink or pen travel in either script, so these are noisy dip proxies, not a pen-load clock. That caps the power of all v18 tests.
- **Lead (Grade C).** In the wider search, a darkening looks slightly like a line start (weaker junction, more line-initial next word). It replicates across page halves and is absent in Latin, but falls to p 0.12 once edge glyphs are regressed out.
- **Lead (Grade C+).** Same-line words 2-4 apart written between two darkenings are slightly more alike than words across one (edit similarity z 3.2 at the surrogate floor, ~1 point). It survives edge-aware residualisation, both page halves and removal of repeats; content-only darkness makes at most half; Latin z 0.9. Would kill: z < 1 on new pages or in a known-language manuscript with short repetitive words.
- **Prior art.** Re-inking weight changes are described qualitatively (Stolfi's notes, voynich.ninja); no measurement of dip points against the text was found.

## Corrections as a rulebook (v24, 4 Oct 2026; `loops/v24_final.txt`)
- **Census (Grade A).** Of eight public transliterations, only ZL marks visible changes: 24 sites (20 `corr?`, 1 `erased?`, 3 notes on added or overwritten glyphs). Only 4 have a recoverable before-state, and 2 of those have unknown direction. The 817 ZL `[a:b]` variants are reading ambiguities, not corrections.
- **Method (controls pass).** The before-form is scored against random single-glyph edits at the same place, over ~1,000-2,000 candidate rules with family-wise correction. A planted generator that repairs only illegal slips shows legality repair (junction z +4.9) and no identity repair. A planted language that repairs non-words shows identity repair (z +2.5) and no legality repair.
- **Real Latin breaks the textbook picture.** The control is 143 single-letter corrections in SCTA diplomatic transcriptions of Plaoul. The scribes' slips were legal and often other real words (word ratio 2.45 vs random edits), fixed for sense. A meaningless generator copied from an exemplar gives the same sign (1.19). So the sign alone cannot separate meaning from mechanism. About 30-50 genuine before/after pairs are needed.
- **Voynich (Grade C).** With 4 pairs, no rule is enforced beyond the null (FW p 0.58). The befores are 3/4 real words and 4/4 legal. This leans copy- or language-like, but ordinary ZL ambiguity pairs do the same 45% of the time.
- **Images (negative).** A local ink-anomaly detector finds planted double-inking (87%), but none of the 5 noted corrections on 25 pages. Its flags sit at word edges where the pen lands and lifts.
- **Lead (C).** Noted corrected words are rare: 10/20 occur elsewhere vs 0.79 for same-length tokens. Transcriber selection is the likely cause.
- **Prediction.** Reading at least 50 erasure or overwrite pairs from the vellum would decide copy-like vs language-like repair. Transcriber variants cannot, since they score Latin-like by construction (2.53).


## Forge it, then catch the forgery (v21, 4 Oct 2026; `loops/v21_final.txt`)
- **No generator passes (Grade A).** Twelve forgers (the v17 junction resynthesis, slot generator, page-topic, line-width, a GRU language model, Timm-Schinner-style copy-and-vary with real edit operations, survivor-targeted rules) were each told apart from the real pages by ~52,000 page discriminators cross-validated by page: best AUC 0.926-0.934 (IT2a 0.928). Controls: forgery vs forgery 0.48-0.52; forger output re-forged by itself 0.54-0.68 (floor); a planted long-range re-use found (0.70-0.78). Latin (Isidore XVI-XVII) and Italian (Brumati 1844) herbals forged the same way are caught at 0.84-0.93.
- **What catches it is not topic (Grade B).** The herbals are always caught by topic re-use (low type-token ratio, words of a paragraph's first line re-used 2-4 lines and paragraphs later). The Voynich never is, in any forger generation; its paragraph-first line is re-used LESS than in forgeries. Its survivors are local copy-and-vary and layout effects: line-width coupling, the line-initial chain, repeats inside a line, near-repeats 2+ lines apart, glyph echoes two words on, word-length autocorrelation, m/g per line, word-final glyph mix shared by neighbouring lines (not a vertical grid). Each closes when one local mechanism is added, and another surfaces. Persistent: near-repeats 2+ lines apart (z +7 to +9) and runs of ch-words vs sh-words (z -4.6 to -5.6).
- **Reading.** Generator-like in kind, but beyond every single known generator: it needs a stack of 6+ local rules and still leaves residue. Leads (C): near_far as scribe variation of words 2+ lines back (would support: a page-wide self-citation window closes it without raising the floor); ch/sh runs as a persisting pen or table state.
- **Learned models.** A CNN on lines and paragraphs barely beats the junction resynthesis (0.54-0.59) but also misses a planted long-range re-use, so it says nothing beyond ~9 glyphs.

## Paragraph program (v22)
- A paragraph is an opening line followed by an unordered body. Word order inside lines adds 33 millibits per word (z ≈ 11). Shuffling lines 2..n while keeping line 1 in place costs nothing. Flexible stages add only 5–11 millibits per word over fixed position bins. An Italian herbal shows a real multi-stage program (+73). Grade B.
- Opening line: 69% of its words contain gallows. sh- words are about 3× as common there, and q- and a-/l- words are rarer. This does not explain the v19 drift.
- Currier A and B share the program: each carries 80–90% of the other's order signal. Grade B.

## Glyphs built from features (v25, 4 Oct 2026; `loops/v25_final.txt`)
- Glyphs that share strokes behave alike: shape-vs-context Mantel r +0.28 to +0.44 from font images (no hand decomposition) and +0.39 from hand strokes, in ZL, IT, Currier A and B, all at the permutation floor and frequency-controlled. Hangul jamo (featural) +0.25 to +0.53; Latin and Greek ~0. Grade A.
- Compositional: an unseen glyph's behaviour is predicted from its strokes (LOO cosine +0.23, p 0.002, like Hangul). The bench (gallows -> benched gallows) and the p/f crossbar act as additive features (parallelogram cosine +0.95 and +0.59). Grade B.
- Carriers: tall strokes (gallows legs, ascenders), the bench, minims and tails; closed loops carry nothing. Grade C as sound features.
- Kill attempt (v28): handwritten manuscripts transcribed as graphic units do show a weaker link (Castilian, German, Latin with abbreviation marks and minims as units: about +0.1 to +0.2, best +0.26 on one model for a Gothic hand split into minims; printed fonts ~0), so the right null is not zero. The Voynich (+0.28 / +0.30 / +0.43) stays 1.5-3x above every rival, including Ethiopic, and keeps the link under every EVA segmentation (benches split: +0.37 / +0.34 / +0.38). Positional and free variants cannot make it (planted variants give ~0). v25 survives. Grade A (survives), B (size of margin). 15th-16th-c. cipher scripts untested (no open corpus).
- Not misreading (needs ~10% confusions; ZL-IT differ on 1.1%; consensus text keeps it), not word position (partial r +0.25). Stroke-edit copying from shape-neutral seeds builds part of it but 0/27 generator settings reproduce the whole profile. Grade B.
- Tandem swap pairs (k/t, l/r, ch/cth) are NOT one stroke apart beyond chance. Grade C, demoted.


## Features do not spread (v29, 4 Oct 2026; `loops/v29_final.txt`)
- No stroke feature spreads through Voynich words like vowel harmony. A blind search over ~2,000 stroke features per text puts the textbook harmony feature on top in Turkish, Hungarian and Finnish (excess agreement +0.4 to +0.8 over a Markov-2 null, left-to-right) and recovers planted harmony down to +0.2; the Voynich has 0 stroke features above the corrected threshold (ZL, IT). Grade A (negative).
- Residues: two gallows in one word tend to share the bench (IT strong, ZL weak, 8-12 pairs; Grade C); gallows avoid each other (dissimilation, Grade B, the one-gallows template).
- Across word junctions the Voynich is strongly dependent (hundreds of features; all controls ~0), mostly in Currier B, not made by copy-and-edit generators. It is a pairing, not agreement: y.q- 1.57x, n.o-/n.ch- 1.4x, while y.y- 0.59x. No sandhi: edges do not adapt to the neighbour (conditional MI 4-20x below Latin, Turkish, Hungarian, Czech; planted sandhi recovered). Grade B.
- Glyphs that share strokes attract only weakly across junctions (r +0.21, p 0.04, not significant after correction). Grade C, unsupported.

## Grow the script in a box (r3, 4 Oct 2026; `loops/r3_final.txt`)
- 62,000 simulated societies evolve scripts for ledger or text worlds; ABC on blind statistics with Ur III, LB, Latin, Italian and shuffled controls.
- The Voynich fits a text world in a ~15-25-sign system: long words of tight length, almost every word ending in a 1-4-sign end set, and a marker sign opening ~30-40% of lines (grade C; marker rate is identifiable, Latin 0.03). The box-1 guess that it needs homophone/null secrecy was killed (0.82 -> 0.06-0.14) once positional grammar was allowed.
- No world reproduces the coupling across the word space (last sign -> next word's first sign, +4.7 sd). Held-out predictions 1/4. Not cracked.

## Arrow of time (v23)
- The method works on its controls. Hebrew in reading order and in display order give opposite signs, and planted reversed Latin flips sign.
- Glyph level, grade A: irreversibility is 2–4× that of any language (4.4 vs 1.5–2.5 bits). An order-3 glyph Markov chain reproduces all of it, so it is local spelling, not language.
- Word level, grade B: the Voynich arrows are spelling and word-junction arrows only. None rides on word frequency (0 of 378 probes, against 48–73 in Latin, Italian and German). There is no arrow two words apart.
- Direction, grade C: the Voynich compresses slightly better reversed, while every language compresses better forward. The effect is weak, and the glyph chain also produces it.
- A paragraph-level "introduce, then reuse" asymmetry was killed: it comes entirely from paragraph-first lines.

## Evolved scribe (v26)
- Adversarial evolution of a modular generator brought detector AUC down to 0.80 (ridge) and 0.84 (boosting). That is below every v21 forger (0.93) but well above the floor (0.52). The Voynich did not reach the floor.
- Grade B: the minimal genome has four mechanisms, and none can be dropped: a rich junction key, line-width filling, the line-initial chain, and paragraph drift (each line's share of q-/a-/l- words follows the line above). Two restarts found them independently.
- On held-out folios the generator fixes 7 of 11 v21 survivor statistics. It predicts glyphs worse than n-grams (2.21 vs 2.03 bits).
- Grade C: even the true genome of a planted generator is caught, and a Latin herbal reached 0.84. So "generable" cannot be certified this way, and the Voynich-versus-herbal contrast is not shown.

## The dialect ladder (v30, 4 Oct 2026; `loops/v30_final.txt`)
- Method: the cheapest rewrite (<= 30 context-sensitive glyph-cluster rules) that turns Currier A into B, scored on held-out pages, put on a ladder of real 1350-1450 pairs run the same way (Commedia copies, Bavarian/Alemannic/Ripuarian from ReF, Latin/Italian from CATMuS, Czech before/after a digraph-to-diacritic spelling change, cipher key changes, unrelated languages).
- Distance, grade B: A->B (0.46 above the floor) sits with Bavarian->Alemannic (0.45), above two scribes copying the same work (0.26), below Latin->Italian (0.78), a spelling reform (0.94) and any cipher key change (1.2-1.9). Random rewrites help A->B as rarely as a dialect pair (1%).
- Shape, grade B: at that distance A->B closes twice as far as any dialect pair (0.47-0.67 vs 0.27-0.35), near a spelling reform (0.64-0.69). In every run the top rules turn A's -chy/-chol/-chor/-or endings into B's -chedy/-chdy/-edy and cut word-initial cth. Edge and interior rules help equally, unlike a letter key or reform (context-free rules 2x better). Learned on herbal pages, the rules carry over to unseen sections.
- B's hands 2 and 3 are at null distance (grade A in this metric). A/B looks symmetric (grade C).
- Reading: A and B are two conventions of one system at dialect distance, a regular ending shift, not two languages or two keys. Not cracked.


## Currier A to B as a dialect ladder (v30)
- The cheapest rewrite turning A into B (up to 30 context-sensitive glyph rules, scored on held-out pages) has raw distance 0.46. Calibration pairs run through the same pipeline: Bavarian to Alemannic German 0.45, two scribes copying the Commedia 0.26, Latin to Italian 0.78, the Czech spelling reform 0.94, cipher key changes 1.2–1.9, unrelated languages 1.3–1.6. Null splits of A against A give 0.00–0.04. Grade B: A and B sit on the dialect rung.
- The rewrite is unusually regular. It closes 47–67% of the gap, against 27–35% for dialect pairs. In 4 of 4 runs the top rules turn A's -chy/-chol/-chor/-or into B's -chedy/-chdy/-edy and cut word-initial cth. Rules learned on herbal pages carry over to unseen sections. Grade B.
- The rewrite is nearly symmetric and is not a letter-for-letter key. Grade C. Hands 2 and 3 within B are at null distance.

## Calendar beat (v32, 4 Oct 2026; `loops/v32_final.txt`)
- Quire 20's starred paragraphs (339 star units in ZL, 288 paragraphs in IT2a) show no period at 7, 12, 27-30, 36 or anywhere from 2 to 40, by folding, autocorrelation combs, a circular HMM with slips and a gap-aware fold, or a day-counter sawtooth (family-wise p 0.18-0.96). Non-starred sections look the same. Grade A negative for written dating at entry openings.
- Controls: a real calendar (Ado of Vienne's martyrology, 355 day entries) is caught at z 45-64 when its dominical letter or Roman date is written, and missed when only its text is kept. Planted day words in at least 20 % of openings are caught at z 17-32. Day vocabulary in 1-4 % of words and length changes of 10-20 % are missed. So an undated or sparsely marked calendar stays open (Grade C).
- The star drawings themselves alternate dotted and plain (comb P2 z 19.6). The 7/8 points and the dark/light colour do not cycle, and the text does not differ between dotted and plain, 7- and 8-point, or dark and light stars (Grade B).

## The lines were cards? (v27, 4 Oct 2026; `loops/v27_final.txt`)
- **No hidden line order (Grade B, negative).** Body lines were treated as shuffled cards and re-sequenced for maximum line-to-line continuity (junction model trained only on within-line word pairs; shared and near-identical words), never looking at the written order. The method puts Italian herbal prose back at 4x chance (Latin 1.6x) and sees planted continuity in >= ~30% of line joins without the written order. Voynich paragraphs (ZL3b, IT2a) sit inside the null band of Markov resyntheses, word-shuffled lines and cross-page lines on every order-blind statistic. Upper bound: continuity across < ~30% of line joins.
- **No cards dealt across pages (Grade B, low power).** A whole-book deck of 3,365 body lines (SA, 1.5e8 moves x 3 restarts) gives the Voynich no more chains, same-page, bifolio or adjacent-page links than its nulls. Caveat: even prose and planted cross-page orders are re-found only as scattered links (<2%), so the book-scale test is weak.
- **Two witnesses (Grade C, underpowered).** ~860 clean random junction hypotheses x 11 corpus versions: junction and line-interior witnesses never agree on hidden successors in the Voynich; Italian shows a small agreement (+0.28 z), Latin and a moderate plant none. A shared tie-break seed made one hypothesis 'agree' at z 15 (caught, excluded).
- **Side result (Grade B).** The written order has a small lexical adjacency excess (neighbouring lines share words; z 9.9 vs 4.8-5.4 in resyntheses) but no junction flow across line breaks (z 0.7; Italian prose 74, Latin 21). Lines do not run on into each other, in the written or any recoverable order.
- **Reading (Grade C guess).** With v22, each body line is a self-contained unit and was written as one. Would support: entry-initial vocabulary on every line; would kill: words demonstrably split across line breaks.


## The texture of a spell (v31, 4 Oct 2026; `loops/v31_final.txt`)
- **Corpora.** 10 voces-magicae corpora (8,681 non-lexical tokens pulled by a lexicon filter from 40+ grimoire transcriptions on esotericarchives.com, e.g. Liber Juratus, Ars notoria, Clavicula MSS, Raziel, Sword of Moses, Romanus-Büchlein, Grimorium Verum, spirit catalogues, and from an English PGM compilation), 37 hand-written gibberish writers (Gaskell & Bowern 2022 data), 48 natural-language corpora (ReF Bavarian/Alemannic MSS 1350-1450, CATMuS, Old Czech, Isidore, Pliny, Leechbook, 20 NT languages), 14 conlang corpora, 14 non-Voynich generators. Test objects: Lingua Ignota nouns, Hélène Smith's "Martian" glossolalia, Trithemius's Steganographia conjurations, v21 forgers. 32 alphabet-free features per 100-token sample.
- **Voces magicae are a real, easy-to-learn texture (Grade A).** Held-out grimoires are classed as magic 10 times out of 10, and 99% of the time by randomly chosen classifiers. Natural languages passed through the same filter are never classed as magic (0 of 11).
- **The Voynich is not a spell (Grade A within these corpora).** The magic posterior is 0.000 in both transliterations. A dedicated spell detector (100% recall, 2% false alarms) gives 0.0000, which is lower than 83% of non-magic corpora (p 0.007 against random pseudo-classes). 0 of 252 surviving random classifiers vote magic.
- **The Voynich looks most like a generator (Grade B).** 83% (ZL) and 86% (IT) of 252 random classifiers that survive held-out testing call it a generator; the rest split between language (7-9%) and gibberish (5-7%). The null with permuted labels leaves no survivors. Its three most extreme features are unit-order rigidity, unit-position MI and word-family density. Only Pinyin and copy-and-vary or wheel generators reach them. Single-model posteriors are generator 0.45-0.53 and gibberish 0.33-0.40, a split the permutation null cannot separate.
- **Sections differ (Grade B).** Stars (SB) leans gibberish (0.57), biological (BB) has the largest language share (0.30), pharma and herbal A lean generator (0.56-0.61). The spell score is 0 in every section.
- **Spell words put through a copy-and-vary procedure do not fit either (Grade B).** 600 random procedures fed four vocabularies were tested. Planted targets were traced back to their vocabulary 17 times out of 20 (magic 4/5). The Voynich picks Latin-type words 78% of the time, Pinyin 19-22% and spell words 0-3%. Its best fit (1.65-1.74) is worse than the planted median (1.25). Whether the source vocabulary is Latin-type is a Grade C lead.
- **Side results.** The Lingua Ignota list scores as magic (1.00), as a list of coined names should. The Steganographia conjurations, a cipher dressed as spells, score 0.69 on the spell detector: dressing works on texture. Hélène Smith's Martian scores as language (0.41).

## Stem and ending grid (v33)
- Grade B: the Voynich stem × ending grid looks like what its own glyph-trigram, slot and v26 generators produce. It sits at 0.9–1.0× its block-to-block spread from them. Latin sits 2.6× away from its own generators, and a classifier separated every language from its fitted generator (28 of 28).
- All 7 languages leave 34–55% of the cells their own trigram model fills empty (gap ratio 0.45–0.66). The Voynich gap ratio is 0.95–1.10: it has almost no lexical gaps. This holds in both transcriptions, in A and B, and across 136 segmentation rules.
- Limit: a planted paradigm on 15–30% of tokens was not detected, so this cannot rule out an inflectional paradigm. Gap closed (v35): invented scripts with arbitrary shapes (Cherokee, Deseret) and real cipher alphabets (Copiale, Borg) are null to weak (-0.01 to +0.18); featural inventions are positive (Tengwar +0.39 to +0.54 but only +0.08 to +0.20 without its vowel marks; Canadian syllabics +0.13 to +0.21 hand; Shavian +0.10 to +0.13), so the Voynich (+0.28 to +0.43) is not explained by being a designed script or cipher (A). New: its most similar glyphs (gallows k~t, p~f, benched forms, ch~sh) are near-interchangeable in context (index 0.81 vs 0.36 frequency-matched, like a planted homophone cipher, unlike any featural script at 0.17-0.37), while the graded link survives without the gallows (+0.33 to +0.37): twins plus features (A measurement, C reading).

## Featural effect against invented and cipher scripts (v35)
- Grade A: invented and cipher scripts do not reproduce the Voynich shape-behaviour effect.
  - Null: Copiale cipher, Cherokee, Deseret.
  - Weak: Shavian +0.10 to +0.13; Borg cipher +0.17/+0.12/0.00, grade B.
  - Positive controls: Canadian syllabics +0.13 to +0.21; Tengwar +0.39 to +0.54, but +0.08 to +0.14 without its vowel marks.
  - The Voynich (+0.28/+0.30/+0.43) sits above every real invented or cipher script once Tengwar's vowel marks are removed.
- Grade A measurement: the most look-alike Voynich glyph pairs (k/t, p/f, the benched gallows, ch/sh) are almost interchangeable in context, index 0.81 against 0.36 for frequency-matched pairs. That matches a planted homophone cipher (0.73), not any featural script (0.17–0.37). Without the gallows the index falls to 0.34–0.36, but the shape link survives (+0.33 to +0.37).
- Reading, grade C: interchangeable twin glyphs plus a featural gradient. Not tested for lack of open data: the Rohonc Codex, Hildegard's script and other DECODE ciphers.

## Is A to B a featural sound change? (v36)
- Rule-level tests have no power on Voynich text: an A-against-A null already scores z +1 to +2, because any learned rule swaps glyphs that behave alike, and those look alike (v25). Dropped.
- An exhaustive single-glyph substitution matrix passed its controls. Featural plants in Hangul and German came out on top with a shared stroke change, and 0 of 7 arbitrary plants and keys were significant.
- Grade B: A to B targets one stroke class. It drops the four benched gallows at word start: per 1,000 words, cth falls from 36 to 2.6, ckh from 13 to 1.3, cph from 7.3 to 1.5 and cfh from 2.2 to 0.3. It holds within the herbal section. No shared stroke change appears among the top swaps, which is the spelling-change pattern, not the sound-change pattern.
- Kill test fired: the bench-removed forms do not rise in B, so B uses different words there. "Featural sound change" is killed.

## Spelling drift as a clock (v41)
- Controls: spelling traits ordered 97 dated German texts by date (rho 0.63–0.75). Within one manuscript the traits did not move together. Planted drift was caught when it rewrote up to about 20% of words. The method cannot tell gradual drift from a mix of two states.
- Grade B: every hand has a coherent A-to-B spelling axis from page to page (z +4 to +11, both transcriptions). Endings and e-load drift together inside a hand. Word-initial benched gallows do not: losing them is a jump between groups.
- Grade B: in Currier B, each bifolio is one moment in time. The two sides of a leaf and conjugate leaves sit close on the clock (z −2.4 to −3.4), fitting v8.
- Grade C: hand 1 rises only between quires (mainly D, G and O). Intermediate pages come in whole quires. Section shifts the clock more than hand does. No page-level writing order is recoverable.

## Collapse the twins (v39, 4 Oct 2026; `loops/v39_final.txt`)
- Question: are the generator-like signatures (v23 no frequency arrow, v31 generator vote, v33 no lexical gaps, v21 forgery battery) artefacts of the v35 twins (k~t, p~f, ckh~cth, cph~cfh, ch~sh) spreading one unit over two glyphs?
- Controls pass (Grade A): Copiale, a real homophone cipher, reads as a generator (P 0.97) and, merged by its published key, as language/conlang (0.92; generator 0.07), with types per 10k halving and the gap ratio falling into the language band; random merges of the same sizes change nothing. A Voynich-matched twin split lowers P(language) in Latin, German and Italian by 0.10-0.16, and an index-driven merge restores it (+0.14 to +0.16, held out). The index ranks the 5 planted pairs top in every language; no real letter pair reaches 0.9.
- Grade A, negative: merging the Voynich twins (each, the gallows pairs, all five, the whole gallows family), merging every pair the data call interchangeable (I >= 0.9 down to 0.5, held-out pages, ZL and IT2a), or any of 260 random merge sets of 1-8 pairs never makes it language-like. P(language or conlang) stays 0.11-0.21, P(generator) rises (0.43 -> 0.47-0.94), the frequency arrow stays at 0-1/400 (ZL), the gap ratio stays 0.92-1.01 (languages 0.46-0.79), forgery AUC 0.92-0.94. 0/12 random-search winners survive held-out in either transcription (planted Latin: 4/12).
- Grade B (promoted from v35 C for k~t and ch~sh): the twins act as variants of fewer units. Merging k=t or ch=sh collapses word types at z -10.6 / -10.1 against frequency-matched merges (all five: types 0.29 -> 0.24, Zipf steeper z -5), as a homophone split does. Would kill: a scribe or section where k-words and t-words share no more types than frequency-matched pairs. p~f and the benched pairs: same index band, too rare to test this way (C).
- Reading: the twins are real near-variants, but the generator texture lies beneath them. Not cracked.

## Among deliberately meaningless books (v42, 4 Oct 2026; `loops/v42_final.txt`)
- Reference book: the Codex Seraphinianus (one hand, 1976-78, declared asemic), from M. Ponzi's neural-network transliteration (44,290 tokens, 33 stroke units, ~18% character error). Every other corpus was also passed through a matched OCR-like noise channel. No other long asemic book is machine-readable (Rohonc: no public transcription; Xu Bing: no transcription).
- Grade B: v31's "generator" verdict is not specific to generators. Most surviving random classifiers call the CS a generator too (61-65% in two unit schemes; the raw stroke scheme reads gibberish).
- Grade B: the Voynich is the nearest neighbour of none of 123 training corpora, but it is the nearest real text to the CS in all three unit schemes. On the classic battery the CS shares 6-8 of the Voynich's 7-9 departures from language: no lexical gaps, low h2/h3, slot rigidity, the same glyph arrow (2.41 vs 2.42) and line effects. Only Pinyin ties it. An ASEMIC class trained only on the two CS volumes absorbs the Voynich at 3000 tokens (0.64-0.98). It absorbs none of 44-91 held-out corpora, none of 3 odd texts, and none of 8 meaningful or generated texts rewritten in Gothic minim strokes (kill test passed).
- Grade B, negative: habits do not harden. The CS drifts away from generator statistics (rho -0.31; volume 1 0.34 -> volume 2 0.18). The Voynich has no trend in folio order. Generator-likeness is present from the first pages.
- Grade B: the books differ in topic. A junction forgery of the CS is caught at 0.99, like a Latin herbal, and mainly by page topic (rare words recur on their page 3.3x the forgery). The Voynich (0.94, or 0.91 at matched noise) has no topic signal. Topic-free pages are therefore not a generic trait of long meaningless writing.
- Grade C: the Voynich belongs with long one-hand asemic books rather than with mechanical generators. Would support: a second, hand-transcribed long asemic book joins the class. Would kill: a long generator or a meaningful minim-script text is absorbed. Limit: n = 1 reference book, machine-read.

## Note on v31, v42 and the "generator-like" results (4 Oct 2026)
The meaningless, asemic and generator readings are long-known hypotheses, not new findings. The v42 resemblance to the Codex Seraphinianus is demoted to background (grade C, not pursued). From here, Voynich loops assume the text carries meaning. They treat the generator-like statistics as an encoding surface and aim at extracting what carries information.

## Similar plants, similar text? (v38, 4 Oct 2026; `loops/v38_final.txt`)
- **Method.** 119 herbal pages (Yale IIIF, low resolution, scratch only). The plant is segmented and described by 6 families (silhouette, paint colour, HOG, edge/LBP, ResNet-18, DINO; later EfficientNet-B0 and DINOv2). Page-to-page drawing similarity is tested against text similarity (5 metrics) by partial Mantel on 11 confounds (language, hand, quire, bifolio, leaf, adjacency, distance, length, drawing area), permuting within language x hand. Real-herbal control: Gerard's Herball (1636), 144 woodcut pages with OCR, same pipeline.
- **Grade A (method).** Hand-crafted shape and colour descriptors see no plant-text link even in a real herbal. Only learned embeddings see it (Gerard z 4.4-7.4).
- **Grade B (negative).** The Voynich text does not track its drawings the way a real herbal does. Gerard's woodcuts predict the words of unseen pages (own-text rank 0.60, z 3.8). They give z 5.6 at the Voynich page count. The Voynich gives rank 0.49 (chance) and z 2.9. There is no same-plant-same-words tail (visual twins z 0.2-0.5), and 2,000 random feature hypotheses do not replicate on held-out leaves (z 0.5; Gerard 2.4).
- **Grade C (lead for meaning).** A faint, diffuse picture-text resemblance remains, carried by Currier A hand 1. It reaches z 2.9-3.0 with four networks and both transliterations (IT2a 2.5-3.0). Production drift (vellum, ink, paint) does not explain it (z 0.2), and neither does layout. It falls to z 1.7-2.4 once handwriting-only page images are partialled out. No single visual trait carries it (no DINO PC > 1.4). Would support: z >= 3 with a fifth network on held-out quires, or the same link from the pharma-section plant parts. Would kill: z < 1 with a CLIP-type network, or its loss when hand 1 is split by painting campaign.
- Caught and withdrawn: a leave-one-out prediction that residualised confounds on all pages leaked (shuffled text scored as high as real). In-fold residualisation fixed it.

## Plant drawings and text (v38)
- Grade B: the herbal text does not track its plant drawings the way a real herbal does. Gerard's woodcuts predict the words of unseen pages (rank 0.60, network composite z 5.6 at the Voynich page count). The Voynich gives rank 0.49 and z 2.9.
- Grade C, kept as a meaning lead: a faint, diffuse picture-text resemblance remains in Currier A hand 1. It reaches z about 3 across four image networks and both transcriptions, and survives production drift and layout controls. It becomes marginal after a handwriting-image partial. No single nameable trait carries it.

## The page chooses the twin? (v40, 4 Oct 2026; `loops/v40_final.txt`)
- Question: if k~t, ch~sh, p~f and the benched gallows are near-interchangeable (v35, v39), what decides which form is written: the word (meaning), the page layout (glyph above, ink load) or position and habit?
- Controls pass: a real context allograph (long s, r rotunda in DTA prints) is attributed entirely to neighbouring letters; real meaning letters (Caesar c/g, n/m) show a word-identity gain of ~0.11 bit; a planted "avoid tall under tall" rule is found at 6% of slots (3% missed); a planted copy rule at 5%; the layout search finds nothing in the long-s text.
- Grade A (measurement, ZL and IT2a): word identity carries no information about k/t or ch/sh beyond the adjacent glyphs (unique gain -0.002 to -0.005 bit). Benched vs plain is decided by neighbouring glyphs (0.22 bit), like long s.
- Grade B negative: no glyph-above or tall-under-tall rule in any twin; no ink-load (pen-dip) effect on 25 imaged pages.
- Grade A measurement / B reading: ch/sh is positional. The same word-shape on the same page is written sh with odds x3.4 in a paragraph's first line, x2.3 in a line's first word, x0.6 in the right half of a line. k/t leans to t in first lines (x1.7) and toward the right (x1.3).
- Grade A measurement / B reading: a short-range hand habit. The previous k/t (or ch/sh) in the line predicts the next in a different word, fading within ~30 glyphs (5 words); it survives all layout features; ch/sh resets at each line. Benched and p/f do not persist.
- Grade C reading: k~t and ch~sh are each one unit whose visible form is chosen by line position and a decaying stylistic run, not by meaning. Would kill: a scribe or section where the same word-shape keeps one form regardless of position and run, or k-/t-words in complementary semantic contexts beyond frequency.

## Find the island (v43, 4 Oct 2026; `loops/v43_final.txt`)
- Method: the v31 classifier score (language or conlang) per 100-token chunk, scanned in 600/1,200/2,400-token windows in reading order and by section, quire, hand, Currier language and bifolio; a subset search on odd lines tested on even lines; 3,000 random classifiers (298 survivors) as independent detectors; v33 gap ratio, frequency arrow, topic re-use and forgery AUC as second tests.
- Controls pass (Grade A): planted German or Latin islands (600-2,400 tokens, transliterated into Voynich units) are found at their place (98.7% of detectors; contiguous 12-page plant recovered on held-out lines, precision 1.0); uniform trigram or forger text gives no island. Limit: 12 scattered planted pages are not recoverable, so scattered or sub-600-token islands stay open.
- Grade B: there is a relative island, biological quire M, hand 2, about f75-f84 (core f76r-f82v), 2,200-4,900 tokens. Both transcriptions find it, 69-91% of the detectors put it in quire M, and it holds on unseen lines. No whole section, quire, hand, language or bifolio stands out once contiguity and the number of groups are corrected.
- Grade A negative: it is not language. On unseen lines it sits at the bottom edge of the language band (0.23-0.34 vs planted German 0.41-0.51), barely above uniform generator text. Its gap ratio is 1.03 like the rest of the book (languages 0.65-0.87), and 0 of 28 windows pass both tests. Its features do not move toward language (17/32, p 0.26; planted German 26/32): it is a repetition texture (fewer word types and endings, more repeated words).
- Reading (C): one kind of text throughout; quire M / hand 2 is the repetitive extreme. Would kill: an M-like island in another hand without the repetition shift.



## The residual is the message? (v45, 4 Oct 2026; `loops/v45_final.txt`)
- Method: the v26 rule model (cross-fitted by pages) predicts every word; the residual is which allowed alternative was chosen. Controls: rule-only and richer generators (no message); Latin and Gerard text pushed through planted verbose codes inside the same rules.
- Grade B: the Voynich pushes its choices off the rules by +0.19 bits/word, like a richer generator scored by a poorer model (+0.24); planted codes push 0.44-1.26.
- Grade B: the residual repeats inside bifolios and quires (z 7-10, ZL and IT2a; generators 0), not by section or illustration, and does not track the herbal drawings (residual below surface; 1,500-reading random search null, where a planted code is z 3.8 at discovery). It looks like writing-unit drift, not content.
- Grade B (measurement): the rule-unpredicted vocabulary concentrates on paragraph-first lines (42% of words outside any allowed choice vs 26% in body lines; generators flat). Grade C: openings hold entry-specific content; its drawing prediction failed (z 1.85 vs generator 2.40).
- Limit: a code choosing in proportion to the rule probabilities, or a letter-level code, is invisible to these tests. Not cracked.

## Note from x4 (cross-script battery, 4 Oct 2026)
The generator-like statistics that earlier pointed at "meaningless" (v31, v33) do not separate meaning from no meaning. Known list languages (Ur III, proto-cuneiform) are also voted "generator" (71–95%) and also lack word-order arrows. Once list references are added, the Voynich's generator vote falls to 41–57%. Grade B: the Voynich statistics fit a list-like or designation-like meaningful text as well as a generator. This supports the working assumption that the text carries meaning, and later loops should treat the Voynich as possibly a list or register rather than prose.

## The rule-breaking words are the subjects? (v49, 4 Oct 2026; `loops/v49_final.txt`)
- Method: words outside every choice the v45 rule model allows were split into an opening-line (paragraph-first) and a body lexicon, then tested as subject terms or borrowed names. The real-herbal control was Brumati's Italian *Flora medico-economica* (1844), which has a Latin binomial in each entry. It was pushed through a planted rule-bound glyph code, both as written and with the name moved to the head of each entry. Other controls: planted subject terms, and a generator with planted rare-word noise.
- Controls pass (Grade A): Latin names come out as rule-breakers (86-87% of their tokens vs 33% of Italian). The subject tests detect planted terms and Latin names (own-page recurrence z 4-6). A search over random spelling views recovers names whose heading spelling differs from later mentions (held-out mean z +2.1).
- Grade B (negative): Voynich opening-line rule-breakers do not behave like subjects. They do not recur on their own page (-0.010, like noise). They show no illustration-tied re-use. They are spread evenly along the line, not front-loaded. Of 4,000 random spelling views, none replicates on held-out pages (ZL +0.24, IT2a -1.0).
- Grade A (measurement): they are not a second language layer. Their extra glyph-level strangeness (+0.21 bits/glyph) disappears once the four gallows are merged and the model is refitted (+0.02, z 0.6). A Latin layer keeps +0.17 under the same merge. With gallows and ch/sh removed they are as separable from body words as planted noise is (AUC 0.56 vs 0.55; Latin 0.66). On a 59-language typology map the two lexicons are no further apart than two halves of the body lexicon.
- Reading (B): the opening line is a spelling regime (more gallows, p/f, gallows+ch) over ordinary body vocabulary. This demotes the v45 Grade C idea that openings hold entry-specific vocabulary. Not excluded: a one-word heading (the paragraph's first word), a subject never re-mentioned, or a foreign layer below about 1 in 4 opening words.

## v52 the book as a database (5 Oct 2026)
Voynich words do not behave like record keys with attribute fields (grade B, negative). Of about 16,000 random slot schemas, none beats a field-free null on held-out pages (0.19–0.20 against 0.20). Controls passed: a planted catalogue, Linear A catalogue numbers and abbreviated Unicode names. Grade A, measurement: the best schema re-finds known effects: initial a- tracks the previous word, q/p/gallows track line position, the ending tracks Currier language, and the long middle tracks nothing external. Grade C: only word edges act like fields, and they point to the neighbouring word. Would kill: the effect goes away if r/s+a- and y+q- are read as misplaced spaces. If meaning is carried anywhere, the word middles are the candidate payload.

## Let a speech model listen (v51, 5 Oct 2026; `loops/v51_final.txt`)
- Text rendered as audio under 565 random phone mappings (formant synthesiser, random durations, pitch and stress rules) and scored by a multilingual speech recogniser trained only on human speech (18,000 clips).
- The listener hears unit-transition structure only: languages beat their own shuffles weakly and unstably (z -0.3 to +3.7), Markov-1 resyntheses do about as well, and Voynich-fitted generators (self-citation, Markov-2) pass while gibberish and a table grille fail. Voynich ZL +1.3 to +2.0, inside both bands (Grade B, negative).
- No mapping found on one half of the Voynich stays speech-like on the other half (p 0.52); the same search cannot recover the vowels of German or Esperanto, so no glyph-class reading is offered (Grade B).
- Sections: stars/recipes > biological > herbal > pharma nominally (stars - pharma z +2.0), but the ranking does not reproduce across mapping halves (rho 0.18). The listener's language guess does not track the source language even for the controls. Not cracked.


## Textual criticism without a language (v54, 5 Oct 2026; `loops/v54_final.txt`)
- Method: every near-repeated 3-word passage (<= 3 edits, C pair search over all 26,600 line trigrams) treated as manuscript witnesses and collated; reduced alphabets scored on held-out pages; stemma by hand and quire; ZL vs IT2a disagreements as a third witness; recurring two-word contexts tested for exact and one-edit-variant returns. Controls: Brumati's herbal with planted medieval spelling by three scribes in opaque symbols, the clean herbal, real ReF German, CATMuS Latin and Italian; nulls: three word shuffles, unit-Markov, self-citation.
- Controls pass (Grade A): planted classes recovered exactly; merging them gains section information (+5 points, z 14) and erases the scribe (z -16); the stemma test finds planted scribes (z 15); German yields its real medieval classes; a Markov generator fails every test.
- Grade B: Voynich near-repeats are 2.2-2.4x the nulls but mostly a similarity field; only 11% of the excess lies within one edit (German 86%), and whole-sample variant classes are the same as in the nulls.
- Grade B: recurring contexts bring back the same word (z 6-7) and a one-edit variant (z 4, both transcriptions), but at 0.85-1.12 variants per exact return, against 0.07-0.15 in real medieval copying and 0.53 for heavy planted spelling. Only a copy-and-modify generator is as high. Merging the variants raises writer information (scribal spelling lowers it) and there is no stemma by hand or quire (z 1-2.5).
- Grade B measurement: the context-anchored variant classes are ch~sh, e~ee~d (the -edy/-eey/-ey ending family), optional q- and cth~d, word-initial and medial. Grade C reading: these are the slots the writer re-rolls when a passage recurs, and the o/a/i/n/y/l/r frame with gallows identity is what is kept. Would kill: the same classes from a junction or line-position model without recurring context.
- Not cracked. The varied slot between two mentions of a passage is the next candidate carrier.

## v50 reading paths across the page grid (5 Oct 2026)
2.27 million paths were searched: columns, diagonals, knight moves, every n-th word, first/last glyphs and random multi-step paths. **A (negative):** no hidden letter-level message. Planted Latin and German acrostics, diagonals and random paths were found exactly (rank 1–3), but on real pages every hit is the known left-margin first-glyph chain (v6), which is not language-like (P 0.04 against 0.99 for a planted acrostic). **C, blind spot:** a word-for-word code word laid along a path ranks near the threshold, so word-level hidden messages are not excluded.

## Where the pen ran dry (v57, 5 Oct 2026; `loops/v57_final.txt`)
- Method: ink fade used as a clock. Writing order was scored over 4,914 guessed orders and with a line-break clock. Re-dips were located below word level on column ink traces of 1,152 Voynich and 915 Latin (CREMMA) lines. Controls: a dark/light mirror null, word-shuffled traces, word-type residualisation, Latin punctuation, and planted re-laid orders (recovered at rank 1 in 4 of 5).
- Grade B (negative): no phrase boundaries in the ink. Latin scribes do not re-dip at punctuation (odds 1.06 vs 1.42 at shuffled junctions), so re-dip position is not a phrase marker even in meaningful text. Segments between Voynich re-dips do not recur. The apparent preference for re-dips between -y and q- words disappears when the same junctions are compared in shuffled lines (z +0.4): it is glyph shape.
- Grade B: pen loads have the same shape in both scripts: a fade, a sudden darkening, then about three heavy words. The strongest re-dips fall at word gaps in both scripts, but most gap darkening is word shape.
- Grade C: consistent with writing in reading order. The ink clock crosses line breaks at about 40% of the Latin strength (z 1.85 vs 3.2). There is no separate pass for line-initial words or columns.
- Grade C+: only the Voynich shows a stable word-level ink sawtooth (increment skew +0.28, z 6.5). It survives word-type control, both page halves and the removal of y|q junctions. Latin's skew is unstable. Would kill: an equal stable skew in another single-hand manuscript with short words.


## v55 the real sky as answer key (5 Oct 2026)
**A (negative):** the zodiac labels, ring texts and circular texts do not fit the real sky of any year from 1290 to 1611. About 48 billion hypotheses were tested against our own ephemeris, which was checked against JPL DE421 and the eclipses of 1406, 1415 and 1433. The hypotheses covered day-by-day labels against Moon sign, phase and weekday; event markers for phases, eclipses and Easter; a horoscope over all 117,603 days; and day runs. Planted almanacs and charts at 50% or more density were found to the exact year and position. 1300 and 1550 score like 1420. The one survivor ("d"-words follow the Moon sign in 1424) failed out of sample (p 0.92). C, still open: sparse codes, or a different label-to-day mapping.

## v53 evolved encoders (5 Oct 2026)
About 164,000 evolved writing programs were run on 9 real plaintexts. **A (method fails):** fitting the statistics cannot recover a planted plaintext's language (German ranked last of 9). Only its layout was recovered. **B (killed):** the best read-back of the Voynich (abbreviated Italian herbal letters) replicates on IT2a, but a meaningless glyph-trigram copy of the Voynich scores just as high. **B:** no evolved program reproduces the Voynich's line-edge effects, neighbour-word coupling (0.12 vs 0.29 bits) or adjacent repeats. Those three features are the part of the surface that simple meaning-preserving encoders do not produce, and they are where any extra machinery would sit. C: no glyph position is marked as padding (low power).

## The book's own Rosetta stone? (v59, 5 Oct 2026; `loops/v59_final.txt`)
- Idea: if Currier A and B are one language written in two ways, the two form a parallel corpus without alignment, and a learned A-to-B translation should carry word contexts, subject matter or drawings across.
- Grade B (negative): no transducer survives. About 45,000 random rule sets and greedy stacks were tested, along with the v30 ending shift. None of them makes A words land in B's line contexts on held-out pages. The same score recovers a Czech spelling reform from scratch and confirms a planted Voynich-size shift (+0.12), though at this size it cannot find that shift by search. The v30 rewrite changes word forms without carrying contexts across: B to A scores +0.014 to +0.019, and unrelated languages reach +0.013 on the same score.
- Grade B (negative): a spelling-blind word aligner recovers word pairs through an opaque verbose encoding (Czech 44-52% P@1, Isidore 24-26%; chance 0.7%). Between A and B it aligns shared words at null level (3.5-4.2%, against 14% between A pages and 1-3% in nulls). The only cross-transcription pair, daiin->chedy, also appears in every null.
- Grade B (measurement): B herbal pages use A-herbal vocabulary (as opposed to A-pharma) more than B's other sections do (AUC 0.71 / 0.77, z 2.4 / 2.6, label-permutation null). The effect is carried by line-interior words and by the first three glyphs of each word (0.76 / 0.81). The last three glyphs carry little (0.56 / 0.64). Grade C reading: word beginnings are shared and endings are re-encoded between A and B. Would kill: an endings-only B->A map doing no better than random ending maps.
- Drawing pivot (word-level plant-embedding centroids): it has no power at 91 vs 27 pages, even in Gerard. Uninformative. Not cracked.

## Hot or cold, wet or dry (v60, 5 Oct 2026; `loops/v60_final.txt`)
- Method: the medieval complexion grid (hot/cold x dry/moist x degree) measured from six real herbals (Old French Circa instans, Banckes, English Macer, Lyte, Gerard, Pechey) and used as an answer key. Searches: 1.3-2.0 million lattice-shaped item quadruples per corpus fitted to the 2x2; 0.5 million complexion-shaped binary labels confirmed by held-out predictability from the rest of the paragraph; 1.5-1.8 million four-way 'degree' sets; a published hot/cold/wet/dry word-family guess as a pre-registered kill test. Controls: real herbals through an opaque verbose code with 35% filler words; recipes and astrology; Markov, word-shuffle, star-section and permuted-target nulls.
- Grade A (measurement): hot-dry is 0.65-0.80 of entries in every herbal; hot share 0.74-0.85; hot vs cold is predictable from the rest of an entry (AUC 0.60-0.79), dry vs moist is not.
- Grade A (method negative): the grid is too low-information to find a slot in 100-300 paragraphs. The coded Circa instans and Macer controls do not recover their own complexion slot (oracle fit below the best noise; at best top 0.1-1% of candidates), and the Voynich equals every null in the lattice and degree searches.
- Grade B (negative): the published families (ched-/shed- hot, saiin/daiin/chol cold, ot- wet, qok- dry) give a hot share of 0.15 (herbal A 0.03) against 0.74-0.85 in every herbal, track Currier language and section, and are no more one-per-paragraph than shuffled text. Killed.
- Grade B (measurement): herbal-A paragraphs have internal vocabulary coherence (labels predictable from the rest of the paragraph, median z 1.2-1.4 vs about 0 in nulls), but astrology and recipe texts show the same, and no label repeats across page splits. Grade C (open): a complexion written as a rare sign or stated per page is not excluded.

## Find the source by its shape (v58, 5 Oct 2026; `loops/v58_final.txt`)
- Idea: an encoded or translated book keeps its shape (the sequence of long and short chapters, recipes, plant entries) whatever happens to its words. Entry-length sequences of 25 works (Pliny, Celsus, Dioscorides, Theophrastus, Hildegard, Regimen Salernitanum, Konrad von Megenberg, Hyginus De astronomia, Apicius, Forme of Cury, Culpeper, Macer floridus, Psalms, Circa instans (Old French), Antidotarium Nicolai (Middle Dutch/Latin), Leechdoms herbarium, De balneis synopsis, Gerard, Isidore, an Italian herbal) were aligned against Voynich paragraph and page lengths by section (local alignment with free scale, merges, skips; block-shuffle null).
- Grade A (method): a translated, opaquely encoded, re-tokenised copy is found from ~87 units (Celsus z 10, Psalms 8.5; 21-24 at 292), also when misbound within quires (z 5.8-13.4) or behind word breaks unrelated to source words (z 19-23). It fails for a different recension padded with commentary (Apicius).
- Grade B (negative): no section matches any of the 25 works (200 tests at the null level, max 3.3; every hit fails IT2a or the translation-partner test); 1,500 guessed leaf orders per search (x7 with nulls) and 8 length rulers change nothing. Not covered: works outside the corpus, glossed or abridged recensions, cross-quire reordering.
- Grade C- (demoted): bio paragraphs ~ Hyginus De astronomia (z ~3-4.5, diffuse, gone on book 2 alone). Not cracked; nothing predicted outside the corpus.

## Which twin is the slip? (v63, 5 Oct 2026; `loops/v63_final.txt`)
- Idea: the exemplar line-width search was already done (v17), so v63 asks which twin of each adjacent pair is the mistake. A copyist who writes a slip and then the intended word leaves the wrong word first. A copy-and-modify generator leaves the source first. A text with no copying leaves no order. Reference: 1,775 real struck words with their replacements from five diplomatic Latin witnesses (SCTA Plaoul, 1.04 M words, with line breaks).
- Grade A (measurement): real copying slips are mostly whole wrong words: other 47%, eye ahead 19%, eye back 13%. Dittography is only 3% and false starts 3%. Slips are rarer than the word written next (dlogf -0.7 to -1.2) and are not tied to the copy's line ends (14.1% vs 14.3%).
- Grade B (negative): Voynich adjacent near-repeats are unordered (dlogf -0.03/-0.04; prefix-first 0.50/0.56). This excludes a copy-from-the-previous-word generator (+0.3) and a copy with 10% or more slips.
- Grade B: about 4 million oriented slip-operator sets, chosen on half the folios and scored on the other half, find a few ordered adjacent alternations beyond line-position drift (held-out z +8.4/+7.7 against a position-matched shuffle at 6.6 +- 0.5). The alternations are o>l, ch>d and ch>cth initials, and -l before -y. Planted 1% glyph slips are recovered with their operators.
- Grade B (negative): these alternations fail every slip test (first twin commoner, deleting the second helps more, they avoid line breaks), and position-matched shuffles behave the same way. They are line-position drift, not copying errors. The 'corrected' text would only lose o/l/s single-glyph tokens and stem+l before stem+y pairs.
- Bounds: systematic glyph slips >= ~1% of words, and slip-and-rewrite of any kind >= ~10%, are excluded. Whole-word slips at real medieval rates (~0.2%) are invisible to this test. Grade C lead: an adjacent stem+l -> stem+y order (chol chy, dal dy, chedal chedy). Would kill: it vanishes for pairs matched for position in the line. Not cracked.

## Undo the sandhi (v61, 5 Oct 2026; `loops/v61_final.txt`)
- Idea: if the junction coupling, word variants and line-edge effects are sandhi, rules that undo the edge changes should exist, make medial words look like the neighbour-free line-edge words (the base forms), and remove the coupling at the same time. About 600 million edge-rewrite hypotheses (endings and beginnings of 0-3 glyphs x 4,000+ context classes) and about 45,000 rule systems were tested, with selection on half the pages and scoring on the other half.
- Grade A (method): Sanskrit (DCS Ramayana, sandhied vs unsandhied, opaque glyphs) gives back its textbook rules with the pausa form at the half-verse end. Its best system matches the true base (edge agreement -60% vs -58%, coupling -56% vs -80%). Italian elision and a planted Voynich sandhi are found; Welsh mutation is found only in part (word-triggered). Undoing true sandhi does not change page prediction, so page gain cannot judge a sandhi reading.
- Grade A (measurement): no coupling crosses a Voynich line break (excess -0.005 bits, z -1.2; verse +0.008 to +0.019). Line ends are true pauses.
- Grade B (negative): line-final words are not base forms. Line-edge pull 0.25 vs 1.07 (Sanskrit) and 1.56 (Italian); nulls 0.04-0.14. Ending systems remove up to two-thirds of the coupling but move medial forms only 13-14% towards the line-final forms (inside the overfit band); random systems move them away. Making medial beginnings look like line-initial ones raises the coupling by half. Sandhi is killed in both directions, extending v29 to multi-glyph edges.
- Grade B: the 'base-form book' (types -6.6%, coupling 0.206 -> 0.064 bits) is built from merges of different words (chol -> chedy, chey -> chetar, qokain -> qoky). Zipf and adjacent repeats do not change. The section gain (+2.5 points) is the same with neighbour-blind rules. Reading: the coupling is a pairing of different words, and line-final and line-initial words come from their own pools.
- Grade C: Currier A's line ends lean to one side of the alternations as much as size-matched Sanskrit (0.42-0.50 vs 0.48; null 0.10-0.13), but this does not become a base-form system (-8 to -11% vs -30%). Would support: an A-only system that passes the joint test in a third transcription. Would kill: the same lean from a line-aware generator without sandhi. Not cracked.

## The text remembers which page came next (v65, 5 Oct 2026; `loops/v65_final.txt`)
- Idea: meaningful prose carries a topic from one page into the next, so the page order with the most carry-over should be the original one and should match the physical codicology. Every physically allowed arrangement of every quire was enumerated (about 130 million scores), with held-out word classes, calibrated on continuous Latin and German herbal prose in an opaque padded code laid into the Voynich page layout, and checked against published physical adjacency evidence (78/81 picture across the Q13 centre, 33/40 drawing and paint, 10/15 spatters, contact transfers, quire signatures).
- Grade A (control): a copy-and-modify generator carries over 20x more than prose (seam z 59 vs 8-14) and gives a stable, binding-matching order. Carry-over cannot separate meaning from generation.
- Grade B (negative): Voynich page seams are weak (z 2-3, a fifth of prose) and have no direction; the binding's leaf-to-leaf neighbours inside quires are no closer than any two pages; no page order is stable across word classes or transcriptions. One-plant-per-page herbals give no signal either, so this does not count against meaning.
- Grade C: with whole words, the bifolio the text puts at the quire centre is the physically documented centre in 5 of 6 runs (Q13 78/81, Q5 33/40, Q2 10/15; top-5% share 0.37-0.47 vs prior 0.20-0.25 and Markov 0.19-0.22); the skeleton keeps only Q13 and bootstraps are weak (0.19-0.39). It would show that the halves of a sheet were written together, which a drifting generator also predicts. Not cracked.

## The Voynich is verse? (v62, 5 Oct 2026; `loops/v62_final.txt`)
- Method: rhyme schemes (2,000-3,000 ending definitions x lags), metre (hill-climbed random glyph weightings, line totals and word-boundary positions) and refrains (equal-gap recurrences of line signatures), trained on odd folios and tested on even folios and IT2a. Controls through an opaque verbose code: Regimen sanitatis, Macer floridus, Dante, a litany, Hildegard's prose herbal, Caesar; nulls: line shuffles, re-flowed breaks, a Markov generator with planted line edges, and planted metre, caesura and within-line likeness in Voynich words.
- Controls pass (Grade A): rhyme found held-out in Regimen (z 3.2) and Dante (84); metre in Macer, Dante and a planted Voynich-word metre (+0.44 to +1.09); the versicle/response litany refrain at z 6-21. Prose and re-flowed verse stay at chance.
- Grade B (negative): no rhyme. The Voynich's best ending definition is below all 9 null maxima and none replicates; neighbouring line ends are slightly less alike than second words (z -1.4 to -2.7, both transcriptions).
- Grade B (negative): no metre or caesura. Line weight totals vary more than in slot-swapped lines (-0.10 to -0.40), the opposite of verse. The boundary-position concentration that first looked metrical (above Dante and Macer) is reproduced by a planted generator in which a line's words share a shape. Grade B measurement: Voynich lines are internally alike (heavy lines and light lines).
- Grade B (negative): no refrain words, lines or skeletons at fixed intervals. Grade C: the known first-glyph margin chain recurs at fixed 2-8-line spacing on even folios only (z 4-5, both transcriptions); would kill: it vanishes without paragraph-first lines.
- Reading (C): the line is a unit of likeness, not of sound. Would support: line modes that predict something outside the line (the drawing, a repeated plant). Not cracked.

## v64 Lullian concept combinations (5 Oct 2026)
**B (negative):** Voynich words are not combinations from a small concept alphabet, at least at the glyph n-gram level. About 353,000 alphabets were scored on held-out pages. Planted Ars brevis and Lullist medical texts gave back their alphabets and wheel rules, and Latin was rejected. On the score (concept model minus glyph-sequence model, bits per word), the Voynich got −0.89/−0.88, about the same as its own Markov-2 resynthesis (−1.00), against +0.89 for the Ars brevis. The best "alphabet" is just the frequent chunks. There is no wheel-step rule, and the ring texts do not turn by rule (low power). C, still open: concepts written as whole words, or as single glyphs hidden in normal spelling.

## The book's mind map matches a real one? (v66, 5 Oct 2026; `loops/v66_final.txt`)
- Idea: meaning lives in which ideas go together, so the Voynich's keyword co-occurrence network should align with the concept network of medieval medicine (body parts, illnesses, vehicles, preparations, plant parts, qualities, sky and time, habitat), built from Culpeper, Konrad, Macer, Circa instans, Celsus and an Italian flora in their own words. Matching by annealed injective graph alignment (about 65,000 restarts), with degree-preserving, Markov and concept-graph-rewiring nulls, held-out halves, IT2a, opaque padded herbals as positive controls and chronicles, novels, philosophy and psalms as kill controls.
- Grade B (method, negative): the alignment score measures how clustered a text's keywords are. Kafka, Caesar and Descartes align with the medical concept graph as well as herbals do, and no herbal control recovers its own word classes, even when the right keywords are handed in (0-2 correct, chance 0.5-0.7). Lexicon-free, keyword-graph shape does not tell a herbal from a chronicle (nearest-neighbour accuracy 0.34-0.59, null 0.48).
- So the Voynich's strong scores (z 7-20 against every null, both transcriptions) carry no information about its subject; this neither supports nor counts against meaning. No keyword-to-concept mapping survives. Not cracked.

## v69 writing care (5 Oct 2026)
Eight measures of writing care were taken on 11,576 Voynich words and 9,409 Latin control words. **B (method fails its control):** Latin copyists also show no extra care for rare words, keywords or first mentions, so the result does not count against meaning. B (upper bound): extra care for Voynich rare words is below about 0.15 SD, and for first mentions below 0.1 SD. Paragraph-first lines are written with less care. The "repeats look alike" lead was killed once line distance was matched.

## v68 the text as a melody (5 Oct 2026)
**B (negative):** no musical reading. Real chant, written through 3,200 random codes, is the worst of 8 sources at producing Voynich-like text. It has too many adjacent repeats and has phrase finals, and **no source under any code reproduces the Voynich's line-initial pool**. A blind hidden-pitch fit recovers coded chant (rho 0.89), but the Voynich's word-to-word coupling is weak (0.11 bits against 0.61 for chant and 0.44 for German). There are no finals or modes on held-out folios. B (measurement): Currier B sections couple neighbouring words 2–4× more than herbal A and pharma A.

## v56 the book as hypertext (6 Oct 2026)
**B (negative, limited power):** no pointer rule maps a word's frame and gallows to a page or quire address beyond the nulls (about 250 million rule evaluations). A planted code was recovered in only 1 of 4 blind runs, so the negative is weak. **B (new measurement):** a word's contexts point back to the same distant page 5–10× less in the Voynich than in two real herbals (z 4.5–10 against 52–57). Padding alone does not explain the gap. C reading: pages behave like independent records, not a cross-referenced treatise. Would kill: a real herbal matched to the Voynich type/token ratio scoring about 5.
