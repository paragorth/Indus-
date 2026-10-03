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
