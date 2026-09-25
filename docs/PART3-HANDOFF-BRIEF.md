# PART 3 HANDOFF BRIEF — The Dictionary Game
## Indus Script Semantic Inference Program

**Prepared:** 22 July 2026, end of Part 2. This brief is the complete context needed to begin Part 3 in a fresh conversation. Upload alongside: `sign-dossiers-top60.json`, `indus-repro-package.zip` (contains all raw data, the analysis pipeline, and `bridge.json`).

---

## 0. The objective, in priority order

1. **PRIMARY: actually read the script — or know precisely why we can't.** The goal is genuine understanding: partial readings of real seals with honest confidence tiers, held by us, even if nobody else ever accepts them. A reading of the form "seal M-15 is [ISSUER-emblem] + [opener formula] + [X] + [terminal complex], where X is one of {name / commodity / office} with these relative odds" is a success if the odds are honestly derived.
2. **SECONDARY: publication.** A side effect. Do not let publishability shape the analysis. If the honest answer is "the posterior over dictionaries is wide," that is the answer we keep.

**The core method (Parag's own, from Logic of Life): adversarial degeneracy measurement.** We do not hunt for *a* dictionary. We generate candidate meaning-assignments, score them against all constraints jointly, and measure how many rival dictionaries survive. Narrow posterior = discovery. Wide posterior = honestly measured impossibility. Both are wins for objective 1. The graveyard of this field is single coherent stories; coherence is cheap, uniqueness is evidence.

---

## 1. What Part 1 and Part 2 established (validated, cite freely)

Corpus: open SQL database (yajnadevam/indus-website), 2,543 objects, 11,280 sign placements, 592 sign codes; validated against published Mahadevan statistics, object-level photo checks (M-14, M-15 verified sign-by-sign), and an 85-sign cross-coding bridge to Mahadevan numbers. Direction: right-to-left on impressions; database stored reversed; corrected everywhere in the frozen pipeline. Full reproducibility package exists; every number regenerates from one script.

**Structural findings (all out-of-sample validated):**
- The corpus is NOT one population. Registers (seals vs other objects, +0.45 nats/sign) and sites (MD vs Harappa seals, +0.29, sequential not inventory-driven) differ measurably. Foreign-findspot texts diverge most. **Consequence for Part 3: all semantic inference runs on the Mohenjo-daro seal register first; other strata are transfer tests, never pooled.**
- Functional slot structure: TEXT = [INITIAL (+connective)] [CORE*] [PRE-TERMINAL] [TERMINAL (+TERMINAL)]. 76% of held-out seals satisfy all class-position constraints (terminal-zone rule) vs 24% shuffled, 28% for random classes. Transfers nearly intact to Harappa seals; partially to non-seals.
- Emblem animals (unicorn/gaur/elephant/zebu) are statistically orthogonal to text content — consistent with issuer-insignia, not message.
- Terminal stacking (TERMINAL→TERMINAL, e.g. jar→man) is real, ~8-14%, register-dependent — compatible with agglutinative morphology, not diagnostic.
- The numeral→counted→terminal construction recovered semi-blind, formally enriched (p=0.001).
- Bias-variance result: 5-class skeleton captures much but strictly not all first-order structure.

**Key sign identities (via the 85-sign bridge; glyph ID = this database's coding):**
| Glyph | Mahadevan | Shape | Role |
|---|---|---|---|
| 740 | M342 | jar, two handles | modal TERMINAL (~33% of endings) |
| 2 | M99/100 | two short strokes | dominant CONNECTIVE (n=501) |
| 817/820/861 | M267 (variants) | diamond/leaf | INITIAL paradigm (opener formula X→2) |
| 90 | M1 | man/person | TERMINAL, stacks after jar |
| 400 | M176 | comb/rake | TERMINAL (86% final) |
| 520 | M211 | arrow/lance | TERMINAL (82% final) |
| 220 | M59 | plain fish | CORE, counted (33% numeral-preceded) |
| 240 | M67 | whiskered fish | CORE |
| 390/405 | M161-169 | tree/branch | CORE, counted, 45% terminal-continuation |
| 900 | M287 | crescent/parenthesis | most-counted sign (47%) |
| 3,4,5 | stroke numerals | 3/4/5 strokes | NUMERAL family (graded freq 100/53/30, initial-heavy) |
| 140 | M17 | person, raised arm | CORE, counted 17% — counted PEOPLE |
| 100 | M8 | three-headed person | PRE-TERMINAL |
| 176 | M48 | seated person | PRE-TERMINAL |

Dossier file: 60 signs, 74% token coverage, 53 bridged to shapes; per-sign: class, final/initial rates, numeral-before %, terminal-after %, MD:H ratio, seal-share, top neighbors both directions.

---

## 2. The constraint set (the rules of the game)

Every candidate meaning-assignment is scored against ALL of these jointly:

**C1. Grammar class.** A TERMINAL cannot mean a countable commodity; a NUMERAL-modified sign must denote something countable; a CONNECTIVE is functional, not lexical; INITIAL signs plausibly mark issuer/locus/document-type.

**C2. Countability.** Signs with high numeral-before rates ({900, 220, 390, 405, 140}) denote enumerable things: measures, commodities, animals, laborers. Counted PERSON signs specifically support labor/ration readings (cf. Rao 2018 proto-Elamite parallels).

**C3. Pictorial identity.** In a logographic/semasiographic reading (Mukhopadhyay's position, which our iconography null supports), shape is evidence: jar→vessel/measure, fish→fish/or rebus, strokes→numbers. In a phonetic reading, shape is rebus bait. Score under BOTH assumptions separately — this is a hyperparameter of the dictionary space, not a settled fact.

**C4. Object-type distribution.** Signs concentrated on pottery → commodities/measures; on tablets → transactional; on seals → identity/authority/function. The database has object-type and material fields.

**C5. The external anchor: Meluhhan trade vocabulary.** Mesopotamian cuneiform texts name Meluhhan imports: carnelian, lapis, gold, silver, copper, timber (mesu-wood), ivory, monkeys, peacocks(?), and mention Meluhhan ships, interpreters, and a Meluhhan village in Sumer. This is a vocabulary list for the export economy written by literate trading partners — the only genuinely external information available. Export-register texts and trade-object texts should preferentially contain signs meaning these things. (Fetch and verify the exact attested list early — Possehl, Parpola, and the Ur III documents are the sources.)

**C6. Register consistency (the stratified adequacy criterion — our own Paper 1).** Any assignment must remain coherent within each register separately and explain, not average away, the differences. The terminal set shared across registers should mean things that make sense on both seals and tablets; register-specific terminals should mean register-specific things.

**C7. Held-out prediction.** An assignment earning belief must predict co-occurrences in texts not used to construct it. Train/test discipline continues in Part 3. ICIT, when access arrives, is the ultimate holdout.

**C8. The two live macro-hypotheses (score dictionaries under each):**
- H-LING (Dravidian-type): terminals = name suffixes/honorifics; texts = names+titles; fish = mīn rebus (star/deity).
- H-ADMIN (Rao/Bonta/Mukhopadhyay-type): terminals = document-function markers; texts = [issuer][quantity][commodity][function]; fish/tree = commodities or measures.
These are not exhaustive and may mix (registers could differ in kind). The degeneracy measurement includes the H-space.

---

## 3. The program (planned steps)

1. **Verify and freeze the Meluhha list** (C5) from primary/secondary sources — this is the one constraint requiring fresh research.
2. **Build the hypothesis space**: for each of the top ~25 signs, enumerate a candidate meaning set (5-15 candidates each) drawn from: pictorial identity, both macro-hypotheses' predictions, the Meluhha list, and null/functional options. Explicitly include "no stable meaning" as a candidate.
3. **Joint scoring engine**: a dictionary = one assignment per sign. Score = sum of constraint satisfactions/violations (C1-C6), computed programmatically against the frozen corpus. MCMC or systematic search over the dictionary space (it is large but the constraints are cheap to evaluate).
4. **Measure the posterior**: how many dictionaries within X% of the best score? Do survivors cluster into interpretable families? Which sign-assignments are stable across ALL survivors (those are the real findings) vs free (honestly unknown)?
5. **Held-out test** (C7): do surviving dictionaries predict co-occurrence structure in the held-out 20%?
6. **Output**: a confidence-tiered lexicon — Tier 1: stable across all surviving dictionaries; Tier 2: stable within a macro-hypothesis; Tier 3: free. Plus actual annotated readings of specific seals (start with M-14, M-15 — already photo-verified).
7. **The reading test (objective 1's finish line)**: take 10 seals never discussed in any session, and produce structured partial readings with odds. If Parag can pick up CISI, open a random page, and say something true and non-trivial about what the inscription is doing — Part 3 has succeeded, publication or not.

---

## 4. Standing guardrails (learned the hard way across Parts 1-2)

- **Coherence is not evidence.** Every crank decipherment was coherent. Only survival against rivals counts.
- **LLMs (including the one reading this) are coherence-generating machines.** Never let narrative fluency substitute for a computed score. All scoring is code, not vibes.
- **Follow the analysis, not the desired conclusion.** Across Parts 1-2, four headline claims flipped under adversarial review (directional asymmetry, export replication, predictive equivalence, smoothing parity) and every flip improved the work. Expect and welcome the same here.
- **Verify before defending.** When a discrepancy is flagged, check it computationally before explaining it away. (A near-miss on exactly this occurred in the audit round: an initially-asserted "denominator explanation" for a flagged number was tested and found false — the number was simply wrong.)
- **Adversarial review loop**: draft → external AI review → run the demanded analyses → concede or rebut with computation → iterate. It works.
- **Register discipline always**: Mohenjo-daro seals first; everything else is transfer.
- **Numbers trace to the pipeline**: extend `run_all.py` (in the repro package) rather than accumulating loose scripts; every reportable number gets a results.json key.

## 5. Context on Parag (so a fresh session works well)

Orthopaedic surgeon, Bristol/Somerset; MSt Healthcare Data Science, Cambridge; this is a serious hobby ("side quest, hobby being an upgrade") — his effort budget is low, the AI's is unlimited; he supplies direction, taste, and adjudication. Voice-input, rapid, direct; low tolerance for hedging or over-explanation. Style of collaboration that worked: AI runs long autonomous loops and reports back when it has something worth his time; he round-trips drafts through external AI reviewers; honest negative results are welcomed. Deep connection exists between this project and his own frameworks (Logic of Life: organised information collapsing toward Kolmogorov complexity; the Inert Residue: in-principle unreadability) — the readable/meaningful anti-correlation in the Indus corpus is a live instance of both, and he enjoys when this is drawn out, but it must never substitute for results.

**Status of everything else:** Papers 1-2 submission-ready, harmonized to the frozen pipeline, audit passed (two stale numbers found by independent LLM review, fixed). Pre-registration frozen (preregistration-frozen.json, in outputs). ICIT access request pending Parag's email to Fuls — when granted, run the pre-registered tests before anything else. Publication actions (Zenodo DOI, RSOS submission) parked, pending Parag's admin.

*Begin Part 3 by: loading the dossiers, verifying the Meluhha vocabulary (step 1), then building the hypothesis space. Constraints first, guesses second, falsification always.*
