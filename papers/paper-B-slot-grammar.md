# Functional Sign Classes in Indus Seal Inscriptions: Out-of-Sample Validation and Cross-Register Transfer

**Parag Garg**

## Abstract

We derive a functional slot structure for Indus inscriptions using a strict train/test protocol confined to deduplicated Mohenjo-daro stamp seals, and evaluate its transfer to other archaeological strata. Sign classes (TERMINAL, INITIAL, PRE-TERMINAL, CONNECTIVE, residual CORE) are defined from positional and transition statistics in a training partition using fixed operational thresholds, locked, and scored on held-out inscriptions against four class-position constraints, including a terminal-zone rule under which terminal stacking is rule-consistent. On the designated test set, 76% of texts contain no violations versus 24% after within-text order randomisation; conditional accuracy per applicable class-token opportunity is 86%, and frequency-quantile-matched random classes of identical sizes (1,000 draws) achieve 28% compliance (null interval 12-44%) and 37% conditional accuracy, so the induced memberships, not the rule template, carry the effect. Context-similarity clustering corroborates three classes as compact paradigms when boundary context is included, but only partially (terminal 2/9, purity 2/3; pre-terminal 3/5) when boundary features are excluded, and is reported as partially independent evidence. A pre-enumerated stroke-family construction (numeral -> counted sign) is significantly enriched in held-out texts against an inventory-preserving null (6 observed vs 1.54 expected; p = 0.001). Transfer of the locked classes is strong across cities within the seal register (Harappa compliance 81%; local terminal inventory wholly contained in the transferred set, with its smaller size quantitatively predicted by downsampling) and weaker across object registers (non-seal terminal recall 52%); the pre-terminal continuation rule alone degrades at Harappa (95% to 73%). Model comparison with per-model tuned smoothing shows a bias-variance crossover: a 400-parameter class model outperforms a 145,160-parameter sign-bigram model in 97% of low-data splits but is decisively inferior at full sample size (+0.46 bits/sign); equivalence is not claimed in any regime. We register nine quantitative predictions, frozen and archived in machine-readable form, divided into locked-class transfer and fresh re-derivation tests, for independent register-matched corpora. No phonetic, linguistic or semantic assignments are made.


**Keywords:** Indus script, functional sign classes, out-of-sample validation, cross-register transfer, parameter efficiency, prospective prediction

## 1. Introduction

Statistical analyses have established directionality, positional sign classes, Zipfian frequencies, and n-gram structure in the Indus script (Mahadevan 1977; Yadav et al. 2010; Rao et al. 2009a, 2009b), and several authors have proposed qualitative segmentations of texts into functional components (Mahadevan 1986; Parpola 1994; Wells 2015; Rao 2018; Mukhopadhyay 2019). Four elements are missing from this literature: (i) sign classes derived by fixed, replicable operational definitions; (ii) out-of-sample validation with rules locked before scoring; (iii) quantitative class-level versus sign-level model comparison with uncertainty; and (iv) predictions registered in advance, with recognition rules, for unseen data. This paper supplies them.

Companion work (Garg, 2026; archived analyses at DOI 10.5281/zenodo.21497936) shows the pooled corpus is not a single statistical population. That result dictates the design: all induction is performed on one homogeneous register - deduplicated Mohenjo-daro stamp seals - and other strata (Harappa seals; Mohenjo-daro non-seal objects; foreign-findspot texts) serve as transfer tests, so the pattern of transfer and failure is itself a finding. Terminology: the primary classes are deterministic threshold-defined categories, not latent states; only the HMM comparison models involve latent variables.

## 2. Corpus and preprocessing

Data, validation, direction correction (three concordant checks: modal-sign positional fingerprint; object-level photographic verification of complete sign order; aggregate boundary-asymmetry match to published values), and deduplication follow the companion paper. Induction register: 762 deduplicated Mohenjo-daro seal texts of length >= 2 (train 609 / test 153, fixed seed). Transfer strata: 272 Harappa seal texts; 193 Mohenjo-daro non-seal texts; 7 foreign-findspot texts (exploratory only).

**Orientation for unseen corpora.** Evidence hierarchy: (1) direct archaeological orientation evidence (impression context, cramping, overlaps) governs where available; (2) otherwise, compute top-10 boundary-sign concentration at each end under both candidate orientations and select the orientation maximising end-concentration, reporting the asymmetry ratio; (3) if the ratio is below 1.5, treat orientation as undetermined and analyse both, reporting divergence.

## 3. Class induction and scored constraints

On the training partition only: TERMINAL = signs with n >= 10 and final-rate >= 0.60; INITIAL = n >= 10, initial-rate >= 0.50, not TERMINAL; PRE-TERMINAL = residual with n >= 10, >= 8 continuations, >= 0.60 of continuations in TERMINAL; CONNECTIVE = n >= 15, residual, final- and initial-rate <= 0.05; CORE = all else. On this register: |TERMINAL| = 9, |INITIAL| = 6, |PRE| = 5, |CONN| = 6, CORE = 355 sign types. Scored constraints: R1 TERMINAL tokens occur only in the terminal zone - defined as final position, or the position immediately preceding a TERMINAL token that is itself in the terminal zone (in practice, final or penultimate-before-final-TERMINAL) - a definition that renders terminal stacking rule-consistent rather than a violation, resolving the tension between the strict-final formulation and the stacking construction (P5). The zone amendment was motivated by the observed stacking phenomenon; it is therefore a data-informed rule refinement rather than a fully a priori choice, which is why both rule versions are reported and the pre-registered band (P3) spans both; R2 INITIAL only text-initial; R3 CONNECTIVE never at an edge; R4 non-final PRE-TERMINAL immediately followed by TERMINAL (final-position PRE-TERMINAL tokens are excluded from the denominator). Compliance (zero violations) is a constraint-satisfaction measure, vacuously satisfiable by texts with few classed signs; we therefore always co-report class applicability, per-opportunity conditional accuracy, and class-wise precision/recall.

## 4. Evaluation methods

Baselines: (a) within-text order randomisation (10 shuffles/text); (b) frequency-quantile-matched random classes - for each true class member, a random sign is drawn from within +/-5 frequency ranks among eligible signs; identical class sizes; 1,000 draws; applicability co-reported. Sequence models: M_sign, first-order sign bigram, (V-1) + V(V-1) = 145,160 free parameters at V = 381; M_class, deterministic-class model P(c1) prod P(ct|ct-1) prod P(st|ct), 4 + 20 + 376 = 400 free parameters, with a single global emission distribution per class (no position conditioning; all sequential information among CORE signs is deliberately discarded); M_HMM(K), EM-trained latent-state models. Smoothing is additive (alpha = 0.5) over a common vocabulary within each comparison; the vocabulary convention (per-split train vocabulary vs fixed register vocabulary) is stated with each result because the class/sign difference is sensitive to it at the 0.01-0.08 bits/sign level. Uncertainty intervals from repeated splits are reported as 95% repeated-split intervals; they summarize partition variability, not population-level confidence, and the equivalence assessment is correspondingly stated as boundary-crossing of the repeated-split interval rather than a formal TOST.

## 5. Results

### 5.1 Same-register validation

Primary locked test (153 pre-designated texts), amended terminal-zone R1: compliance 76% vs 24% shuffled; micro-averaged conditional accuracy per class-token opportunity 86% (macro 86%); rule-wise opportunities/violations: R1 132/18, R2 57/18, R3 104/6, R4 21/1 (i.e., PRE->TERMINAL continuation 20/21 = 95%). Terminal-zone precision 114/132 (86%); terminal recall (endings in the terminal class) 69%. Under the superseded strict-final R1, compliance was 71%; both are reported for continuity, and the amended rule is used throughout. Applicability 93%. Frequency-quantile-matched random classes (1,000 draws, sampled without replacement, classes disjoint): compliance 28% (null interval 12-44%) and conditional accuracy 37% (19-60%) versus induced 76% and 86% - the induced memberships carry the effect on both the text-level and opportunity-normalised metrics.

### 5.2 Context-similarity paradigms, with a circularity control

Mutual-kNN clustering (k = 3; cosine > 0.70; L2-normalised left/right neighbour count vectors over the full symbol dimension; membership identical across cosine 0.60-0.75 and k = 3-5) of vectors that include text-boundary tokens recovers the TERMINAL paradigm {90, 151, 156, 400, 520, 740}, the PRE-TERMINAL paradigm {100, 176, 760, 904, 923}, the INITIAL paradigm {817, 820, 861}, and a COUNTED paradigm {390, 405, 407} within CORE. Because boundary tokens encode the same positional signal used in induction, we repeated clustering with boundary features excluded (interior neighbours only): TERMINAL recovery falls to 2/9 (the recovered pair {90, 400} sits in a three-member family with one non-terminal sign; purity 2/3) and PRE-TERMINAL to 3/5. The clustering therefore provides partially independent corroboration - strong when boundary context is admitted, partial when it is excluded - and is not claimed as fully independent evidence. The CONNECTIVE class is dominated by one sign (glyph 2 = Mahadevan 99; n = 501) and is not recovered as a multi-sign paradigm; CORE is residual and no naturalness is claimed for it.

### 5.3 The numeral-counted-terminal construction: a semi-blind recovery

Disclosure: candidate families were graphically pre-enumerated as repeated-stroke sets (a visual taxonomy prepared before distributional analysis); the recovery is therefore semi-blind, not purely distributional. Procedure: among the pre-enumerated stroke families, one exhibited monotonically decreasing member frequencies and family final-rate < 5%; signs immediately following it were ranked by lift over unigram expectation and clustered by context; the construction was locked; enrichment was then measured on the held-out test set only and tested formally against a null preserving each text's length and sign inventory (within-text order randomisation, 2,000 permutations): 6 numeral-to-counted bigrams observed, in 6 distinct texts (pair breakdown: 5->390 x2, 5->900 x2, 4->900, 3->900), versus a null mean of 1.54; p = 0.001. Counted-family terminal-continuation 10/23 (43%). Frequency-matched control families showed no comparable follower clustering. Via the cross-coding bridge, consulted only after locking, the construction corresponds to the numeral + fish/tree + terminal template of the prior literature.

### 5.4 Class-level versus sign-level models: a bias-variance crossover

Model comparison proves sensitive to smoothing methodology, and we report the methodologically strongest version. With the additive constant tuned per model by nested validation on training data (selected: alpha = 0.05 for M_sign, 0.10 for M_class; fixed register vocabulary), the sign-bigram model is decisively better at full sample size: median Delta = L(M_class) - L(M_sign) = +0.458 bits/sign (repeated-split interval +0.342 to +0.622; class model favoured in 0% of 30 splits). At 10% of training data the ordering reverses: median Delta = -0.206 (-0.398 to -0.011), with the class model favoured in 97% of splits. An earlier analysis using a shared smoothing constant showed near-parity between the models; per-model tuning revealed this to be an artifact of sub-optimal smoothing penalising the larger model, and the comparison reported here supersedes it (analysis history archived with the code). The corrected conclusion (Figure 2) is a classical bias-variance crossover: the 400-parameter class model (4 + 20 + 376 free parameters; V = 381) is superior when data are scarce, and the 145,160-parameter sign model is superior at the full register size. The class structure therefore captures a substantial but strictly partial share of first-order predictive structure; sign-specific information beyond class membership is real and exploitable at this corpus size. Equivalence is not claimed in any regime.

**Latent-state comparison** (secondary; not parameter-matched to the deterministic-class model). Held-out bits/sign for M_HMM(K) improves monotonically from 6.95 (K = 2) to 6.37 (K = 10) with no elbow at five; five classes are the minimal interpretable skeleton, and latent gains are consistent with the COUNTED paradigm and further CORE substructure.

### 5.5 Cross-register and cross-site transfer

Under the amended terminal-zone rule, with all denominators reported (Figure 1):

| Metric | Held-out MD seals | Harappa seals | MD non-seals |
|---|---|---|---|
| Compliance (vs shuffled) | 76% (24%) | 81% (28%) | 73% (35%) |
| Applicability | 93% | 87% | 79% |
| Conditional accuracy (micro / macro) | 86% / 86% | 87% / 82% | 80% / 81% |
| Terminal recall / zone-precision | 69% / 86% | 67% / -- | 52% / -- |
| PRE->TERMINAL continuation (ok/opportunities) | 20/21 (95%) | 27/37 (73%) | 28/30 (93%) |

Terminal, initial and connective behaviour transfer across cities with little loss; the pre-terminal continuation rule degrades specifically at Harappa (95% to 73%) - transfer is strong but not uniform. Locally re-derived inventories in the transfer strata are analysed by containment and retention rather than symmetric overlap, since frequency thresholds bind at smaller n: Harappa's locally derived terminal class {90, 400, 520, 740} is wholly contained in the transferred set (containment 4/4; retention 4/9), and downsampling the Mohenjo-daro register to Harappa's sample size yields a derived terminal inventory of mean 4.9 signs (range 3-6 over 30 draws) - the smaller local inventory is quantitatively what sample size predicts, supporting a shared high-frequency terminal core rather than a divergent one. Mohenjo-daro non-seal texts show containment 2/3, retention 2/9. The seven foreign texts (terminal coverage 29%, roughly two endings) are exploratory only.

## 6. Prospective predictions

The prediction set below is frozen as of 22 July 2026 and archived in machine-readable form (preregistration-frozen.json) with the code and seeds; any subsequent amendment will be listed as such in the archive.

Predictions are divided into locked-class transfer tests (primary, requiring a compatible sign coding or completed sign map: apply the induced inventories directly) and re-derivation tests (secondary: re-run Section 3 thresholds within the new corpus). Corpus requirements: register-matched (stamp seals), home-region, n >= 200 deduplicated texts of length >= 2, orientation per Section 2, signs with n < 10 unclassed, all resampling >= 500 bootstrap replicates.

P1 (re-derivation). The modal ending sign - the single sign occupying the largest number of final positions - will take 28-45% of endings (observed 33%, interval 29-37%) and will satisfy the terminal threshold.

P2 (re-derivation). The re-derived terminal class's joint share of endings: 50-75% (observed 61%, interval 58-65%).

P3 (both; primary falsifier). Zero-violation compliance under the terminal-zone rule: 65-82% (observed 76%; band = observed value +/- approximately twice the repeated-split standard deviation, widened toward the strict-rule value of 71% to remain conservative under rule-version ambiguity in independent implementations), and >= 25 points above the within-text shuffle baseline; report applicability and conditional accuracy alongside.

P4 (exploratory; excluded from the falsification set). Given a graphically pre-enumerated repeated-stroke taxonomy prepared before analysis, at least one family of >= 3 members will show monotonically decreasing frequencies and family final-rate < 5%.

P5 (transfer). Texts ending in two consecutive terminal-class signs (locked inventory where coding permits; identical-sign repetitions excluded), as a fraction of all eligible texts: 4-18%; failure would undermine the generality of terminal stacking and weaken, not falsify, morphological interpretations.

P6 (re-derivation). An [initial-class sign] -> [high-frequency strictly-medial sign] bigram in the corpus top ten.

P7 (re-derivation). Final-position entropy (Miller-Madow estimator; length-5 stratum or nearest adequately sampled stratum; minimum count 30) lower than initial-position entropy by >= 0.3 bits.

P8 (transfer; effect size primary). Foreign-findspot texts, where present, will show elevated mean per-text surprisal under the home model; no significance claim for n < 10.

P9 (equivalence). Emblem-animal classes within the seal register will show weighted-JSD association with terminal distributions below delta = 0.10 bits - half the observed seal/non-seal register ending-JSD (0.200 bits) - with a bias-corrected bootstrap CI below delta.

Failure of P3 falsifies the induced structure; failure of P1-P2 falsifies the generality of its terminal organisation.

## 7. Discussion

A deterministic set of distributionally induced positional classes captures substantial ordering regularity in held-out Mohenjo-daro stamp-seal inscriptions. Three of the explicitly defined classes correspond to compact paradigms in contextual-similarity space, although the independence of that corroboration depends on whether boundary features are admitted to the vectors. The induced constraints transfer strongly to Harappa stamp seals and less completely to non-seal inscriptions from Mohenjo-daro, suggesting a common seal-register structure with object-specific extensions; foreign material is too sparse for more than exploratory description. A class-transition model with 0.3% of the parameters is superior under data scarcity and decisively inferior at full sample size under best-practice smoothing - a bias-variance crossover indicating that the class skeleton captures a substantial but strictly partial share of the first-order structure. These findings establish a compact, prospectively testable representation of sign ordering; they do not establish an optimal grammar, linguistic structure, phonetic values or semantic assignments.

**Applications.** Although the classes assign no meanings, they have three immediate uses. First, they permit structural parsing of arbitrary seal inscriptions: any text in the register can be segmented into its formulaic frame (initial complex, connective, terminal complex) and its variable medial payload, localising where message-bearing content, if any, resides — the medial core is also the region of highest positional entropy. Second, they constrain restoration: for a damaged or illegible sign position, the class-position rules and class-transition statistics restrict the set of class-consistent candidates, providing a principled prior for epigraphic reconstruction. Third, they function as necessary conditions on future proposals: any semantic or phonetic assignment scheme must be consistent with the class memberships, positional constraints, and cross-register transfer pattern reported here, which substantially narrows the space of admissible assignments without asserting any particular one.

## 8. Limitations

Compilation-database provenance (validated; ICIT replication pending). Compliance is constraint satisfaction, not generation. CORE is residual and internally unmodelled. Boundary-feature circularity limits the independence of the clustering evidence. Chronology uncontrolled. Foreign stratum exploratory. Repeated-split intervals are not population-level confidence intervals; object-level bootstrap wrapping the full pipeline is a planned strengthening.

## 9. Data and code availability

The complete reproducibility package - single-entry-point pipeline regenerating every cited number, canonical results file, the 85-sign cross-coding bridge, audit report, and the frozen prospective predictions - is permanently archived at DOI 10.5281/zenodo.21497936 (MIT license). Raw corpus sources are public GitHub repositories (yajnadevam/indus-website; mayig/indus-valley-script-corpus); fetch instructions and commit guidance are in the archived README.

## References

Mahadevan, I. (1977). *The Indus Script: Texts, Concordance and Tables.* Memoirs of the ASI 77. New Delhi.

Mahadevan, I. (1986). Towards a grammar of the Indus texts. *Tamil Civilization* 4(3-4), 15-30.

Ansumali Mukhopadhyay, B. (2019). Interrogating Indus inscriptions to unravel their mechanisms of meaning conveyance. *Palgrave Communications* 5(1), 73, 1-37.

Parpola, A. (1994). *Deciphering the Indus Script.* Cambridge University Press.

Rao, R. P. N., Yadav, N., Vahia, M. N., Joglekar, H., Adhikari, R., & Mahadevan, I. (2009a). Entropic evidence for linguistic structure in the Indus script. *Science* 324, 1165.

Rao, R. P. N., Yadav, N., Vahia, M. N., Joglekar, H., Adhikari, R., & Mahadevan, I. (2009b). A Markov model of the Indus script. *PNAS* 106, 13685-13690.

Rao, R. P. N. (2018). The Indus script and economics. In *Walking with the Unicorn*, 518-525. Oxford: Archaeopress.

Wells, B. K. (2015). *The Archaeology and Epigraphy of Indus Writing.* Oxford: Archaeopress.

Yadav, N., Joglekar, H., Rao, R. P. N., Vahia, M. N., Adhikari, R., & Mahadevan, I. (2010). Statistical analysis of the Indus script using n-grams. *PLoS ONE* 5(3), e9506.
