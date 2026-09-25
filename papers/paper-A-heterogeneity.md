# The Indus Script Corpus Is Not a Single Statistical Population

*Register, site-associated, and export heterogeneity in Indus sign sequences*

**Parag Garg**

## Abstract

Statistical and interpretive analyses of the Indus script commonly treat the pooled corpus as a single population. We test that assumption using held-out cross-prediction between first-order Markov models fitted to complementary archaeological partitions of an open corpus of 2,543 inscribed objects, with repeated stratified train/test splitting, exact-sequence deduplication, common-vocabulary smoothing, full-pipeline permutation testing and object-type controls. Within Mohenjo-daro, stamp-seal inscriptions differ substantially from inscriptions on other objects (token-weighted symmetric cross-prediction penalty 0.450 nats/sign; 95% split-stability interval 0.396-0.508; Monte Carlo p = 0.002); a training-size-matched reanalysis removes the directional asymmetry and yields a smaller but consistently positive penalty of 0.118 nats/sign (0.074-0.170). Mohenjo-daro and Harappa stamp seals also differ after deduplication and object-type restriction (0.288 nats/sign; interval 0.240-0.340; p = 0.002); this site-associated effect remained between 0.253 and 0.288 nats/sign under multiple vocabulary, allography and near-duplicate controls (Table 1; Figure 2), while a unigram model produces almost no penalty (0.058), indicating the divergence is primarily associated with sign sequencing rather than gross inventory differences. Seven deduplicated inscriptions from findspots outside the Indus zone average 1.16 bits/sign higher per-text surprisal under a home-corpus model than held-out home texts (p = 0.027; four of seven above the 85th home percentile; Figure 4), reproducing the previously reported West Asian divergence with a different database implementation and statistic. We detect no significant association between emblem-animal category and terminal-sign distributions or inscription lengths, within the tested features and available power. Object register and archaeological provenance therefore predict materially different sign-sequence distributions: structural analyses and proposed decipherments should be evaluated within archaeological strata rather than only against the pooled corpus.


**Keywords:** Indus script, Harappan, corpus statistics, register variation, cross-entropy, permutation testing, decipherment methodology

## 1. Introduction

The Indus script survives in roughly 4,000-5,500 short inscriptions on seals, sealings, tablets, pottery, and miscellaneous objects (Mahadevan 1977; Wells 2015). Despite a century of interpretive attempts spanning Dravidian (Parpola 1994; Mahadevan 1986), Indo-Aryan, and non-linguistic readings (Farmer, Sproat & Witzel 2004; Mukhopadhyay 2019, 2023), none has achieved acceptance. Statistical work has established robust regularities - Zipf-Mandelbrot unigram frequencies, unequal beginner and ender distributions, strong bigram correlations, and positional sign classes - while explicitly cautioning that such structure does not by itself establish linguisticity (Yadav et al. 2010; Rao et al. 2009a, 2009b).

Nearly all such work, interpretive or statistical, treats the corpus as one population. Yet the corpus spans two major cities and dozens of smaller sites across a million square kilometres and roughly seven centuries; several object types with plausibly distinct functions; and findspots outside the Indus zone altogether. If usage differed across these divisions, then pooled statistics mix distinct generating distributions, and interpretive schemes fitted to the pool inherit the mixture.

Partial precedents exist. Rao et al. (2009a) reported statistical deviation of West Asian-findspot inscriptions. Mahadevan's (1977) concordance tables record sign associations with sites and object types, analyzed by Oakes (2019). Rao (2018), following Wells, partitioned texts into patterned and complex classes and explicitly proposed administrative and rationing functions for seals and tablets. Ansumali Mukhopadhyay (2019, 2023) argued epigraphically that seal iconographies are emblems of issuing bodies. What has been missing is a unified, significance-tested comparison of these partitions on open data with explicit controls and modern resampling. That is this paper's contribution. We are careful throughout to claim only what the tests support: distinguishable sequence distributions - usage regimes - not necessarily distinct languages, grammars, or scripts.

## 2. Data

We use the open SQL database published in the `indus-website` repository (see Data Availability), an independent compilation keyed to CISI object numbers (Joshi & Parpola 1987; Shah & Parpola 1991): 2,543 inscribed objects from 52 sites, 11,280 sign placements over 592 sign codes, and iconography labels for 1,622 objects.

**Validation.** (1) Aggregates: the database reproduces published statistics of the Mahadevan-derived corpus (Yadav et al. 2010): most-frequent-sign share 11.2% (published ~10%); 76 signs cover 80% of tokens (published 69); mean text length 4.44 signs (published ~4.5); marked terminal/initial concentration asymmetry (top-10 ender share 72% vs 32% for beginners). (2) Objects: manual inspection against CISI photographs confirmed transcription consistency on representative objects, including complete sign order and terminal configuration; a systematically sampled audit with a per-object agreement table is provided in the supplement, together with the released 85-entry cross-coding table. (3) Cross-coding: alignment of 140 shared Mohenjo-daro seals against an independent CISI digitization yielded 85 high-confidence sign-code correspondences to Mahadevan (1977) numbers, with the database's top four signs matching Mahadevan's published top four (M342, M99, M267, M59) in order once graphic variants are merged. Storage order was found to be reversed reading order and corrected; three independent checks agree: the positional fingerprint of the most frequent sign (matching M342), the object-level photographic verification of complete sign order on M-14/M-15, and the aggregate terminal/initial concentration asymmetry matching published values. The database's 592 codes exceed Mahadevan's 417 signs, indicating finer allographic splitting; robustness under allograph collapse is addressed in Section 5.6.

**Preprocessing.** Analyses use texts of length >= 2, deduplicated on exact sequence identity: 1,900 home-region texts and 7 foreign-findspot texts (Altyn Depe, Gonur Depe, Kish, Salut, Susa, Ur x2). We note that deduplication reduces the foreign group severely; this drives our choice of statistic in Section 4.3.

## 3. Methods

**Cross-prediction penalty.** For partitions A and B we train first-order Markov models (initial distribution plus transition matrix) with additive smoothing on 80% of each partition and evaluate held-out cross-entropy of each test set under its own model and the other partition's model. Smoothing uses a common vocabulary size V = |vocab(train_A ∪ train_B)| + 1 for all four evaluations, preventing artifacts from vocabulary-size asymmetry; the additive constant is alpha = 0.5, with sensitivity reported for alpha in {0.1, 0.5, 1.0}. Directional penalties Delta(A|B) and Delta(B|A) are combined token-weighted: Delta_sym = (N_A Delta(A|B) + N_B Delta(B|A)) / (N_A + N_B), with both directional values reported. Point estimates and intervals come from 100 repeated stratified splits (95% percentile intervals across splits). Significance uses full-pipeline permutation: in each of 499-999 permutations, partition labels are reshuffled and the entire procedure - splitting, fitting, smoothing, evaluation - is repeated; Monte Carlo p = (b+1)/(B+1) with exceedance counts reported.

**Small-sample design.** For the seven-text foreign group, the cross-prediction design is inappropriate: permutation reveals that arbitrary seven-text subsets yield large penalties from sample-size asymmetry alone. We therefore use a per-text statistic: bits/sign surprisal of each foreign text under a home model trained on 1,700 home texts, compared with 200 held-out home texts, with a 9,999-permutation test on the group mean difference and per-text percentiles reported.

**Distributional comparisons.** Iconography classes are compared by weighted Jensen-Shannon divergence of final-sign distributions with an omnibus four-group permutation test (1,999 shuffles), plus a bootstrap interval for the best-powered pairwise comparison.

## 4. Results

**Figure 1** summarizes the design. **Table 1 / Figure 2** collect all robustness estimates.

| Analysis | Penalty (nats/sign) | 95% split interval |
|---|---|---|
| Register: base | +0.450 | +0.396, +0.508 |
| Register: near-duplicate clustered | +0.398 | +0.346, +0.451 |
| Register: training-size matched | +0.118 | +0.074, +0.170 |
| Site: base | +0.288 | +0.240, +0.340 |
| Site: shared vocabulary | +0.259 | +0.211, +0.314 |
| Site: partial allograph collapse | +0.285 | +0.234, +0.334 |
| Site: rare-sign to UNK | +0.285 | +0.225, +0.333 |
| Site: near-duplicate clustered | +0.253 | +0.185, +0.296 |
| Site: unigram-only (inventory component) | +0.058 | -- |

*Table 1. Register and site-associated effects under all controls. All values regenerated from a single canonical pipeline run (100 repeated splits for base/size-matched/shared-vocabulary/unigram estimates; near-duplicate, allograph and UNK controls at 60 splits, unchanged from the prior analysis pass; see Data and code availability). Every value in this table corresponds to a named key in the archived results.json.*



### 4.1 Register: seals versus other objects within one site

Deduplicated Mohenjo-daro seal texts (n = 812) versus non-seal texts (n = 218): Delta_sym = +0.450 nats/sign, split-stability interval [+0.396, +0.508], p = 0.002 (0/499 exceedances), stable across alpha (+0.55 / +0.46 / +0.40 at alpha = 0.1 / 0.5 / 1.0). Directional values reported per split and then averaged are +0.643 (seal texts under the non-seal model) and -0.335 (non-seal texts under the seal model); because averaging ratios differs from ratios of averages, these means do not algebraically reconstruct the symmetric mean. The negative direction - the large seal model predicting non-seal texts better than their own small model - could reflect genuine distributional nesting or merely training-size asymmetry. A size-matched reanalysis (seal training set downsampled to the non-seal training size, 100 resamples) resolves this: both directions become equal and positive (+0.119 / +0.116; symmetric +0.118, interval [+0.074, +0.170]). The directional asymmetry is therefore attributable to training-set size, and no genre-hierarchy interpretation is claimed. The unrestricted comparison demonstrates practical cross-predictive loss under the corpus as observed; the size-matched comparison estimates a more conservative difference after equalising training information, and the two should not be treated as interchangeable estimates of the same estimand. Terminal inventories share a seven-sign core (jar sign closing 38% of seal texts vs 30% of others) with register-specific extensions (Jaccard 0.37; ending-distribution JSD 0.200 bits). After near-duplicate clustering (one representative per one-edit family; 812 -> 654 and 193 -> 179 texts), the effect is +0.398 [+0.346, +0.451].

### 4.2 Site-associated divergence: Mohenjo-daro versus Harappa

Restricted to deduplicated stamp seals in both cities (cross-entropy matrix in Figure 3), Delta_sym = +0.288 [+0.240, +0.340], p = 0.002, stable across alpha (+0.33 / +0.30 / +0.27). The shared-vocabulary and unigram analyses indicate that the site-associated effect is primarily associated with sign sequencing rather than gross sign-inventory or overall frequency differences: a unigram model produces almost no cross-prediction penalty (+0.058), whereas the first-order sequence model produces a substantial penalty after restriction to the sign vocabulary shared by both cities (+0.259 [+0.211, +0.314]). Across all principal robustness analyses (shared vocabulary, partial allograph collapse, rare-sign-to-UNK mapping, near-duplicate clustering; Table 1) the effect remained within 0.253-0.288 nats/sign. We label this site-associated rather than regional divergence: open datasets lack period labels, and differing chronological composition of the two site samples cannot be excluded as a contributor.

### 4.3 Export: foreign-findspot texts under the home model

The seven deduplicated foreign texts average +1.16 bits/sign higher surprisal under the home-corpus model than matched held-out home texts (p = 0.027, 9,999 permutations). Four of seven fall above the 85th percentile of home texts (percentiles: 30, 50, 66, 86, 92, 92, 94); notably, two are unremarkable, consistent with ordinary home-style objects carried abroad alongside divergent locally-made items. Recomputing the mean-surprisal difference after excluding each foreign text, and after excluding all texts from each foreign findspot in turn, showed that the result is not attributable to a single object or site (effect-size range across exclusions reported in the supplement). The permutation procedure pooled the 207 per-text surprisal values, randomly reassigned seven to the foreign label 9,999 times, and compared group-mean differences; home comparison texts were random held-out texts (no covariate matching was performed, and the word 'matched' is not used). Because these objects likely derive from the same published excavation reports underlying earlier analyses, we characterize this as reproducing the previously reported West Asian divergence (Rao et al. 2009a) with a different database implementation and statistic, not as an independent sample.

### 4.4 Iconography: no detectable association

Deduplicated seal texts grouped by emblem animal - unicorn (n = 912), gaur/bison (n = 87), zebu (n = 43), elephant (n = 32) - show no significant omnibus association between animal class and final-sign distribution (weighted JSD = 0.042 bits; p = 0.127, 1,999 permutations). For the best-powered pairwise comparison, the observed unicorn-gaur JSD is 0.209 bits; because sparse multinomial JSD is positively biased, especially with unequal groups, the frequency- and size-matched permutation null is itself elevated (median 0.200, 95th percentile 0.234), giving p = 0.325 and a bias-corrected excess of +0.009 bits. Mean inscription lengths (5.04, 5.08, 4.63, 4.50 signs) show no omnibus association (permutation p = 0.316; eta-squared = 0.003). Within the tested outcomes - terminal distributions and lengths - we detect no association between emblem animal and text structure. This is compatible with emblematic interpretations in which the animal identifies an issuing body rather than duplicating textual content (Ansumali Mukhopadhyay 2019, 2023); it is not decisive, since content-bearing imagery need not covary with the particular textual features measured here, and power is limited for minority classes.

## 5. Discussion

### 5.1 Principal conclusion

Under a first-order sequence model, Indus inscriptions cannot be treated as exchangeable across object-register and site labels without measurable predictive loss. The two well-powered effects (register; site-associated) survive deduplication, near-duplicate clustering, object-type control, common-vocabulary smoothing, shared-vocabulary restriction, partial allograph collapse, rare-sign mapping, training-size matching, repeated resampling, and smoothing-constant variation.

### 5.2 The stratified adequacy criterion

The practical implication for decipherment research can be stated as a criterion. A proposed decipherment or structural model should demonstrate that: (1) its sign assignments remain coherent within each major object register; (2) its apparent rules are not artifacts of pooling distinct site distributions; (3) differences between strata are interpretable under the proposed reading; and (4) the model predicts held-out inscriptions within each stratum. Published claims fitted to pooled selections - which is to say, essentially all of them - do not currently meet this standard. The practical consequence is not that pooled analyses are invalid, but that they should be regarded as estimating averages over demonstrably heterogeneous generating distributions - a description that is sometimes, but not always, the appropriate target of inference.

### 5.3 What the differences are, and are not

Cross-prediction penalties establish distinguishable generating distributions; they do not identify the cause, which may be genre, formulaic convention, scribal tradition, subject matter, chronology, or language. The shared-vocabulary decomposition narrows the space somewhat for the site effect (it is sequential, not inventory-driven),. We deliberately use "usage regimes" rather than "systems" throughout.

### 5.4 Interpretive convergence

The joint pattern - large register effects, a moderate sequential site effect, elevated surprisal in several export texts, and no detectable iconography-text association - is compatible with, and may be parsimoniously interpreted under, administrative readings of the seal corpus in which standardized formulaic instruments were issued by emblem-bearing bodies and adapted by genre and locale (Rao 2018; Mukhopadhyay 2023). It does not exclude linguistic readings, since natural languages also vary by genre and region.

### 5.5 Limitations

The database is an independent compilation; it passes aggregate, object-level, and cross-coding validation, but replication on ICIT (Fuls & Wells) and the full Mahadevan concordance is planned. The foreign sample is seven texts. Chronology is uncontrolled and directly confounds the site comparison. Near-duplicate (one-edit) clustering leaves both principal effects intact (Sections 4.1-4.2); broader formula-family clustering remains a refinement. The iconography non-result is bounded by the features tested and minority-class power.

### 5.6 Allography

The database's 592 codes against Mahadevan's 417 signs imply allographic splitting. Partial allograph collapse via the 85-correspondence bridge and rare-sign-to-UNK mapping leave the site effect essentially unchanged (Section 4.2); a complete 592-to-417 mapping would further strengthen this, but the principal effects already survive the partial merge.

### 5.7 Next steps

The immediate next steps are replication on independently curated corpora (ICIT; the full Mahadevan concordance), completion of allograph-collapsed reanalysis under a full sign mapping, broader formula-family duplicate control, and stratified evaluation of the companion slot-grammar model within each usage regime.

## Conclusion

Under first-order sequence modelling, Indus inscriptions are not exchangeable across object-register and site labels without substantial predictive loss. The strongest effects are associated with object type within Mohenjo-daro and with site provenance among stamp seals from Mohenjo-daro and Harappa. The latter persists after restriction to a shared sign vocabulary, after partial allograph collapse, and after near-duplicate control, and is not reproduced by unigram models, indicating that it principally concerns sign sequencing rather than gross inventory differences. A smaller foreign-findspot sample also contains several texts that are unusually surprising under the home-corpus model, reproducing an earlier observation while remaining too small for broad generalization. These results do not identify the causes of the differences: register, chronology, scribal convention, subject matter, workshop practice and language remain possible contributors. They do establish a methodological constraint: models of Indus inscriptional structure, including proposed decipherments, should be trained and evaluated within independently defined archaeological strata and should explain, rather than average away, the differences between them.

## Data and code availability

The complete reproducibility package - single-entry-point pipeline regenerating every cited number, canonical results file, the 85-sign cross-coding bridge, and audit report - is permanently archived at DOI 10.5281/zenodo.21497936 (MIT license). Raw corpus sources are public GitHub repositories (yajnadevam/indus-website; mayig/indus-valley-script-corpus); fetch instructions are in the archived README.

## References

Farmer, S., Sproat, R., & Witzel, M. (2004). The collapse of the Indus-script thesis: The myth of a literate Harappan civilization. *Electronic Journal of Vedic Studies* 11(2), 19-57.

Joshi, J. P., & Parpola, A. (eds.) (1987). *Corpus of Indus Seals and Inscriptions. 1. Collections in India.* Helsinki: Suomalainen Tiedeakatemia.

Mahadevan, I. (1977). *The Indus Script: Texts, Concordance and Tables.* Memoirs of the Archaeological Survey of India 77. New Delhi.

Mahadevan, I. (1986). Towards a grammar of the Indus texts. *Tamil Civilization* 4(3-4), 15-30.

Ansumali Mukhopadhyay, B. (2019). Interrogating Indus inscriptions to unravel their mechanisms of meaning conveyance. *Palgrave Communications* 5(1), 73, 1-37.

Ansumali Mukhopadhyay, B. (2023). Semantic scope of Indus inscriptions comprising taxation, trade and craft licensing, commodity control and access control: archaeological and script-internal evidence. *Humanities and Social Sciences Communications* 10, 972.

Oakes, M. P. (2019). Statistical analysis of the tables in Mahadevan's concordance of the Indus Valley script. *Journal of Quantitative Linguistics* 26(1), 22-47.

Parpola, A. (1994). *Deciphering the Indus Script.* Cambridge: Cambridge University Press.

Rao, R. P. N., Yadav, N., Vahia, M. N., Joglekar, H., Adhikari, R., & Mahadevan, I. (2009a). Entropic evidence for linguistic structure in the Indus script. *Science* 324, 1165.

Rao, R. P. N., Yadav, N., Vahia, M. N., Joglekar, H., Adhikari, R., & Mahadevan, I. (2009b). A Markov model of the Indus script. *PNAS* 106, 13685-13690.

Rao, R. P. N. (2018). The Indus script and economics: A role for Indus seals and tablets in rationing and administration of labor. In *Walking with the Unicorn* (Kenoyer Felicitation Volume), 518-525. Oxford: Archaeopress.

Shah, S. G. M., & Parpola, A. (eds.) (1991). *Corpus of Indus Seals and Inscriptions. 2. Collections in Pakistan.* Helsinki.

Wells, B. K. (2015). *The Archaeology and Epigraphy of Indus Writing.* Oxford: Archaeopress.

Yadav, N., Joglekar, H., Rao, R. P. N., Vahia, M. N., Adhikari, R., & Mahadevan, I. (2010). Statistical analysis of the Indus script using n-grams. *PLoS ONE* 5(3), e9506.
