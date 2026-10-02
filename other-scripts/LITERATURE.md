# Literature on three other undeciphered systems

Proto-Elamite, Linear A and the Voynich manuscript. Compiled 2 Oct 2026.

Scope: the most recent work (2020–2026) plus key older baselines. For each item:
**M** = method and data, **R** = result or claim, **C** = control, test or rebuttal.
We use these papers for their data and methods. Following our rules of evidence, we do not adopt their interpretations.

Sources checked: Crossref API, Semantic Scholar API, arXiv export API, ACL Anthology, CEUR-WS, publisher pages, and full text where it was open (Kelley et al. 2022; Davis 2024; Gaskell & Bowern 2022; Bowern & Gaskell 2022; Boxer 2022).
Items marked *(abstract only)* were read only at abstract or press level.

---

## 1. Proto-Elamite (Iran, c. 3100–2900 BCE; c. 1,700 tablets)

1. **Damerow, P. & Englund, R. K. 1989.** *The Proto-Elamite Texts from Tepe Yahya.* American School of Prehistoric Research Bulletin 39. Review: https://doi.org/10.2307/282252
   - M: close reading of the tablets, plus arithmetic reconstruction of the numerical sign systems (sexagesimal, bisexagesimal, capacity and others), checked against totals.
   - R: Proto-Elamite uses several numerical systems that depend on the commodity, inherited from proto-cuneiform. These systems are the best-understood part of the script.
   - C: internal arithmetic control (entries must add up to the stated totals). This is the standard baseline and it has not been rebutted.

2. **Englund, R. K. 2004.** "The state of decipherment of proto-Elamite." In S. Houston (ed.), *The First Writing*, CUP, 100–149.
   - M: survey of the corpus, the sign list (based on Meriggi), the numerical systems and the tablet formats (header, then entries, then total).
   - R: the script is administrative. Signs fall into numerals, "object" or commodity signs, and a large residue of non-numerical signs. Some sign sequences may be personal names.
   - C: descriptive survey. It is the baseline that later computational work tries to replicate.

3. **Dahl, J. L. 2009.** "Early writing in Iran, a reappraisal." *Iran* 47. https://doi.org/10.1080/05786967.2009.11864757. **Dahl 2018**, "The proto-Elamite writing system," in *The Elamite World*, https://doi.org/10.4324/9781315658032-20. **Dahl 2019**, *Tablettes et fragments proto-élamites* (TCL 32, Louvre). Also the CDLI Proto-Elamite corpus and sign list (Oxford "Proto-Elamite project", with RTI imaging).
   - M: a new edition of the Louvre tablets and CDLI transliterations, sign-list revision, and analysis of how signs are used in entries.
   - R: Dahl proposes a small "syllabary", a set of common signs that spell names, so Proto-Elamite would be partly glottographic. He also gives a revised view of how scribes were organised.
   - C: no formal statistical control. Kelley et al. 2022 (item 7) find that Dahl's syllabary and Desset's sign correspondences overlap only partly. The syllabary is still a hypothesis.

4. **Born, L., Kelley, K., Kambhatla, N., Chen, C., Sarkar, A. 2019.** "Sign clustering and topic extraction in Proto-Elamite." LaTeCH@NAACL. https://doi.org/10.18653/v1/w19-2516
   - M: CDLI transliterations; hierarchical clustering of signs by context, n-gram frequencies and LDA topic models.
   - R: replicates groupings that experts had found by hand (for example commodity sign families) and suggests new sign relationships.
   - C: the control is replication of known expert findings. There is no null model of a non-linguistic system.

5. **Born, L., Kelley, K., Monroe, M. W., Sarkar, A. 2021.** "Compositionality of complex graphemes in the undeciphered Proto-Elamite script using image and text embedding models." *Findings of ACL 2021.* https://doi.org/10.18653/v1/2021.findings-acl.362 (This is the "Born & Kelley 2021" paper.)
   - M: a language model over sequences of sign images (and image plus label), used to study complex graphemes (ligatures).
   - R: complex graphemes are at least partly compositional, and new formation rules were found. A model over images works better than one over sign names, which suggests that **the conventional sign names hide signal**.
   - C: compares text-only, image-only and multimodal models. Not rebutted.

6. **Born, Monroe, Kelley, Sarkar 2022.** "Sequence models for document structure identification in an undeciphered script." EMNLP 2022. https://aclanthology.org/2022.emnlp-main.620/
   - M: unsupervised neural and n-gram sequence models over CDLI texts to test the expert category of "header" signs.
   - R: header signs are confirmed without supervision. The paper also finds which features predict a header and flags tablets where the experts' labels and the model disagree.
   - C: the expert labels serve as an independent check. This is a good template for testing a "frame slot" in a script.

7. **Kelley, K., Born, L., Monroe, M. W., Sarkar, A. 2022.** "On newly proposed Proto-Elamite sign values." *Iranica Antiqua* 57. https://doi.org/10.2143/IA.57.0.3291506 (PDF: https://anoopsarkar.github.io/papers/pdf/IA57001.pdf)
   - M: applies Desset et al.'s Linear Elamite sound values to the whole Proto-Elamite corpus and checks the resulting "readings" in context.
   - R: no decipherment follows. A sizeable minority of the proposed signs are rare, hapax or even absent from CDLI. **61 of the 86 proposed correspondences are signs that can stand alone in entries**, so any sound value would apply only sometimes. Repeated n-grams become very few once object signs are excluded.
   - C: this is itself the test, and its outcome is negative or unproven for transferring LE values to PE.

8. **Desset, F., Tabibzadeh, K., Kervran, M., Basello, G. P., Marchesi, G. 2022.** "The decipherment of Linear Elamite writing." *ZA* 112: 11–60. https://doi.org/10.1515/za-2022-0003
   - M: eight inscribed silver beakers (published 2018) with recurring royal names and titles known from cuneiform (Puzur-Sušinak, Šilhaha and others). Values were set by aligning names, then extended. Linear Elamite was written c. 2300–1880 BCE.
   - R: a near-complete phonetic reading of Linear Elamite as the Elamite language (an alpha-syllabary of about 80–110 graphemes). The paper also gives a chart linking some LE signs to PE signs and says PE and LE may be "one system at two stages".
   - C: the anchor is bilingual names, which is the strongest kind of control. The LE reading is broadly accepted but still disputed in places (Dahl contests the completeness). The PE link is **not supported** by Kelley et al. 2022 and is disputed by Dahl 2023.

9. **Dahl, J. L. 2023.** "Proto-Elamite and Linear Elamite, a misunderstood relationship?" *Akkadica.* https://doi.org/10.21825/akkadica.99694 *(abstract and summary only)*
   - M: chronology and palaeography.
   - R: PE (3100–2900) and LE (2100–1900) are separated by about 500+ years with no continuous use. LE scribes may have reinvented a script, using old PE tablets and Old Akkadian cuneiform as models. Graphic similarity therefore does **not** imply that sound values carried over.
   - C: argument, not a statistical test. It directly opposes Desset's continuity claim.

10. **Born, Monroe, Kelley, Sarkar 2023.** "Disambiguating numeral sequences to decipher ancient accounting corpora." CAWL 2023, https://aclanthology.org/2023.cawl-1.9/ (arXiv 2502.00090). Also **"Learning the character inventories of undeciphered scripts using unsupervised deep clustering"**, CAWL 2023, https://aclanthology.org/2023.cawl-1.11/
    - M: (a) list every possible reading of each PE numeral notation under the 4+ numerical systems, then disambiguate using document structure (totals) and bootstrapped classifiers, with a hand-built test set. (b) deep clustering of sign images to recover the sign inventory, validated on known modern and ancient scripts first.
    - R: (a) confirms experts' intuitions about which system goes with which commodity, and finds new correlations between tablet content and the size of numbers. Numerals outnumber text tokens in PE. (b) the models beat the previous state of the art at recovering inventories, and are used to test PE sign-variant hypotheses.
    - C: (a) a held-out test set; (b) calibration on scripts with known inventories. Both are good control designs to copy.

11. **Monroe, M. W., Kelley, K., Born, L., Sarkar, A. 2025.** "Recent progress in deciphering Proto-Elamite." *Near Eastern Archaeology* 88(4). https://doi.org/10.1086/738240. **Kelley, K. 2026.** *Proto-Elamite* (Cambridge Elements). https://doi.org/10.1017/9781009614559 *(abstract only)*
    - M: synthesis of the computational programme (items 4–7, 10), mathematics and comparative study, plus digitisation of the corpus.
    - R: "only very partly deciphered". Progress is in the numerals, document structure, sign inventory and commodity semantics, not in a language reading.
    - C: these are reviews, and they are the current state of the field.

Other: Jamshidi Yeganeh et al. 2025, *JAS Reports*, https://doi.org/10.1016/j.jasrep.2025.104973 (compositional clay data from sealings and tablets; administration, not script). We found **no 2023–2026 paper claiming a phonetic decipherment of PE that has passed review.**

### Lessons for our approach: Proto-Elamite
- **Already done (do not claim novelty):** sign clustering and n-grams (2019); header detection by sequence models (2022); numeral-system disambiguation using totals (2023); compositional ligatures (2021); unsupervised sign-inventory recovery (2023); applying LE values to PE (2022, negative). Commodity-specific numeral systems have been known since 1989. Our S70 "two counting systems" is a **parallel** to that finding, not a discovery about PE.
- **Open questions:** does PE contain a syllabary at all, and if so which signs? What do the non-numerical "residue" signs in entries encode (persons, offices, institutions)? Is there a principled, frequency-calibrated test of the Dahl syllabary? (None was published as of 2026.)
- **Datasets they used:** CDLI ATF transliterations (the same source as our `proto-elamite/data/pe_raw.atf`) and CDLI images. Their numeral test set is described in the CAWL 2023 paper.
- **Methods to copy as controls:**
  1. Born et al. 2022: confirm an expert slot category (for us, the GRAMMAR.md opener, marker and closer slots) with an unsupervised sequence model, and list the disagreements.
  2. Born et al. 2023: list the possible readings and use document-internal totals as the arbiter. Indus has no totals, so any numeral claim needs a different internal check.
  3. Kelley et al. 2022: before trusting a cross-script sign match, count how many proposed signs are hapax or rare, and how many can stand alone as logograms. Apply this to any Indus–foreign sign correspondence (ANCHORS.md §19–21).

---

## 2. Linear A (Crete, c. 1800–1450 BCE; about 1,400 inscriptions, about 7,500 signs)

1. **Bennett, E. L. 1950.** "Fractional quantities in Minoan bookkeeping." *AJA* 54. **Packard, D. W. 1974.** *Minoan Linear A* (reprinted 2023). https://doi.org/10.2307/jj.8501487
   - M: Packard did an early computer frequency analysis. He compared sign frequencies and positions in Linear A with Linear B to test whether Linear B values carry over. Bennett did the first systematic work on the fraction signs.
   - R: Packard found that the frequencies of homomorphs are consistent with LB values being roughly right. KU-RO is the "total" word, because the numbers it heads equal the sum of the entries.
   - C: KU-RO = total rests on an arithmetic control (sums match) and is accepted. LB-value transfer is supported statistically, but it gives no known language.

2. **Davis, B. 2013/2014.** "Syntax in Linear A: the word-order of the 'Libation Formula'." *Kadmos* 52. https://doi.org/10.1515/kadmos-2013-0003. Also *Minoan Stone Vessels with Linear A Inscriptions* (Aegaeum 36, 2014).
   - M: distributional analysis of the recurring formula on stone libation vessels, using LB values for the signs.
   - R: the Minoan language is argued to be **verb-initial (VSO)**, with prefixing and suffixing morphology. This is a language property obtained without a translation.
   - C: internal comparison of the formula's variants. It depends on the segmentation and on LB values. It has not been refuted, but it has not been independently replicated either.

3. **Davis, B. 2018.** "The Phaistos Disk: a new way of viewing the language behind the script." *Oxford J. Archaeology.* https://doi.org/10.1111/ojoa.12151. **Davis, B. 2024.** "Investigations into the language(s) behind Cretan Hieroglyphic and Linear A." In Civitillo, Ferrara & Meissner (eds), *Cretan Hieroglyphic*, CUP, ch. 8. https://doi.org/10.1017/9781009490122.010
   - M: **syllabotactics.** Find homomorphs in two scripts, tabulate which word-internal sign pairs occur in each, and score the overlap (out of 60 pairs). Then build a null distribution from 1,000,000 random relabellings of the homomorph set.
   - R (2024): the **control** works. Cypriot Syllabic (Greek) against a Linear B benchmark gives 52/60, +2.27σ, p = 0.023. Cypriot against Linear A gives 30/60, p = 0.72. Main test: Cretan Hieroglyphic against Linear A gives 32/60, +2.91σ, p = 0.0028, while Cretan Hieroglyphic against Linear B is not significant. Conclusion: CH and LA probably encode the same language, and it is not Greek. (2018: the Phaistos Disk and LA also match.)
   - C: it has a built-in positive and negative control, which is exemplary. The weaknesses are a single small target text, only 10–21 homomorphs, and one-sided p-values that are not corrected across tests.

4. **Steele, P. M. & Meißner, T. 2017.** "From Linear B to Linear A: the problem of the backward projection of sound values." In Steele (ed.), *Understanding Relations Between Scripts.* https://doi.org/10.2307/j.ctvh1dr51.11
   - M: a critique of the method, using script-adaptation cases such as cuneiform and Cypriot.
   - R: when a script is adapted, signs can change value. Projecting LB values back onto LA is therefore an assumption, not a fact, and it is safer for common signs than for rare ones.
   - C: methodological. It is the standard caution, and Davis's homomorph method tries to answer it.

5. **Salgarella, E. 2020.** *Aegean Linear Script(s): Rethinking the Relationship between Linear A and Linear B.* CUP. https://doi.org/10.1017/9781108783477 (review: Bennet 2021, https://doi.org/10.1017/s0009840x21002614). **Salgarella 2019**, "Drawing lines," *Kadmos*, https://doi.org/10.1515/kadmos-2019-0004. **Salgarella 2025**, *Writing in Bronze Age Crete* (Cambridge Element), https://doi.org/10.1017/9781009520041
   - M: systematic palaeography (sign-shape variation by site and scribe) and a structural comparison of the two scripts' administrative uses.
   - R: Linear B is an adaptation of Linear A, with some reinterpretation and reorganisation of signs. Homomorphs mostly keep approximate values. Linear A can be "read" (roughly sounded out) but not understood.
   - C: qualitative, and well received. It gives the sign-identity base that statistical tests need.

6. **Salgarella, E. & Castellan, S. 2021.** "SigLA: The Signs of Linear A. A palæographical database." *Grapholinguistics and Its Applications* 4. https://doi.org/10.36824/2020-graf-salg. Database: https://sigla.phis.me/
   - M: every sign token is drawn as a vector and tagged with document, site, scribe, position and LB homomorph.
   - R: this is infrastructure. It allows allograph studies and per-site sign statistics.
   - C: not applicable. **It is the dataset to use for any sign-level Linear A test.**

7. **Corazza, M., Ferrara, S., Montecchi, B., Tamburini, F., Valério, M. 2021.** "The mathematical values of fraction signs in the Linear A script: a computational, statistical and typological approach." *J. Archaeological Science* 125: 105214. https://doi.org/10.1016/j.jas.2020.105214 (ERC INSCRIBE)
   - M: four independent strands: palaeography, distribution statistics (which fractions combine, and which totals must close), a computational constraint search, and typology (attested fraction systems elsewhere).
   - R: a systematic set of values for the fraction signs. Values for J (1/2), E (1/4), F (1/8) and others were confirmed, and the problematic ones were assigned.
   - C: the four strands "converge", which works as a cross-check. The values have been accepted widely but not universally. This is the best example of **converging independent methods on a non-linguistic sub-system**.

8. **Corazza, M., Tamburini, F., Valério, M., Ferrara, S. 2022.** "Unsupervised deep learning supports reclassification of Bronze Age Cypriot writing system." *PLOS ONE.* https://doi.org/10.1371/journal.pone.0269544. Also Ravanelli, Corazza et al. 2025, "New methods for old worlds," https://doi.org/10.4324/9781003584155-13 *(abstract only)*
   - M: an unsupervised convolutional network on sign images of Cypro-Minoan, a sister script, with no prior labels.
   - R: the conventional three-way split CM1/CM2/CM3 is not supported. The differences are better explained by medium and palaeography than by different languages.
   - C: unsupervised by design. It shows how sign-list and allograph decisions can be checked from images.

9. **Luo, J., Cao, Y., Barzilay, R. 2019.** "Neural decipherment via minimum-cost flow: from Ugaritic to Linear B." ACL. https://doi.org/10.18653/v1/p19-1303. **Luo, Hartmann, Santus, Cao, Barzilay 2021.** "Deciphering undersegmented ancient scripts using phonetic prior." *TACL.* https://doi.org/10.1162/tacl_a_00354
   - M: cognate alignment between a lost script and a **known related language**. 2021 adds IPA-geometry embeddings and joint segmentation.
   - R: Linear B to Greek gives 67.3% of cognates right. Ugaritic and Gothic are recovered. For Iberian there is **no strong support for Basque**, which agrees with scholarly consensus. These models have not been applied to Linear A with a positive result.
   - C: proper positive controls on known decipherments. The limitation is that the method needs a candidate relative, which Linear A lacks.

10. **Nepal, A. & Perono Cacciafoco, F. 2024.** "Minoan cryptanalysis: computational approaches to deciphering Linear A…" *Information* 15(2): 73. https://doi.org/10.3390/info15020073. **Manoj, P. & Perono Cacciafoco, F. 2025.** "Minoan and the machines," *Analele Univ. Craiova, Lingvistică.* https://anale-lingvistica.reviste.ucv.ro/index.php/laucv/article/view/134
    - M: feature-based sign-shape similarity (LA against Carian and Cypriot), then consonant-skeleton matching of LA words against lexicons of Egyptian, Luwian, Hittite, Proto-Celtic, Uralic and others. The 2025 paper is a review.
    - R: lists of "possible matches" in several languages, with no single winner.
    - C: **no proper null.** With many languages and loose consonant matching, chance matches are expected. Treat this as a cautionary example.

11. **Grey literature, 2026.** Papanikolaou, "Structure before language: a pre-registered computational study of Minoan Linear A," Zenodo, https://zenodo.org/records/22867213. Not peer reviewed. Also press reports of an engineer's "Minoan is Semitic" claim (GreekReporter, Aug 2026). Not peer reviewed and uncontrolled.
    - The Zenodo study has locked pre-registration and a same-lexicon control. A Knossos-derived Mycenaean lexicon covers 18.2% of Pylos words but only 9.2% of readable LA words, so **Greek is rejected**. Unsupervised sign-role recovery fails (ARI 0.007). Six other language hypotheses were inconclusive. This is useful as a design pattern only, until it is reviewed.

### Lessons for our approach: Linear A
- **Already done (do not claim novelty):** KU-RO = total by arithmetic; fraction values by converging methods (2021); LB-value transfer for homomorphs (Packard 1974 onward); "LA is not Greek" (many studies, including Davis 2024 and the 2026 same-lexicon control); CH and LA as the same language via syllabotactics (Davis 2018/2024); VSO word order (Davis 2013). Any totals or fraction test we run on `linear-a/data/corpus.json` is a **replication**, and must be called that.
- **Open questions:** what language family Minoan belongs to (no test has passed controls); what the transaction words other than KU-RO, KI-RO and PO-TO-KU-RO mean; how stable LB values are for rare signs; how the sign inventory varies by site and scribe (SigLA makes this testable).
- **Datasets they used:** GORILA (Godart & Olivier) as the standard edition; SigLA (sign tokens); lineara.xyz (mwenge, which is our source); DĀMOS for the Linear B benchmark (we have `damos_items.jsonl`).
- **Methods to copy as controls:**
  1. **Davis's syllabotactic permutation test.** It has a positive control (Cypriot against LB), a negative control (Cypriot against LA) and a 10^6-permutation null over homomorph relabellings. We should use this exact design for any Indus–foreign sign-pair claim.
  2. **The same-lexicon control** (coverage of a known-related corpus against the target) for any lexicon-matching claim.
  3. **Corazza-style convergence:** require independent strands (shape, distribution, typology) to agree before a value is graded A.

---

## 3. The Voynich manuscript (Beinecke MS 408, parchment 1404–1438; about 38,000 tokens)

1. **Rugg, G. 2004.** "An elegant hoax? A possible solution to the Voynich manuscript." *Cryptologia* 28. https://doi.org/10.1080/0161-110491892755. **Rugg & Taylor 2017**, "Hoaxing statistical features of the Voynich manuscript," https://doi.org/10.1080/01611194.2016.1206753. **Schinner 2007**, https://doi.org/10.1080/01611190601133539
   - M: a Cardan grille moved over a table of prefixes, midfixes and suffixes to generate meaningless words. Schinner used random-walk and token-distance statistics.
   - R: a low-technology hoax can reproduce word structure, low entropy and repetitiveness.
   - C: it was a demonstration, not a fit. **Zandbergen 2021** (arXiv 2104.12548) refined the grille and showed that the near-binomial word-length distribution follows naturally. But **Parisel 2026** (item 11) found that grille generators fail to reproduce four positional and directional signatures together.

2. **Montemurro, M. A. & Zanette, D. H. 2013.** "Keywords and co-occurrence patterns in the Voynich manuscript: an information-theoretic analysis." *PLOS ONE.* https://doi.org/10.1371/journal.pone.0066344. Also **Amancio, Altmann, Rybski et al. 2013**, *PLOS ONE*, https://doi.org/10.1371/journal.pone.0067310
   - M: the information carried by word placement across sections (the entropy of a word's distribution compared with **randomly shuffled text**). Keywords are words concentrated in sections, and these are then linked into networks. Amancio used network and word-recurrence statistics compared with many languages and shuffled texts.
   - R: Voynich words cluster by section, matching the illustrations, much as keywords do in real books. This is "compatible with a genuine message".
   - C: the shuffle null is a good control. Rebuttal: Timm & Schinner 2020 showed that a self-citation generator also produces section-clustered vocabulary, because local copying creates topical drift. Topic structure alone is therefore **not specific to meaning**.

3. **Hauer, B. & Kondrak, G. 2016.** "Decoding anagrammed texts written in an unknown language and script." *TACL* 4. https://doi.org/10.1162/tacl_a_00084 (press wave in Jan 2018)
   - M: language identification of monoalphabetic ciphertexts (97% accuracy on 380 UDHR languages), plus anagram decoding. Applied to the Voynich.
   - R: "Hebrew" as the source language, with a decoded first line run through machine translation.
   - C: the controls were on synthetic ciphers, not on Voynich-like generated text. Hebrew specialists rejected the decoded lines (for example Mosaic Magazine, Feb 2018), and the authors themselves say it may be an artifact of anagramming plus language models. **Regarded as rebutted.**

4. **Bax, S. 2014.** "A proposed partial decoding of the Voynich script" (self-published, stephenbax.net).
   - M: plant and star names matched to illustrations to give about 10 words and 14 glyph values.
   - R: claimed partial phonetic values.
   - C: not peer reviewed. The values do not produce a coherent reading of other pages. **Not accepted.**

5. **Cheshire, G. 2019.** "The language and writing system of MS408 (Voynich) explained." *Romance Studies* 37. https://doi.org/10.1080/02639904.2019.1599566
   - M: glyph-by-glyph reading as "proto-Romance", with ad hoc lexical matches.
   - R: claimed full decipherment.
   - C: **rebutted.** Medievalists, including L. F. Davis, called it circular, and the university withdrew its press release. There is no control and no reproducible mapping.

6. **Timm, T. & Schinner, A. 2020.** "A possible generating algorithm of the Voynich manuscript." *Cryptologia* 44. https://doi.org/10.1080/01611194.2019.1596999. **Timm & Schinner 2023**, "Discussion of text creation hypotheses," https://doi.org/10.1080/01611194.2023.2225716. **Timm 2026**, *Cryptologia*, https://doi.org/10.1080/01611194.2026.2693462 *(abstract only)*
   - M: a network analysis of word similarity (edit distance) by position on the page. Then a "self-citation" generator: copy a nearby word and modify it.
   - R: a generator a scribe could run reproduces both Zipf laws, similar words lying near each other, and line effects. The authors argue the text is meaningless.
   - C: the generator is the control. Later tests found **gaps**: Rozanova & Temerev 2026 show that self-citation text has fewer hapaxes (59–60% singleton types against 70% in the Voynich) and lacks the coupling between glyphs at token edges.

7. **Davis, L. F. 2020.** "How many glyphs and how many scribes? Digital paleography and the Voynich manuscript." *Manuscript Studies* 5(1). https://doi.org/10.1353/mns.2020.0011. Also **Layfield & Davis 2026**, "Latent semantic analysis applied to the Voynich manuscript," *DHQ*, https://doi.org/10.63744/2ezxpskcezq4, and **"Singulion structure…"**, *Digital Medievalist* 2026.
   - M: digital palaeography (letterform measurement across folios).
   - R: **five scribes.** Scribe identity lines up with Currier's A/B "languages", which Davis calls "dialects". So A/B is partly a scribe effect.
   - C: independent physical evidence. It is widely used as a covariate. Farrugia et al. (VOY2022) and Parisel 2026 (an unsupervised mixture model that predicts held-out A/B labels with 89% accuracy) confirm the split.

8. **Bowern, C. & Lindemann, L. 2021.** "The linguistics of the Voynich manuscript." *Annual Review of Linguistics* 7. https://doi.org/10.1146/annurev-linguistics-011619-030613. Also **Lindemann & Bowern 2020**, "Character entropy in modern and historical texts," arXiv 2010.14697, and **Sterneck, Polish & Bowern 2021**, "Topic modeling in the Voynich manuscript," arXiv 2107.02858. Reply: Timm & Schinner 2021, https://doi.org/10.1080/01611194.2021.1911875
   - M: comparison corpora (294 Wikipedia languages and 18 historical texts), conditional character entropy, Currier, scribe and transcription partitions, and topic models (LDA, LSA, NMF).
   - R: character entropy in the Voynich is lower than in **every** comparison text. No script choice, transcription choice or substitution cipher closes the gap, because characters are highly constrained by position in the word. Topics track **scribe plus illustration subject.** Their reading: probably meaningful, possibly with phonemic merger or abbreviation.
   - C: large reference corpora, which is good. The topic result is confounded by scribe, as they note. Timm & Schinner dispute the "meaningful" conclusion.

9. **Gaskell, D. E. & Bowern, C. L. 2022.** "Gibberish after all? Voynichese is statistically similar to human-produced samples of meaningless text." *VOY2022*, CEUR-WS 3313, paper 4. https://ceur-ws.org/Vol-3313/paper4.pdf. Companion: **Bowern & Gaskell 2022**, "Enciphered after all? Word-level text metrics are compatible with some types of encipherment," paper 6. Also **Boxer 2022**, "Fingerprinting gibberish," paper 1.
   - M: 42 volunteers wrote gibberish by hand. About 20 metrics (entropy, Zipf fit, compression, repeats, character and word position bias within lines, word-length autocorrelation) were compared across gibberish, 5 Voynich transcriptions and many meaningful texts, using a random-forest classifier. The companion paper ran 22 cipher and manipulation methods. Boxer compared the Voynich with Dee and Kelley's "Enochian" in Sloane MS 3188.
   - R: human gibberish is **not random**. Depending on the writer, it can match either language or Voynichese on almost all metrics, **including Zipf's law**. What separates gibberish: lower information content, more repeated words and characters (including triples), stronger position bias within lines, **positive autocorrelation of word lengths**, and a weaker Zipf fit. Most of these features are also seen in the Voynich, and the classifier put all 5 Voynich samples nearer gibberish, but with low confidence. The companion paper shows that ciphers which merge phonemes or use bigraphs lower entropy to Voynich levels, so odd character statistics do **not** prove gibberish. Boxer finds that the running fraction of hapaxes separates Enochian gibberish from language, and that the Voynich sides with language on this measure.
   - C: an explicit positive control set of known non-language, which is exactly the right design. The limit is that the samples were short (page-level structure was untested).

10. **Greshko, M. A. 2025.** "The Naibbe cipher: a substitution cipher that encrypts Latin and Italian as Voynich manuscript-like ciphertext." *Cryptologia.* https://doi.org/10.1080/01611194.2025.2566408. Code: https://github.com/greshko/naibbe-cipher *(abstract and press only; the publisher page refused fetch)*
    - M: a die roll splits the plaintext into unigrams and bigrams (about half each). A card draw picks one of six substitution tables. Tested on Latin and Italian texts and compared with Voynich statistics.
    - R: a **historically plausible verbose cipher** reproduces glyph frequencies, word lengths and positional glyph behaviour at the same time. The author says it is not a solution and that it "fails in several major ways", notably in fully replicating Currier B.
    - C: it is an existence proof (a "compatible cipher" control). Rozanova & Temerev 2026 test it: like self-citation, it lacks the hapax-rich vocabulary (41% singletons against 70%) and the coupling of glyphs at token edges.

11. **2024–2026 arXiv preprints (not yet reviewed):**
    - **Rozanova & Temerev 2026**, "A glyph is not a letter, a token is not a word, a space is not a space," arXiv 2608.17096. Uses the ZL transliteration with matched prose, cipher and pseudo-text controls and quire-level resampling. Conditional glyph entropy is 2.7 bits, against about 3.5 for Latin, Italian and English. The identity of one token predicts the next by **under 1%** of token entropy, against 2–10% in controls. Glyphs at token edges share 0.2 bits. Uncertain spaces are physically narrower (AUC 0.905). The Naibbe cipher and self-citation each reproduce several features but not edge coupling or hapax richness. **This is the best-controlled recent paper.**
    - **Parisel 2025–2026**, arXiv 2509.10573, 2604.19762 and 2604.25979: directional n-gram perplexity, layered right-to-left within words and left-to-right across word boundaries; slot generators and Cardan grilles fail a four-signature joint test; Currier A/B is recovered without labels (89% on held-out folios) and is driven by a once-per-folio "switch".
    - **Steckley & Steckley 2024**, arXiv 2404.13069: token distributions shift with line and paragraph position **and with the position of plant drawings**.
    - **Turenne 2026**, arXiv 2609.20835: uses LLM-based image–text alignment to compare Voynich plants with Pseudo-Apuleius herbals. Argues for a "pastiche", a structured imitation of language. There is no gibberish control, so it is weak.
    - **Matlach et al. 2022**, *PLOS ONE*, https://doi.org/10.1371/journal.pone.0260948: some glyphs behave like compounds, so the alphabet may be smaller. Proposes steganographic encoding.
    - LLM "translations" circulating 2024–2026 (blogs, OSF and SSRN preprints, for example Caspari & Faccini 2025, Gropper 2026): none has a held-out or gibberish control. **Ignore them.**

### Lessons for our approach: Voynich
- **Already done (do not claim novelty):** Zipf fits, word-length binomiality, low character entropy, section and topic clustering, the Currier A/B split, scribe counts, both hoax generators (grille and self-citation), cipher generators (Naibbe, merged-phoneme ciphers), hand-written gibberish controls, and direction tests. "Zipf's law holds" is **known not to discriminate** language from gibberish (Gaskell & Bowern 2022).
- **Open questions:** whether token-edge coupling and a hapax-rich vocabulary can be produced by any generator; whether A/B is a code switch or a dialect; whether the spaces are real; whether page-level and illustration-linked structure exceeds what generators produce.
- **Datasets they used:** the Zandbergen–Landini ZL transliteration (IVTFF, our `ZL3b-n.txt`), Takahashi IT (our `IT2a-n.txt`), the Yale corpora of 294 Wikipedia languages and 18 historical texts, Gaskell & Bowern's 42 gibberish samples, Sloane MS 3188 Enochian, and the Naibbe Python code (GitHub).
- **Methods to copy as controls:**
  1. **A gibberish and generator panel.** For any "it is language" claim, compute the same metric on human gibberish, self-citation output, a grille, and a verbose cipher (Naibbe). Report where each falls.
  2. **The discriminating metrics:** word-length autocorrelation (positive suggests gibberish), the running hapax fraction, triple repeats, position bias within lines, and next-token predictability against matched controls.
  3. **Covariate control:** partition by scribe and by "language" (for Indus, by site and by object type) before calling a cluster "topical". Sterneck et al. found topics followed scribes.

---

## 4. Cross-cutting: separating language from non-linguistic or generated sign systems

The literature now agrees on several points that bear directly on our Indus result (signs as a business code, not speech).

1. **Common statistics are not specific to language.** Zipf's law, low or medium conditional entropy, positional asymmetry and n-gram structure all appear in hand-written gibberish (Gaskell & Bowern 2022), in algorithmic generators (Timm & Schinner 2020; Rugg 2004; Zandbergen 2021), and in ciphers (Naibbe 2025; Bowern & Gaskell 2022).
   - **Raghavendra 2026** ("On the non-specificity of statistical measures used in script decipherment," arXiv 2608.02999) tested this on the Indus case directly. It built a 3,000-text emblem system with compositional meanings and **no phonology**. It fell in the same category as Indus on every published Indus test that could be reproduced (54 methods registered in advance). A sequential "decipherment" reached high dictionary coverage for English, Sanskrit **and** Tamil on that same non-linguistic corpus.
   - Consequence: entropy and n-gram arguments in either direction (for example Rao et al. 2009 for language) cannot settle the question alone. Our claims must rest on **positive features of administrative notation** (slots, commodity-specific numerals, frames), not on "it is not like language".
2. **What does discriminate** (so far): word-length autocorrelation; the running hapax fraction; repetition, including triples; how much one token predicts the next compared with matched controls; coupling at token edges; and, for administrative systems, **arithmetic closure** (totals that add up: KU-RO in Linear A, PE totals). Arithmetic closure is the one internal check that proves a sub-system has meaning without phonology.
3. **Accounting scripts are the normal case for early writing.** PE is mostly numerals and object signs. Glottographic elements (a syllabary for names), if they exist at all, are a minority and still unproven after 100+ years (Kelley et al. 2022). This matches our Indus "business code" reading. A small phonetic subset for names cannot be excluded on statistics alone. Absence of evidence in a sample is an upper bound, not proof.
4. **Controls that worked:** (a) positive and negative script pairs with a permutation null (Davis 2024); (b) held-out test sets and expert labels as independent checks (Born et al. 2022, 2023); (c) convergence of independent strands (Corazza et al. 2021); (d) a same-lexicon coverage control (Linear A 2026, unreviewed); (e) a generator panel (Rozanova & Temerev 2026). Claims without such controls failed: Cheshire 2019, Hauer & Kondrak 2016 (Hebrew), Bax 2014, the 2024 consonant-skeleton matching, and the LLM "translations".
5. **Independence of transcriptions.** Voynich work uses several transliterations (ZL, IT, Currier, v101) and reports results across them, as Lindemann & Bowern 2020 do. This matches our rule that Wells and IM77 agreement is "transcription-robust", not "replicated".
6. **Sign names can mislead.** Born et al. 2021 found that image-based models beat label-based models in PE. Our Wells and Mahadevan numbering and allograph merges are a similar risk. The rule of running on `seq_raw`, `seq_strong` and `seq` is the right guard, and image-level checks would be a further one.

## Key links
- CDLI Proto-Elamite: https://cdli.mpiwg-berlin.mpg.de/ (bulk data: github.com/cdli-gh/data)
- SigLA: https://sigla.phis.me/; lineara.xyz: https://github.com/mwenge/lineara.xyz
- Voynich transliterations (Zandbergen): https://www.voynich.nu/transcr.html
- VOY2022 proceedings: https://ceur-ws.org/Vol-3313/
- Naibbe code: https://github.com/greshko/naibbe-cipher
