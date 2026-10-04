# Three undeciphered scripts: what the data-only attacks found (2–3 Oct 2026)

Every claim below has a control. The details are in each script's FINDINGS.md. LITERATURE.md has the related papers.

## Proto-Elamite (CDLI, 1,585 tablets, 95% from Susa)
**Holds:**
- Strong slot grammar: an optional prefix, then a middle, then a final class sign, then the number. M288 ends 237 of 250 entries.
- The final class sign predicts the number system: measured with capacity units (M297, M002…) or counted (M263, M264…).
- Openers: M157 heads 37% of tablet headers.
- Middles are as varied as free combination, which fits names or spelled words.
- Counting units 1/10/100/300 (grade B).
- Decimal or sexagesimal is chosen per tablet (p = 0.0005).

**Failed:**
- Linear Elamite values do not make the middles read as Elamite names (p = 0.65 against value shuffles).
- The arithmetic joint fit missed its pre-set 70% bar.
- No fixed ration per head.
- No seasonal time order (pe11): a blind search for a yearly herd/harvest cycle across undated tablets finds only a loop of commodity types that a timeless mixture reproduces; calibrated on Ur III Drehem tablets with known months, the method needs a season about 40x stronger than real herd accounts carry.
- Names are not built from a used-up set of counters or stamps (pe9): a sign used once on a tablet becomes 2-3x more likely to recur there (held out on 52 tablets), never less; 0 of 90 signs behave as depleted tokens. What remains is a per-tablet name pool that is reused freely (grade B).
- Not written from a fading memory and not copied (pe13): a flat per-tablet topic fits, with a dip at the next entry (M288 and M346 alternate) that no control corpus shows; the reverse recalls the obverse rather than resetting; a weak shared order of entries holds out of sample (grade B).
- Administration as a food web (pe15): name x goods webs are nested and modular only as far as their degrees force, unlike Ur III and Linear B. Three sign 'offices' (grain, class-sign, bare-count) do replicate on reserved tablets (+0.13 bits/entry; none of 3,000 random partitions does better). The signs of a new name predict its goods class (grade B); hidden links are predicted by popularity alone.
- Hunger as the ruler (pe16): sizing units from human and animal food needs recovers the Ur III sila blind (0.83 l, truth ~1 l; daily and monthly issues sorted on 97-100% of tablets), but random rulers fit as well and month/year alias 1 time in 3. On the PE two-line records N39C = 4.6 l (90% 1.5-14) on a lattice x{1/30, 1, 12}; the scale follows which units scribes used. Only the /30 point makes a unit bowl-sized (N30C ~ 0.6 l, N24 ~ one adult day), a check with a 43% chance hit rate. Grade C.
- The scribe knew the length? (pe21): clay area grows only with about the square root of the text on PE (elasticity 0.40; 0.65 on lists of 8+ lines), as in proto-cuneiform (0.44-0.49) and below Ur III (receipts 0.91, compiled accounts 0.72). About 42% of PE lists of 5+ lines sit on a sized-to-fit line (re-dealt texts 2-7%), and they are no particular genre, system, office, seal or site. Token piles written unexchanged are excluded (< 1%), and there is no hand-counting cap at 5. Totals go to the reverse while the obverse still has room. Grade B for these points; nothing is read.
- Counting the unseen signs (pe22): about 750 [680-850] base signs including compounds, of which 551 are seen. The simple signs are a closed set of about 425 (353 seen). Most entry strings are unseen (coverage 0.22). Hashed predictions made from pre-2000 tablets matched the later tablets: 6 new signs against 8 [3-14] predicted, 37 new strings against 39 [33-43], and a known-sign AUC of 0.92. These tablets behave like any batch of the old archive. Sofalin and Ozbaki add no genuine new sign. A further hashed prediction covers 89 untransliterated Tehran tablets. Grade B; nothing is read.
- Forensic accounting (pe23, `proto-elamite/loops/pe23_final.txt`): Digit fingerprints were computed (round-number heaping, mod 5, leading digit, repeats) against a smooth null that keeps the notation. They separate Ur III counted, measured, rule-allocated and computed-estimate numbers (0.91-0.97, shuffled 0.29-0.38), but Ur III estimates are less round than measurements, not more. PE counts look counted and PE capacity looks measured (B). No PE sign, office or site carries estimated or target numbers (B, for the ~8 signs large enough to test). The top-edge '1(N34)' is a constant tag of M157-headed tablets (21% vs 7%, p 4e-10) and never equals the entry sum (B). Nothing is read.

**Open:** the phonetic layer. The varied middles say "maybe", and the name test says "not with these values".

## Linear A (lineara.xyz, 1,721 inscriptions; DAMOS Linear B 94%)
**Holds:**
- KU-RO = total: 9 of 30 sections add up exactly, against 0.5 by chance.
- KI-RO = a residual amount or a list heading.
- The fraction signs have a strict writing order (0 of 500 random relabellings match it).
- Which fraction sign appears depends on the commodity.
- Linear B shared words: 12 of 3+ signs against 2.0 by chance.
- Commodity pecking order (la20): logograms are written in a fixed order (CYP, GRA > VIR, OLE > OLIV, VIN). This holds on unseen tablets (0.715 against 0.50; FWER 0.005) and beats a large-amounts-first null (P 0.002). The order is by commodity, not by size (A; grain-first sequences known in kind).
- Affixes are real (z ≈ 6). -TE, -ME and JA- belong to objects, not tablets.

**Failed:**
- The fraction values cannot be solved from the totals.
- The "J < 1/2" argument rests on two doubtful J+J readings (photo check).
- The anchors are too few to separate places from persons.
- The affixes do not predict new forms.
- Tablets as pieces of split ledgers (la12): no cross-tablet sum beats chance; with 2-3 tablets almost any total can be matched by chance, so arithmetic alone cannot rejoin ledgers.
- Linear A as its own dialects (la13): no cross-site or cross-scribe sign substitution beats nulls, and no sign grid emerges; sites share too few words (43 types) for even a planted rule to show. Words differing by one sign are site- and hand-local (B).
- Counting the words like wildlife (la15): at least ~4,500 word types were in use (925 seen; ABC 5,200-82,000) and ~170-200 syllabic signs (139 seen). Words are site-endemic and anti-nested like Linear B's. A hashed prediction of the post-1985 documents got new words (33 vs 32), new signs (1 vs 0) and returning words (AUC 0.76) right; new findspots were underpredicted (1 vs 6). Census, not a reading.
- Private spellings as a Rosetta stone (la16): one-sign variants are not hand-private even in Linear B, so no sign equation can be read off scribal differences; no LA equation survives held-out re-test. LA hands are recoverable from shared words (B), not spellings.
- Words that avoid each other (la18): in Linear B, forms of one word share a tablet (3x) but avoid the same entry, and tablets keep to one case. Linear A shows neither (calibrated, B); perfect avoidance cannot be seen at LA size. SI- prefix alternation stays C.
- The palace spoke many tongues (la19): mixtures of sign-level phonotactic models choose K = 4 for Linear A, but 33/36 single real languages also choose K >= 2, and LA is less mixed and less stable than the median language. Its components track nothing outside the model, and foreign-looking outliers (Phaistos, roundels) are rare-sign effects. No multilingual lexicon detected (B, calibrated).
- Words spread like epidemics (la17): SI/SIR outbreaks fitted by ABC (~390,000 simulations) to the site x word and site x sign incidence cannot recover a planted source (1/20, 0/20) or an alternative-generator history (chance), and the Linear B control (Knossos first) passes only where shuffled data pass too. No Linear A source or adoption order is identifiable; the inferred orders do not match the MM II-LM IB deposit dates (P >= 0.14). Grade C: time-separated sites share fewer words.

- The pecking order (la20): no word, entry-word or first-sign hierarchy recurs across tablets (0.51 vs Linear B Pylos at Linear A size 0.61). People and places are not listed in a fixed rank or tour order (B, calibrated); a tour that was only ever written in pieces would not be detectable.
- Universals build the grid (la21): a blind grid from phonotactic universals recovers Linear B's consonant rows (not its vowel columns). Linear A's blind rows agree with the LB-derived consonant series (B), but no specific new grouping replicates across sites; QA acts like a pure vowel (C).
- The rings know who talked (la26): the published seal-ring network (7 shared seals) does not predict shared words, sign variants or entry structures, and multi-site words do not follow ring paths (P 0.16-0.66 against all graphs; the doc-perm null passes 38% of random graphs on Linear B). Only a ring process ~4x the real cross-site sharing would be visible (upper bound). Khania is an island in signs and entry structure too (B).
- Predict what the fire destroyed (la22): random-ensemble restorers, calibrated on hidden known items at the real break pattern, restore signs at 28 % top-1 / 52 % top-5 (Linear B 45 %) from word-internal regularities only (shuffled tablets give the same), and numbers and commodity signs from the accounting skeleton (gain gone in shuffled tablets) (B). Frozen predictions (hashed) for 974 break edges do not anticipate SigLA's later re-readings (1/19 in top 5, B negative); edge hits 1/4 clean (C).
- Count the people (la23): capture-recapture calibrated on Linear B (whole-archive name counts predicted, 5/6) puts the Hagia Triada tablets at ~1,000 [510-2,430] distinct small-number entry words; perhaps a few hundred people, the size of a town (C). Persons cannot be separated blindly from other entry words at LA size (AUC 0.59). Prediction: ~55 % of such words on a new HT tablet will be new.
- The clerk ran an apportionment algorithm (la24): about 90 M share-rule fits find no tablet built by dividing a total by share classes. The MDL search fails its planted and Ur III ration controls, so it is blind at Linear A list lengths. Lists are not built on one unit share, although Linear B lists are (calibrated, B). Fraction-absorption fits prefer simple values, not the conventional ones as such. Grade C: HT 9b scales three of HT 9a's recipients by 4/5, which needs J = 1/2.
- The fire set the clock (la25): ABC over ~1.5x10^7 random crop calendars x destruction months cannot date any LM IB archive from its commodity mix: planted months are not recovered even at 64x corpus size (base rate and season are confounded), and the Linear B spring control fails. Whether the sites burned in one season is untestable (A, calibrated negative).
- The balance pans know the numbers (la27): 70 published Minoan weight masses show no quantum (no 61 g unit, no mina step), and Linear A integers do not heap on the 1:8:60 unit-mina-talent ladder or on any of 10^4 random ladders beyond decimal counting. Planted 5 % heaping is found (B, calibrated negative). The Linear B ceiling rule (a sub-unit is written at most r-1 times) recovers M:N = 4 and LANA:M = 3 from numbers alone. On Linear A it finds D written 2-4 times and K, L2, JE, F, H never repeated (B). So D and B are counted parts (>= 3 per next unit), consistent with D = 1/5 and B = 1/3, not with binary values (C).
- The receipts fed the ledgers (la28): the terms on roundels and nodules reappear on tablets of the same site no more than vessel words do (z 1.7 vs 2.4). A Linear B sealing control passes (z 4.2), and planted 10 % copying is found (B, calibrated negative). Khania roundel classes never appear on Khania tablets. 19 % of receipt types use signs no tablet uses, and 9 of 14 receipt amounts are fraction-only (A). Receipts and ledgers are separate layers (B). Sums cannot be tested (4 integer receipt amounts).
- The winds carried the words (la31): 1.48 M simulated voyages over ERA5 daily winds, plus Tobler walking times, give travel matrices for 29 find-sites. Symmetrised, sailing time ranks site pairs exactly like km (rho >= 0.96), so only direction can separate wind from distance. Inside Crete, nearer sites share more rare signs and logograms (P 0.023, B), but no travel model, season, month or leg beats km. Words do not sit downwind (P 0.91). The directional test has AUC 0.68 at Linear A size (planted), so this is an upper bound. Linear B behaves the same.

**Open:** the fraction values. They need specialist re-reading of PH 9b, PH 22a, ZA 8 and HT 104 in GORILA.

## Voynich (ZL3b and IT2a transliterations, 39,019 words)
**Holds:**
- Letters are too predictable for simple or verbose substitution of Latin or Italian. On letter statistics it equals a no-language Markov text.
- The last glyph of a word predicts the next word's first glyph within a line, and this resets at every line break, even between ordinary words.
- Lines have opener and closer vocabularies. m/g works as about one end marker per line.
- The two word classes (q-type and a-type) are mostly a property of section and page.
- No hidden letter stream in word lengths, gaps, line counts or line totals (v15: ~10 M random codebooks, 7 languages; planted Caesar recovered at +0.3 to +1.3 bits/letter, Voynich +0.03). Grade A, negative.
- The spaces are not a mechanical lie: no re-spacing rule (80,000 tried) or free MDL re-segmentation beats the written spaces beyond a Markov-2 null, while planted spacings on Latin/Italian are beaten 2-3x harder. The written words save nothing over a glyph-bigram code (Latin words save 0.4-0.65 bits/glyph); the junction effect survives re-spacing (v16). Grade B.
- Nothing is sorted: no page, window, line-initial run, label set or margin list is in any glyph order (v19: ~1.2 M order optimisations; planted lists and Isidore's alphabetical Etymologiae X found under substitution). Grade A negative for runs of >= ~25 entries. Side result: q-initial words drift up and a-initial words down within pages and paragraphs (Grade B, cause open).
- Not written to a quota: no glyph, pair, word class or linear combination is held even per page, paragraph or bifolio (v20: ~2,150 features x 4 levels, planted quotas found; Voynich pages 2-5x more uneven than natural texts). Grade A negative for hard quotas. The one conserved per-line balance is reproduced by a word-bigram chain (Grade B).
- No calendar beat in the starred paragraphs (v32): quire 20 entries repeat nothing at 7, 12, 27-30, 36 or any period from 2 to 40 (folding, combs, circular HMM with a lost-bifolio gap, day-counter; p 0.18-0.96 in ZL and IT2a). A real martyrology is caught at z 45-64 when its date is written but not from its text alone, and planted day words in at least 20 % of entry openings are caught. Grade A negative for written dating; undated calendars remain untested. The star drawings alternate dotted and plain, and the text ignores this.
- The pen does not reset the text (v18): word choice, length, junction and repeats flow across ink-darkness dip proxies (25 Yale IIIF pages, 6,048 detector settings) as in Latin manuscripts; a planted generator restarting at each dip is found 3/3. Grade B negative. Leads: a darkening looks slightly like a line start (C); words on one pen-load are slightly more alike (C+).
- Forge it, then catch it (v21): twelve forgers, from the junction resynthesis to a GRU and Timm-Schinner copy-and-vary, are all told from real pages by ~52,000 page discriminators (best AUC 0.93; forgery vs forgery 0.50; Latin and Italian herbals forged the same way 0.84-0.93). The herbals are caught by topic re-use, the Voynich never is: only local copy-and-vary and layout effects survive, each closable by one more rule until another appears. Generator-like in kind, beyond any single known generator. Grade A (no forger passes), B (no topic signal).
- Corrections are too few to read as a rulebook (v24): only 24 marked in all transliterations, 4 with a before-state, and ink anomalies do not find them. Controls show real Latin corrections fix legal, often real-word slips, and a copied meaningless generator does the same at half the strength. Telling them apart needs about 50 genuine pairs. Grade A census, C for any Voynich lean.
- Glyphs are built from features (v25): glyphs that share strokes behave alike (font-image shape vs context r +0.28 to +0.44, ZL, IT, Currier A and B), as in Hangul (+0.25 to +0.53) and unlike Latin or Greek (~0); a held-out glyph's behaviour is predictable from its strokes, and the bench and p/f crossbar act as additive features. Not misreading or position; stroke-edit copying explains part. Grade A (shape predicts behaviour), B (compositional), C (sound values). Swap pairs are not one stroke apart (demoted). Kill attempt (v28): handwritten graphic-unit transcriptions (allographs, abbreviation marks, minims) give a weaker real link (+0.1 to +0.26, printed ~0); the Voynich stays 1.5-3x above them and Ethiopic, under any EVA segmentation; positional and free variants cannot produce it. Survives.
- The lines were not a shuffled deck (v27): re-sequencing body lines for continuity (junction trained only within lines, shared words; exact DP / annealing, also over the whole book) restores Italian prose at 4x chance and finds planted continuity in >= ~30% of joins, but the Voynich matches its Markov, word-shuffle and cross-page nulls everywhere; no chains across pages. Lines do not run on into each other in any recoverable order. Grade B negative (book scale low power).
- Features do not spread (v29): no stroke feature agrees through words as vowel harmony does (Turkish, Hungarian, Finnish and planted harmony found blind; Voynich 0), and the strong word-junction dependence is a pairing (y.q-, n.o-, n.ch-) with same-glyph avoidance and no sandhi, not feature agreement. Grade A negative / B.
- The dialect ladder (v30): the cheapest A->B glyph rewrite costs what Bavarian->Alemannic costs (ReF, 1350-1450), more than two scribes copying the Commedia, far less than Latin->Italian, a spelling reform or any cipher key change. But it is unusually regular: in every run the top rules turn A's -chy/-chol/-chor endings into B's -chedy/-edy, and the rules learned on herbal pages carry over to unseen sections. Two conventions of one system, not two languages or keys. Grade B. B's hands 2 and 3 are at null distance.

**Already published:** most of the above (arXiv 2608.17096, 2604.19762; Vogt 2012). Ours are independent replications.

**Possibly new, small:**
- the m/g one-per-line quota;
- the ch↔e trade-off by line;
- the split of word-class variance across section, page, paragraph and line;
- the verbose-cipher merge test with encrypted-language controls.

## Random programs (r2)
- ~8.9 M random and evolved small programs (copy rules, counters, lookup tables, L-systems), scored by held-out description length against Kneser-Ney, with shuffled, Markov-resynthesised, planted (grille, ledgers) and natural (Latin, Linear B) controls. Planted generators recovered 3/3. In no script does a program beat the baselines beyond its nulls: LA and PE programs only 'copy from the entry above' (gone against a document cache), Voynich gains nothing over a 3-glyph Markov chain. Grade A negative for short generators of this kind.

## Grow the script in a box (r3)
- 62,000 simulated societies evolve scripts (iterated learning) for ledger or text worlds; ABC on blind statistics. Controls: Latin/Italian come out as text and LB as a ledger; Ur III's 'many people, few goods' is never recovered, and people/goods counts and learning history cannot be identified from these sample sizes. Inferred worlds (all C): PE = mostly nameless counted lines with 100+ logograms; LA = named numbered ledger, no secrecy ('one commodity per tablet' killed by a held-out miss); Voynich = text in a small alphabet with word-final end sets and line-initial markers (~30-40% of lines). The Voynich 'needs secrecy' guess was killed. Held-out predictions: PE 3/4, LA 3/4, Voynich 1/4.

## Which script teaches which (x3)
- Predict the clay (pe17, `proto-elamite/loops/pe17_final.txt`): a blind, hash-frozen ranking from 2,000 random classifiers (plateau vs Susa writing) was scored against hXRF clay provenance (Yeganeh et al. 2025). A weak plateau writing fingerprint exists (held-out AUC 0.69, B). The five chemically foreign tablets point the right way in three of four tests (ST-11 top 10%), but Fisher p is 0.14 and power is only about 12% (C). There is no itinerant-scribe signal.
- Blind seriation (pe19, `proto-elamite/loops/pe19_final.txt`): hash-frozen orders of all PE tablets by evolving traits agree with the CDLI stratigraphic levels above random (Susa rho +0.47, Malyan AUC 0.92), but mostly through tablet size. A planted 40-trait drift is invisible, and Uruk IV/III is not seriated in the right direction. No relative chronology of signs (C). One prediction stands: the order of the 5 CahDAFI 8 Acropole tablets.
- Flock arithmetic (pe20, `proto-elamite/loops/pe20_final.txt`): 8 herd signs were assigned blind to animal categories by herd biology (200,000 draws plus Gibbs). Herd counts are coupled out of sample (held-out p 0.008; imputation +0.86 bits per cell, B). M362 and M362~a are reference counts, and the numbers alone recover the scribe's plain vs ~a split (B-). Biology does not name the categories, because the same search mislabels Ur III ewes and kids, so ewe/ram/young readings stay C.
- Transfer test (x3, `proto-elamite/loops/x3_final.txt`): transfer learning across 10 corpora with relabelling controls. Accounting corpora teach LA and PE by format, not by sign identity; Voynich learns from nothing; identity transfer appears only where transliterations share labels (LA-LB +0.12 bits/token, a replication; LA-Greek none). Frequency rank cannot reveal shared signs (planted control), so LA-PE-Voynich sign sharing is untestable this way.

## Bottom line
The cheap data-only attacks on these three scripts are largely spent:
- **Voynich:** the field is saturated. What we found replicates published work.
- **Proto-Elamite and Linear A:** the limit is clean data (few balancing totals, small corpora), not method.

The next real progress needs:
1. specialist re-reading of a short list of named tablets (listed in each FINDINGS.md);
2. new material.

The Indus finding (a structured business code, not speech) remains our strongest result. Proto-Elamite and Linear A both show the same kind of accounting skeleton: commodity classes, totals and fixed slots. Proto-Elamite, unlike Indus, also has name-like variable middles.
