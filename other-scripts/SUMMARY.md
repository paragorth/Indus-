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

**Open:** the phonetic layer. The varied middles say "maybe", and the name test says "not with these values".

## Linear A (lineara.xyz, 1,721 inscriptions; DAMOS Linear B 94%)
**Holds:**
- KU-RO = total: 9 of 30 sections add up exactly, against 0.5 by chance.
- KI-RO = a residual amount or a list heading.
- The fraction signs have a strict writing order (0 of 500 random relabellings match it).
- Which fraction sign appears depends on the commodity.
- Linear B shared words: 12 of 3+ signs against 2.0 by chance.
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
- The pen does not reset the text (v18): word choice, length, junction and repeats flow across ink-darkness dip proxies (25 Yale IIIF pages, 6,048 detector settings) as in Latin manuscripts; a planted generator restarting at each dip is found 3/3. Grade B negative. Leads: a darkening looks slightly like a line start (C); words on one pen-load are slightly more alike (C+).
- Forge it, then catch it (v21): twelve forgers, from the junction resynthesis to a GRU and Timm-Schinner copy-and-vary, are all told from real pages by ~52,000 page discriminators (best AUC 0.93; forgery vs forgery 0.50; Latin and Italian herbals forged the same way 0.84-0.93). The herbals are caught by topic re-use, the Voynich never is: only local copy-and-vary and layout effects survive, each closable by one more rule until another appears. Generator-like in kind, beyond any single known generator. Grade A (no forger passes), B (no topic signal).
- Corrections are too few to read as a rulebook (v24): only 24 marked in all transliterations, 4 with a before-state, and ink anomalies do not find them. Controls show real Latin corrections fix legal, often real-word slips, and a copied meaningless generator does the same at half the strength. Telling them apart needs about 50 genuine pairs. Grade A census, C for any Voynich lean.
- Glyphs are built from features (v25): glyphs that share strokes behave alike (font-image shape vs context r +0.28 to +0.44, ZL, IT, Currier A and B), as in Hangul (+0.25 to +0.53) and unlike Latin or Greek (~0); a held-out glyph's behaviour is predictable from its strokes, and the bench and p/f crossbar act as additive features. Not misreading or position; stroke-edit copying explains part. Grade A (shape predicts behaviour), B (compositional), C (sound values). Swap pairs are not one stroke apart (demoted).

**Already published:** most of the above (arXiv 2608.17096, 2604.19762; Vogt 2012). Ours are independent replications.

**Possibly new, small:**
- the m/g one-per-line quota;
- the ch↔e trade-off by line;
- the split of word-class variance across section, page, paragraph and line;
- the verbose-cipher merge test with encrypted-language controls.

## Random programs (r2)
- ~8.9 M random and evolved small programs (copy rules, counters, lookup tables, L-systems), scored by held-out description length against Kneser-Ney, with shuffled, Markov-resynthesised, planted (grille, ledgers) and natural (Latin, Linear B) controls. Planted generators recovered 3/3. In no script does a program beat the baselines beyond its nulls: LA and PE programs only 'copy from the entry above' (gone against a document cache), Voynich gains nothing over a 3-glyph Markov chain. Grade A negative for short generators of this kind.

## Bottom line
The cheap data-only attacks on these three scripts are largely spent:
- **Voynich:** the field is saturated. What we found replicates published work.
- **Proto-Elamite and Linear A:** the limit is clean data (few balancing totals, small corpora), not method.

The next real progress needs:
1. specialist re-reading of a short list of named tablets (listed in each FINDINGS.md);
2. new material.

The Indus finding (a structured business code, not speech) remains our strongest result. Proto-Elamite and Linear A both show the same kind of accounting skeleton: commodity classes, totals and fixed slots. Proto-Elamite, unlike Indus, also has name-like variable middles.
