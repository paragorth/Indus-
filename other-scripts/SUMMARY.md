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

**Already published:** most of the above (arXiv 2608.17096, 2604.19762; Vogt 2012). Ours are independent replications.

**Possibly new, small:**
- the m/g one-per-line quota;
- the ch↔e trade-off by line;
- the split of word-class variance across section, page, paragraph and line;
- the verbose-cipher merge test with encrypted-language controls.

## Bottom line
The cheap data-only attacks on these three scripts are largely spent:
- **Voynich:** the field is saturated. What we found replicates published work.
- **Proto-Elamite and Linear A:** the limit is clean data (few balancing totals, small corpora), not method.

The next real progress needs:
1. specialist re-reading of a short list of named tablets (listed in each FINDINGS.md);
2. new material.

The Indus finding (a structured business code, not speech) remains our strongest result. Proto-Elamite and Linear A both show the same kind of accounting skeleton: commodity classes, totals and fixed slots. Proto-Elamite, unlike Indus, also has name-like variable middles.
