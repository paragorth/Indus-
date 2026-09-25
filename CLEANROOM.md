# Clean-room analysis (no Mahadevan, no Parpola, no sign labels, no language prior)

Data: Yajnadevam's transcription only (Wells sign IDs as opaque numbers), reversed to reading order, deduplicated. Segments split at damage and line breaks: 3,534 segments, 12,817 signs. Code: `cleanroom.py`.

## 1. Word-like units (MDL segmentation)
Adjacent units are merged while the total description length (lexicon + corpus) falls.
- Fitted separately on each half: 64 and 60 multi-sign units. **43 are found independently in both halves.** On within-text-shuffled halves: 0 and 1 units, **0 shared**. The recurring chunks are real sequence structure.
- Full corpus: 95 multi-sign units. Most frequent (reading order): 861-002 ×142, 817-002 ×120, 740-400 ×97, 760-740 ×94, 820-002 ×87, 740-090 ×74, 032-220 ×64, 176-740 ×47, 705-033 ×46, 100-740 ×46, 798-740 ×39, 692-060 ×38, 590-390-740 ×37, 235-240 ×36, 920-060-741 ×35, 220-520 ×30, 705-033-520 ×30, 255-435-690-740 ×23.
- Segments contain 1–4 units in 84% of cases.

## 2. Typology without a candidate language
For 353 units seen at least 5 times: distinct left contexts 4,460, right contexts 4,273, **right/left = 0.96** (shuffled control 1.00). There is no suffix-side or prefix-side concentration of variation. Suffixing languages would show one, and every candidate usually proposed (Dravidian, Indo-Aryan, Elamite) is suffixing. The corpus behaves like a list of fixed labels (formula + name + title) more than like inflected sentences. That is a constraint on genre, not on language.
