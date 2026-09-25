# Clean-room analysis (no Mahadevan, no Parpola, no sign labels, no language prior)

Data: Yajnadevam's transcription only (Wells sign IDs as opaque numbers), reversed to reading order, deduplicated. Segments split at damage and line breaks: 3,534 segments, 12,817 signs. Code: `cleanroom.py`.

## 1. Word-like units (MDL segmentation)
Adjacent units are merged while the total description length (lexicon + corpus) falls.
- Fitted separately on each half: 64 and 60 multi-sign units. **43 are found independently in both halves.** On within-text-shuffled halves: 0 and 1 units, **0 shared**. The recurring chunks are real sequence structure.
- Full corpus: 95 multi-sign units. Most frequent (reading order): 861-002 ×142, 817-002 ×120, 740-400 ×97, 760-740 ×94, 820-002 ×87, 740-090 ×74, 032-220 ×64, 176-740 ×47, 705-033 ×46, 100-740 ×46, 798-740 ×39, 692-060 ×38, 590-390-740 ×37, 235-240 ×36, 920-060-741 ×35, 220-520 ×30, 705-033-520 ×30, 255-435-690-740 ×23.
- Segments contain 1–4 units in 84% of cases.

## 2. Typology without a candidate language
For 353 units seen at least 5 times: distinct left contexts 4,460, right contexts 4,273, **right/left = 0.96** (shuffled control 1.00). There is no suffix-side or prefix-side concentration of variation. Suffixing languages would show one, and every candidate usually proposed (Dravidian, Indo-Aryan, Elamite) is suffixing. The corpus behaves like a list of fixed labels (formula + name + title) more than like inflected sentences. That is a constraint on genre, not on language.

## 3. Replication on Mahadevan's transcription (IM77, used as data only)
- 2,827 segments, 11,008 signs. MDL units fitted per half: 85 and 75. **Shared: 51 real vs 7 shuffled.**
- Most frequent units (Mahadevan numbers): 267-99 ×238, 89-328 ×84, 391-99 ×81, 347-342 ×77, 87-59 ×65, 336-89 ×62, 87-328 ×58, 342-1, 342-176, 150-123, 48-342, 245-245, 336-89-211 …
- **Cross-transcription agreement:** of the 75 units from the Yajnadevam transcription whose signs map to Mahadevan numbers (via the bridge file, a symbol-to-symbol mapping only), **69 are also units in IM77**. Two independent transcriptions give the same word inventory.
- Right/left context variety: 0.97 (Yajnadevam 0.96, shuffled 1.00). Same result: no affix-side asymmetry.

Rule for this file: data and symbols from anyone (Mahadevan, Wells, Yajnadevam), no one's interpretations (no readings, no language assumption, no functional labels).

## 4. Data-derived grammar, validated on held-out texts (`cleanroom_grammar.py`)
Units and classes are fitted on the training half only. Held-out texts are segmented with the same merges.
- Each unit gets a position class from training (≥60% initial → I; ≥60% final or alone → F; else M). On held-out texts the class **predicts the unit's position correctly 63.1% (Yajnadevam) and 64.9% (IM77), against chance 34.4% and 34.9%**.
- Classes: Yajnadevam 24 I / 149 M / 51 F; IM77 32 I / 110 M / 60 F.
- **Open slots (paradigms):** frames with ≥3 alternating fillers: 285 (Yajnadevam), 234 (IM77). The largest non-trivial slot in both is **[start] _ [the commonest final sign]** (Wells 740 = M342), with **59 and 39 different fillers**. The next is [start] _ [002] (44 fillers). These are one-slot positions filled from a large open set, which is how a name slot behaves in a labelling system.

Summary without anyone's interpretation: the script has (a) a stable inventory of multi-sign units that replicates across halves and across two independent transcriptions, (b) a three-position grammar (initial units, a large open middle, a small set of final units) that predicts unseen texts at almost twice chance, and (c) a large open slot right before the commonest final sign. It fixes the *structure* of the texts. It says nothing about sound or language, and no amount of internal data will.

## 5. What fills the open slot before the commonest final sign
Texts ending in Wells 740 / M342: 884 (Yajnadevam) and 762 (IM77).
- Filler length before that sign: 1 sign 12% / 9%. **2–5 signs about 75%**, up to 8.
- Multi-sign fillers draw on the 50 most frequent signs at the corpus-wide rate (70% vs 68%; 72% vs 72%): they are built from the common inventory.
- Single-sign fillers are mostly rare signs (63% / 60% outside the top 50).
Consistent with a mixed system (rare word-signs for some slot values, combinations of common signs for others). It does not decide between spelling by sound and word-signs.
