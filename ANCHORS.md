# Outside anchors: inventory and first tests (25 Sept 2026)

The goal is anything outside the corpus that could fix what a sign *says*. There is no bilingual. What exists falls into four groups.

## 1. Indus texts found abroad (computed here: `anchors_foreign.py`)

Seals and graffiti from Ur, Kish, Susa, Nippur, Umma, Failaka, Bahrain, Oman, Luristan, Altyn Depe. Each is compared with length-matched home texts (10,000 draws).

| measure | foreign | home | p | corpus |
|---|---|---|---|---|
| adjacent pairs attested at home | 0.55 | 0.64 | 0.12 | merged (Wells codes), 23 texts |
| starts with an opener (817/861/820) | 0.00 | 0.14 | 0.03 | merged |
| adjacent pairs attested at home | 0.42 | 0.75 | <0.0001 | IM77 (Mahadevan), 16 lines |
| starts with an opener (M267/M391) | 0.00 | 0.20 | 0.025 | IM77 |
| jar-final | 0.17 / 0.25 | 0.27 / 0.34 | n.s. | both |

The sign repertoire is the Indus one: almost no signs unseen at home. The **grammar is not**. There is no opener formula, and sign pairs never seen in the Indus valley are common. This replicates across two independent transcriptions. It fits the view that seals made or used abroad (many are round, Gulf-type) wrote different content, probably **local names or another language**, in Indus signs. These about 40 texts are where a named individual attested in cuneiform could someday be matched, and so where an anchor is most likely to be found.

## 2. Meluhhan people in cuneiform

| item | source | use as anchor |
|---|---|---|
| **Samar**, **Nanaza**: "men of Meluhha", royal slaves, bezoar shepherds at Ur, c. 2037–2029 BC; Samar's wife Ali-ahi (Akkadian name) | CUSAS 40-2 1354, 1582; Nisaba 15/2 0371; UCTIS 099; Laursen & Steinkeller 2017, *Babylonia, the Gulf Region and the Indus* | The only two probably Meluhhan names in Mesopotamian records. Too short and too late to tie to a particular seal. Useful only if a foreign-found text can be dated and placed at Ur in that period. |
| **Lu-sunzida**, "man of the just buffalo cow", a man of Meluhha | Ur III legal text (Simo Parpola, Asko Parpola & Brunswig 1977) | A Sumerian *translation* of an Indian name: it gives a **meaning** ("buffalo"), not a sound. Testable: is any sign enriched on buffalo-emblem seals? Q10 says no sign is tied to any emblem (0 surviving pairs in Mahadevan's coding), so this lead is currently negative. |
| Meluhhan village (Guabba) at Lagash, from c. 2062 BC; "Ur-Lama, son of Meluhha" and others with Sumerian names | Vermaak 2008; Ur III archives | Shows Meluhhans took Sumerian names: most "Meluhhan" individuals carry no Meluhhan language. |
| Šu-ilišu, "interpreter of Meluhha" (cylinder seal, Akkadian period) | Louvre AO 22310 | Proves Meluhhan was not Akkadian. No words. |

## 3. Proposed loanwords (contested, sound only)

- Sumerian *ilu* / Akkadian *ellu* "sesame oil", compared with Dravidian *eḷ/eḷḷu* "sesame". Witzel proposes a para-Munda source instead (*jar-tila*).
- Sumerian words for ivory and some woods have been proposed as Meluhhan loans. None is agreed.

They are candidates for what Indus texts *could* contain. But none is tied to a sign, and the key search showed that a dictionary match alone doesn't identify a language.

## 4. Not yet searched

- **Weights.** The Indus binary-then-decimal weight standard (about 0.87 g base) against the counts on tablets (I–IIII + leaf, and the three-denomination series). This is a candidate numeric anchor for what the leaf unit is.
- **Dated foreign contexts.** Assign each foreign-found text its excavation date where published, to see whether any overlaps Samar and Nanaza at Ur (2037–2029 BC).

## 5. Primary cuneiform evidence (CDLI bulk dump, 135,254 texts, Aug 2022)

`data/derived/cdli-meluhha-attestations.json` holds all 214 tablets mentioning Meluhha, with period, find-spot and context: 86 from the 3rd millennium, 48 Old Babylonian, 73 Neo-Assyrian.

- **Samar and Nanaza, primary text.** CDLI P453801 = *Nisaba 15, 0371* (Irisagrig; Šu-Suen year 6, about 2032 BC): 1 sila oil *na-na-sa₃*; 1 sila *sa₆-ma-ar*; ½ sila *a-li-a-hi* his wife; "oil rations of the men of Meluhha"; reverse: "royal offering, shepherds of the bezoar goats". These are the only Meluhhan names in the record that are neither Sumerian nor Akkadian. Sound shapes: **/nanasa/ ~ /nanaza/, /samar/**.
- **Other named Meluhhans**, all with Sumerian or Akkadian names, so no language evidence: Ur-Lamma son of Meluhha (Girsu, about 15 tablets), Ur-Igalim son of Meluhha (Girsu), Lu-sunzida man of Meluhha (CT 50, 076, Old Akkadian: pays 10 shekels for a broken tooth), Lu-marza son of Meluhha (CUSAS 26, 259, Ur, Old Akkadian), Ili-ahi man of Meluhha (CUSAS 35, 288, Adab). One man is simply named *Meluhha* (JESHO 20, 145 12). A second Irisagrig list, Nisaba 15, 0951, has a ration line for "men of Meluhha".
- **Goods "of Meluhha"**: ab-ba wood (chairs, footstools), mes wood, esi (ebony), carnelian (*gug gi-rin*), gold, copper (*uruda me-luh-ha*, UET 3, 0368), ivory, the Meluhha bird (*dar-mušen*), the "speckled dog/leopard of Meluhha", goats and sheep, Meluhha ships (Sargon, Adab). These are Sumerian words for foreign goods. None is a Meluhhan word.
- **The Indus seal with a cuneiform legend** (Ur, U.7683 = BM 120573; square, gaur). Gadd 1932 reads *sag-ku-zi*, which occurs **0 times** in the CDLI corpus. An equally possible reading, *inim-ma-zi*, is an attested Sumerian personal name (OIP 104, 014; TIMA 1, 045; NYPL 085). So the seal most likely names a Sumerian-named owner, and does not preserve a Meluhhan word.

## Verdict so far

No anchor yet fixes a sign's sound or meaning. The strongest real lead is group 1: about 40 Indus texts written abroad that break the home grammar. If any of them was written for a person attested by name in cuneiform, that is the first anchor. That match can only come from excavation records (find-spot, level and date for each foreign seal), not from more computation on the texts.

## 6. Linear Elamite (network opened, 25 Sept 2026)

The full Hatamti-Linear Elamite Database (Liège; 45 inscriptions, readings by Desset et al.) is saved as `data/derived/linear-elamite-corpus-hled.json`.

- **Seals naming owners in a readable script, from the Indus contact zone:**
  - **LEI 22**: chlorite *Persian Gulf seal* (Ligabue Collection; early 2nd millennium BC; 3 signs, right to left) reads ***za-ši-ri₂*, "Zaširi"**, a personal name (CDLI P247651; Winkelmann 1999).
  - **LEI 33**: *Central Asian gold seal* (early 2nd millennium BC; 6 signs) reads ***R~Haši-zana-niri***.
  - LEI 45: a cylinder seal (12 signs), *Has-han miki lazp kela-h*.

  Gulf and Central Asian seals of the Indus trade sphere wrote their owners' names in 3–6 signs, the same length as Indus seal texts. That supports the name-and-title reading of Indus seals, and specifically of the foreign-found Indus texts that break the Indus grammar (section 1). It is a strong structural analogy, but it doesn't identify any Indus sign.
- **Konar Sandal tablets** (LEI 28–31, end of 3rd millennium) contain names (*Zulari*, *Pa(a)reri*). The Konar Sandal cylinder seal with Indus iconography comes from the same site.
- **Claimed Linear Elamite–Indus sign matches.** The Zenodo preprint by Scott Dunn (2026, records 21611160 / 21632153) lists about 6 matches to Mahadevan signs (M87, M101, M202, M236, M303, M214/228). The sound values are the author's own, assuming Indo-Iranian, and the paper also rereads Puzur-Sušinak as Indo-Iranian. They are not Desset's values and not an independent key, so not testable as an external anchor. The peer-reviewed comparison (IJAS, usb.ac.ir) concludes the two scripts are "not directly related". That server refused connections from here.
- **Verdict:** no shape-based Linear Elamite to Indus key exists that is independent of Indus-internal guesses. Similar shapes in neighbouring scripts don't carry sound values across, and Linear Elamite's own sign values were fixed from royal names in cuneiform bilinguals, which the Indus script lacks.

## 7. The Ur seal with a cuneiform legend, from the primary publication (Gadd 1932, archive.org in.gov.ignca.33779)

- BM 120573: grey steatite, perforated ridge ("button") back, **short-horned bull with head lowered**, with a horizontal line under the legend (never found with Indus script). Gadd dates the cuneiform "in any case pre-Sargonic". Find context unknown (Woolley).
- The legend has 3 signs and possibly a trace of a 4th: first SAG(K) or KA; second KU or possibly LU; **third "almost certainly ŠI"**. Gadd's best reading is ***sak-ku-ši(-…)***, and he states it "does not seem to be any Sumerian or Akkadian name".
- CDLI check (135,254 texts): no exact match for any combination of SAG/SAK/KA × KU/LU × ŠI. The only string sharing a prefix is Akkadian *ka-lu-ši-na* "all of them", which is not a name. **This corrects section 5**: the *inim-ma-zi* reading needs a ZI, which Gadd rules out.
- So the only Indus-type seal carrying a readable legend gives its owner a **non-Mesopotamian name**, /sakkuši/ or /kaluši/. With *Samar* and *Nanaza* (section 5) and the Gulf seal's *Zaširi* (section 6), this is a third name, and a probable fourth, from the Meluhha–Gulf sphere. All are sound shapes only. None of them is written in Indus signs, so none fixes a sign.
- Gadd's catalogue also gives find-spots for the Ur Indus seals. No. 2 (BM 122187): round seal, short-horned bull, 5 Indus signs. No. 18 (BM 123059): bought via a Baghdad dealer, 5 signs including two "man" signs and a fish, above a unique mating-bull scene. These are the foreign-found texts that section 1 tests.

## 8. Reduplication in foreign-found texts (weak, not replicated)

Two of the attested Meluhhan names are reduplicated (*na-na-sa₃*). If Indus seals abroad spelled foreign names by sound, adjacent repeated signs should be commoner there. Share of texts with an adjacent repeated sign, against length-matched home texts (20,000 draws):
- IM77: foreign 0.25 (4/16) vs home 0.073, p = 0.023.
- Merged corpus: foreign 0.043 (1/23) vs home 0.014, p = 0.27.

Suggestive in Mahadevan's transcription only. Not counted as a finding.

## 9. Gadd 1932 catalogue: find contexts of the 18 Indus-style seals from Babylonia

| Gadd no. | museum / excavation no. | type | Indus signs? | context and date |
|---|---|---|---|---|
| 1 | BM 120573 / U.7683 | square button seal, short-horned bull | **cuneiform** *sak-ku-ši* | no context. Script "pre-Sargonic" |
| 2 | BM 122187 | round, short-horned bull | 5 signs | bought at Ur 1928–9, no find-spot |
| 3 | BM 122946 / U.17342 | button seal, half | yes | Ur 1930–1, no context |
| 4 | BM 122188 | fragment | fish + 1 | Ur 1930–1, no context |
| 5 | Penn / U.17341 | lower half | yes | Ur, Diqdiqqah area |
| 6 | BM 122947 | cylinder: humped bull, palm, scorpion, snakes, rayed figure | — | **Larsa tomb** cut into Amar-Suen's (Bur-Sin) annex: after c. 2000 BC |
| 7 | U.11958 | cylinder: unicorn bull, tree | cross-hatched fish (+2?) | Ur |
| 8 | BM 118704 / U.6020 | round Gulf-type, Sumerian sacrifice scene | — | loose near surface |
| 9–13 | BM 122945, 120576; Penn U.16397, U.16747; BM 122841 | round Gulf-type | no. 11 scorpion + "eye"; **no. 12 water-carrier drawn as a picture version of script signs** | no context (no. 12: "Kassite? level" rubbish) |
| 14 | Penn CBS 16301 / U.7027 | round, Babylonian scene (bull-men, seated god) | — | Ur |
| 15 | Penn / U.8685 | round, bull | crowded inscription with a unique first sign | **grave, probably Sargonic** (gold double-crescent earrings, carnelian): c. 2300 BC |
| 16 | BM 123208 / U.17649 | round, bull, no manger | 4 well-known signs | **tomb-shaft fill, "Second Dynasty of Ur"** (Woolley) |
| 17 | BM (unprovenanced, Babylonia) | round, short-horned bull with manger | 5 signs "in the best Indian style" | unknown |
| 18 | BM 123059 | round, mating bull and cow | 5 signs: 2 "men", fish, + 2 | Baghdad dealer, presumed Babylonia |

Datable: no. 15 (Sargonic), no. 6 (Larsa), no. 16 (early). **None is securely Ur III**, the period of Samar and Nanaza (who were at Irisagrig, not Ur). So no seal–person match is possible from these contexts.

**Gadd no. 12** shows a water-carrier with yoke and two pots between "parenthesis" marks, and Gadd identifies each element with a script sign (Mahadevan pl. cxxix, ccclxxxvi ff.; clxxviii; cccxvii; xx). A scene drawn as the full picture of signs fixes the **pictorial referent** of those signs (the man with a yoke is a water or burden carrier, glyphs 151/156, M12/M15), but not their sound.

## 10. Gulf Type seals (Laursen 2010, *AAE* 21: 96–134; PDF supplied by the user) and the person sign

**Facts from the paper (objects, not interpretations):**
- **Table 1, no. 28** (Winkelmann 1999, abb. 2; the Linear Elamite database's LEI 22, Ligabue Collection) is a Gulf Type seal with the standard Indus layout, a bull in profile below a text band, **but the text band is in Linear Elamite**, read *za-ši-ri* (Linear Elamite database). On this seal type, the slot that elsewhere holds Indus text holds a personal name.
- **Table 1, no. 11** (Bahrain National Museum, Karzakkan mound cemetery, excavated 1999/2005) carries **pseudo-script**: one clear "twins" sign (two standing figures) first, then random strokes imitating Indus writing. A cutter who could not write Indus still reproduced the first sign.
- Laursen counts 27–29 Indus-inscribed Gulf Type seals. The "twins" sign occurs 8 times, and **more Bahrain texts begin with it (4) than all Indus Valley texts do**.
- There are three morphological groups of Gulf Type seals by production area (Indus Valley, Iran, Mesopotamia/Bahrain). Inscribed seals are measurably larger (22.49 ± 0.77 mm, n = 54; Wilcoxon p = 2.8e-6).

**Our test on the corpora (symbol identity only, length-matched home texts, 20,000 draws):**
| measure | foreign | home | p | corpus |
|---|---|---|---|---|
| begins with a person sign (Wells 90 or 91) | **0.261** | 0.007 | <0.0001 | merged (Wells), n = 23 |
| contains 90 or 91 | **0.435** | 0.040 | <0.0001 | merged |
| begins with M1 (standing person) | 0.062 | 0.003 | 0.045 | IM77, n = 16 |
| contains M1 | 0.188 | 0.045 | 0.032 | IM77 |
| M1 doubled (1 1) | 0.062 | 0.002 | 0.030 | IM77 |

**What the data supports, with no one's reading:** Indus-sign texts found abroad (1) lack the home opener (section 1), (2) use unattested sign pairs, (3) are dominated by the person signs, often at the start and often doubled, and (4) sit on a seal type whose text slot demonstrably held a personal name (the Linear Elamite *Zaširi* seal). Together: **the western texts are names, written with a different content structure from home texts, led by a person or "twins" sign.** This is the closest thing yet to a functional anchor. Still no sound value: we know *what* these texts are, not how they were pronounced.

**What would give sounds:** a Gulf Type seal in Indus signs whose owner is named in a cuneiform text from the same place and time (Bahrain, Failaka, Ur; about 2050–1900 BC). Next source: Kjærum 1983/1994 for all Gulf Type seals, cross-checked against Dilmunite names in Old Babylonian texts. CDLI has those, and I can search them.

## 11. The only external key, tested blind: Linear Elamite look-alike signs (Fuls 2024, IJAS 14(1): 3–16, PDF supplied by the user)

Fuls, Fig. 9: 17 Indus signs (Wells/ICIT numbers) with graphically similar Linear Elamite signs and Desset et al. 2022 values: 850 a · 502 li · 455 hi · 370/371 la · 012 na · 360 ši · 440 še · 323 ni · 400/402 ni · 684 š · 167 n · 899/900 pa · 335 po · 032 zu · 615 zo · 831 r · 482 re. Saved as `data/derived/fuls2024-le-indus-key.json`. Fuls himself labels several as "graphic universals" (simple shapes found in many scripts) and concludes the scripts are "not directly related".

This key was never fitted to Indus texts, so a blind test is fair. Test, no dictionary needed: do Indus adjacent pairs of mapped signs reproduce syllable pairs attested in the Linear Elamite corpus (HLED, 2,127 syllables, 465 distinct bigrams), more than keys with the same values reshuffled among the signs (20,000)?
- Indus pairs with both signs mapped: **only 25** of 9,283 (the key covers mostly rare signs).
- Pairs matching an attested Linear Elamite syllable pair: **real key 2; shuffled keys 4.1 ± 2.7; P = 0.83.** **Fails.**

Shape similarity between Indus and Linear Elamite signs does not carry sound values. Fuls also notes **Susa "text O"** (Louvre Sb 9382), a red clay tablet with about 20 Indus-like graphemes. One corresponds to Indus 853, which is text-initial in 10 of 11 Indus texts, and it sits initial on text O as well. This is a mixed Indus / Linear Elamite / Sibri (Mehrgarh VIII) object of the early 2nd millennium, too eroded to read.

## 12. Laursen 2016 (*AAE* 27: 2–18, PDF supplied by the user): Dilmun after the Indus script
- From about 1950 BC Dilmun used a **symbol system** (palm branches, palm trees, altars) on pottery, tokens and seals, developing from single symbols to sequences forming "irregular logograms" and pseudo-script. It was exclusive to the royal house and temple (cult of Inzak), in a society Laursen calls illiterate. It is a **successor, not a script**: no sound values.
- **Dilmun adopted the Indus weight standard about 2050 BC** (Babylonian mercantile texts, Indus cubical weights in Bahrain). CDLI has Ur III references to *na₄ dilmun* ("Dilmun stone", e.g. UET 3, 0672 *kišib₃ na₄ dilmun*), which can be a Dilmun-standard weight stone or seal stone. That is a numeric link (Indus weights ↔ Mesopotamian units), not a phonetic one.
- The Indus script in Dilmun: the high frequency of the "twins" sign (section 10); the script itself "never gained a foothold".

## 13. Laursen 2010, Fig. 11: the Gulf Type inscriptions (read from the drawings, shapes only)
Drawings "as seen on the impression". The rightmost sign is first in reading order.
- **Person / doubled-person ("twins") sign in first position:** Bahrain 6, 7, 8, 11; Iran 15; Mesopotamia 18, 20 (person with dot), 25. It also appears inside sequences: Bahrain 9 (twice), 10; Mesopotamia 24 (doubled, mid-text), 26, 27; Failaka 12, 13; Iran 14.
- **Recurring block, doubled figure + short-stroke cluster:** Bahrain 7 (`⋮⋮` + twins), 9 (`||||` … twins … person), 10 (`⋮⋮` twins `⋮⋮`), 11 (`| | \ \ | |` + twins). Four of the six Bahrain inscriptions share a "figure + number" block. It is absent from the home Indus-Valley Group 1 drawings (1, 3, 4, 5).
- **Seal 28** (Iran group): the same layout with a 3-sign Linear Elamite text, *za-ši-ri* (section 10).
- **Seal 11:** pseudo-script after an initial twins sign.

Structural reading, no language assumed: Bahrain seal texts are built as [doubled figure] + [number-like stroke group] + [1–2 other signs]. A fixed lineage or affiliation marker plus a count plus an individual element is the shape of the "ordered family sequences" Vidale (2005) predicted. **Test to run next:** transcribe these ~29 texts into Wells IDs (they are partly in the merged corpus already: Failaka, Qala'at al-Bahrain, Ur, Kish, Susa, Karzakan) and check whether the stroke count after the figure varies between owners while the figure stays fixed. That is the signature of a family or rank numbering.

## 14. BM 120573 photographed (British Museum reg. 1928,1009.56; photos supplied by the user, `data/images/`)
Impression: **three cuneiform signs** in the upper register (from left: a wedge-headed sign, a box-shaped sign, an arrow-like sign), a **divider line** with faint traces below them (Gadd's possible fourth sign), and a **short-horned bull** with head lowered and a **band over its back**. **No Indus signs.** The reverse has the perforated ridge ("button") of Indus-type seals. The photo confirms Gadd's description. It does not allow a better reading than *sak/ka – ku/lu – ši(-…)*, which needs collation of the object by an Assyriologist.
