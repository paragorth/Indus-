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
