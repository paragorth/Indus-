# Contemporary non-Harappan artifacts with Indus motifs (c. 2600–1900 BC)

What's been found, and what each could contribute as an anchor. Sources are web-search summaries of published work. Primary papers were blocked in this environment, so every entry needs checking against the publication before use.

## Traditions that also had writing (possible anchors)

| tradition | what is Indus-like | writing | anchor value |
|---|---|---|---|
| **Linear Elamite** (Susa, Fars, Kerman/Jiroft; c. 2300–1880 BC) | A published comparison ([IJAS](https://ijas.usb.ac.ir/article_8463.html); [academia](https://www.academia.edu/125297406/Comparison_of_Linear_Elamite_and_Indus_Writing_Systems)) finds shared sign-modification methods and reading direction. A separate claim that ">10 signs match Indus signs one-to-one in form and position" comes from a Zenodo preprint that also argues for a "Sanskrit sister language", so treat it with caution. | **Deciphered 2022**: Desset, Tabibzadeh, Kervran, Basello & Marchesi, *ZA* 112: 11–60. About 72 phonetic signs, 45 inscriptions, 2,000+ tokens (Hatamti database, Liège). | **Highest.** The only script neighbouring the Indus with known sound values. The test to run: take the signs that match Indus in shape, attach their Linear Elamite sound values, and test that key on Indus texts. It comes from outside the corpus, so it avoids the overfitting that sank the key search. Needs the sign concordance, which is blocked here. |
| **Konar Sandal "Geometric" tablets** (Jiroft, 3rd millennium) | Konar Sandal South also produced a white-marble cylinder seal carved with Indus animal-series iconography in local style ([Vidale & Frenez 2015, *South Asian Studies* 31](https://www.tandfonline.com/doi/full/10.1080/02666030.2015.1008820)). | A separate, undeciphered "geometric" script ([Madjidzadeh](https://www.researchgate.net/publication/287575331_A_new_writing_system_discovered_in_3rd_millennium_bce_iran_The_konar_sandal_'geometric'_tablets)) | Low until deciphered, but it places Indus craftsmen or merchants at a literate Iranian city. |
| **Gulf / Dilmun seals** (Failaka, Bahrain, Saar; c. 2100–1800 BC) | Round "Gulf type" seals derived from Indus seals. The only animal on them is the short-horned bull (gaur). Some carry Indus signs. About 400 seals from Failaka F3/F6 and Qala'at al-Bahrain ([Laursen 2010, *AAE* 21](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0471.2010.00329.x)). | Indus signs on the early ones; Dilmun also used **cuneiform** | **High.** Our corpus test shows texts found abroad break the Indus grammar (see `ANCHORS.md`). Dilmunite personal names are attested in cuneiform, so a Gulf seal with Indus signs and a Dilmun owner attested by name would fix sign values. |
| **Indus seal with a cuneiform legend** (Ur, U.7683 = BM 120573; square, gaur) | Indus seal form and animal | Cuneiform, 3 signs (Gadd: *sag-ku-zi*) | Checked in `ANCHORS.md`: *sag-ku-zi* has 0 attestations in the CDLI corpus. The alternative reading *inim-ma-zi* is a Sumerian name. Probably not a Meluhhan word. |
| **Tepe Yahya potters' marks** | Marks compared with Indus signs | Marks, not writing | Low: short marks match by chance. |

## Traditions with shared motifs but no writing (context only)

- **Mesopotamian cylinder seals** with Indus fauna, e.g. Tell Asmar (Eshnunna): elephant, rhinoceros, gharial in Indus carving style, animals absent from Sumerian art ([harappa.com](https://www.harappa.com/blog/indus-cylinder-seals)). Reverse traffic too: a Mesopotamian-style cylinder seal at Kalibangan.
- **Master of animals / tiger-strangler.** Mohenjo-daro seals show a man gripping two tigers, the Indus counterpart of the Mesopotamian hero with two lions. A horned bull-man fighting a tiger is compared to Enkidu ([harappa.com](https://www.harappa.com/blog/bare-handed-tiger-wrestling-seals)). This is a shared motif, not shared meaning.
- **Oxus civilisation / BMAC** (Gonur Depe, Altyn Depe; 2400–1600 BC). Two provincial Indus seals at Altyn Depe (one is in our corpus: signs 415·390); Indus pipal-leaf motif on seals, metal and mosaics at Gonur; elephant ivory ([Penn Museum, *Expedition*](https://www.penn.museum/sites/expedition/the-middle-asian-interaction-sphere/)).
- **Helmand / Baluchistan** (Shahr-i Sokhta, Mundigak, Nal, Kulli). Indus-fashion steatite beads, Indus-related vessels, Nal pottery at Shahr-i Sokhta and Tepe Yahya ([Vidale et al.](https://www.researchgate.net/publication/261979101)).

## What to do next (in order)

1. **Linear Elamite key test.** Get the published sign list with sound values (Desset et al. 2022; Hatamti database) and the Indus–Linear Elamite shape concordance. Build a key only from shape matches fixed *before* looking at Indus sequences. Test it with the Q14 criterion (value-shuffled keys) and the cross-language baseline from `RESULTS.md`. The likely outcome is a fail, because similar shapes in neighbouring scripts rarely share sound values. But it is the one test anyone can run today with an external key.
2. **Gulf seal catalogue.** List every Gulf-type seal with Indus signs (Laursen 2010; Kjærum's Failaka catalogue), with find-spot and date, and cross-check against Dilmunite names in cuneiform (Old Babylonian Ur and Dilmun texts; the CDLI dump already in hand).

Both need hosts this environment blocks: `hatamti-elam.uliege.be`, `zenodo.org`, `www.academia.edu`, `onlinelibrary.wiley.com`, `www.tandfonline.com`, `archive.org`.

## Objects not tagged Indus, in museum open data (Met Open Access, searched 25 Sept 2026)

Method: all 10,854 Met objects dated 3300–1500 BC. Kept the Ancient Near Eastern and Asian departments, dropped anything labelled Indus or Harappan, and searched titles, object names and tags for Indus motifs. That left 64 matches (`data/derived/met-non-indus-motif-matches.json`). Titles are short, so this finds labelled motifs only.

**Motifs specific to South Asia, on non-Indus objects:**
| Met no. | object | date | place | why it matters |
|---|---|---|---|---|
| [2014.717](http://www.metmuseum.org/art/collection/search/329087) | Chlorite vessel with calcite inlay: **two zebu** | 2600–2350 BC | "probably Persian Gulf" | The humped zebu is South Asian. Chlorite with inlay is the Iranian "intercultural style" (Jiroft / Tepe Yahya). An Indus animal on a Gulf/Iranian object. |
| [2015.789](http://www.metmuseum.org/art/collection/search/328186) | Copper-alloy plate: **reclining zebu** | late 3rd–early 2nd mill. BC | Eastern Iran | Zebu again, in the Helmand / Baluchistan corridor. |
| [1989.281.43](http://www.metmuseum.org/art/collection/search/327430) | Steatite plaque with gold foil: **tiger** | late 3rd–early 2nd mill. BC | Bactria-Margiana (Oxus) | The tiger is an Indus animal (tiger-strangler seals, the "horned tiger"), not a Central Asian one. |
| [1983.535.92](http://www.metmuseum.org/art/collection/search/326410) | Compartmented copper stamp seal | late 3rd–early 2nd mill. BC | Bactria-Margiana | The compartmented seal type spans Baluchistan, the Helmand, the Oxus and the Indus. |
| [1996.353](http://www.metmuseum.org/art/collection/search/327527); [1989.281.2](http://www.metmuseum.org/art/collection/search/327391) | Human-headed **bison**; bison figure | c. 2080 BC; 3rd mill. BC | Mesopotamia / Syria | The gaur is the "short-horned bull" of the western-trade seals. |
| [2015.505](http://www.metmuseum.org/art/collection/search/39126) | Bronze | 2000–1750 BC | Kausambi, India | Gangetic, late-/post-Harappan horizon. Catalogued as "India", not Indus. |

**Shared but generic motifs** (Mesopotamia, Anatolia, Syria): bull-men and heroes grasping animals (master of animals, 11 objects), scorpions (4), bull processions and bull heads (about 30). These match the Indus tiger-strangler and horned-figure scenes in composition, but they are native Mesopotamian themes, so they carry no specific Indus signal.

**Absent:** no unicorn, elephant, rhinoceros or gharial on any non-Indus Met object of the period. In the Met's holdings the most diagnostic Indus animals stay inside the Indus sphere, and only the zebu and the tiger travel. This fits the Gulf-seal pattern: abroad, the gaur/short-horned bull replaces the unicorn.

**None of these objects carries writing**, so none can fix a sign value. Their use is mapping where Indus iconography went: the Gulf, eastern Iran and the Oxus. That is also where inscribed objects from the same workshops, the Gulf seals and the Linear Elamite sites, might carry names.

## Further sources searched (25 Sept 2026)

- **Cleveland Museum of Art open access** (68,778 objects; full descriptions searched): only 305 objects fall in 3300–1500 BC, and none outside the Indus carries an Indus motif. Its three Indus seals (1964.104, 1973.160, 1973.161) and the Quetta ibex jar (2001.1) are already Indus-labelled.
- **Smithsonian open access:** the GitHub repository only points to off-site storage (blocked here).
- **CDLI catalogue, all fields** (for elephant, zebu, rhinoceros, gharial, unicorn, Indus, Harappan, Meluhha, bison): 8 hits.
  - **P387604** (UCLA YRL SC 1826-Bx3-8, Old Babylonian, unprovenanced): a clay tablet with a **hand-drawn zebu**, the South Asian humped bull, on a Mesopotamian school or practice tablet. Unpublished. Worth an image check.
  - **P500711 = BM 110450** (NABU 2017/033), catalogued "Harappan?": published by Proust 2017 as an ordinary Babylonian metrological scratch pad. The Harappan label looks like a catalogue error. Dead end.
  - **P302053 = ENES 1089 / NCBS 00876** (Yale, Newell Collection): a steatite button-handled stamp catalogued as Harappan, held in a *Babylonian* seal collection. Its Mesopotamian find context, if recorded, would make it one of the foreign-found Indus seals.
  - **CUSAS 40, 1354 and 1582** (Irisagrig, Šu-Suen 6, months 3 and 12): the other two ration texts for "men of Meluhha" (shepherds). They are catalogued but not transliterated in this dump, so any further Meluhhan names on them need the publication (George 2019, CUSAS 40).
- **Met Oxus (BMAC) stamp seals:** about 150 stamp seals, mostly compartmented, almost all unprovenanced 1980s acquisitions. Titles carry no motif detail beyond a few (snake-holding figure, winged hero dominating snakes, **seated monkey**, boar, griffin). Checking them for Indus signs needs the images, which are hosted on metmuseum.org (blocked here). One **Dilmun** seal (1987.96.22: hunters and goats) is untagged as Indus and shows no listed Indus motif.

## Checked on CDLI's live site (25 Sept 2026)
- **P387604** (UCLA zebu tablet): photo inspected. A bovine drawn in outline on a round clay tablet, **no signs of any script**. Not an anchor.
- **P302053** (Yale NCBS 00876, "Harappan" stamp): CDLI records "no linguistic content". **No Indus signs.** Drops out.
- **CUSAS 40, 1354 and 1582** (Irisagrig, Šu-Suen 6, months 3 and 12): same three people as Nisaba 15, 0371, *na-na-sa₃*, *sa₆-ma-ar*, *a-li-a-hi dam-a-ni*, "men of Meluhha", bezoar shepherds. Three tablets across one year, **no further Meluhhan names**.

Not reachable from this environment (site-side blocks, not the network setting): harappa.com and academia.edu (403 to scripts), Wiley and Taylor & Francis (paywalled), ijas.usb.ac.ir (resets), metmuseum.org pages (429 rate limit; the Met data API works). Kjærum 1983, Kjærum 1994 and Laursen & Steinkeller 2017 are printed books.

## Met 2014.717, "Vessel with two zebu" (photo supplied by the user)
Chlorite with drilled white-stone inlay, about 2600–2350 BC, "probably Persian Gulf region". A humped bull with long curving horns, **head frontal and body in profile** (the Iranian "intercultural style" of Jiroft / Tepe Yahya, not Indus), scale-pattern "mountains" above, wavy water lines. **No writing.** It is a South Asian animal carved by Iranian workshops for Gulf trade: evidence of contact, not an anchor.

In the Indus corpus the zebu ("humped bull", IM77 field symbols 31/32) is on **54 objects** (51 seals, 3 sealings; 46 Mohenjo-daro, 6 Harappa, 2 Kalibangan), about 5% of the unicorn count. Its texts are structurally identical to unicorn texts (mean 4.4 vs 4.8 signs; jar-final 33% vs 36%; opener-initial 22% vs 21%). The emblem does not select a different text type (consistent with the null emblem–sign result, Q10).

## Met 1987.96.22, Dilmun stamp seal "hunters and goats, rectangular pen (?)" (photo supplied by the user)
Round Dilmun Type seal, early 2nd millennium BC, with an impression. Rearing long-horned goats or ibexes, human figures (hunters) at left and top, a **central gridded rectangle** with a reclining long-horned animal, and short stroke fringes at the edges. **No Indus signs, no cuneiform**: a pictorial Dilmun seal from after the Indus-sign phase, not an anchor. Shape note only: the gridded rectangle resembles the Indus grid sign (Wells 615 / M249, which takes tall-stroke counts on tablets) and may be one of the Indus-derived symbols of the Dilmun system (Laursen 2016).
