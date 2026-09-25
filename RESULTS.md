## Run started 2026-09-25 00:01:15Z

corpus: 3250 unique texts (1625 train / 1625 held-out), 567 signs in train; lexicons: Tamil 62868 forms, Sanskrit 163998 forms; MINLEN={'full': 5, 'skel': 3}; restarts=20, fakes=30, iterations=400*N

### 2026-09-25 00:01:53Z  published key (Yajnadevam xlits.csv) vs sanskrit, full
real=0.0104, scrambled=0.0121, fake_mean=0.0103, fake_sd=0.0014, z_fixed=0.0218, pass_fixed=False

### 2026-09-25 00:02:16Z  published key (Yajnadevam xlits.csv) vs sanskrit, skel
real=0.2694, scrambled=0.2522, fake_mean=0.2633, fake_sd=0.0030, z_fixed=2.0511, pass_fixed=False

### 2026-09-25 00:02:27Z  published key (Yajnadevam xlits.csv) vs tamil, full
real=0.0068, scrambled=0.0053, fake_mean=0.0052, fake_sd=0.0010, z_fixed=1.5628, pass_fixed=False

### 2026-09-25 00:02:33Z  published key (Yajnadevam xlits.csv) vs tamil, skel
real=0.0743, scrambled=0.0778, fake_mean=0.0744, fake_sd=0.0009, z_fixed=-0.0699, pass_fixed=False

### 2026-09-25 00:05:28Z  `control-planted-tamil-full-80-noise0.0`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 174s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3010 | 0.0103 | 0.2203 | 0.2046 | 0.0151 | 6.3886 | 0.3170 | 3514 | 2741 | 20 | 30 |

fixed-key / verdict: z_fixed=12.8601, pass_fixed=True, fakefit_z_fixed=5.9265, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +6.39 SD -> PASS**

### 2026-09-25 00:07:22Z  `control-planted-tamil-full-80-noise0.2`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 114s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2001 | 0.0096 | 0.1593 | 0.1323 | 0.0128 | 5.3104 | 0.2153 | 2497 | 1873 | 20 | 30 |

fixed-key / verdict: z_fixed=9.3935, pass_fixed=True, fakefit_z_fixed=6.6879, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +5.31 SD -> PASS**

### 2026-09-25 00:08:36Z  `control-planted-tamil-full-80-noise0.4`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 74s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1289 | 0.0066 | 0.1126 | 0.0782 | 0.0055 | 9.2427 | 0.1420 | 1647 | 1145 | 20 | 30 |

fixed-key / verdict: z_fixed=8.4025, pass_fixed=True, fakefit_z_fixed=7.0309, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +9.24 SD -> PASS**

### 2026-09-25 00:12:27Z  `control-planted-sanskrit-full-80-noise0.0`  (control)
method: full key, sanskrit, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 231s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2325 | 0.0119 | 0.1511 | 0.1893 | 0.0127 | 3.4091 | 0.2621 | 2965 | 2366 | 20 | 30 |

fixed-key / verdict: z_fixed=7.8791, pass_fixed=True, fakefit_z_fixed=7.9374, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +3.41 SD -> PASS**

### 2026-09-25 00:15:04Z  `control-planted-sanskrit-full-80-noise0.2`  (control)
method: full key, sanskrit, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 157s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1484 | 0.0077 | 0.1010 | 0.1219 | 0.0097 | 2.7376 | 0.1684 | 1898 | 1595 | 20 | 30 |

fixed-key / verdict: z_fixed=7.7945, pass_fixed=True, fakefit_z_fixed=6.5265, fakefit_pass_fixed=True, PASS=False

**gap-over-fake = +2.74 SD -> fail**

### 2026-09-25 00:16:58Z  `control-planted-sanskrit-full-80-noise0.4`  (control)
method: full key, sanskrit, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 114s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0854 | 0.0049 | 0.0643 | 0.0663 | 0.0078 | 2.4359 | 0.0928 | 1150 | 1021 | 20 | 30 |

fixed-key / verdict: z_fixed=5.6270, pass_fixed=True, fakefit_z_fixed=5.7226, fakefit_pass_fixed=True, PASS=False

**gap-over-fake = +2.44 SD -> fail**

### 2026-09-25 00:19:31Z  `control-planted-tamil-skel-80-noise0.0`  (control)
method: skel key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 153s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2624 | 0.0047 | 0.2034 | 0.2594 | 0.0046 | 0.6630 | 0.2692 | 3172 | 3159 | 20 | 30 |

fixed-key / verdict: z_fixed=3.3322, pass_fixed=True, fakefit_z_fixed=3.4119, fakefit_pass_fixed=True, PASS=False

**gap-over-fake = +0.66 SD -> fail**

### 2026-09-25 00:22:39Z  `control-planted-tamil-mixed-150-noise0.2`  (control)
method: mixed key, tamil, top-150 signs, 20 real restarts + 30 fake-lexicon searches, 60000 SA iterations each, 188s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2300 | 0.0133 | 0.1892 | 0.1428 | 0.0143 | 6.1120 | 0.2533 | 3019 | 2188 | 20 | 30 |

fixed-key / verdict: z_fixed=9.7940, pass_fixed=True, fakefit_z_fixed=8.6955, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +6.11 SD -> PASS**

### 2026-09-25 00:23:16Z  `control-planted-tamil-logo-150-noise0.2`  (control)
method: logo key, tamil, top-150 signs, 20 real restarts + 30 fake-lexicon searches, 60000 SA iterations each, 37s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2019 | 0.0144 | 0.1487 | 0.2401 | 0.0229 | -1.6688 | 0.2204 | 1453 | 1672 | 20 | 30 |

fixed-key / verdict: PASS=False

**gap-over-fake = -1.67 SD -> fail**

### 2026-09-25 00:25:58Z  `tamil-full-80`
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 162s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2663 | 0.0063 | 0.1941 | 0.1705 | 0.0175 | 5.4703 | 0.2782 | 3161 | 2532 | 20 | 30 |

fixed-key / verdict: z_fixed=6.4052, pass_fixed=True, fakefit_z_fixed=4.9557, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +5.47 SD -> PASS**

most stable assignments across real restarts (share of restarts, sign, value, share of fake runs giving that value, also modal under fakes?): 0.75 032->'tan' (0.07); 0.75 031->'tan' (0.07); 0.70 368->'tan' (0.03); 0.70 001->'tan' (0.17); 0.60 235->'kan' (0.10); 0.55 233->'kan' (0.17); 0.55 231->'kan' (0.13, fake-modal); 0.55 140->'tan' (0.00)

**STOP: tamil-full-80 passed. See STOP.md. Halting.**

## Run started 2026-09-25 00:26:16Z (IM77; queued before the STOP, aborted by hand at once so that break tests could run; no condition finished)

corpus (im77): 2286 unique texts (1143 train / 1143 held-out), 340 signs in train; lexicons: Tamil 62868 forms, Sanskrit 163998 forms; MINLEN={'full': 5, 'skel': 3}; restarts=20, fakes=30, iterations=400*N

## Break tests for `tamil-full-80` (2026-09-25 00:26:51Z)

### 2026-09-25 00:29:27Z  `break-english-full-80`  (control)
method: full key, english, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 156s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1232 | 0.0120 | 0.0567 | 0.1062 | 0.0102 | 1.6570 | 0.1460 | 1688 | 1384 | 20 | 30 |

fixed-key / verdict: z_fixed=17.0025, pass_fixed=True, fakefit_z_fixed=11.1638, fakefit_pass_fixed=True, PASS=False

**gap-over-fake = +1.66 SD -> fail**

### 2026-09-25 00:31:38Z  `break-iid-tamil-full-80`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 131s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2403 | 0.0055 | 0.2456 | 0.1697 | 0.0370 | 1.9108 | 0.2501 | 2888 | 2616 | 20 | 30 |

fixed-key / verdict: z_fixed=3.7481, pass_fixed=False, fakefit_z_fixed=2.5435, fakefit_pass_fixed=False, PASS=False

**gap-over-fake = +1.91 SD -> fail**

### 2026-09-25 00:34:22Z  `break-im77-tamil-full-80`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 163s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2772 | 0.0172 | 0.1978 | 0.1802 | 0.0177 | 5.4890 | 0.3079 | 3288 | 2408 | 20 | 30 |

fixed-key / verdict: z_fixed=7.0229, pass_fixed=True, fakefit_z_fixed=5.6488, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +5.49 SD -> PASS**

### 2026-09-25 00:37:48Z  `break-sanskrit-full-80`  (control)
method: full key, sanskrit, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 206s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1954 | 0.0122 | 0.1152 | 0.1548 | 0.0121 | 3.3589 | 0.2111 | 2449 | 2121 | 20 | 30 |

fixed-key / verdict: z_fixed=13.5405, pass_fixed=True, fakefit_z_fixed=6.0744, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +3.36 SD -> PASS**

## Markov-null tests (2026-09-25 00:38:06Z): fake lexicons generated by a 2-letter-context character model trained on the real lexicon (same size, same length distribution)

### 2026-09-25 01:05:39Z  `markov-planted-0.0-tamil-full-80`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 151s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3010 | 0.0103 | 0.2203 | 0.3182 | 0.0170 | -1.0122 | 0.3170 | 3514 | 4169 | 20 | 30 |

fixed-key / verdict: z_fixed=5.7330, pass_fixed=True, fakefit_z_fixed=3.6632, fakefit_pass_fixed=True, PASS=False

**gap-over-fake = -1.01 SD -> fail**

### 2026-09-25 01:07:13Z  `markov-planted-0.2-tamil-full-80`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 92s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2001 | 0.0096 | 0.1593 | 0.2121 | 0.0116 | -1.0297 | 0.2153 | 2497 | 2916 | 20 | 30 |

fixed-key / verdict: z_fixed=4.1573, pass_fixed=True, fakefit_z_fixed=3.3962, fakefit_pass_fixed=True, PASS=False

**gap-over-fake = -1.03 SD -> fail**

### 2026-09-25 01:09:33Z  `markov-tamil-full-80`
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 141s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2663 | 0.0063 | 0.1941 | 0.2856 | 0.0128 | -1.5133 | 0.2782 | 3161 | 3639 | 20 | 30 |

fixed-key / verdict: z_fixed=3.5220, pass_fixed=True, fakefit_z_fixed=3.0628, fakefit_pass_fixed=True, PASS=False

**gap-over-fake = -1.51 SD -> fail**

most stable assignments across real restarts (share of restarts, sign, value, share of fake runs giving that value, also modal under fakes?): 0.75 032->'tan' (0.13); 0.75 031->'tan' (0.13); 0.70 368->'tan' (0.17); 0.70 001->'tan' (0.17); 0.60 235->'kan' (0.67, fake-modal); 0.55 233->'kan' (0.57, fake-modal); 0.55 231->'kan' (0.43, fake-modal); 0.55 140->'tan' (0.10)

(markov_test.py stopped by hand after markov-tamil-full-80: the planted-Tamil controls fail this null, so its remaining conditions could not be informative)

## Cross-language controls (2026-09-25 01:09:54Z): planted language X, searched with lexicon Y (noise 0.2, letter-shuffle null)

### 2026-09-25 01:11:45Z  `cross-planted-sanskrit-searched-tamil-full-80`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 109s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1941 | 0.0064 | 0.1588 | 0.1346 | 0.0123 | 4.8181 | 0.2033 | 2319 | 1825 | 20 | 30 |

fixed-key / verdict: z_fixed=8.8484, pass_fixed=True, fakefit_z_fixed=2.5720, fakefit_pass_fixed=False, PASS=True

**gap-over-fake = +4.82 SD -> PASS**

### 2026-09-25 01:13:00Z  `cross-planted-english-searched-tamil-full-80`  (control)
method: full key, tamil, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 74s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1300 | 0.0051 | 0.0937 | 0.0943 | 0.0068 | 5.2276 | 0.1368 | 1629 | 1268 | 20 | 30 |

fixed-key / verdict: z_fixed=7.1376, pass_fixed=True, fakefit_z_fixed=4.0571, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +5.23 SD -> PASS**

### 2026-09-25 01:15:44Z  `cross-planted-tamil-searched-sanskrit-full-80`  (control)
method: full key, sanskrit, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 163s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1528 | 0.0056 | 0.1024 | 0.1251 | 0.0078 | 3.5484 | 0.1696 | 2031 | 1757 | 20 | 30 |

fixed-key / verdict: z_fixed=8.0487, pass_fixed=True, fakefit_z_fixed=6.2930, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +3.55 SD -> PASS**

### 2026-09-25 01:17:34Z  `cross-planted-tamil-searched-english-full-80`  (control)
method: full key, english, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 109s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0945 | 0.0079 | 0.0474 | 0.0844 | 0.0054 | 1.8781 | 0.1077 | 1281 | 1167 | 20 | 30 |

fixed-key / verdict: z_fixed=7.9647, pass_fixed=True, fakefit_z_fixed=13.3662, fakefit_pass_fixed=True, PASS=False

**gap-over-fake = +1.88 SD -> fail**

## Final entry (2026-09-25 01:18Z, about 1 h 20 min of compute)

**No key survived.**

`tamil-full-80` met every pre-set criterion (+5.47 SD gap over fake lexicons, fixed-key z 6.4, held-out > scrambled), and the run halted with STOP.md as instructed. The break tests then showed the criterion does not identify a language:

| lexicon | gap on Indus corpus (SD) | gap on a planted corpus of a *different* language (SD) |
|---|---|---|
| Tamil | +5.47 (Mahadevan transcription: +5.49) | +4.82 on planted Sanskrit, +5.23 on planted English |
| Sanskrit | +3.36 | +3.55 on planted Tamil |
| English | +1.66 | +1.88 on planted Tamil |

Each lexicon scores on the Indus corpus about what it scores on a corpus written in another language. So the pass measures "language-like, formulaic sequences" (plus Tamil's repetitive phonotactics: the best key maps 18/80 signs to `tan` and reads `tantan`, `natan`, `tantam` on the opener and jar formulae), not Tamil. Supporting results:
- On a structure-free corpus (signs drawn i.i.d. from real frequencies), Tamil still beats its letter-shuffled fakes by 7.1 points, three-quarters of its 9.6-point margin on the real corpus.
- A null that keeps phonotactics (Markov fake lexicons) makes everything fail, including planted Tamil (-1.0 SD), so it has no power either.

Calibrated verdict: a language is supported only if its gap on the Indus corpus clearly exceeds its gap on corpora of other languages. None does.

Best gap-over-fake reached per language and search type (Yajnadevam corpus, letter-shuffle null, top-80 syllabic keys):
- Tamil / full syllabic: +5.47 SD, **not above the cross-language baseline (+4.8 to +5.2)**
- Sanskrit / full syllabic: +3.36 SD, **not above the cross-language baseline (+3.55)**
- English (control) / full syllabic: +1.66 SD

Not run after the STOP: N = 150/250 syllabic, mixed, logographic and skeleton conditions on the real corpus. The criterion they would be judged by is now known not to discriminate languages. The skeleton ("inherent vowel") and logographic searches have no power anyway: their planted controls fail (+0.66 and -1.67 SD). The Sanskrit search detects planted Sanskrit only without noise (+3.41 SD at 0% noise; fails at 20% and 40%).

Published key: Yajnadevam's xlits.csv fails at the calibrated settings (Sanskrit full-string z 0.02, real below scrambled; Q14 value-shuffle criterion: rejected on all three corpora). At a 3-letter minimum it passes the fixed-key test (z 3.6), which says more about that test than about the key.
## Strategy 8: new candidate languages via CDLI lexicons (2026-09-25 21:43:43Z)

### 2026-09-25 21:45:14Z  `s8-sumerian-power-planted-sumerian-full-80`  (control)
method: full key, sumerian, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 90s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2024 | 0.0070 | 0.1715 | 0.1254 | 0.0133 | 5.7913 | 0.2161 | 2476 | 1867 | 20 | 30 |

fixed-key / verdict: z_fixed=6.2782, pass_fixed=True, fakefit_z_fixed=5.4942, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +5.79 SD -> PASS**

### 2026-09-25 21:47:14Z  `s8-sumerian-indus-full-80`  (control)
method: full key, sumerian, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 120s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2467 | 0.0096 | 0.1759 | 0.1521 | 0.0118 | 8.0237 | 0.2671 | 3006 | 2018 | 20 | 30 |

fixed-key / verdict: z_fixed=8.6012, pass_fixed=True, fakefit_z_fixed=5.2708, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +8.02 SD -> PASS**

### 2026-09-25 21:48:35Z  `s8-sumerian-cross-planted-tamil-full-80`  (control)
method: full key, sumerian, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 81s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1882 | 0.0083 | 0.1465 | 0.1215 | 0.0116 | 5.7473 | 0.2069 | 2471 | 1786 | 20 | 30 |

fixed-key / verdict: z_fixed=10.0471, pass_fixed=True, fakefit_z_fixed=5.3903, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +5.75 SD -> PASS**

### 2026-09-25 21:50:13Z  `s8-akkadian-power-planted-akkadian-full-80`  (control)
method: full key, akkadian, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 96s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1526 | 0.0269 | 0.1139 | 0.1006 | 0.0168 | 3.0911 | 0.2119 | 2350 | 1651 | 20 | 30 |

fixed-key / verdict: z_fixed=5.0759, pass_fixed=True, fakefit_z_fixed=5.1250, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +3.09 SD -> PASS**

### 2026-09-25 21:52:09Z  `s8-akkadian-indus-full-80`  (control)
method: full key, akkadian, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 116s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1578 | 0.0258 | 0.0937 | 0.1099 | 0.0148 | 3.2303 | 0.2158 | 2461 | 1691 | 20 | 30 |

fixed-key / verdict: z_fixed=6.7966, pass_fixed=True, fakefit_z_fixed=9.1277, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +3.23 SD -> PASS**

### 2026-09-25 21:53:21Z  `s8-akkadian-cross-planted-tamil-full-80`  (control)
method: full key, akkadian, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 72s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1172 | 0.0128 | 0.0728 | 0.0846 | 0.0092 | 3.5341 | 0.1505 | 1757 | 1193 | 20 | 30 |

fixed-key / verdict: z_fixed=8.3466, pass_fixed=True, fakefit_z_fixed=4.9163, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +3.53 SD -> PASS**

## Strategy 8b: Sumerian cross-language baselines (2026-09-25 21:56:33Z)

### 2026-09-25 21:57:58Z  `s8b-sumerian-cross-planted-sanskrit-full-80`  (control)
method: full key, sumerian, top-80 signs, 20 real restarts + 30 fake-lexicon searches, 32000 SA iterations each, 80s

| real_held_mean | real_held_sd | real_scr_mean | fake_held_mean | fake_held_sd | gap_sd | best_real_held | best_real_train | best_fake_train | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1846 | 0.0075 | 0.1472 | 0.1259 | 0.0139 | 4.2167 | 0.1987 | 2272 | 1791 | 20 | 30 |

fixed-key / verdict: z_fixed=7.3541, pass_fixed=True, fakefit_z_fixed=5.4942, fakefit_pass_fixed=True, PASS=True

**gap-over-fake = +4.22 SD -> PASS**

