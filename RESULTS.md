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

