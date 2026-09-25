# STOP: `tamil-full-80` passed, then was broken

**Retracted as evidence (2026-09-25 01:18Z).** The same pipeline gives the Tamil lexicon +4.82 SD on a corpus of planted *Sanskrit* and +5.23 SD on planted *English*, the same as its +5.47 SD on the Indus corpus. The pass measures language-like formulaic structure, not Tamil. See RESULTS.md, 'Final entry'. Nothing below is a reading.

## Original record

2026-09-25 00:25:58Z

```
{
 "real_held_mean": 0.2663195408682773,
 "real_held_sd": 0.00625004517256824,
 "real_scr_mean": 0.1941343453978833,
 "fake_held_mean": 0.17054830448332253,
 "fake_held_sd": 0.017507554543751677,
 "gap_sd": 5.470280623465762,
 "best_real_held": 0.2781634731093215,
 "best_real_train": 3161,
 "best_fake_train": 2532,
 "n_real": 20,
 "n_fake": 30
}
{
 "z_fixed": 6.40520488799183,
 "pass_fixed": true,
 "fakefit_z_fixed": 4.955716994321637,
 "fakefit_pass_fixed": true,
 "PASS": true
}
```

Key and all runs: results/tamil-full-80.json

Reproduce:

```
./scripts/fetch_data.sh
python run_all.py --only tamil-full-80
```
