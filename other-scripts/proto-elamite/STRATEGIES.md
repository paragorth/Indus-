# Proto-Elamite: strategies log

One row per strategy: method and control, result, verdict. Earlier work is summarised in FINDINGS.md.

| ID | Method and control | Result | Verdict |
|---|---|---|---|
| pe2:PE-1.1 | Assignment score (mean cosine of 60 pairs), all features; C1 = each feature column shuffled within each script (200x) | real 0.702 vs C1 mean 0.657, max 0.682 (p < 0.005) | beats C1, but C1 only shows both scripts share a feature covariance structure (e.g. sole-sign <-> header); it does not show correct pairings |
| pe2:PE-2.1 | Same score; C2 = PE numeral groups shuffled between entries of the same tablet (30x), profiles rebuilt | real 0.702 vs C2 mean 0.704, max 0.712 (p = 0.70) | NULL: destroying the PE sign-system link does not lower the score. The pairing is not driven by commodity/number-system behaviour |
| pe2:PE-3.1 | Same without position features (PC transliteration order is editorial) | real 0.736; C1 mean 0.723 (p = 0.055); C2 mean 0.739 (p = 0.63) | NULL on both controls |
| pe2:PE-4.1 | Planted, within proto-cuneiform: PC tablets split at random, one half relabelled and sub-sampled to 1,585 tablets (PE size); top-60 relabelled signs assigned into top-200 of the other half (5 seeds) | 14, 9, 16, 19, 18 of 60 recovered (mean 15.2 = 25%; chance ~0.3) | method CAN recover identity when both halves come from one script and one archive mix |
| pe2:PE-5.1 | Planted, cross-site: Uruk tablets as reference, non-Uruk PC tablets (Jemdet Nasr, Umma, uncertain) relabelled | 4 of 60 recovered (7%) | recovery collapses when archives differ even inside one script; a cross-script bridge between Uruk and Susa is expected to recover far less |
| pe2:PE-6.1 | Per-pair p (assigned similarity vs PE sign's best similarity against column-shuffled PC, 200x) | 6 of 60 pairs p < 0.01 (M388-NAM2, M297-SZIM, M066-PIRIG, M206-ZATU753, M048-AN, M002-TAR); ~0.6 expected by chance under this lenient null | lenient null only; tested against held-out data and stability in cycle 2 |
| pe2:PE-1.2 | Stability: pair from half A and from half B separately (all features); share of PE signs with same PC partner. Reference: two halves of PC paired into PC | PE 12.7% of signs keep their partner; PC self-halves 72.5% | PE partners are mostly unstable; only M388 (10/10), M376 (9/10), M297 (8/10), M124 (7/10) keep a partner |
| pe2:PE-2.2 | Held-out commodity test: pair on A WITHOUT own-system features (position, tablet context, header, reverse, magnitude); rank of partner among 200 PC signs by similarity of PC system profile to the PE sign's system profile on held-out B (0 = best, random = 0.5). Control: same pairing after PE-A numeral groups shuffled within tablet | real mean rank 0.423, top-10% 19.3%; control 0.404, top-10% 21.7% | NULL: the pairing predicts held-out number-system behaviour no better than the control. The prediction that exists (rank < 0.5) comes from tablet context, which the shuffle keeps |
| pe2:PE-3.2 | Known-function partners (PC sign names: grain, beer, small/large cattle, person, fish, textile...): is the PE sign's held-out capacity share above the PE baseline exactly when the partner's is above the PC baseline? | 53/88 sign-split cases (60%; binomial p = 0.04, but cases repeat across splits and the null is not uniform) | weak, not independent; proper null in cycle 3 |
| pe2:PE-4.2 | Per-sign leaders: no-system partner mode and held-out top-10% count | M036 -> TAR 8/10, top-10% 10/10; M002 -> GUG2 6/10, 8/10; M243 -> NIN 5/10, 8/10; M010 -> |ZATU714xHI@g| 4/10, 8/10; M376 -> AL 9/10 but held-out 2/10; M388 -> NAM2 9/10 but held-out 2/10 | the signs that predict well are the PE capacity class (M036, M002, M243, M010, M297) -- class-level role, partners vary; needs a per-sign null (cycle 3) |

## Summary

See FINDINGS.md for the state of knowledge before 4 Oct 2026.
