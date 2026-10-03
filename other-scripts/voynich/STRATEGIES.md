# Voynich manuscript: strategies log

One row per strategy: method and control, result, verdict. Earlier work is summarised in FINDINGS.md.

| ID | Method and control | Result | Verdict |
|---|---|---|---|
| v2:V-1.1 | Pharma jar (Lc, 40) vs fragment (Lf, 194) labels, 15 pages: same-role minus cross-role similarity; null permutes role within page. Neg. control: locus-number parity. Power check: 30% of jar labels given a shared 2-glyph suffix | edit z = -0.82, prefix z = -0.07, suffix z = +1.21 (all n.s.). Parity control z = 0.6/-0.6/0.9. Injection detected at z = 2.5 (edit) and 6.0 (suffix) | Jar labels are not more like other jar labels than like fragment labels. The test would see a 30%-strength shared marker. No class marker in the label string |
| v2:V-1.2 | Bio nymph (Ln) vs tube (Lt) labels, 7 pages, same design | edit z = 0.37, prefix -0.14, suffix 0.66 | No class signal by similarity. (Descriptive: tube labels start with o 81% vs 51% for nymphs, see V-2.x) |
| v2:V-1.3 | Zodiac inner vs outer ring, 12 pages (259 labels), same design; parity neg. control | edit z = 1.28 (p = 0.10), prefix -0.28, suffix 0.24; control -0.1/-0.7/-1.3 | No ring signal |
| v2:V-1.4 | Zodiac clock adjacency: within 26 page-rings (290 labels), mean edit similarity of circularly adjacent labels minus all pairs; null = shuffle labels within ring (500) | +0.006, z = 0.98, p = 0.16 | Neighbouring figures do not get related labels: no serial/sequential progression detectable |
| v2:V-1.5 | Parallel zodiac pages: same ring, clock within 45 min vs > 2 h, across pages; null = random rotation of each ring (100) | edit z = -0.87, prefix 0.31, suffix -1.19 | The figure at the same position on different zodiac pages does not get a related label (rules out a fixed positional list) |
| v2:V-1.6 | Pharma jar-group cohesion: jar + its fragments vs other groups on the same page; null permutes group within page (1000) | +0.017, z = 1.45, p = 0.086 | Weak, not significant. At most a slight tendency for labels in one row to resemble each other (could be writing-time drift) |
| v2:V-1.7 | Label words (722 tokens that occur in text, of 1,163) in running text of same page / same section vs words of matched text frequency (+-25%), placed on the same page (1000 draws) | Same page: 358 vs 250 expected (x1.43, z = 4.7). Same section x1.32, z = 4.4. Words with text freq <= 100: x2.45 page, z = 7.1. By class: zodiac x2.6 (z 7.5), cosmo x2.5 (z 6.8), nymph x1.7 (z 2.2), frag x1.2 (n.s.), star x1.6 (p .07), tube x1.1, other x0.5 | Labels do reappear in their own page's text, strongly for zodiac and cosmological pages. BUT not yet controlled for ordinary local burstiness of Voynich vocabulary: cycle 2 must test whether any text word recurs on its own page this much |

## Summary

See FINDINGS.md for the state of knowledge before 4 Oct 2026.
