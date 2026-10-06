#!/usr/bin/env python3
"""v83 cycle 2 report: per test, the headline statistic in each version (legacy, all, clean, agree) with the
random-thinning band (thin-clean-1..3, thin-agree-1..3), both transcriptions, and whether the B criterion holds.
A version 'departs' when its value lies outside min-max of its thinning runs widened by 25% of the band's centre
distance from 'all' (reported, not used for the verdict). The verdict uses the pre-stated B criterion only."""
import os, sys, json, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v83_parse as P

D = collections.defaultdict(dict)
for f in glob.glob(os.path.join(P.CK, 'c2', '*.json')):
    t, v, n = os.path.basename(f)[:-5].split('__')
    D[(t, n)][v] = json.load(open(f))

# (test, statistic, criterion text, function value -> bool)
CRIT = [
    ('v78', 'spike', 'spike 0.85-1.15 (no neighbour rule; languages 0.04-0.23, doubling mechanisms >= 1.5)', lambda x: 0.85 <= x <= 1.15),
    ('v78', 'line', 'line re-use > 1.1', lambda x: x > 1.1),
    ('v78', 'rate1', 'doubling rate > 0.012 (generators; real systems <= 0.004)', lambda x: x > 0.012),
    ('v61', 'cross_z', 'no coupling crosses the line break: z < 2', lambda x: x < 2),
    ('v61', 'within_excess', 'within-line junction excess > 0.1 bits', lambda x: x > 0.1),
    ('v72', 'excess', 'held-out sequence excess > 0.024 (junction generator)', lambda x: x > 0.024),
    ('v72', 'gain_page', 'page gain < 0.055 (every plant)', lambda x: x < 0.055),
    ('v82', 'gain_d0', 'onset link gain > 0 ...', lambda x: x > 0),
    ('v82', 'z_d0', '... with z > 3 (d0)', lambda x: x > 3),
    ('v82', 'z_d1', 'z > 3 (d1)', lambda x: x > 3),
    ('v82', 'diag', 'onset repeats dominate: diagonal share > 0.3', lambda x: x > 0.3),
    ('v82', 'cross_z', 'link stops at the line break: cross z < 2', lambda x: x < 2),
    ('v54', 'ratio_best_null', 'near-repeat passages > 1.5x best null', lambda x: x > 1.5),
    ('v54', 'var_per_exact', 'variants per exact return > 0.4 (real copying <= 0.15, planted 0.53)', lambda x: x > 0.4),
    ('v54', 'variant_z', 'one-edit variant return z > 2.5', lambda x: x > 2.5),
    ('v56', 'z_hit_word', 'page tie z_hit < 20 (herbals 52-57)', lambda x: x < 20),
    ('v59', 'auc', 'B herbal ~ A herbal AUC > 0.65', lambda x: x > 0.65),
    ('v59', 'z', 'label-permutation z > 2', lambda x: x > 2),
    ('v59', 'auc_first3', 'first 3 glyphs AUC > last 3 glyphs (checked jointly below)', lambda x: True),
    ('v79', 'rep_ratio', 'line-start mark avoids the mark above: ratio < 0.8', lambda x: x < 0.8),
    ('v79', 'mi_z', 'first-order successor table z > 5', lambda x: x > 5),
    ('v68', 'mi_lf', 'line-initial pool MI > 0.08 (coded plaintexts <= 0.044)', lambda x: x > 0.08),
    ('v68', 'nb_ratio_B_over_A', 'Currier B sections couple neighbours more than A (ratio > 1.3; proxy)', lambda x: x > 1.3),
    ('v71', 's4_mean', 'page-order trace well below self-citation (checked against its value)', lambda x: True),
]
VER = ['legacy', 'all', 'glyph', 'clean', 'agree']
out = []
for test, stat, crit, ok in CRIT:
    for n in ('ZL3b', 'IT2a'):
        d = D.get((test, n), {})
        if not d: continue
        row = {v: d[v].get(stat) for v in VER if v in d}
        tc = [d[v].get(stat) for v in d if v.startswith('thin-clean')]
        ta = [d[v].get(stat) for v in d if v.startswith('thin-agree')]
        tg = [d[v].get(stat) for v in d if v.startswith('thin-glyph')]
        if test == 'v59' and stat == 'auc_first3':
            holds = {v: d[v]['auc_first3'] > d[v]['auc_last3'] for v in row}
        elif test == 'v71':
            holds = {v: d[v]['s4_mean'] < 0.5 * d[v]['selfcit_s4_mean'] for v in row}
        else:
            holds = {v: (x is not None and ok(x)) for v, x in row.items()}
        fmt = lambda x: 'NA' if x is None else ('%.3f' % x)
        band = lambda t: ('%s..%s' % (fmt(min(t)), fmt(max(t)))) if t else '-'
        line = '%-4s %-17s %-2s | %s | thin-g %s | thin-c %s | thin-a %s | holds %s' % (
            test, stat, n[:2], ' '.join('%s=%s' % (v[:3], fmt(row.get(v))) for v in VER), band(tg), band(tc), band(ta),
            ''.join('Y' if holds.get(v) else ('n' if v in holds else '.') for v in VER))
        print(line)
        out.append(dict(test=test, stat=stat, name=n, crit=crit, values=row, thin_glyph=tg, thin_clean=tc, thin_agree=ta, holds=holds))
json.dump(out, open(os.path.join(P.CK, 'c2_report.json'), 'w'), indent=1)
