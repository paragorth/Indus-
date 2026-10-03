"""Analyse the bootstrap output of dark7_fingerprint.py: fingerprint statistics, Indus residue, controls, Voynich."""
import json, os, sys, math
import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data/derived/dark')
R = dict(np.load(os.path.join(D, 'loop7_boot.npz')))
meta = json.load(open(os.path.join(D, 'loop7_meta.json')))
names = meta['stats']
m = len(names)
ZCRIT = norm.isf(0.025 / m)          # Bonferroni over statistics, two-sided
ZAGREE = 2.0

mu = {k: np.nanmean(v, 0) for k, v in R.items()}
sd = {k: np.nanstd(v, 0) + 1e-12 for k, v in R.items()}


def z(a, b):
    return (mu[a] - mu[b]) / np.sqrt(sd[a] ** 2 + sd[b] ** 2)


def fmt(k, j):
    return f'{mu[k][j]:.3g}±{sd[k][j]:.2g}'


def report(fh, title, idx, cols):
    fh.write(f'\n## {title}  (n={len(idx)})\n')
    for j in idx:
        fh.write(f'  [{j}] {names[j]}\n      ' + '  '.join(f'{c}={fmt(c, j)}' for c in cols) + '\n')


def main(ind='IND_raw', tag='cycle1'):
    acc = ['PE', 'LA']
    lang = ['LB_words', 'LB_syll', 'UR3_words', 'UR3_syll']
    out = []
    # (a) accounting fingerprint: Indus, PE, LA agree (all pairwise |z|<2); each differs from every language corpus (|z|>ZCRIT) same sign
    fp, fp_loose, resid, resid_loose = [], [], [], []
    for j in range(m):
        if any(np.isnan(mu[k][j]) for k in [ind] + acc + lang):
            continue
        agree = max(abs(z(a, b)[j]) for a in [ind] + acc for b in [ind] + acc if a < b) < ZAGREE
        zs = {(a, b): z(a, b)[j] for a in [ind] + acc for b in lang}
        strict = all(abs(v) > ZCRIT for v in zs.values()) and len({np.sign(v) for v in zs.values()}) == 1
        loose = (all(abs(z(ind, b)[j]) > ZCRIT for b in lang) and len({np.sign(z(ind, b)[j]) for b in lang}) == 1
                 and all(any(abs(z(a, b)[j]) > ZCRIT for b in lang) for a in acc)
                 and len({np.sign(z(a, b)[j]) for a in [ind] + acc for b in lang}) == 1)
        if agree and strict:
            fp.append(j)
        elif agree and loose:
            fp_loose.append(j)
        zr = {b: z(ind, b)[j] for b in acc + lang}
        if all(abs(v) > ZCRIT for v in zr.values()) and len({np.sign(v) for v in zr.values()}) == 1:
            resid.append(j)
        elif all(abs(v) > ZCRIT for v in zr.values()):
            resid_loose.append(j)
    # calibration: Indus split halves, PE vs itself not available -> use IND halves; also MDH vs other sites
    cal = int(np.sum(np.abs(z('IND_halfA', 'IND_halfB')) > ZCRIT))
    cal_sites = int(np.sum(np.abs(z('IND_MDH', 'IND_otherSites')) > ZCRIT))
    merge = int(np.sum(np.abs(z('IND_raw', 'IND_all')) > ZCRIT))
    # order dependence: does the statistic change when texts are shuffled within?
    def order_dep(j, k):
        return abs(z(k, k + '_SHUF')[j]) > ZCRIT
    # pairwise distance matrix (mean |z| over all stats), robust: median |z|
    keys = [ind, 'PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll', 'VOY_words', 'VOY_chars']
    fh = open(os.path.join(D, f'loop7_{tag}.txt'), 'w')
    fh.write(f'S-DARK-7 {tag}: Indus corpus = {ind}; {m} random statistics; B={meta["B"]} bootstraps of N={meta["N"]} '
             f'length-matched texts (len 2-8, Indus weights); z = diff/sqrt(sd1^2+sd2^2); Bonferroni z_crit={ZCRIT:.2f}\n')
    fh.write('corpora: ' + '; '.join(f'{k}: {v["n_used"]} texts, {v["V"]} types' for k, v in meta['infos'].items()) + '\n')
    fh.write(f'\nCalibration: Indus split-half A vs B exceed z_crit on {cal}/{m} statistics; MD+H vs other sites {cal_sites}/{m}; '
             f'seq_raw vs seq_all {merge}/{m}\n')
    fh.write('\nMedian |z| between corpora (all statistics):\n      ' + ' '.join(f'{k[:9]:>9}' for k in keys) + '\n')
    for a in keys:
        fh.write(f'{a[:9]:>9} ' + ' '.join(f'{np.nanmedian(np.abs(z(a, b))):9.2f}' for b in keys) + '\n')
    fh.write('\nFraction of statistics separated (|z|>z_crit):\n      ' + ' '.join(f'{k[:9]:>9}' for k in keys) + '\n')
    for a in keys:
        fh.write(f'{a[:9]:>9} ' + ' '.join(f'{np.nanmean(np.abs(z(a, b)) > ZCRIT):9.2f}' for b in keys) + '\n')
    cols = [ind, 'PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll', 'VOY_words', 'VOY_chars']
    report(fh, 'Accounting-notation fingerprint (strict: Indus~PE~LA agree; each differs from LB and Ur III at both granularities, same direction)', fp, cols)
    report(fh, 'Accounting-notation fingerprint (loose: Indus differs from all four language corpora; PE and LA from at least one each, same direction)', fp_loose[:40], cols)
    report(fh, 'Indus-specific residue (Indus differs from PE, LA, LB x2, Ur III x2, all same direction)', resid, cols)
    report(fh, 'Indus-specific residue (mixed directions)', resid_loose[:40], cols)
    # order dependence for the fingerprint / residue
    fh.write('\n## Order dependence (statistic changes when tokens are shuffled within texts; |z|>z_crit) for fingerprint and residue statistics\n')
    for j in fp + fp_loose[:40] + resid:
        fh.write(f'  [{j}] ' + ' '.join(f'{k}:{"ORDER" if order_dep(j, k) else "inv"}' for k in [ind, 'PE', 'LA', 'LB_words', 'UR3_words', 'VOY_words']) + '\n')
    # Voynich placement
    fh.write('\n## Voynich placement on fingerprint statistics (z vs group mean of Indus/PE/LA, and vs LB_words, UR3_words)\n')
    for j in fp + fp_loose[:40]:
        g = np.mean([mu[k][j] for k in [ind] + acc]); gs = math.sqrt(np.mean([sd[k][j] ** 2 for k in [ind] + acc]))
        for v in ['VOY_words', 'VOY_chars']:
            zz = (mu[v][j] - g) / math.sqrt(sd[v][j] ** 2 + gs ** 2)
            fh.write(f'  [{j}] {v}: z vs accounting group {zz:+.1f}; z vs LB_words {z(v, "LB_words")[j]:+.1f}; z vs UR3_words {z(v, "UR3_words")[j]:+.1f}\n')
    # Voynich: nearest corpus over all stats
    fh.write('\n## Nearest corpus to Voynich (median |z| over all statistics)\n')
    for v in ['VOY_words', 'VOY_chars']:
        fh.write(f'  {v}: ' + ', '.join(f'{k}={np.nanmedian(np.abs(z(v, k))):.2f}' for k in [ind, 'PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll']) + '\n')
    fh.write('\n## Nearest corpus to Indus (median |z| over all statistics; and over order-dependent statistics only)\n')
    od = np.abs(z(ind, ind + '_SHUF')) > ZCRIT
    fh.write(f'  order-dependent statistics in Indus: {int(od.sum())}/{m}\n')
    for k in ['PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll', 'VOY_words', 'VOY_chars']:
        zz = np.abs(z(ind, k))
        fh.write(f'  {k}: all {np.nanmedian(zz):.2f}; order-dependent {np.nanmedian(zz[od]):.2f}; separated {np.nanmean(zz > ZCRIT):.2f}\n')
    fh.close()
    json.dump({'fp': fp, 'fp_loose': fp_loose, 'resid': resid, 'resid_loose': resid_loose, 'cal': cal, 'cal_sites': cal_sites, 'merge': merge},
              open(os.path.join(D, f'loop7_{tag}_sel.json'), 'w'))
    print(f'{tag}: fp strict {len(fp)}, loose {len(fp_loose)}, residue {len(resid)} (+{len(resid_loose)} mixed); calibration {cal}, sites {cal_sites}, merge {merge}; zcrit {ZCRIT:.2f}')


if __name__ == '__main__':
    main(*(sys.argv[1:3]))
