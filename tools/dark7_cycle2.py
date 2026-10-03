"""S-DARK-7 cycle 2: held-out-site replication of the residue; unigram-null control (if loop7_boot_uni.npz exists)."""
import json, os, sys
import numpy as np
from scipy.stats import norm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(ROOT, 'data/derived/dark')
R = dict(np.load(os.path.join(D, 'loop7_boot.npz')))
uni = os.path.join(D, 'loop7_boot_uni.npz')
if os.path.exists(uni):
    R.update(dict(np.load(uni)))
meta = json.load(open(os.path.join(D, 'loop7_meta.json'))); names = meta['stats']; m = len(names)
ZC = norm.isf(0.025 / m)
mu = {k: np.nanmean(v, 0) for k, v in R.items()}
sd = {k: np.maximum(np.nanstd(v, 0), 1e-3 * np.maximum(np.abs(np.nanmean(v, 0)), 1e-6) + 1e-4) for k, v in R.items()}
z = lambda a, b: (mu[a] - mu[b]) / np.sqrt(sd[a] ** 2 + sd[b] ** 2)
NUMP = ('is_num', 'next_is_num', 'prev_is_num', 'dist_to_nearest_num')
isnum = np.array([any(t in n for t in NUMP) for n in names])
others = ['PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll']
sel = json.load(open(os.path.join(D, 'loop7_cycle1_sel.json')))
fh = open(os.path.join(D, 'loop7_cycle2_tables.txt'), 'w')
def resid(ind):
    out = []
    for j in range(m):
        zs = [z(ind, b)[j] for b in others]
        if all(abs(v) > ZC for v in zs) and len({np.sign(v) for v in zs}) == 1:
            out.append(j)
    return out
rA, rB, rM, rO = resid('IND_halfA'), resid('IND_halfB'), resid('IND_MDH'), resid('IND_otherSites')
r1 = set(sel['resid'])
rep_half = sorted(r1 & set(rA) & set(rB)); rep_site = sorted(r1 & set(rM) & set(rO))
fh.write(f'Residue (cycle 1): {len(r1)}. Replicates in both random halves: {len(rep_half)}; in both MD+H and other sites (held-out): {len(rep_site)}\n')
fh.write(f'Residue found independently in other-sites-only Indus (800 texts): {len(rO)}; in MD+H only: {len(rM)}\n')
for j in sorted(r1):
    fh.write(f'  [{j}] {"NUM" if isnum[j] else "non"} {names[j]}\n      halves {"ok" if j in rA and j in rB else "FAIL"}; sites {"ok" if j in rM and j in rO else "FAIL"}; '
             f'z(MDH,other)={z("IND_MDH","IND_otherSites")[j]:+.1f}; min|z|(Indus,X)={min(abs(z("IND_raw",b)[j]) for b in others):.1f}; '
             f'otherSites={mu["IND_otherSites"][j]:.3g} MDH={mu["IND_MDH"][j]:.3g}\n')
# residue found in held-out sites alone, not in cycle 1
new = sorted(set(rO) & set(rM) - r1)
fh.write(f'\nStatistics where BOTH MD+H and other sites separate from all six others (same direction) but full-corpus did not: {len(new)}\n')
for j in new[:20]:
    fh.write(f'  [{j}] {names[j]}: MDH={mu["IND_MDH"][j]:.3g} other={mu["IND_otherSites"][j]:.3g} ' + ' '.join(f'{b}={mu[b][j]:.3g}' for b in others) + '\n')
# unigram control
if 'IND_raw_UNI' in R:
    fh.write('\nUnigram-null control (tokens drawn iid from each corpus unigram distribution, same lengths).\n')
    fh.write('For each residue statistic: does the Indus value differ from its own unigram null (|z|>zc)? Does Indus_UNI still differ from all six other _UNI corpora (=> inventory/Zipf effect, not sequence/composition)?\n')
    for j in sorted(r1):
        zu = [z('IND_raw_UNI', b + '_UNI')[j] for b in others]
        inv = all(abs(v) > ZC for v in zu) and len({np.sign(v) for v in zu}) == 1
        fh.write(f'  [{j}] Indus vs own unigram null z={z("IND_raw","IND_raw_UNI")[j]:+.1f}; residue survives in unigram nulls: {"YES (inventory effect)" if inv else "no (needs real texts)"}; Indus_UNI={mu["IND_raw_UNI"][j]:.3g}\n')
    # how much of each corpus's statistics are explained by unigram: fraction of stats where real differs from own null
    fh.write('\nFraction of 600 statistics on which each real corpus differs from its own unigram null (sequence+composition structure):\n')
    for k in ['IND_raw', 'PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll', 'VOY_words']:
        fh.write(f'  {k}: {np.nanmean(np.abs(z(k, k + "_UNI")) > ZC):.2f} (median |z| {np.nanmedian(np.abs(z(k, k + "_UNI"))):.2f})\n')
    # distances between unigram nulls: does the Indus~LB_syll closeness persist with inventory alone?
    keys = ['IND_raw', 'PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll', 'VOY_words']
    fh.write('\nMedian |z| between UNIGRAM nulls (inventory-only distances):\n      ' + ' '.join(f'{k[:9]:>9}' for k in keys) + '\n')
    for a in keys:
        fh.write(f'{a[:9]:>9} ' + ' '.join(f'{np.nanmedian(np.abs(z(a + "_UNI", b + "_UNI"))):9.2f}' for b in keys) + '\n')
    # structure-only distances: compare (real - own unigram) profiles
    fh.write('\nMedian |z| between corpora on "structure residuals" (real minus own unigram null, per statistic; bootstrap sds combined):\n      ' + ' '.join(f'{k[:9]:>9}' for k in keys) + '\n')
    for a in keys:
        row = []
        for b in keys:
            d = (mu[a] - mu[a + '_UNI']) - (mu[b] - mu[b + '_UNI'])
            s = np.sqrt(sd[a] ** 2 + sd[a + '_UNI'] ** 2 + sd[b] ** 2 + sd[b + '_UNI'] ** 2)
            row.append(np.nanmedian(np.abs(d / s)))
        fh.write(f'{a[:9]:>9} ' + ' '.join(f'{v:9.2f}' for v in row) + '\n')
fh.close()
print(open(os.path.join(D, 'loop7_cycle2_tables.txt')).read())
