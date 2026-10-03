"""S-DARK-7 cycle 4: whole-space geometry. Hierarchical clustering of corpora on median |z| distance over the 600
statistics (all, order-dependent only, non-numeral only); nearest neighbours; Voynich placement on the Indus residue
axes; which real corpus is nearest to Indus on the 'structure residual' (real minus own unigram null)."""
import json, os, sys
import numpy as np
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(ROOT, 'data/derived/dark')
R = dict(np.load(os.path.join(D, 'loop7_boot.npz')))
for f in ('loop7_boot_uni.npz', 'loop7_boot_c3.npz'):
    p = os.path.join(D, f)
    if os.path.exists(p):
        R.update(dict(np.load(p)))
meta = json.load(open(os.path.join(D, 'loop7_meta.json'))); names = meta['stats']; m = len(names)
ZC = norm.isf(0.025 / m)
mu = {k: np.nanmean(v, 0) for k, v in R.items()}
sd = {k: np.maximum(np.nanstd(v, 0), 1e-3 * np.maximum(np.abs(np.nanmean(v, 0)), 1e-6) + 1e-4) for k, v in R.items()}
z = lambda a, b: (mu[a] - mu[b]) / np.sqrt(sd[a] ** 2 + sd[b] ** 2)
NUMP = ('is_num', 'next_is_num', 'prev_is_num', 'dist_to_nearest_num')
isnum = np.array([any(t in n for t in NUMP) for n in names])
keys = [k for k in ['IND_raw', 'IND_seals', 'IND_nonseal', 'PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll',
                    'UR3_names_syll', 'LE_syll', 'VOY_words', 'VOY_chars'] if k in R]
od = np.abs(z('IND_raw', 'IND_raw_SHUF')) > ZC
fh = open(os.path.join(D, 'loop7_cycle4_tables.txt'), 'w')


def dend(mask, title):
    n = len(keys)
    M = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i != j:
                M[i, j] = np.nanmedian(np.abs(z(a, b))[mask])
    M = (M + M.T) / 2
    Lk = linkage(squareform(M, checks=False), 'average')
    dn = dendrogram(Lk, labels=keys, no_plot=True)
    fh.write(f'\n## {title}: average-linkage clustering on median |z| ({int(mask.sum())} statistics)\n')
    fh.write('  leaf order: ' + ' | '.join(dn['ivl']) + '\n')
    # print merges
    clusters = {i: [k] for i, k in enumerate(keys)}
    for step, (a, b, h, _) in enumerate(Lk):
        a, b = int(a), int(b)
        clusters[n + step] = clusters[a] + clusters[b]
        fh.write(f'  merge at {h:.2f}: {{{", ".join(clusters[a])}}} + {{{", ".join(clusters[b])}}}\n')
    fh.write('  nearest neighbour of each corpus: ' + '; '.join(
        f'{a}->{keys[int(np.argmin(np.where(np.arange(n) == i, 9e9, M[i])))]} ({np.min(np.where(np.arange(n) == i, 9e9, M[i])):.2f})'
        for i, a in enumerate(keys)) + '\n')


dend(np.ones(m, bool), 'All statistics')
dend(~isnum, 'Non-numeral statistics')
dend(od, 'Order-dependent statistics (Indus value changes under within-text shuffle)')
dend(od & ~isnum, 'Order-dependent, non-numeral statistics')

# Voynich on the residue axes
sel = json.load(open(os.path.join(D, 'loop7_cycle1_sel.json')))
fh.write('\n## Voynich on the Indus-residue axes (z of Voynich vs Indus, and vs the mean of the six comparison corpora)\n')
others = ['PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll']
for j in sel['resid']:
    if isnum[j]:
        continue
    g = np.mean([mu[k][j] for k in others]); gs = np.sqrt(np.mean([sd[k][j] ** 2 for k in others]))
    fh.write(f'  [{j}] {names[j]}\n      ' + '  '.join(
        f'{v}: vs Indus {z(v, "IND_raw")[j]:+.1f}, vs others {(mu[v][j] - g) / np.sqrt(sd[v][j] ** 2 + gs ** 2):+.1f}' for v in ['VOY_words', 'VOY_chars']) + '\n')

# PCA on standardized profiles (pooled bootstrap sd)
X = np.array([mu[k] for k in keys])
ok = ~np.isnan(X).any(0)
X = X[:, ok]
S = np.sqrt(np.mean(np.array([sd[k] for k in keys])[:, ok] ** 2, 0))
Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
U, s, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
pc = U * s
fh.write(f'\n## PCA of z-standardized statistic profiles (variance explained PC1 {s[0]**2/np.sum(s**2):.2f}, PC2 {s[1]**2/np.sum(s**2):.2f}, PC3 {s[2]**2/np.sum(s**2):.2f})\n')
for i, k in enumerate(keys):
    fh.write(f'  {k:>15}: PC1 {pc[i,0]:+7.2f}  PC2 {pc[i,1]:+7.2f}  PC3 {pc[i,2]:+7.2f}\n')
idx = np.where(ok)[0]
for c in range(2):
    top = np.argsort(-np.abs(Vt[c]))[:6]
    fh.write(f'  PC{c+1} top loadings: ' + '; '.join(f'{names[idx[t]]} ({Vt[c][t]:+.2f})' for t in top) + '\n')
fh.close()
print(open(os.path.join(D, 'loop7_cycle4_tables.txt')).read())
