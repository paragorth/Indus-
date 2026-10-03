"""S-DARK-32 cycle 3 audit: which features separate the real Indus from its own Markov-2 / bigram / slot-shuffled surrogates
(z over the 8 subsample rows of the control), and where does each of those features put Indus relative to the class ranges?
Also: is Indus closer to any reference corpus than to its Markov-2 surrogate in the full standardised space?
Reads loop32_features.json. Output: loop32_c3_audit.txt
"""
import os, sys, json, collections, statistics as st
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from dark_loop32 import TYPE, TRAIN3, TRAIN_EXCLUDE, LEN_FEATS
OUTD = os.path.join(ROOT, 'data/derived/dark')
rows = json.load(open(os.path.join(OUTD, 'loop32_features.json')))
feats = sorted(k for k in rows[0]['feats'] if k not in LEN_FEATS and k not in ('n_texts', 'bits_bigram', 'bits_slot'))
corp = sorted({r['corpus'] for r in rows})
X = {c: np.array([[r['feats'][f] for f in feats] for r in rows if r['corpus'] == c]) for c in corp}
M = {c: X[c].mean(axis=0) for c in corp}
train = [c for c in corp if TYPE[c] in TRAIN3 and c not in TRAIN_EXCLUDE]
lab = {c: TRAIN3[TYPE[c]] for c in train}
out = []; P = out.append
P('S-DARK-32 cycle 3 audit: real Indus vs its surrogates, feature by feature (Indus = mean of seq_raw/strong/all; IM77 shown apart)')
ind = np.mean([M[c] for c in ('indus_seq_raw', 'indus_seq_strong', 'indus_seq_all')], axis=0)
for ctrl in ('indus_markov2', 'indus_bigram', 'indus_slotshuf'):
    P(f'\n== Indus vs {ctrl}: z = (Indus - control mean) / control sd over 8 rows; sorted by |z|; class ranges over corpus means ==')
    sd = X[ctrl].std(axis=0) + 1e-9
    zs = sorted(((abs((ind[i] - M[ctrl][i]) / sd[i]), i) for i in range(len(feats))), reverse=True)
    for z, i in zs[:14]:
        rng = {k: (min(M[d][i] for d in train if lab[d] == k), max(M[d][i] for d in train if lab[d] == k)) for k in ('L', 'D', 'A')}
        side = 'L-range' if rng['L'][0] <= ind[i] <= rng['L'][1] else 'outside L'
        P(f'  {feats[i]:24s} Indus {ind[i]:7.3f} IM77 {M["indus_im77"][i]:7.3f} | {ctrl[6:]:9s} {M[ctrl][i]:7.3f} z {(ind[i]-M[ctrl][i])/sd[i]:+7.1f} | '
          f'L [{rng["L"][0]:.2f}-{rng["L"][1]:.2f}] D [{rng["D"][0]:.2f}-{rng["D"][1]:.2f}] A [{rng["A"][0]:.2f}-{rng["A"][1]:.2f}] {side}')
# distances in the full standardised space (z over training corpus means)
mu = np.mean([M[c] for c in train], axis=0); sd = np.std([M[c] for c in train], axis=0) + 1e-9
Z = {c: (M[c] - mu) / sd for c in corp}
P('\n== Full-space distances (z over corpus means, all structural features): Indus to its surrogates vs to the nearest references ==')
for c in ('indus_seq_raw', 'indus_seq_all', 'indus_im77'):
    d = sorted(((np.linalg.norm(Z[c] - Z[o]), o) for o in corp if o != c and not o.startswith('indus_seq') or o in ('indus_markov2', 'indus_bigram', 'indus_slotshuf')))
    P(f'  {c:14s} ' + ', '.join(f'{o}[{TYPE[o]}] {v:.1f}' for v, o in d[:8]))
P('  (an Indus level to another Indus level: ' + f'{np.linalg.norm(Z["indus_seq_raw"]-Z["indus_seq_all"]):.1f}; raw to IM77: {np.linalg.norm(Z["indus_seq_raw"]-Z["indus_im77"]):.1f})')
# the features on which Indus is outside the whole language range AND the controls are inside it (or the reverse)
P('\n== Features where Indus lies outside the range of every language corpus (corpus means) ==')
for i, f in enumerate(feats):
    lo = min(M[d][i] for d in train if lab[d] == 'L'); hi = max(M[d][i] for d in train if lab[d] == 'L')
    if not lo <= ind[i] <= hi:
        where = {k: 'in' if min(M[d][i] for d in train if lab[d] == k) <= ind[i] <= max(M[d][i] for d in train if lab[d] == k) else 'out' for k in ('D', 'A')}
        P(f'  {f:24s} Indus {ind[i]:7.3f} (IM77 {M["indus_im77"][i]:.3f}) vs L [{lo:.3f}-{hi:.3f}]; D {where["D"]}, A {where["A"]}; markov2 {M["indus_markov2"][i]:.3f} slotshuf {M["indus_slotshuf"][i]:.3f}')
txt = '\n'.join(out); open(os.path.join(OUTD, 'loop32_c3_audit.txt'), 'w').write(txt + '\n'); print(txt)
