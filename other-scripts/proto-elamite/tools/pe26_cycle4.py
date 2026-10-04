"""pe26 cycle 4: colour-blind clay. CDLI photos carry no internal colour reference (black background clipped to L=0;
white labels are visible on <1% of NMI thumbnails; none on ST-11), so a lighting cast cannot be removed. Texture
(fine relief, mottling, local binary patterns) is nearly blind to white balance. Re-run the two cycle-2 tests on
texture only, and on colour only, and on lightness-free chromaticity:
 (a) Yahya vs the Susa tablets of the same processing run (AUC by leave-one-out diagonal discriminant; null = 2,000
     random sets of 4 Susa tablets from the same run);
 (b) ST-11 percentile among NMI Susa on the Yahya direction (campaign-centred), null = pseudo-plateau directions.
Planted: a texture-only shift (+1 sd in mottle, hp_std) on 4 random Susa tablets must be found by (a)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe26_common import *
rng = np.random.default_rng(2604)
rows = [r for r in load() if r['group'] == 'PE' and r.get('is_grey', 0) < 0.5 and r['museum'] == 'NMI' and r['site'] in ('Susa', 'Yahya')]
def bkey(r):
    parts = r['batch'].split('|'); return parts[2][:22] + '|' + parts[3][:7]
bat = [bkey(r) for r in rows]; ids = [r['id'] for r in rows]; ix = {p: i for i, p in enumerate(ids)}
SETS = {'texture': TEXTURE, 'colour': COLOUR, 'chromaticity': ['a_med', 'b_med', 'hue', 'r_chrom', 'g_chrom', 'a_over_L', 'b_over_L', 'chroma']}
run = rows[ix['P009532']]['batch']
out = {}
for name, cols in SETS.items():
    X = within_batch_centre(X_of(rows, cols), bat); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Y = [ix[p] for p in ('P009532', 'P009535', 'P009538')]  # YT-01 held out
    S = [i for i, r in enumerate(rows) if r['site'] == 'Susa' and r['id'] != 'P009157']
    same = [i for i in S if rows[i]['batch'] == run]
    def w_of(Yi, Si):
        d = X[Yi].mean(0) - X[Si].mean(0); return d / (X[Si].std(0) + 1e-6) ** 2
    def loo_auc(pos, neg):
        sc_pos = [X[i] @ w_of([j for j in pos if j != i], neg) for i in pos]
        w = w_of(pos, neg); sc_neg = X[neg] @ w
        return np.mean([[sp > sn for sn in sc_neg] for sp in sc_pos])
    allY = Y + [ix['P009536']]
    real = loo_auc(allY, same)
    nul = [loo_auc(list(rng.choice(same, 4, replace=False)), None) if False else None for _ in range(0)]
    nul = []
    for k in range(500):
        fake = list(rng.choice(same, 4, replace=False)); rest = [i for i in same if i not in fake]
        nul.append(loo_auc(fake, rest))
    nul = np.array(nul); p_a = (np.sum(nul >= real) + 1) / (len(nul) + 1)
    st = ix['P009157']; NS = S + [st]
    def pct(w): s = X[NS] @ w; return np.mean(s <= X[st] @ w)
    real_b = pct(w_of(Y, S))
    ps = np.array([pct(w_of(list(rng.choice(same, 3, replace=False)), S)) for _ in range(1000)])
    p_b = (np.sum(ps >= real_b) + 1) / 1001
    res = dict(yahya_auc=float(real), yahya_null_q95=float(np.percentile(nul, 95)), p_yahya=float(p_a), st11_pct=float(real_b), p_st11=float(p_b))
    if name == 'texture':
        pl = []
        for k in range(100):
            fake = list(rng.choice(same, 4, replace=False)); Xs = X.copy()
            for c in ('mottle', 'hp_std'): X[fake, cols.index(c)] += 1.0
            rest = [i for i in same if i not in fake]; pl.append(loo_auc(fake, rest)); X[:] = Xs
        res['planted_auc'] = float(np.mean(pl)); res['planted_power'] = float(np.mean(np.array(pl) >= np.percentile(nul, 95)))
    out[name] = res
    print(name, json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
json.dump(out, open(os.path.join(CK, 'cycle4.json'), 'w'), indent=1)
