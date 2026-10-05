"""v55 cycle 5: PRE-REGISTERED out-of-sample test of the best cycle-1 survivor.
Fixed before looking at the ring texts (from c1_real: classifier has_d, feature Moon sign, year 1424, forward,
rotation 2; labels A 5.64 / held-out B 3.53; secondary 1482 rotation 4, has_al x has_d -> has_d).
Prediction: in the 964 words of the zodiac CIRCULAR texts, laid on the same days the same way, words containing d
follow the Moon's sign in 1424 the way the labels do (enriched with the Moon in Gemini, Scorpio, Sagittarius,
depleted in Cancer). Scores: G-test at the fixed (year, alignment); its rank among all 322 years at that alignment
and among all 322 x 60 (year, alignment) pairs; Spearman correlation of the 12-sign d-rate profile of the labels
with that of the ring words; nulls = within-sign shuffles and sign-block permutations of the ring words."""
import sys, os, json
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v55_lib as V

T = V.get_sky('real'); dd = V.degree_days(T)
ms = V.day_features(T)['moon_sign'][0]
labels = V.load_labels(); words = V.load_ring_words()


def profile(units, idx, y, a):
    c = np.array(['d' in x['label'] for x in units])
    m = ms[idx[y - V.Y0, a]]
    n = np.array([[((m == k) & c).sum(), ((m == k) & ~c).sum()] for k in range(12)], float)
    return n


def G(n):
    rs = n.sum(1, keepdims=True); cs = n.sum(0, keepdims=True); N = n.sum()
    e = rs * cs / N
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2 * np.nansum(np.where(n > 0, n * np.log(n / e), 0))


def allG(units, idx):
    c = np.array(['d' in x['label'] for x in units]).astype(int)
    M = ms[idx]  # (Y, 60, L)
    out = np.zeros(M.shape[:2])
    for k in range(12):
        pass
    code = M * 2 + c[None, None, :]
    cnt = np.zeros(M.shape[:2] + (24,))
    for v in range(24):
        cnt[..., v] = (code == v).sum(2)
    n = cnt.reshape(M.shape[0], 60, 12, 2)
    rs = n.sum(3, keepdims=True); cs = n.sum(2, keepdims=True); N = n.sum((2, 3), keepdims=True)
    e = rs * cs / N
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2 * np.where(n > 0, n * np.log(n / e), 0).sum((2, 3))


res = {}
li = V.alignments(labels, dd)
for tag, (y, a) in {'primary_1424_r2': (1424, 2), 'secondary_1482_r4': (1482, 4)}.items():
    pl = profile(labels, li, y, a)
    rl = pl[:, 0] / np.maximum(pl.sum(1), 1)
    out = {}
    for cond in ['real'] + [f'shuf{k}' for k in range(200)] + [f'sperm{k}' for k in range(60)]:
        u = words
        if cond.startswith('sperm'):
            u = V.sign_permute(words, int(cond[5:]))
        wi = V.alignments(u, dd)
        if cond.startswith('shuf'):
            u = V.make_condition(cond, u, wi, T)
        pw = profile(u, wi, y, a)
        rw = pw[:, 0] / np.maximum(pw.sum(1), 1)
        g = G(pw)
        rec = dict(G=float(g), rho=float(spearmanr(rl, rw)[0]))
        if cond == 'real':
            Gall = allG(u, wi)
            rec['rank_year_at_align'] = int((Gall[:, a] >= g - 1e-9).sum())
            rec['rank_all_pairs'] = int((Gall >= g - 1e-9).sum())
            rec['n_pairs'] = int(Gall.size)
            rec['labels_rate'] = [round(float(v), 2) for v in rl]
            rec['ring_rate'] = [round(float(v), 2) for v in rw]
            rec['ring_counts'] = pw[:, 0].astype(int).tolist(), pw.sum(1).astype(int).tolist()
        out[cond] = rec
    real = out['real']
    nul = [v for k, v in out.items() if k != 'real']
    real['p_G_vs_nulls'] = float(np.mean([v['G'] >= real['G'] for v in nul]))
    real['p_rho_vs_nulls'] = float(np.mean([v['rho'] >= real['rho'] for v in nul]))
    res[tag] = real
    print(tag, json.dumps(real))
json.dump(res, open(os.path.join(V.CK, 'c5.json'), 'w'))
