"""pe18 cycle 5: is the format gain of cycle 3 an artefact of how a system becomes visible?
DEC and SEX can only be told apart on numbers >= 60 (mostly totals), so 'has a total / reverse /
columns' may predict the LABEL without predicting the CHOICE.  Test the choice alone:
5a DEC vs SEX (both visible), binary logistic, 20 x 5-fold CV, goods vs goods + family; null = labels
   permuted within size bin x dominant class sign (100x).  Same for C vs C@ (capacity variant).
5b planted accent: a real variant form switched on in 30% of DEC tablets must be found by goods+VAR.
"""
import json, sys, os, collections, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('NSETS', '10')
import pe18_cycle3 as c3
from pe18_common import strata_perm, CK
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

rng = np.random.default_rng(185)
SU = c3.SU; Xz = c3.Xz; FAMS = c3.FAMS; GOODS = c3.GOODS
ms = np.array([c3.main_sys(t) for t in SU])


def cvbits(cols, y, rows, reps=4, C=0.1):
    tot = []
    for r in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=r)
        b = 0.0
        for tr, te in skf.split(rows, y):
            mdl = LogisticRegression(C=C, max_iter=500).fit(Xz[rows[tr]][:, cols], y[tr])
            p = np.clip(mdl.predict_proba(Xz[rows[te]][:, cols])[np.arange(len(te)), y[te]], 1e-4, 1)
            b += -np.log2(p).sum()
        tot.append(b / len(rows))
    return float(np.mean(tot))


def test(a, b, nnull=60, Xplant=None):
    rows = np.where((ms == a) | (ms == b))[0]
    y = (ms[rows] == a).astype(int)
    st = c3.STR[rows]
    res = {'n_' + a: int(y.sum()), 'n_' + b: int((1 - y).sum())}
    p0 = y.mean(); res['prior_bits'] = round(float(-(p0 * np.log2(p0) + (1 - p0) * np.log2(1 - p0))), 4)
    res['goods_bits'] = round(cvbits(GOODS, y, rows), 4)
    for f in ['FMT', 'SITE', 'HDR', 'VAR', 'VOC', 'SEAL']:
        cols = np.concatenate([GOODS, np.where(FAMS == f)[0]])
        g = res['goods_bits'] - cvbits(cols, y, rows)
        nl = []
        for _ in range(nnull):
            y2 = strata_perm(y, st, rng)
            nl.append(cvbits(GOODS, y2, rows, 1) - cvbits(cols, y2, rows, 1))
        nl = np.array(nl)
        res[f] = {'gain': round(g, 4), 'null_mean': round(float(nl.mean()), 4), 'null_sd': round(float(nl.std()), 4),
                  'z': round(float((g - nl.mean()) / (nl.std() + 1e-9)), 2), 'p': float((1 + (nl >= g).sum()) / (nnull + 1))}
        print(a, b, f, res[f], flush=True)
    return res


out = {}
out['DEC_vs_SEX'] = test('DEC', 'SEX')
out['C@_vs_C'] = test('C@', 'C')
# planted: switch a mid-frequency variant form on in 30% of DEC tablets
VARC = np.where(FAMS == 'VAR')[0]
prev = c3.X_all[:, VARC].mean(0)
pc = int(rng.choice(VARC[(prev > 0.03) & (prev < 0.08)]))
dec = np.where(ms == 'DEC')[0]
on = rng.choice(dec, int(0.3 * len(dec)), replace=False)
col = c3.X_all[:, pc].copy(); col[on] = 1
saved = Xz[:, pc].copy(); Xz[:, pc] = (col - col.mean()) / (col.std() + 1e-9)
out['planted_feature'] = c3.NAMES[pc]
pl = test('DEC', 'SEX', 30)
out['planted_DEC_vs_SEX_VAR'] = pl['VAR']
Xz[:, pc] = saved
json.dump(out, open(os.path.join(CK, 'c5.json'), 'w'), indent=1)
