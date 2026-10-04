"""v43 cycle 4: what makes the island, and does it pass the rest of the battery?
(a) Feature direction: for each of the 32 v31 features, does the island (biological quire M, hand 2, chunks whose
    page is in quire M) move from the Voynich rest toward the LANG class mean or toward the GEN class mean?
    Null: 200 random contiguous Voynich stretches of the same number of chunks (outside the island).  Control: a
    planted German island (cycle 1 plant) must move most features toward LANG.
(b) Forgery detectability per 20-page stretch: v21 F3 forger fitted on the whole text, ridge AUC real vs forged
    pages inside the stretch; island vs 8 other contiguous 20-page stretches vs 20-page stretches of the v21 Latin
    and Italian herbals (forger fitted on the whole herbal).
Usage: python3 v43_cycle4.py"""
import sys, time, random
import numpy as np
from collections import Counter
import v43_lib as L
import v31_lib as V
t0 = time.time()
def log(*a): print(round(time.time() - t0), *a, flush=True)

R = V.load('feats_N100.json'); KEYS = sorted(R[0]['F'].keys())
def mean_cls(c):
    X = np.array([[r['F'][k] for k in KEYS] for r in R if r['cls'] == c], float); X[~np.isfinite(X)] = 0
    return X.mean(0), X.std(0)
mL, sL = mean_cls('LANG'); mG, sG = mean_cls('GEN')
def mat(F):
    X = np.array([[f.get(k, 0.0) for k in KEYS] for f in F], float); X[~np.isfinite(X)] = 0; return X

P = L.voynich('ZL3b'); S = L.stream_of_pages(P); cp = [S[i * 100 + 50][1] for i in range(len(S) // 100)]
X = mat(L.load('F_ZL.json'))
isl = np.array([P[cp[j]]['quire'] == 'M' and P[cp[j]]['hand'] == '2' for j in range(len(cp))])
n = int(isl.sum()); idx = np.where(isl)[0]
log('island chunks', n, P[cp[idx[0]]]['id'], P[cp[idx[-1]]]['id'])
axis = (mL - mG)                               # LANG minus GEN class means
sc = 0.5 * (sL + sG); sc[sc == 0] = 1


def toward(sel):
    d = X[sel].mean(0) - X[~sel].mean(0)
    return np.sign(d) == np.sign(axis), d / sc


tw, dz = toward(isl)
res = {'n_chunks': n, 'toward_lang': int(tw.sum()), 'nfeat': len(KEYS)}
# weighted: projection of the shift on the LANG-GEN axis in class-sd units
proj = float((dz * np.sign(axis) * np.abs(axis / sc)).sum() / np.abs(axis / sc).sum())
res['proj'] = proj
rng = random.Random(0); nt, npj = [], []
for _ in range(200):
    while True:
        s = rng.randrange(0, len(cp) - n)
        sel = np.zeros(len(cp), bool); sel[s:s + n] = True
        if not (sel & isl).any(): break
    t, d = toward(sel); nt.append(int(t.sum()))
    npj.append(float((d * np.sign(axis) * np.abs(axis / sc)).sum() / np.abs(axis / sc).sum()))
res['null_toward'] = [float(np.percentile(nt, q)) for q in (5, 50, 95)]
res['null_proj'] = [float(np.percentile(npj, q)) for q in (5, 50, 95)]
res['p_proj'] = float((np.array(npj) >= proj).mean())
# top features of the shift
order = np.argsort(-np.abs(dz))
res['top'] = [(KEYS[i], round(float(dz[i]), 2), 'toward LANG' if tw[i] else 'toward GEN') for i in order[:10]]
# control: planted German (cycle 1, 2400 tokens at chunk 60)
Xp = X.copy(); Xp[60:84] = mat(L.load('F_P_L_msG_Alem_2400_60.json'))
selp = np.zeros(len(cp), bool); selp[60:84] = True
X_save = X; X = Xp
tp, dp = toward(selp)
res['plant_toward'] = int(tp.sum())
res['plant_proj'] = float((dp * np.sign(axis) * np.abs(axis / sc)).sum() / np.abs(axis / sc).sum())
X = X_save
log('features', res)

# (b) forgery AUC per 20-page stretch
import v21_lib as W
def stretch_auc(Pall, pages, seed=0):
    FZ = W.Featurizer(Pall)
    F = W.Forger(Pall, scope='sec', pos=True, name='F3')
    out = []
    for s in (seed, seed + 1):
        Q = F.forge(pages, random.Random(s))
        Xr, keys = FZ.matrix(pages); Xf, _ = FZ.matrix(Q, keys)
        ok = np.isfinite(Xr).all(0) & np.isfinite(Xf).all(0)
        out.append(W.cv_auc(Xr[:, ok], Xf[:, ok], 'ridge', seed=s))
    return float(np.mean(out))

ipages = [i for i, p in enumerate(P) if p['quire'] == 'M' and p['hand'] == '2']
auc = {'island': stretch_auc(P, [P[i] for i in ipages[:20]])}
starts = [s for s in range(0, len(P) - 20, 20) if not set(range(s, s + 20)) & set(ipages)][:8]
auc['others'] = [stretch_auc(P, P[s:s + 20], seed=s) for s in starts]
log('auc voynich', auc)
for nm, fn in (('latin', W.latin_herbal), ('italian', W.italian_herbal)):
    Q = fn()
    auc[nm] = [stretch_auc(Q, Q[s:s + 20], seed=s) for s in range(0, min(len(Q) - 20, 80), 20)]
    log('auc', nm, auc[nm])
res['auc'] = auc
L.save('cycle4.json', res)
log('done')
