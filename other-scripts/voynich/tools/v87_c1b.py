"""v87 cycle 1b: generator-control power on plants matched to Voynich word length (wl_mean 3.6-5.3, h2 < 2.6)."""
import json, os, numpy as np, v87_lib as L, v87_common as C
R, F, P = C.load_bank('base'); sid = np.array([r['sid'] for r in R]); scale = L.scale_of(F[:, C.TI])
wl = F[:, C.FE.index('wl_mean')]; h2 = F[:, C.FE.index('h2')]
cand = np.where((wl > 3.6) & (wl < 5.3) & (h2 < 2.6))[0]
rs = np.random.default_rng(3); pick = rs.choice(cand, min(100, len(cand)), replace=False)
sh = []
for i in pick:
    _, _, _, lines = C.regenerate(int(R[i]['seed']))
    a = L.abc(F[:, C.TI], P, C.fvec(lines)[C.TI], scale, mask=(sid != sid[i]))
    b = L.abc(F[:, C.TI], P, C.fvec(C.trigram_resynth(lines))[C.TI], scale, mask=(sid != sid[i]))
    sh.append([a['is_list'], b['is_list'], a['ptt'], b['ptt'], P[i, 0], P[i, 1], a['pwl'], b['pwl'], P[i, 2]])
sh = np.array(sh)
o = dict(n_cand=int(len(cand)), n=len(sh), abs_dlist=float(np.abs(sh[:, 1] - sh[:, 0]).mean()), abs_dptt=float(np.abs(sh[:, 3] - sh[:, 2]).mean()),
         r_ptt_real=float(np.corrcoef(sh[:, 2], sh[:, 5])[0, 1]), r_ptt_resynth=float(np.corrcoef(sh[:, 3], sh[:, 5])[0, 1]),
         r_pwl_real=float(np.corrcoef(sh[:, 6], sh[:, 8])[0, 1]), r_pwl_resynth=float(np.corrcoef(sh[:, 7], sh[:, 8])[0, 1]),
         frac_abs_dptt_below_0p07=float((np.abs(sh[:, 3] - sh[:, 2]) < 0.07).mean()))
print(o); json.dump(o, open(os.path.join(L.CK, 'c1b.json'), 'w'), indent=1)
