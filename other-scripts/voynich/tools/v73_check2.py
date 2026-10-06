import time, numpy as np, v73_lib as L, v72_lib as V
P = V.voynich('ZL3b')
pp, tr = L.plant(P, 'FREE', 'SELFCIT', 'VOY', which='antid', seed=273)
t = time.time(); C = L.extend(L.Corpus(pp, 'x')); print('extend', time.time() - t)
idx = L.icm_select(C, C.half == 0); print('icm', time.time() - t)
w = L.fit_rule(C, idx, C.half == 0); print('fit', time.time() - t)
import v73_c1_report as R
ti = R.truth_index(C, tr)
print('icm recovery disc', np.mean((idx == ti)[C.half[C.line_page] == 0]), 'rule recovery hold', np.mean((C.select(w) == ti)[C.half[C.line_page] == 1]))
print(sorted(zip(np.round(w, 2), L.ALLF))[-6:])
