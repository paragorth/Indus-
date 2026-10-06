import numpy as np, v73_lib as L, v72_lib as V, v73_c3 as C3, v73_c1_report as R
P = V.voynich('ZL3b')
for key in ('first', 'last'):
    pp, tr, tab = C3.plant_ptr(P, 'SELFCIT', key)
    C = L.Corpus(pp, 'x'); ti = R.truth_index(C, tr)
    print(key, tab, 'recovery with true table', np.mean(C3.select_ptr(C, key, np.array(tab)) == ti))
