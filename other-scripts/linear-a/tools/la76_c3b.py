"""la76 c3b: magnitude null for the frozen OLE-GRA and VIR-GRA log-value gaps: 40 fits on no-economy data."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import la76_c3 as T
import numpy as np
fz = json.load(open(os.path.join(T.D, 'la76_values.json'))); g = fz['goods']; v = fz['log_value_median_top50']
real = {p: v[g.index(p[0])] - v[g.index(p[1])] for p in T.PRED}
nul = {p: [] for p in T.PRED}
for s in range(40):
    b, _ = T.fit(T.mk.permute(np.random.default_rng(2000 + s)), 3000 + s)
    for p in T.PRED: nul[p].append(b[T.gi[p[0]]] - b[T.gi[p[1]]])
for p in T.PRED:
    a = np.array(nul[p]); print(p, 'real %.2f null mean %.2f sd %.2f P(null>=real) %.3f' % (real[p], a.mean(), a.std(), (1 + (a >= real[p]).sum()) / 41))
