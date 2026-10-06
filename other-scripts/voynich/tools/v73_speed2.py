import sys, time, numpy as np, v73_lib as L, v72_lib as V
C=L.Corpus(V.voynich('ZL3b'),'ZL'); w=np.random.randn(L.NF).astype(np.float32)
print(C.F.dtype, C.F.flags['C_CONTIGUOUS'])
for f in [lambda: C.F@w, lambda: np.maximum.reduceat(C.F@w, C.line_start)]:
    t=time.time(); [f() for _ in range(20)]; print((time.time()-t)/20)
