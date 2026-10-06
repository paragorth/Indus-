"""stats of the TRUE item stream in each plant vs the random-word-per-line baseline (method power check)."""
import sys, numpy as np, v73_lib as L, v72_lib as V, v73_c1_report as R
P = V.voynich('ZL3b')
for which in ('antid', 'apic'):
    for fil in ('MK2', 'SELFCIT'):
        for code in ('VOY', 'NOVEL'):
            pp, tr = L.plant(P, 'FREE', fil, code, which=which)
            C = L.Corpus(pp, 'x'); ti = R.truth_index(C, tr)
            ti = np.where(ti >= 0, ti, C.random_select(5))
            out = []
            for h in (0, 1):
                m = C.half == h; b = L.baseline(C, m); s = L.stream_stats(C, ti, m)
                out.append({k: round(s[k] - b[k], 3) for k in L.STATS})
            # same for a pure-filler corpus: best possible is 0
            print(which, fil, code, 'pages', C.npg, out, flush=True)
