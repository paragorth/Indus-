"""v70 cycle 3c: calibrate the cycle-3b anatomy NMI on the real Latin scans (letters as truth, same width
estimator), at theta 0/2.5/4 and K 40; also on V and SV at all thetas, and with a 12-label width-only
'clustering' as floor."""
import sys, os, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v70_c3b as B
from v70_c1 import embed
from v70_lib import CK
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score as NMI


def glists(name, D):
    if name == 'L':
        return [list(r['word']) for r in D['recs']]
    return _orig(name, D)


_orig = B.glyph_lists
B.glyph_lists = glists
out = {}
for name in ('L', 'V', 'SV'):
    for th in (0, 2.5, 4):
        D, U, gl, G2U, spans = B.glyph_units(name, th)
        E, _, _ = embed(name, th)
        lab = MiniBatchKMeans(40, random_state=0, n_init=1, batch_size=4096).fit_predict(E)
        est = [''] * len(U)
        for wi, m in G2U.items():
            for pos_, (_, _, ui) in enumerate(sorted(spans[wi])):
                est[ui] = '+'.join(g for g, mm in zip(gl[wi], m) if mm == pos_) or '0'
        ok = [i for i, t in enumerate(est) if t]
        wd = np.array([U[i][1] for i in ok])
        wl = np.searchsorted(np.quantile(wd, np.linspace(0, 1, 41)[1:-1]), wd)
        single = np.mean([('+' not in est[i]) and est[i] != '0' for i in ok])
        r = {'nmi': float(NMI([est[i] for i in ok], lab[ok])), 'nmi_width_only': float(NMI([est[i] for i in ok], wl)),
             'single_glyph_units': float(single), 'n_labels': len(set(est[i] for i in ok))}
        out[f'{name}_t{th}'] = r
        print(name, th, r, flush=True)
json.dump(out, open(os.path.join(CK, 'c3c.json'), 'w'), indent=1)
