"""LA-79 cycle 3 control: forward vs backward share-positive under 10 within-site label shuffles and
10 planted universal grammars (fresh hypotheses each), vs the real labels (3 hypothesis seeds)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la79_c2 as C2, la78_engine as E
from la79_c3 import GRP, GS
from la79_common import CK
E.MODE['mode'] = 'prior'
p50 = C2.year0 <= 1950

def fb(y, seed, N=1500):
    rng = np.random.default_rng(seed); Y = [(y, 4)]; r = []
    for i in range(N):
        f, K = C2.gen(rng, C2.FAMS[i % len(C2.FAMS)])
        r.append([E.score(f, Y, p50, ~p50), E.score(f, Y, ~p50, p50)])
    r = np.array(r)
    return [float((r[:, 0] > 0).mean()), float((r[:, 1] > 0).mean())]

out = dict(real=[], lshuf=[], plant=[])
for s in range(3):
    out['real'].append(fb(C2.y1, 100 + s))
for s in range(10):
    rng = np.random.default_rng(200 + s)
    ys = C2.y1.copy()
    for g in GS:
        m = np.where(GRP == g)[0]; ys[m] = ys[rng.permutation(m)]
    out['lshuf'].append(fb(ys, 300 + s))
    LP = np.zeros((len(ys), 4))
    for g in GS:
        c = np.bincount(C2.y1[GRP == g], minlength=4) + 1.0; LP[GRP == g] = np.log(c / c.sum())
    part = rng.integers(0, 3, len(C2.signs)); key = C2.last if s % 2 == 0 else C2.first
    Em = rng.normal(0, 1, (3, 4)); lg = LP + Em[part[key]]
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    yp = (p.cumsum(1) > rng.random((len(ys), 1))).argmax(1)
    out['plant'].append(fb(yp, 400 + s))
    print(s, out['lshuf'][-1], out['plant'][-1], flush=True)
json.dump(out, open(os.path.join(CK, 'c3_fb.json'), 'w'))
for k, v in out.items():
    v = np.array(v); print(k, 'fwd', v[:, 0].round(3).tolist(), 'bwd', v[:, 1].round(3).tolist(), 'bwd-fwd', (v[:, 1] - v[:, 0]).round(3).tolist())
