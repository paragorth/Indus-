"""Frame test: for each (positions, features, k>=2) item, contrast = best offset score - second best offset score.
A framed message (k-glyph letters) has one right frame; a frame-free mood or sequence does not."""
import sys, os, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v91_excess as E
def frames(X):
    g = defaultdict(dict)
    for (p, f, k, o), s in X.items():
        if k >= 2: g[(p, f, k)][o] = s
    out = {}
    for it, d in g.items():
        if len(d) < it[2]: continue
        v = sorted(d.values(), reverse=True)
        out[it] = (v[0] - v[1], v[0], max(d, key=d.get))
    return out
if __name__ == '__main__':
    for nm in sys.argv[1].split(','):
        X, d = E.load(nm); F = frames(X)
        c = np.array([v[0] for v in F.values()])
        top = sorted(F.items(), key=lambda kv: -min(kv[1][0], kv[1][1]))[:5]
        print('== %s items %d  contrast max %.1f p99.9 %.1f n>40 %d n>80 %d' % (nm, len(c), c.max(), np.percentile(c, 99.9), (c > 40).sum(), (c > 80).sum()))
        for it, v in top: print('    ', it, 'contrast %.1f best %.1f offset %d' % v)
