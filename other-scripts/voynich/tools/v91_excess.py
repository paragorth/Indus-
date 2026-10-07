"""Per-mask excess over nulls: ex(X) = X[m] - max over null set of null[m]."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v91_lib as V
def load(nm, pre='c1'):
    d = pickle.load(open(os.path.join(V.CK, '%s_%s.pkl' % (pre, nm)), 'rb'))
    return {(r[3], r[4], r[5], r[6]): r[0] + r[1] for r in d['res']}, d
def excess(X, nulls):
    out = {}
    for m, s in X.items():
        ns = [n.get(m) for n in nulls]
        if any(v is None for v in ns): continue
        out[m] = s - max(ns)
    return out
if __name__ == '__main__':
    targets = sys.argv[1].split(','); nullnames = sys.argv[2].split(',')
    nulls = [load(n)[0] for n in nullnames]
    for t in targets:
        X, d = load(t)
        ex = excess(X, nulls)
        v = np.array(sorted(ex.values(), reverse=True))
        print('==', t, 'vs', nullnames, 'n', len(v), 'max %.1f p99.9 %.1f p99 %.1f n>0 %d' % (v[0], np.percentile(v, 99.9), np.percentile(v, 99), (v > 0).sum()))
        for m, e in sorted(ex.items(), key=lambda kv: -kv[1])[:10]:
            print('   ex %.1f raw %.1f' % (e, X[m]), m)
