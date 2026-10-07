"""v96 cycle 4: replication of the cycle-3 cross-stratum ratio on the train half (h=0) with the frozen cycle-3 views.
Prediction frozen in data/v96_frozen_c4.json before this run."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v96_lib as L, v96_c3 as C3


def run(n):
    out = os.path.join(L.CK, 'c4_%s.json' % n)
    if os.path.exists(out): return n
    views = json.load(open(C3.FZ))['views']; pages = L.corpus3(n)
    json.dump([C3.stats(pages, v, 0) for v in views], open(out, 'w')); return n


if __name__ == '__main__':
    if sys.argv[1:] == ['report']:
        out = {}
        for n in C3.names() + L.GENS + L.VOY:
            r = json.load(open(os.path.join(L.CK, 'c4_%s.json' % n)))
            g = lambda k: float(np.nanmedian([x[k] - 0.5 if x else np.nan for x in r]))
            rat = float(np.nanmedian([(x['aRC'] - .5) / (x['aCM'] - .5) if x and x['aCM'] - .5 > 0.05 else np.nan for x in r] + [np.nan]))
            out[n] = dict(w=C3.world(n), aRC=g('aRC'), aCM=g('aCM'), aRM=g('aRM'), ratio=rat)
        for n in L.VOY + L.GENS: print(n, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[n].items()})
        for w in ('W2', 'W12', 'W1CK', 'W1', 'W0'):
            x = sorted(round(d['ratio'], 2) for d in out.values() if d['w'] == w and d['aCM'] > 0.1)
            print(w, x)
        json.dump(out, open(os.path.join(L.CK, 'c4_report.json'), 'w'))
        sys.exit()
    from multiprocessing import Pool
    with Pool(2) as P:
        for n in P.imap_unordered(run, C3.names() + L.GENS + L.VOY): pass
    print('done')
